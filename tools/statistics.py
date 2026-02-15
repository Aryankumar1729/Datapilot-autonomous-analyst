"""
statistics.py — Core Statistical Analysis Engine (Quick Mode)

Production implementation of the LOCKED statistical framework.
Implements: Schema Profiling, Data Quality, Numeric Summary, 
            Categorical Summary, Auto-KPI Detection

All output schemas are FINAL and must not be modified.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CONSTANTS
# =============================================================================

# Column type inference thresholds
HIGH_CARDINALITY_THRESHOLD = 100
HIGH_CARDINALITY_RATIO = 0.5  # If unique values > 50% of rows
ID_COLUMN_PATTERNS = re.compile(
    r"(^id$|_id$|^uuid$|^guid$|^key$|_key$|^index$)", re.IGNORECASE
)
REVENUE_PATTERNS = re.compile(
    r"(revenue|sales|amount|price|value|cost|profit|income|total)", re.IGNORECASE
)
COUNT_PATTERNS = re.compile(
    r"(count|quantity|qty|units|orders|items|number|num_|n_)", re.IGNORECASE
)

# Data quality thresholds
MISSING_FLAG_THRESHOLD = 0.05  # Flag columns with > 5% missing
COMPLETE_ROWS_FLAG_THRESHOLD = 0.80  # Flag if < 80% complete rows
DUPLICATE_FLAG_THRESHOLD = 0.01  # Flag if > 1% duplicates

# Numeric summary thresholds
MIN_NUMERIC_OBSERVATIONS = 10
SKEWNESS_THRESHOLD = 1.0  # |skew| > 1 is notably skewed


# =============================================================================
# TYPE INFERENCE
# =============================================================================

def _infer_column_type(series: pd.Series, row_count: int) -> str:
    """
    Infer semantic type of a column.
    
    Returns one of: "numeric", "categorical", "temporal", "text", "boolean", "id"
    """
    col_name = str(series.name).lower()
    dtype = series.dtype
    nunique = series.nunique()
    
    # Check for ID columns first (by name pattern)
    if ID_COLUMN_PATTERNS.search(col_name):
        return "id"
    
    # Boolean detection
    if dtype == bool or (nunique <= 2 and set(series.dropna().unique()).issubset({0, 1, True, False, "True", "False", "true", "false", "yes", "no", "Yes", "No", "Y", "N"})):
        return "boolean"
    
    # Temporal detection
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "temporal"
    
    # Try parsing as datetime if object type
    if dtype == object:
        sample = series.dropna().head(100)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return "temporal"
            except (ValueError, TypeError):
                pass
    
    # Numeric detection
    if pd.api.types.is_numeric_dtype(dtype):
        # Check if it's actually an ID (high cardinality integers)
        if pd.api.types.is_integer_dtype(dtype):
            if nunique > HIGH_CARDINALITY_THRESHOLD and nunique / row_count > HIGH_CARDINALITY_RATIO:
                return "id"
        return "numeric"
    
    # Categorical vs Text detection
    if dtype == object or pd.api.types.is_categorical_dtype(dtype):
        # High cardinality strings are likely text/IDs
        if nunique > HIGH_CARDINALITY_THRESHOLD or nunique / max(row_count, 1) > HIGH_CARDINALITY_RATIO:
            # Check average string length - long strings are text
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                return "text"
            return "id"
        return "categorical"
    
    # Default fallback
    return "text"


def _get_sample_values(series: pd.Series, n: int = 3) -> list[Any]:
    """Get n non-null sample values from a series."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    samples = non_null.head(n).tolist()
    # Convert numpy types to Python native types for JSON serialization
    return [
        x.item() if hasattr(x, "item") else x 
        for x in samples
    ]


# =============================================================================
# SCHEMA PROFILING
# =============================================================================

def profile_schema(df: pd.DataFrame) -> dict:
    """
    Profile the schema of a DataFrame.
    
    Args:
        df: Input DataFrame (any structure)
        
    Returns:
        dict matching schema_profile specification:
        {
            row_count: int,
            col_count: int,
            columns: list[{name, inferred_type, original_dtype, sample_values}],
            type_summary: {numeric: int, categorical: int, ...}
        }
    """
    if df is None or df.empty:
        return {
            "row_count": 0,
            "col_count": 0,
            "columns": [],
            "type_summary": {
                "numeric": 0,
                "categorical": 0,
                "temporal": 0,
                "text": 0,
                "boolean": 0,
                "id": 0,
            },
        }
    
    row_count = len(df)
    col_count = len(df.columns)
    
    columns = []
    type_counts = {
        "numeric": 0,
        "categorical": 0,
        "temporal": 0,
        "text": 0,
        "boolean": 0,
        "id": 0,
    }
    
    for col in df.columns:
        series = df[col]
        inferred_type = _infer_column_type(series, row_count)
        type_counts[inferred_type] += 1
        
        columns.append({
            "name": str(col),
            "inferred_type": inferred_type,
            "original_dtype": str(series.dtype),
            "sample_values": _get_sample_values(series, 3),
        })
    
    return {
        "row_count": row_count,
        "col_count": col_count,
        "columns": columns,
        "type_summary": type_counts,
    }


# =============================================================================
# DATA QUALITY SCAN
# =============================================================================

def scan_data_quality(df: pd.DataFrame) -> dict:
    """
    Scan DataFrame for data quality issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict matching data_quality specification:
        {
            overall_score: float,
            missing_summary: {total_missing_cells, total_cells, missing_pct},
            column_quality: list[{name, missing_pct, missing_count, is_flagged}],
            complete_rows_pct: float,
            duplicate_rows: int,
            constant_columns: list[str],
            quality_warnings: list[str]
        }
    """
    if df is None or df.empty:
        return {
            "overall_score": 0.0,
            "missing_summary": {
                "total_missing_cells": 0,
                "total_cells": 0,
                "missing_pct": 0.0,
            },
            "column_quality": [],
            "complete_rows_pct": 0.0,
            "duplicate_rows": 0,
            "constant_columns": [],
            "quality_warnings": ["DataFrame is empty"],
        }
    
    row_count = len(df)
    col_count = len(df.columns)
    total_cells = row_count * col_count
    
    # Missing value analysis
    missing_per_col = df.isnull().sum()
    total_missing = missing_per_col.sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
    
    # Column-level quality
    column_quality = []
    for col in df.columns:
        missing_count = int(missing_per_col[col])
        col_missing_pct = (missing_count / row_count * 100) if row_count > 0 else 0.0
        column_quality.append({
            "name": str(col),
            "missing_pct": round(col_missing_pct, 2),
            "missing_count": missing_count,
            "is_flagged": col_missing_pct > MISSING_FLAG_THRESHOLD * 100,
        })
    
    # Complete rows (rows with zero nulls)
    complete_rows = df.dropna().shape[0]
    complete_rows_pct = (complete_rows / row_count * 100) if row_count > 0 else 0.0
    
    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    
    # Constant columns (only 1 unique value)
    constant_columns = [
        str(col) for col in df.columns 
        if df[col].nunique(dropna=True) <= 1
    ]
    
    # Generate warnings
    quality_warnings = []
    
    flagged_cols = [cq["name"] for cq in column_quality if cq["is_flagged"]]
    if flagged_cols:
        quality_warnings.append(
            f"High missing values (>{MISSING_FLAG_THRESHOLD*100:.0f}%) in: {', '.join(flagged_cols[:5])}"
            + (f" and {len(flagged_cols)-5} more" if len(flagged_cols) > 5 else "")
        )
    
    if complete_rows_pct < COMPLETE_ROWS_FLAG_THRESHOLD * 100:
        quality_warnings.append(
            f"Only {complete_rows_pct:.1f}% of rows are complete (no missing values)"
        )
    
    duplicate_pct = (duplicate_rows / row_count * 100) if row_count > 0 else 0.0
    if duplicate_pct > DUPLICATE_FLAG_THRESHOLD * 100:
        quality_warnings.append(
            f"{duplicate_rows} duplicate rows detected ({duplicate_pct:.1f}%)"
        )
    
    if constant_columns:
        quality_warnings.append(
            f"Constant columns (no analytical value): {', '.join(constant_columns[:5])}"
            + (f" and {len(constant_columns)-5} more" if len(constant_columns) > 5 else "")
        )
    
    # Calculate overall score (weighted composite)
    # Components: missing %, complete rows %, duplicates %, constant columns %
    missing_score = max(0, 1 - (missing_pct / 100))
    complete_score = complete_rows_pct / 100
    duplicate_score = max(0, 1 - (duplicate_pct / 100) * 10)  # Penalize duplicates heavily
    constant_score = max(0, 1 - (len(constant_columns) / max(col_count, 1)))
    
    overall_score = (
        missing_score * 0.35 +
        complete_score * 0.35 +
        duplicate_score * 0.15 +
        constant_score * 0.15
    )
    
    return {
        "overall_score": round(overall_score, 3),
        "missing_summary": {
            "total_missing_cells": int(total_missing),
            "total_cells": total_cells,
            "missing_pct": round(missing_pct, 2),
        },
        "column_quality": column_quality,
        "complete_rows_pct": round(complete_rows_pct, 2),
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "quality_warnings": quality_warnings,
    }


# =============================================================================
# NUMERIC SUMMARY
# =============================================================================

def _interpret_skewness(skew: float) -> str:
    """Interpret skewness value."""
    if abs(skew) <= SKEWNESS_THRESHOLD:
        return "symmetric"
    elif skew > SKEWNESS_THRESHOLD:
        return "right-skewed"
    else:
        return "left-skewed"


def numeric_summary(df: pd.DataFrame) -> list[dict]:
    """
    Generate univariate summary statistics for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        list of dicts matching numeric_summary specification:
        [{
            column: str,
            mean: float,
            median: float,
            std: float,
            min: float,
            max: float,
            skewness: float,
            skew_interpretation: str
        }, ...]
    """
    if df is None or df.empty:
        return []
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summaries = []
    for col in numeric_cols:
        series = df[col].dropna()
        
        # Skip if insufficient observations
        if len(series) < MIN_NUMERIC_OBSERVATIONS:
            continue
        
        # Skip if constant (std = 0)
        if series.nunique() <= 1:
            continue
        
        # Compute statistics
        try:
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std())
            min_val = float(series.min())
            max_val = float(series.max())
            skewness_val = float(stats.skew(series, nan_policy="omit"))
        except (ValueError, TypeError):
            # Skip columns that fail computation
            continue
        
        # Handle edge cases (inf, nan)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        if not np.isfinite(std_val):
            std_val = 0.0
        if not np.isfinite(skewness_val):
            skewness_val = 0.0
        
        summaries.append({
            "column": str(col),
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "skewness": round(skewness_val, 4),
            "skew_interpretation": _interpret_skewness(skewness_val),
        })
    
    return summaries


# =============================================================================
# CATEGORICAL SUMMARY
# =============================================================================

def categorical_summary(df: pd.DataFrame) -> list[dict]:
    """
    Generate frequency summary for categorical columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        list of dicts matching categorical_summary specification:
        [{
            column: str,
            cardinality: int,
            is_high_cardinality: bool,
            top_values: list[{value, count, pct}],
            other_pct: float,
            concentration: str
        }, ...]
    """
    if df is None or df.empty:
        return []
    
    row_count = len(df)
    summaries = []
    
    # Select object/categorical columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    for col in cat_cols:
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        cardinality = series.nunique()
        
        # Skip high-cardinality columns (likely IDs or text)
        is_high_cardinality = (
            cardinality > HIGH_CARDINALITY_THRESHOLD or 
            cardinality / max(row_count, 1) > HIGH_CARDINALITY_RATIO
        )
        
        if is_high_cardinality:
            # Still include in summary but mark it
            summaries.append({
                "column": str(col),
                "cardinality": cardinality,
                "is_high_cardinality": True,
                "top_values": [],
                "other_pct": 100.0,
                "concentration": "dispersed",
            })
            continue
        
        # Compute value counts
        value_counts = series.value_counts()
        total = len(series)
        
        # Top 5 values
        top_n = min(5, len(value_counts))
        top_values = []
        top_sum = 0
        
        for i, (val, count) in enumerate(value_counts.head(top_n).items()):
            pct = (count / total * 100)
            top_sum += count
            top_values.append({
                "value": str(val),
                "count": int(count),
                "pct": round(pct, 2),
            })
        
        other_pct = ((total - top_sum) / total * 100) if total > 0 else 0.0
        
        # Determine concentration
        top_1_pct = top_values[0]["pct"] if top_values else 0
        if top_1_pct > 50:
            concentration = "high"
        elif top_1_pct > 25:
            concentration = "moderate"
        else:
            concentration = "dispersed"
        
        summaries.append({
            "column": str(col),
            "cardinality": cardinality,
            "is_high_cardinality": False,
            "top_values": top_values,
            "other_pct": round(other_pct, 2),
            "concentration": concentration,
        })
    
    return summaries


# =============================================================================
# AUTO-KPI DETECTION
# =============================================================================

def _format_number(value: float | int, is_currency: bool = False) -> str:
    """Format a number for display."""
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    
    abs_val = abs(value)
    prefix = "$" if is_currency else ""
    
    if abs_val >= 1_000_000_000:
        formatted = f"{value/1_000_000_000:.1f}B"
    elif abs_val >= 1_000_000:
        formatted = f"{value/1_000_000:.1f}M"
    elif abs_val >= 1_000:
        formatted = f"{value/1_000:.1f}K"
    elif isinstance(value, float):
        formatted = f"{value:.2f}"
    else:
        formatted = f"{value:,}"
    
    return f"{prefix}{formatted}"


def detect_auto_kpis(df: pd.DataFrame, schema_profile: dict) -> list[dict]:
    """
    Automatically detect and compute KPIs from the DataFrame.
    
    Args:
        df: Input DataFrame
        schema_profile: Output from profile_schema()
        
    Returns:
        list of dicts matching auto_kpis specification:
        [{
            name: str,
            value: float | int | str,
            formatted_value: str,
            source_column: str,
            computation: str,
            confidence: float
        }, ...]
    """
    if df is None or df.empty or not schema_profile:
        return []
    
    kpis = []
    columns_info = {c["name"]: c for c in schema_profile.get("columns", [])}
    row_count = schema_profile.get("row_count", len(df))
    
    # Always add total records KPI
    kpis.append({
        "name": "Total Records",
        "value": row_count,
        "formatted_value": _format_number(row_count),
        "source_column": "_rows_",
        "computation": "count",
        "confidence": 1.0,
    })
    
    # Detect revenue/value columns
    for col_info in schema_profile.get("columns", []):
        col_name = col_info["name"]
        col_type = col_info["inferred_type"]
        
        if col_type != "numeric":
            continue
        
        series = df[col_name].dropna()
        if len(series) == 0:
            continue
        
        # Check for revenue patterns
        if REVENUE_PATTERNS.search(col_name):
            total_val = float(series.sum())
            avg_val = float(series.mean())
            
            # Determine if it looks like currency (positive values, reasonable range)
            is_currency = series.min() >= 0 and "price" in col_name.lower() or "revenue" in col_name.lower() or "amount" in col_name.lower()
            
            # Title case the column name for display
            display_name = col_name.replace("_", " ").title()
            
            kpis.append({
                "name": f"Total {display_name}",
                "value": round(total_val, 2),
                "formatted_value": _format_number(total_val, is_currency),
                "source_column": col_name,
                "computation": "sum",
                "confidence": 0.85,
            })
            
            kpis.append({
                "name": f"Avg {display_name}",
                "value": round(avg_val, 2),
                "formatted_value": _format_number(avg_val, is_currency),
                "source_column": col_name,
                "computation": "mean",
                "confidence": 0.80,
            })
            continue
        
        # Check for count patterns
        if COUNT_PATTERNS.search(col_name):
            total_val = float(series.sum())
            display_name = col_name.replace("_", " ").title()
            
            kpis.append({
                "name": f"Total {display_name}",
                "value": int(total_val),
                "formatted_value": _format_number(int(total_val)),
                "source_column": col_name,
                "computation": "sum",
                "confidence": 0.80,
            })
    
    # Detect date range from temporal columns
    for col_info in schema_profile.get("columns", []):
        col_name = col_info["name"]
        col_type = col_info["inferred_type"]
        
        if col_type != "temporal":
            continue
        
        try:
            # Try to parse as datetime
            date_series = pd.to_datetime(df[col_name], errors="coerce").dropna()
            if len(date_series) == 0:
                continue
            
            min_date = date_series.min()
            max_date = date_series.max()
            span_days = (max_date - min_date).days
            
            date_range_str = f"{min_date.strftime('%b %Y')} – {max_date.strftime('%b %Y')}"
            
            kpis.append({
                "name": "Date Range",
                "value": date_range_str,
                "formatted_value": date_range_str,
                "source_column": col_name,
                "computation": "range",
                "confidence": 0.95,
            })
            
            kpis.append({
                "name": "Time Span",
                "value": span_days,
                "formatted_value": f"{span_days} days",
                "source_column": col_name,
                "computation": "range",
                "confidence": 0.95,
            })
            break  # Only use first temporal column
        except (ValueError, TypeError):
            continue
    
    # Detect unique segment counts from categorical columns
    for col_info in schema_profile.get("columns", []):
        col_name = col_info["name"]
        col_type = col_info["inferred_type"]
        
        if col_type != "categorical":
            continue
        
        cardinality = df[col_name].nunique()
        
        # Only include low-cardinality categoricals as segment KPIs
        if 2 <= cardinality <= 50:
            display_name = col_name.replace("_", " ").title()
            
            kpis.append({
                "name": f"Unique {display_name}s",
                "value": cardinality,
                "formatted_value": str(cardinality),
                "source_column": col_name,
                "computation": "count",
                "confidence": 0.70,
            })
    
    # Sort by confidence (highest first) and limit to reasonable number
    kpis.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Deduplicate by name (keep highest confidence)
    seen_names = set()
    unique_kpis = []
    for kpi in kpis:
        if kpi["name"] not in seen_names:
            seen_names.add(kpi["name"])
            unique_kpis.append(kpi)
    
    return unique_kpis[:10]  # Cap at 10 KPIs


# =============================================================================
# QUICK MODE ORCHESTRATOR
# =============================================================================

def run_quick_analysis(df: pd.DataFrame) -> dict:
    """
    Run all Quick mode analyses in optimal order.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict containing all Quick mode outputs:
        {
            schema_profile: dict,
            data_quality: dict,
            numeric_summary: list[dict],
            categorical_summary: list[dict],
            auto_kpis: list[dict],
            warnings: list[str],
            skipped_analyses: list[dict]
        }
    """
    warnings = []
    skipped = []
    
    # 1. Schema profiling (must run first - others depend on it)
    try:
        schema = profile_schema(df)
    except Exception as e:
        warnings.append(f"Schema profiling failed: {str(e)}")
        schema = profile_schema(pd.DataFrame())  # Empty fallback
        skipped.append({"name": "schema_profile", "reason": str(e)})
    
    # 2. Data quality scan
    try:
        quality = scan_data_quality(df)
        warnings.extend(quality.get("quality_warnings", []))
    except Exception as e:
        warnings.append(f"Data quality scan failed: {str(e)}")
        quality = scan_data_quality(pd.DataFrame())
        skipped.append({"name": "data_quality", "reason": str(e)})
    
    # 3. Numeric summary (depends on having numeric columns)
    try:
        num_summary = numeric_summary(df)
        if not num_summary and schema.get("type_summary", {}).get("numeric", 0) > 0:
            warnings.append("Numeric columns exist but summary could not be computed")
    except Exception as e:
        warnings.append(f"Numeric summary failed: {str(e)}")
        num_summary = []
        skipped.append({"name": "numeric_summary", "reason": str(e)})
    
    # 4. Categorical summary
    try:
        cat_summary = categorical_summary(df)
    except Exception as e:
        warnings.append(f"Categorical summary failed: {str(e)}")
        cat_summary = []
        skipped.append({"name": "categorical_summary", "reason": str(e)})
    
    # 5. Auto-KPI detection (depends on schema profile)
    try:
        kpis = detect_auto_kpis(df, schema)
    except Exception as e:
        warnings.append(f"Auto-KPI detection failed: {str(e)}")
        kpis = []
        skipped.append({"name": "auto_kpis", "reason": str(e)})
    
    return {
        "schema_profile": schema,
        "data_quality": quality,
        "numeric_summary": num_summary,
        "categorical_summary": cat_summary,
        "auto_kpis": kpis,
        "warnings": warnings,
        "skipped_analyses": skipped,
    }


# =============================================================================
# =============================================================================
#                         STANDARD MODE ANALYTICS
# =============================================================================
# =============================================================================

# Standard mode thresholds
MIN_OBSERVATIONS_NORMALITY = 30
MIN_OBSERVATIONS_CORRELATION = 30
MIN_TEMPORAL_PERIODS = 2
MIN_SEGMENT_SIZE = 20
CORRELATION_STRONG_THRESHOLD = 0.7
CORRELATION_MODERATE_THRESHOLD = 0.5
CORRELATION_WEAK_THRESHOLD = 0.3
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0


# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================

def _interpret_kurtosis(kurt: float) -> str:
    """Interpret kurtosis value."""
    if kurt > 1:
        return "heavy-tailed"
    elif kurt < -1:
        return "light-tailed"
    return "normal-tailed"


def _classify_distribution(skewness: float, kurtosis: float, is_normal: bool) -> str:
    """Classify distribution type based on shape statistics."""
    if is_normal:
        return "normal"
    
    if abs(skewness) <= 0.5 and abs(kurtosis) <= 1:
        return "normal"
    elif skewness > 1:
        return "right-skewed"
    elif skewness < -1:
        return "left-skewed"
    elif kurtosis > 3:
        return "heavy-tailed"
    elif abs(skewness) <= 0.5:
        return "symmetric"
    else:
        return "skewed"


def distribution_analysis(df: pd.DataFrame) -> list[dict]:
    """
    Perform distribution analysis on numeric columns.
    
    Computes quartiles, IQR, percentiles, kurtosis, and normality tests.
    
    Args:
        df: Input DataFrame
        
    Returns:
        list of dicts matching distributions specification:
        [{
            column: str,
            quartiles: {q1, q2, q3},
            iqr: float,
            percentiles: {p5, p95},
            kurtosis: float,
            is_normal: bool,
            normality_p_value: float,
            distribution_type: str
        }, ...]
    """
    if df is None or df.empty:
        return []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    distributions = []
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        # Need minimum observations for meaningful analysis
        if len(series) < MIN_OBSERVATIONS_NORMALITY:
            continue
        
        # Skip constant columns
        if series.nunique() <= 1:
            continue
        
        try:
            # Quartiles
            q1 = float(series.quantile(0.25))
            q2 = float(series.quantile(0.50))  # median
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            
            # Percentiles
            p5 = float(series.quantile(0.05))
            p95 = float(series.quantile(0.95))
            
            # Kurtosis (Fisher's definition, normal = 0)
            kurt = float(stats.kurtosis(series, nan_policy="omit"))
            
            # Skewness for classification
            skewness = float(stats.skew(series, nan_policy="omit"))
            
            # Normality test
            # Use Shapiro-Wilk for n < 5000, D'Agostino-Pearson for larger
            if len(series) < 5000:
                # Shapiro-Wilk (sample if too large)
                sample = series.sample(min(len(series), 5000), random_state=42)
                _, p_value = stats.shapiro(sample)
            else:
                # D'Agostino-Pearson
                _, p_value = stats.normaltest(series)
            
            is_normal = p_value > 0.05
            
            # Handle non-finite values
            if not np.isfinite(kurt):
                kurt = 0.0
            if not np.isfinite(p_value):
                p_value = 0.0
                is_normal = False
            
            distribution_type = _classify_distribution(skewness, kurt, is_normal)
            
            distributions.append({
                "column": str(col),
                "quartiles": {
                    "q1": round(q1, 4),
                    "q2": round(q2, 4),
                    "q3": round(q3, 4),
                },
                "iqr": round(iqr, 4),
                "percentiles": {
                    "p5": round(p5, 4),
                    "p95": round(p95, 4),
                },
                "kurtosis": round(kurt, 4),
                "is_normal": is_normal,
                "normality_p_value": round(p_value, 6),
                "distribution_type": distribution_type,
            })
            
        except (ValueError, TypeError, RuntimeWarning):
            continue
    
    return distributions


# =============================================================================
# TEMPORAL TRENDS
# =============================================================================

def _detect_time_grain(date_series: pd.Series) -> str:
    """Detect the granularity of a datetime series."""
    if len(date_series) < 2:
        return "unknown"
    
    # Sort and compute differences
    sorted_dates = date_series.sort_values()
    diffs = sorted_dates.diff().dropna()
    
    if len(diffs) == 0:
        return "unknown"
    
    # Median difference in days
    median_diff = diffs.dt.total_seconds().median() / 86400  # Convert to days
    
    if median_diff <= 1.5:
        return "daily"
    elif median_diff <= 8:
        return "weekly"
    elif median_diff <= 35:
        return "monthly"
    elif median_diff <= 100:
        return "quarterly"
    else:
        return "yearly"


def _compute_trend_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Compute linear trend slope using OLS regression.
    
    Returns:
        (slope, slope_pct_per_period, p_value)
    """
    if len(x) < 3:
        return 0.0, 0.0, 1.0
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate percentage change per period
        y_mean = np.mean(y)
        if y_mean != 0:
            slope_pct = (slope / y_mean) * 100
        else:
            slope_pct = 0.0
        
        return float(slope), float(slope_pct), float(p_value)
    except (ValueError, RuntimeWarning):
        return 0.0, 0.0, 1.0


def temporal_trends(df: pd.DataFrame) -> dict | None:
    """
    Analyze temporal trends in the data.
    
    Detects time grain, computes period-over-period changes, and trend direction.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict matching temporal_analysis specification or None if no temporal data:
        {
            date_column: str,
            detected_grain: str,
            date_range: {start, end, span_days},
            trends: list[{metric_column, aggregation, direction, slope, ...}],
            seasonality: {detected, pattern, coefficient_of_variation}
        }
    """
    if df is None or df.empty:
        return None
    
    # Find temporal columns
    temporal_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            temporal_cols.append(col)
        else:
            # Try parsing
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    temporal_cols.append(col)
            except (ValueError, TypeError):
                continue
    
    if not temporal_cols:
        return None
    
    # Use first temporal column
    date_col = temporal_cols[0]
    date_series = pd.to_datetime(df[date_col], errors="coerce").dropna()
    
    if len(date_series) < MIN_TEMPORAL_PERIODS:
        return None
    
    # Detect grain
    detected_grain = _detect_time_grain(date_series)
    
    # Date range
    min_date = date_series.min()
    max_date = date_series.max()
    span_days = (max_date - min_date).days
    
    # Find numeric columns for trend analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    trends = []
    for num_col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        try:
            # Create time-indexed series
            temp_df = df[[date_col, num_col]].dropna()
            if len(temp_df) < MIN_TEMPORAL_PERIODS:
                continue
            
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
            temp_df = temp_df.dropna()
            
            # Resample based on grain
            resample_map = {
                "daily": "D",
                "weekly": "W",
                "monthly": "M",
                "quarterly": "Q",
                "yearly": "Y",
            }
            resample_freq = resample_map.get(detected_grain, "M")
            
            # Aggregate by time period
            temp_df = temp_df.set_index(date_col)
            aggregated = temp_df[num_col].resample(resample_freq).sum()
            aggregated = aggregated.dropna()
            
            if len(aggregated) < MIN_TEMPORAL_PERIODS:
                continue
            
            # Compute trend
            x = np.arange(len(aggregated))
            y = aggregated.values
            
            slope, slope_pct, p_value = _compute_trend_slope(x, y)
            
            # Determine direction
            if p_value > 0.05:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
            
            # Period over period comparison
            if len(aggregated) >= 2:
                current_val = float(aggregated.iloc[-1])
                previous_val = float(aggregated.iloc[-2])
                
                if previous_val != 0:
                    change_pct = ((current_val - previous_val) / previous_val) * 100
                else:
                    change_pct = 0.0
                
                pop = {
                    "current_period": str(aggregated.index[-1].date()),
                    "previous_period": str(aggregated.index[-2].date()),
                    "current_value": round(current_val, 2),
                    "previous_value": round(previous_val, 2),
                    "change_pct": round(change_pct, 2),
                }
            else:
                pop = None
            
            trends.append({
                "metric_column": str(num_col),
                "aggregation": "sum",
                "direction": direction,
                "slope": round(slope, 4),
                "slope_pct": round(slope_pct, 2),
                "p_value": round(p_value, 6),
                "period_over_period": pop,
            })
            
        except (ValueError, TypeError, KeyError):
            continue
    
    # Seasonality detection (simple coefficient of variation by period)
    seasonality = {"detected": False, "pattern": None, "coefficient_of_variation": 0.0}
    
    if detected_grain in ["daily", "weekly", "monthly"] and len(date_series) >= 12:
        try:
            # Group by month and compute CV
            temp_df = df[[date_col, numeric_cols[0]]].dropna() if numeric_cols else None
            if temp_df is not None and len(temp_df) > 0:
                temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
                temp_df["month"] = temp_df[date_col].dt.month
                monthly_means = temp_df.groupby("month")[numeric_cols[0]].mean()
                
                if len(monthly_means) > 0 and monthly_means.mean() != 0:
                    cv = (monthly_means.std() / monthly_means.mean()) * 100
                    
                    if cv > 20:  # Significant variation
                        seasonality["detected"] = True
                        seasonality["coefficient_of_variation"] = round(float(cv), 2)
                        
                        # Find peak period
                        peak_month = monthly_means.idxmax()
                        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        if 1 <= peak_month <= 12:
                            seasonality["pattern"] = f"{month_names[peak_month-1]} typically highest"
        except (ValueError, TypeError, KeyError):
            pass
    
    return {
        "date_column": str(date_col),
        "detected_grain": detected_grain,
        "date_range": {
            "start": str(min_date.date()),
            "end": str(max_date.date()),
            "span_days": span_days,
        },
        "trends": trends,
        "seasonality": seasonality,
    }


# =============================================================================
# CATEGORICAL SEGMENTATION
# =============================================================================

def categorical_segmentation(df: pd.DataFrame) -> list[dict]:
    """
    Analyze numeric KPIs across categorical segments.
    
    Computes segment means, contributions, and identifies top/bottom performers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        list of dicts matching segmentation specification:
        [{
            segment_column: str,
            kpi_column: str,
            aggregation: str,
            segments: list[{value, count, count_pct, kpi_value, kpi_contribution_pct}],
            insights: {top_performer, bottom_performer, performance_spread, concentration_index}
        }, ...]
    """
    if df is None or df.empty:
        return []
    
    # Find categorical columns (low to medium cardinality)
    categorical_cols = []
    row_count = len(df)
    
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        
        # Include if categorical and cardinality between 2 and 20
        if (dtype == object or pd.api.types.is_categorical_dtype(dtype)) and 2 <= nunique <= 20:
            categorical_cols.append(col)
    
    if not categorical_cols:
        return []
    
    # Find numeric KPI columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to likely KPI columns (using revenue/count patterns)
    kpi_cols = [col for col in numeric_cols if REVENUE_PATTERNS.search(col) or COUNT_PATTERNS.search(col)]
    
    # If no pattern matches, use first numeric column
    if not kpi_cols and numeric_cols:
        kpi_cols = numeric_cols[:1]
    
    if not kpi_cols:
        return []
    
    segmentations = []
    
    for cat_col in categorical_cols[:3]:  # Limit to 3 categorical columns
        for kpi_col in kpi_cols[:2]:  # Limit to 2 KPI columns
            try:
                # Group by segment
                grouped = df.groupby(cat_col, dropna=True)[kpi_col].agg(["sum", "count", "mean"])
                grouped = grouped.reset_index()
                
                if len(grouped) < 2:
                    continue
                
                total_kpi = grouped["sum"].sum()
                total_count = grouped["count"].sum()
                
                if total_kpi == 0 or total_count == 0:
                    continue
                
                segments = []
                for _, row in grouped.iterrows():
                    segments.append({
                        "value": str(row[cat_col]),
                        "count": int(row["count"]),
                        "count_pct": round((row["count"] / total_count) * 100, 2),
                        "kpi_value": round(float(row["sum"]), 2),
                        "kpi_contribution_pct": round((row["sum"] / total_kpi) * 100, 2),
                    })
                
                # Sort by KPI value descending
                segments.sort(key=lambda x: x["kpi_value"], reverse=True)
                
                # Insights
                top_performer = {
                    "value": segments[0]["value"],
                    "kpi_value": segments[0]["kpi_value"],
                }
                bottom_performer = {
                    "value": segments[-1]["value"],
                    "kpi_value": segments[-1]["kpi_value"],
                }
                
                # Performance spread (ratio of top to bottom)
                if bottom_performer["kpi_value"] != 0:
                    spread = top_performer["kpi_value"] / bottom_performer["kpi_value"]
                else:
                    spread = float("inf")
                
                # Concentration index (HHI-like)
                shares = [s["kpi_contribution_pct"] / 100 for s in segments]
                hhi = sum(s ** 2 for s in shares)
                
                segmentations.append({
                    "segment_column": str(cat_col),
                    "kpi_column": str(kpi_col),
                    "aggregation": "sum",
                    "segments": segments[:10],  # Limit to top 10 segments
                    "insights": {
                        "top_performer": top_performer,
                        "bottom_performer": bottom_performer,
                        "performance_spread": round(spread, 2) if np.isfinite(spread) else None,
                        "concentration_index": round(hhi, 4),
                    },
                })
                
            except (ValueError, TypeError, KeyError):
                continue
    
    return segmentations


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def _interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient strength."""
    abs_r = abs(r)
    
    if abs_r >= CORRELATION_STRONG_THRESHOLD:
        prefix = "strong"
    elif abs_r >= CORRELATION_MODERATE_THRESHOLD:
        prefix = "moderate"
    elif abs_r >= CORRELATION_WEAK_THRESHOLD:
        prefix = "weak"
    else:
        return "none"
    
    suffix = "positive" if r > 0 else "negative"
    return f"{prefix}_{suffix}"


def correlation_analysis(df: pd.DataFrame) -> dict | None:
    """
    Compute correlation matrix and identify significant correlations.
    
    Uses Pearson correlation with significance testing.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict matching correlations specification or None:
        {
            matrix: dict,
            significant_pairs: list[{column_a, column_b, correlation, p_value, strength, interpretation}],
            top_correlations: list,
            multicollinearity_warning: bool
        }
    """
    if df is None or df.empty:
        return None
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()
    
    # Need at least 2 columns and sufficient observations
    if len(numeric_cols) < 2 or len(df) < MIN_OBSERVATIONS_CORRELATION:
        return None
    
    # Limit columns to prevent excessive computation
    numeric_cols = numeric_cols[:15]
    numeric_df = numeric_df[numeric_cols]
    
    # Remove constant columns
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
    numeric_cols = numeric_df.columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    # Compute correlation matrix
    try:
        corr_matrix = numeric_df.corr(method="pearson")
    except (ValueError, TypeError):
        return None
    
    # Convert matrix to dict format
    matrix_dict = {}
    significant_pairs = []
    multicollinearity = False
    
    for i, col_a in enumerate(numeric_cols):
        for j, col_b in enumerate(numeric_cols):
            if i >= j:  # Skip diagonal and duplicates
                continue
            
            r = corr_matrix.loc[col_a, col_b]
            
            if not np.isfinite(r):
                continue
            
            # Store in matrix
            key = f"{col_a}|{col_b}"
            matrix_dict[key] = round(r, 4)
            
            # Compute p-value
            try:
                valid_mask = numeric_df[[col_a, col_b]].notna().all(axis=1)
                n = valid_mask.sum()
                
                if n < 3:
                    p_value = 1.0
                else:
                    _, p_value = stats.pearsonr(
                        numeric_df.loc[valid_mask, col_a],
                        numeric_df.loc[valid_mask, col_b]
                    )
            except (ValueError, TypeError):
                p_value = 1.0
            
            strength = _interpret_correlation(r)
            
            # Check for multicollinearity
            if abs(r) > 0.9:
                multicollinearity = True
            
            # Add significant pairs
            if abs(r) >= CORRELATION_WEAK_THRESHOLD and p_value < 0.05:
                interpretation = f"{col_a} and {col_b} show {strength.replace('_', ' ')} correlation"
                
                significant_pairs.append({
                    "column_a": str(col_a),
                    "column_b": str(col_b),
                    "correlation": round(r, 4),
                    "p_value": round(p_value, 6),
                    "strength": strength,
                    "interpretation": interpretation,
                })
    
    # Sort by absolute correlation
    significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    # Top correlations
    top_correlations = significant_pairs[:5]
    
    return {
        "matrix": matrix_dict,
        "significant_pairs": significant_pairs,
        "top_correlations": top_correlations,
        "multicollinearity_warning": multicollinearity,
    }


# =============================================================================
# BASIC ANOMALY DETECTION
# =============================================================================

def basic_anomaly_detection(df: pd.DataFrame) -> dict | None:
    """
    Detect outliers and suspicious values using IQR and Z-score methods.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict matching anomalies specification or None:
        {
            outlier_summary: {total_outliers, outlier_pct},
            column_outliers: list[{column, outlier_count, outlier_pct, lower_bound, upper_bound, outlier_examples, direction}],
            suspicious_values: list[{column, issue, count, examples}]
        }
    """
    if df is None or df.empty:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return None
    
    total_outliers = 0
    column_outliers = []
    suspicious_values = []
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        if len(series) < MIN_OBSERVATIONS_NORMALITY:
            continue
        
        if series.nunique() <= 1:
            continue
        
        try:
            # IQR method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - IQR_MULTIPLIER * iqr
            upper_bound = q3 + IQR_MULTIPLIER * iqr
            
            # Find outliers
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers_mask.sum()
            outlier_pct = (outlier_count / len(series)) * 100
            
            total_outliers += outlier_count
            
            if outlier_count > 0:
                # Determine direction
                low_outliers = (series < lower_bound).sum()
                high_outliers = (series > upper_bound).sum()
                
                if low_outliers > 0 and high_outliers > 0:
                    direction = "both"
                elif high_outliers > 0:
                    direction = "high"
                else:
                    direction = "low"
                
                # Get examples (most extreme values)
                outlier_values = series[outliers_mask].sort_values()
                examples = []
                
                # Get 3 most extreme (low and high)
                if direction in ["low", "both"]:
                    for idx, val in outlier_values.head(2).items():
                        examples.append({"row_index": int(idx), "value": round(float(val), 4)})
                
                if direction in ["high", "both"]:
                    for idx, val in outlier_values.tail(2).items():
                        if {"row_index": int(idx), "value": round(float(val), 4)} not in examples:
                            examples.append({"row_index": int(idx), "value": round(float(val), 4)})
                
                examples = examples[:3]  # Limit to 3
                
                column_outliers.append({
                    "column": str(col),
                    "outlier_count": int(outlier_count),
                    "outlier_pct": round(outlier_pct, 2),
                    "lower_bound": round(float(lower_bound), 4),
                    "upper_bound": round(float(upper_bound), 4),
                    "outlier_examples": examples,
                    "direction": direction,
                })
            
            # Check for suspicious values
            # Unexpected zeros in likely non-zero columns
            if REVENUE_PATTERNS.search(col) or COUNT_PATTERNS.search(col):
                zero_count = (series == 0).sum()
                if zero_count > 0 and (zero_count / len(series)) < 0.5:  # Not majority zeros
                    suspicious_values.append({
                        "column": str(col),
                        "issue": "unexpected_zeros",
                        "count": int(zero_count),
                        "examples": [0],
                    })
            
            # Unexpected negatives in typically-positive columns
            if REVENUE_PATTERNS.search(col) and not any(x in col.lower() for x in ["change", "diff", "delta", "growth"]):
                negative_count = (series < 0).sum()
                if negative_count > 0:
                    negative_examples = series[series < 0].head(3).tolist()
                    suspicious_values.append({
                        "column": str(col),
                        "issue": "unexpected_negatives",
                        "count": int(negative_count),
                        "examples": [round(float(x), 2) for x in negative_examples],
                    })
            
        except (ValueError, TypeError):
            continue
    
    if not column_outliers and not suspicious_values:
        return None
    
    row_count = len(df)
    outlier_pct = (total_outliers / (row_count * len(numeric_cols))) * 100 if row_count > 0 else 0
    
    return {
        "outlier_summary": {
            "total_outliers": total_outliers,
            "outlier_pct": round(outlier_pct, 2),
        },
        "column_outliers": column_outliers,
        "suspicious_values": suspicious_values,
    }


# =============================================================================
# STANDARD MODE ORCHESTRATOR
# =============================================================================

def run_standard_analysis(df: pd.DataFrame) -> dict:
    """
    Run all Standard mode analyses (includes Quick mode + Standard extensions).
    
    Standard mode builds on Quick mode by adding:
    - Distribution analysis (quartiles, normality tests)
    - Temporal trends (time series patterns)
    - Categorical segmentation (segment performance)
    - Correlation analysis (relationship detection)
    - Basic anomaly detection (outliers)
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict containing all Quick + Standard mode outputs:
        {
            # Quick mode outputs
            schema_profile: dict,
            data_quality: dict,
            numeric_summary: list[dict],
            categorical_summary: list[dict],
            auto_kpis: list[dict],
            
            # Standard mode outputs
            distributions: list[dict] | None,
            temporal_analysis: dict | None,
            segmentation: list[dict] | None,
            correlations: dict | None,
            anomalies: dict | None,
            
            # Meta
            warnings: list[str],
            skipped_analyses: list[dict]
        }
    """
    # Start with Quick mode analysis
    result = run_quick_analysis(df)
    
    warnings = result.get("warnings", [])
    skipped = result.get("skipped_analyses", [])
    
    # Add Standard mode analyses
    
    # 1. Distribution analysis
    try:
        distributions = distribution_analysis(df)
        if not distributions:
            warnings.append("Distribution analysis skipped: insufficient numeric data")
    except Exception as e:
        warnings.append(f"Distribution analysis failed: {str(e)}")
        distributions = []
        skipped.append({"name": "distributions", "reason": str(e)})
    
    # 2. Temporal trends
    try:
        temporal = temporal_trends(df)
        if temporal is None:
            warnings.append("Temporal analysis skipped: no temporal columns detected")
    except Exception as e:
        warnings.append(f"Temporal analysis failed: {str(e)}")
        temporal = None
        skipped.append({"name": "temporal_analysis", "reason": str(e)})
    
    # 3. Categorical segmentation
    try:
        segmentation = categorical_segmentation(df)
        if not segmentation:
            warnings.append("Segmentation analysis skipped: no suitable categorical/numeric pairs")
    except Exception as e:
        warnings.append(f"Segmentation analysis failed: {str(e)}")
        segmentation = []
        skipped.append({"name": "segmentation", "reason": str(e)})
    
    # 4. Correlation analysis
    try:
        correlations = correlation_analysis(df)
        if correlations is None:
            warnings.append("Correlation analysis skipped: insufficient numeric columns")
    except Exception as e:
        warnings.append(f"Correlation analysis failed: {str(e)}")
        correlations = None
        skipped.append({"name": "correlations", "reason": str(e)})
    
    # 5. Basic anomaly detection
    try:
        anomalies = basic_anomaly_detection(df)
        if anomalies is None:
            warnings.append("Anomaly detection skipped: insufficient data")
    except Exception as e:
        warnings.append(f"Anomaly detection failed: {str(e)}")
        anomalies = None
        skipped.append({"name": "anomalies", "reason": str(e)})
    
    # Extend result with Standard mode outputs
    result["distributions"] = distributions if distributions else None
    result["temporal_analysis"] = temporal
    result["segmentation"] = segmentation if segmentation else None
    result["correlations"] = correlations
    result["anomalies"] = anomalies
    result["warnings"] = warnings
    result["skipped_analyses"] = skipped
    
    return result