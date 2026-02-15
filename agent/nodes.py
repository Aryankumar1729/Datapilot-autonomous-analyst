# nodes.py — Agent intelligence (individual graph node functions)
# Steps: load → profile → analyze → synthesize → present
"""
nodes.py — LangGraph Agent Nodes

Production implementation of agent nodes for the Autonomous Data Analyst.
Each node is a pure function that takes AgentState and returns state updates.

Node Responsibilities:
- ingest_data_node: Parse CSV, validate, prepare for analysis
- profile_data_node: Run schema profiling and data quality scan
- resolve_goal_node: Determine analysis goal (user-provided or inferred)
- analyze_quick_node: Execute Quick mode statistical analysis
- synthesize_insights_node: Transform stats into business insights
- handle_error_node: Graceful error handling and recovery

All nodes are defensive and never crash the graph.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from tools.data_loader import safe_load_csv
from tools.statistics import (
    profile_schema,
    scan_data_quality,
    numeric_summary,
    categorical_summary,
    detect_auto_kpis,
)
from tools.validators import (
    validate_file_extension,
    validate_dataframe,
    sanitize_user_goal,
    validate_goal_against_schema,
    sanitize_dict_for_json,
)


# =============================================================================
# PROGRESS HELPERS
# =============================================================================

def _emit_progress(
    state: dict,
    node: str,
    progress: float,
    message: str,
    status: str = "running",
) -> None:
    """
    Emit a progress update via the callback if available.
    
    Args:
        state: Current agent state
        node: Current node name
        progress: Progress value (0.0 - 1.0)
        message: Human-readable progress message
        status: "running" | "complete" | "failed"
    """
    callback = state.get("progress_callback")
    if callback and callable(callback):
        try:
            callback({
                "node": node,
                "status": status,
                "progress": progress,
                "message": message,
            })
        except Exception:
            pass  # Never let callback errors crash the node


def _create_error_state(
    state: dict,
    node: str,
    error_msg: str,
    error_type: str,
    recovery_hint: str,
) -> dict:
    """
    Create state update for error routing.
    """
    return {
        "error": error_msg,
        "error_type": error_type,
        "failed_node": node,
        "recovery_hint": recovery_hint,
        "current_node": node,
        "partial_results": _has_partial_results(state),
    }


def _has_partial_results(state: dict) -> bool:
    """Check if state has any usable partial results."""
    return any([
        state.get("schema_profile"),
        state.get("data_quality"),
        state.get("numeric_summary"),
        state.get("categorical_summary"),
        state.get("auto_kpis"),
    ])


# =============================================================================
# NODE: INGEST DATA
# =============================================================================

def ingest_data_node(state: dict) -> dict:
    """
    Parse and validate the uploaded CSV file.
    
    Input state:
        - raw_file: bytes (required)
        - filename: str (required)
        
    Output state updates:
        - dataframe: pd.DataFrame
        - row_count: int
        - col_count: int
        - current_node: str
        - progress: float
        
    On error:
        - error, error_type, failed_node, recovery_hint
    """
    node_name = "ingest_data"
    _emit_progress(state, node_name, 0.02, "Loading your data...")
    
    # Validate inputs exist
    raw_file = state.get("raw_file")
    filename = state.get("filename", "unknown.csv")
    
    if raw_file is None:
        return _create_error_state(
            state, node_name,
            "No file provided",
            "DATA_MISSING",
            "Please upload a CSV file to analyze.",
        )
    
    # Validate file extension
    is_valid_ext, ext_error = validate_file_extension(filename)
    if not is_valid_ext:
        return _create_error_state(
            state, node_name,
            ext_error,
            "DATA_INVALID",
            "Please upload a valid CSV file.",
        )
    
    _emit_progress(state, node_name, 0.05, "Parsing CSV...")
    
    # Load CSV
    df, load_error = safe_load_csv(raw_file, filename)
    
    if load_error:
        return _create_error_state(
            state, node_name,
            load_error,
            "DATA_INVALID",
            "Check that your file is a valid CSV with UTF-8 or Latin-1 encoding.",
        )
    
    # Validate DataFrame
    is_valid_df, df_error = validate_dataframe(df)
    if not is_valid_df:
        return _create_error_state(
            state, node_name,
            df_error,
            "DATA_EMPTY",
            "The file appears to be empty or invalid. Please check and re-upload.",
        )
    
    _emit_progress(state, node_name, 0.10, "Data loaded successfully", "complete")
    
    return {
        "dataframe": df,
        "row_count": len(df),
        "col_count": len(df.columns),
        "current_node": node_name,
        "progress": 0.10,
        "progress_message": f"Loaded {len(df):,} rows × {len(df.columns)} columns",
    }


# =============================================================================
# NODE: PROFILE DATA
# =============================================================================

def profile_data_node(state: dict) -> dict:
    """
    Run schema profiling and data quality scan.
    
    Input state:
        - dataframe: pd.DataFrame (required)
        
    Output state updates:
        - schema_profile: dict
        - data_quality: dict
        - warnings: list[str] (extended)
        - current_node: str
        - progress: float
        
    On error:
        - error, error_type, failed_node, recovery_hint
    """
    node_name = "profile_data"
    _emit_progress(state, node_name, 0.12, "Profiling columns...")
    
    df = state.get("dataframe")
    
    if df is None:
        return _create_error_state(
            state, node_name,
            "No DataFrame available for profiling",
            "DATA_MISSING",
            "Please re-upload your file.",
        )
    
    warnings = list(state.get("warnings", []))
    
    # Schema profiling
    try:
        schema = profile_schema(df)
    except Exception as e:
        return _create_error_state(
            state, node_name,
            f"Schema profiling failed: {str(e)}",
            "ANALYSIS_FAILED",
            "The data structure could not be analyzed. Check for malformed columns.",
        )
    
    _emit_progress(state, node_name, 0.18, "Scanning data quality...")
    
    # Data quality scan
    try:
        quality = scan_data_quality(df)
        # Add quality warnings to state warnings
        warnings.extend(quality.get("quality_warnings", []))
    except Exception as e:
        # Non-fatal: continue with empty quality
        quality = {
            "overall_score": 0.0,
            "missing_summary": {"total_missing_cells": 0, "total_cells": 0, "missing_pct": 0},
            "column_quality": [],
            "complete_rows_pct": 0,
            "duplicate_rows": 0,
            "constant_columns": [],
            "quality_warnings": [f"Quality scan failed: {str(e)}"],
        }
        warnings.append(f"Data quality scan incomplete: {str(e)}")
    
    _emit_progress(state, node_name, 0.25, "Profiling complete", "complete")
    
    return {
        "schema_profile": sanitize_dict_for_json(schema),
        "data_quality": sanitize_dict_for_json(quality),
        "warnings": warnings,
        "current_node": node_name,
        "progress": 0.25,
        "progress_message": f"Profiled {schema.get('col_count', 0)} columns",
    }


# =============================================================================
# NODE: RESOLVE GOAL
# =============================================================================

def resolve_goal_node(state: dict) -> dict:
    """
    Determine analysis goal: use user-provided or infer from data.
    
    Input state:
        - user_goal: str | None
        - schema_profile: dict (required)
        
    Output state updates:
        - resolved_goal: str
        - goal_source: str ("user" | "inferred")
        - goal_confidence: float
        - current_node: str
        - progress: float
        
    On error:
        - error, error_type, failed_node, recovery_hint
    """
    node_name = "resolve_goal"
    _emit_progress(state, node_name, 0.27, "Understanding your data...")
    
    schema = state.get("schema_profile")
    user_goal = state.get("user_goal")
    
    if not schema:
        return _create_error_state(
            state, node_name,
            "Schema profile not available",
            "DATA_MISSING",
            "Please re-run the analysis.",
        )
    
    # Sanitize user goal
    clean_goal = sanitize_user_goal(user_goal)
    
    if clean_goal:
        # User provided a goal - validate it
        is_valid, confidence, matched_cols = validate_goal_against_schema(
            clean_goal, schema
        )
        
        if is_valid and confidence >= 0.4:
            _emit_progress(state, node_name, 0.35, "Goal validated", "complete")
            return {
                "resolved_goal": clean_goal,
                "goal_source": "user",
                "goal_confidence": confidence,
                "current_node": node_name,
                "progress": 0.35,
                "progress_message": "Using your specified analysis goal",
            }
        # Low confidence - fall through to inference
    
    # Infer goal from schema
    inferred_goal, confidence = _infer_goal_from_schema(schema)
    
    if confidence < 0.4:
        # Cannot infer meaningful goal
        return _create_error_state(
            state, node_name,
            "Unable to infer analysis goal from data structure",
            "DATA_AMBIGUOUS",
            "Please specify what you'd like to analyze (e.g., 'Analyze revenue trends').",
        )
    
    _emit_progress(state, node_name, 0.35, "Goal inferred", "complete")
    
    return {
        "resolved_goal": inferred_goal,
        "goal_source": "inferred",
        "goal_confidence": confidence,
        "current_node": node_name,
        "progress": 0.35,
        "progress_message": f"Auto-detected goal: {inferred_goal[:50]}...",
    }


def _infer_goal_from_schema(schema: dict) -> tuple[str, float]:
    """
    Infer analysis goal from schema profile.
    
    Returns:
        (inferred_goal, confidence)
    """
    type_summary = schema.get("type_summary", {})
    columns = schema.get("columns", [])
    
    has_numeric = type_summary.get("numeric", 0) > 0
    has_temporal = type_summary.get("temporal", 0) > 0
    has_categorical = type_summary.get("categorical", 0) > 0
    
    # Find column names by type for goal construction
    numeric_cols = [c["name"] for c in columns if c.get("inferred_type") == "numeric"]
    temporal_cols = [c["name"] for c in columns if c.get("inferred_type") == "temporal"]
    categorical_cols = [c["name"] for c in columns if c.get("inferred_type") == "categorical"]
    
    # Goal inference logic (priority order)
    if has_numeric and has_temporal and has_categorical:
        # Rich dataset: trends + segmentation
        goal = f"Analyze trends in {numeric_cols[0]} over time, segmented by {categorical_cols[0]}"
        confidence = 0.85
    elif has_numeric and has_temporal:
        # Time series analysis
        goal = f"Analyze trends and patterns in {numeric_cols[0]} over time"
        confidence = 0.80
    elif has_numeric and has_categorical:
        # Segmentation analysis
        goal = f"Compare {numeric_cols[0]} across {categorical_cols[0]} segments"
        confidence = 0.75
    elif has_numeric:
        # Pure numeric analysis
        goal = f"Analyze distribution and patterns in numeric fields ({', '.join(numeric_cols[:3])})"
        confidence = 0.65
    elif has_categorical:
        # Categorical analysis
        goal = f"Analyze category distributions and relationships in {', '.join(categorical_cols[:3])}"
        confidence = 0.55
    else:
        # Cannot infer
        goal = "General data profiling and quality assessment"
        confidence = 0.35
    
    return goal, confidence


# =============================================================================
# NODE: ANALYZE QUICK
# =============================================================================

def analyze_quick_node(state: dict) -> dict:
    """
    Execute Quick mode statistical analysis.
    
    Input state:
        - dataframe: pd.DataFrame (required)
        - schema_profile: dict (required)
        
    Output state updates:
        - numeric_summary: list[dict]
        - categorical_summary: list[dict]
        - auto_kpis: list[dict]
        - warnings: list[str] (extended)
        - current_node: str
        - progress: float
        
    On error:
        - error, error_type, failed_node, recovery_hint
    """
    node_name = "analyze_quick"
    _emit_progress(state, node_name, 0.40, "Running statistical analysis...")
    
    df = state.get("dataframe")
    schema = state.get("schema_profile")
    
    if df is None or schema is None:
        return _create_error_state(
            state, node_name,
            "Data or schema not available for analysis",
            "DATA_MISSING",
            "Please re-run the analysis from the beginning.",
        )
    
    warnings = list(state.get("warnings", []))
    
    # Numeric summary
    _emit_progress(state, node_name, 0.45, "Analyzing numeric columns...")
    try:
        num_summary = numeric_summary(df)
    except Exception as e:
        num_summary = []
        warnings.append(f"Numeric analysis incomplete: {str(e)}")
    
    # Categorical summary
    _emit_progress(state, node_name, 0.52, "Analyzing categorical columns...")
    try:
        cat_summary = categorical_summary(df)
    except Exception as e:
        cat_summary = []
        warnings.append(f"Categorical analysis incomplete: {str(e)}")
    
    # Auto-KPI detection
    _emit_progress(state, node_name, 0.60, "Detecting key metrics...")
    try:
        kpis = detect_auto_kpis(df, schema)
    except Exception as e:
        kpis = []
        warnings.append(f"KPI detection incomplete: {str(e)}")
    
    # Check if we have any results
    if not num_summary and not cat_summary and not kpis:
        return _create_error_state(
            state, node_name,
            "No statistical analysis could be performed",
            "ANALYSIS_FAILED",
            "The data may lack numeric or categorical columns suitable for analysis.",
        )
    
    _emit_progress(state, node_name, 0.65, "Analysis complete", "complete")
    
    return {
        "numeric_summary": sanitize_dict_for_json(num_summary),
        "categorical_summary": sanitize_dict_for_json(cat_summary),
        "auto_kpis": sanitize_dict_for_json(kpis),
        "warnings": warnings,
        "current_node": node_name,
        "progress": 0.65,
        "progress_message": f"Generated {len(kpis)} KPIs, analyzed {len(num_summary)} numeric and {len(cat_summary)} categorical columns",
    }


# =============================================================================
# NODE: SYNTHESIZE INSIGHTS
# =============================================================================

def synthesize_insights_node(state: dict) -> dict:
    """
    Transform statistical results into business insights.
    
    NOTE: This is a template-based implementation for Quick mode.
    LLM-powered synthesis would be added for Standard/Deep modes.
    
    Input state:
        - auto_kpis: list[dict]
        - numeric_summary: list[dict]
        - categorical_summary: list[dict]
        - data_quality: dict
        - resolved_goal: str
        
    Output state updates:
        - insights: list[dict]
        - executive_summary: str
        - ui_payload: dict
        - current_node: str
        - progress: float
    """
    node_name = "synthesize_insights"
    _emit_progress(state, node_name, 0.70, "Generating insights...")
    
    kpis = state.get("auto_kpis", [])
    num_summary = state.get("numeric_summary", [])
    cat_summary = state.get("categorical_summary", [])
    quality = state.get("data_quality", {})
    schema = state.get("schema_profile", {})
    goal = state.get("resolved_goal", "General analysis")
    warnings = list(state.get("warnings", []))
    
    insights = []
    llm_available = False
    llm = None
    
    # Try to initialize LLM for goal-specific answers
    try:
        from config.llm_config import get_llm
        llm = get_llm(temperature=0.5)  # Auto-selects Groq or Ollama
        if llm and llm.is_available():
            llm_available = True
    except Exception:
        pass
    
    # FIRST: Try to directly answer the user's question if they have one
    if llm_available and goal and goal != "General analysis":
        _emit_progress(state, node_name, 0.72, "Answering your question...")
        goal_answer = _answer_user_question(
            llm=llm,
            goal=goal,
            kpis=kpis,
            num_summary=num_summary,
            cat_summary=cat_summary,
            segmentation=[],  # Quick mode doesn't have segmentation
            schema=schema,
        )
        if goal_answer:
            insights.append(goal_answer)
    
    # Generate insights from KPIs
    _emit_progress(state, node_name, 0.75, "Analyzing key metrics...")
    for kpi in kpis[:5]:  # Top 5 KPIs
        insight = _generate_kpi_insight(kpi, llm=llm if llm_available else None, warnings=warnings)
        if insight:
            insights.append(insight)
    
    # Generate insights from numeric analysis
    _emit_progress(state, node_name, 0.80, "Analyzing distributions...")
    for num in num_summary[:3]:  # Top 3 numeric columns
        insight = _generate_numeric_insight(num)
        if insight:
            insights.append(insight)
    
    # Generate insights from categorical analysis
    for cat in cat_summary[:3]:  # Top 3 categorical columns
        if not cat.get("is_high_cardinality"):
            insight = _generate_categorical_insight(cat)
            if insight:
                insights.append(insight)
    
    # Generate data quality insight if issues exist
    if quality.get("overall_score", 1.0) < 0.8:
        insights.append(_generate_quality_insight(quality))
    
    # Generate executive summary
    _emit_progress(state, node_name, 0.85, "Writing summary...")
    executive_summary = _generate_executive_summary(
        schema, quality, kpis, num_summary, cat_summary, goal
    )
    
    # Build UI payload
    _emit_progress(state, node_name, 0.92, "Preparing report...")
    ui_payload = _build_ui_payload(state, insights, executive_summary)
    
    _emit_progress(state, node_name, 0.95, "Insights ready", "complete")
    
    return {
        "insights": sanitize_dict_for_json(insights),
        "executive_summary": executive_summary,
        "ui_payload": sanitize_dict_for_json(ui_payload),
        "warnings": warnings,
        "current_node": node_name,
        "progress": 0.95,
        "progress_message": f"Generated {len(insights)} insights",
    }


def _generate_kpi_insight(kpi: dict, llm=None, warnings: list = None) -> dict | None:
    """Generate insight from a KPI, optionally using LLM for richer text."""
    name = kpi.get("name", "")
    value = kpi.get("formatted_value", "")
    
    if not name or not value:
        return None
    
    # Try LLM-enhanced insight if available
    if llm:
        try:
            body = llm.generate_insight(
                statistical_finding=f"{name}: {value}",
                context="This is a key business metric from the dataset."
            )
            if body and len(body) > 10:
                return {
                    "title": name,
                    "body": body,
                    "severity": "info",
                    "category": "kpi",
                    "supporting_data": kpi,
                    "llm_enhanced": True,
                }
        except Exception as e:
            if warnings is not None:
                warnings.append(f"LLM insight generation failed: {str(e)[:100]}")
    
    # Template fallback
    return {
        "title": name,
        "body": f"**{value}** detected as a key metric in your data.",
        "severity": "info",
        "category": "kpi",
        "supporting_data": kpi,
    }


def _generate_numeric_insight(num: dict) -> dict | None:
    """Generate insight from numeric summary."""
    col = num.get("column", "")
    mean = num.get("mean", 0)
    median = num.get("median", 0)
    skew = num.get("skew_interpretation", "symmetric")
    
    if not col:
        return None
    
    # Detect mean-median divergence
    if mean != 0 and abs((mean - median) / mean) > 0.2:
        severity = "warning"
        body = f"`{col}` shows **{skew}** distribution (mean: {mean:.2f}, median: {median:.2f}). Consider using median for typical values."
    else:
        severity = "info"
        body = f"`{col}` has mean **{mean:.2f}** with {skew} distribution."
    
    return {
        "title": f"Distribution: {col}",
        "body": body,
        "severity": severity,
        "category": "distribution",
        "supporting_data": num,
    }


def _generate_categorical_insight(cat: dict) -> dict | None:
    """Generate insight from categorical summary."""
    col = cat.get("column", "")
    cardinality = cat.get("cardinality", 0)
    concentration = cat.get("concentration", "")
    top_values = cat.get("top_values", [])
    
    if not col or not top_values:
        return None
    
    top = top_values[0] if top_values else {}
    top_value = top.get("value", "")
    top_pct = top.get("pct", 0)
    
    if concentration == "high":
        severity = "warning"
        body = f"`{col}` is **highly concentrated**: '{top_value}' accounts for {top_pct:.1f}% of data. Consider whether this reflects reality or sampling bias."
    else:
        severity = "info"
        body = f"`{col}` has {cardinality} unique values. Most common: '{top_value}' ({top_pct:.1f}%)."
    
    return {
        "title": f"Segments: {col}",
        "body": body,
        "severity": severity,
        "category": "segmentation",
        "supporting_data": cat,
    }


def _generate_quality_insight(quality: dict) -> dict:
    """Generate insight from data quality issues."""
    score = quality.get("overall_score", 0)
    missing_pct = quality.get("missing_summary", {}).get("missing_pct", 0)
    warnings_list = quality.get("quality_warnings", [])
    
    return {
        "title": "Data Quality Alert",
        "body": f"Data quality score: **{score*100:.0f}%**. {missing_pct:.1f}% of cells are missing. " + 
                (warnings_list[0] if warnings_list else "Review data before making decisions."),
        "severity": "warning",
        "category": "quality",
        "supporting_data": {"score": score, "issues": warnings_list[:3]},
    }


def _generate_executive_summary(
    schema: dict,
    quality: dict,
    kpis: list,
    num_summary: list,
    cat_summary: list,
    goal: str,
) -> str:
    """Generate one-paragraph executive summary."""
    row_count = schema.get("row_count", 0)
    col_count = schema.get("col_count", 0)
    quality_score = quality.get("overall_score", 0) * 100
    
    # Count by type
    type_summary = schema.get("type_summary", {})
    num_cols = type_summary.get("numeric", 0)
    cat_cols = type_summary.get("categorical", 0)
    
    # Top KPI
    top_kpi = kpis[0] if kpis else None
    top_kpi_str = f"Key metric: {top_kpi['name']} = {top_kpi['formatted_value']}. " if top_kpi else ""
    
    summary = (
        f"Analyzed **{row_count:,} records** across **{col_count} columns** "
        f"({num_cols} numeric, {cat_cols} categorical). "
        f"Data quality: **{quality_score:.0f}%**. "
        f"{top_kpi_str}"
        f"Analysis goal: {goal[:100]}."
    )
    
    return summary


def _build_ui_payload(state: dict, insights: list, summary: str) -> dict:
    """Build Streamlit-ready UI payload."""
    return {
        "filename": state.get("filename", "data.csv"),
        "row_count": state.get("row_count", 0),
        "col_count": state.get("col_count", 0),
        "schema_profile": state.get("schema_profile"),
        "data_quality": state.get("data_quality"),
        "kpis": state.get("auto_kpis", []),
        "numeric_summary": state.get("numeric_summary", []),
        "categorical_summary": state.get("categorical_summary", []),
        "insights": insights,
        "executive_summary": summary,
        "warnings": state.get("warnings", []),
        "goal": state.get("resolved_goal"),
        "goal_source": state.get("goal_source"),
        "analysis_depth": state.get("analysis_depth", "quick"),
        "dashboard_plan": state.get("dashboard_plan", []),
    }


# =============================================================================
# NODE: HANDLE ERROR
# =============================================================================

def handle_error_node(state: dict) -> dict:
    """
    Handle errors gracefully and prepare user-facing error payload.
    
    Input state:
        - error: str
        - error_type: str
        - failed_node: str
        - recovery_hint: str
        - partial_results: bool
        
    Output state updates:
        - ui_payload: dict (error payload for UI)
        - current_node: str
        - progress: float
    """
    node_name = "handle_error"
    _emit_progress(state, node_name, 0.99, "Handling error...", "failed")
    
    error = state.get("error", "An unknown error occurred")
    error_type = state.get("error_type", "UNKNOWN")
    failed_node = state.get("failed_node", "unknown")
    recovery_hint = state.get("recovery_hint", "Please try again or contact support.")
    has_partial = state.get("partial_results", False)
    
    # Build error payload for UI
    error_payload = {
        "is_error": True,
        "error_message": error,
        "error_type": error_type,
        "failed_node": failed_node,
        "recovery_hint": recovery_hint,
        "has_partial_results": has_partial,
    }
    
    # Include partial results if available
    if has_partial:
        error_payload["partial_results"] = {
            "schema_profile": state.get("schema_profile"),
            "data_quality": state.get("data_quality"),
            "kpis": state.get("auto_kpis"),
            "warnings": state.get("warnings", []) + [f"Analysis incomplete: {error}"],
        }
        error_payload["message"] = "Partial results available despite error."
    
    return {
        "ui_payload": sanitize_dict_for_json(error_payload),
        "current_node": node_name,
        "progress": 1.0,
        "progress_message": f"Error: {error_type}",
    }


# =============================================================================
# NODE: ANALYZE STANDARD
# =============================================================================

def analyze_standard_node(state: dict) -> dict:
    """
    Execute Standard mode statistical analysis.
    
    Standard mode extends Quick mode with:
    - Distribution analysis (quartiles, normality tests)
    - Temporal trend analysis
    - Categorical segmentation
    - Correlation analysis
    - Anomaly detection
    
    Input state:
        - dataframe: pd.DataFrame (required)
        - schema_profile: dict (required)
        
    Output state updates:
        - standard_analysis: dict (full Standard mode results)
        - numeric_summary: list[dict]
        - categorical_summary: list[dict]
        - auto_kpis: list[dict]
        - distributions: list[dict]
        - temporal_analysis: dict | None
        - segmentation: list[dict]
        - correlations: dict | None
        - anomalies: dict | None
        - warnings: list[str] (extended)
        - current_node: str
        - progress: float
        
    On error:
        - error, error_type, failed_node, recovery_hint
    """
    # Import here to avoid circular imports
    from tools.statistics import run_standard_analysis
    
    node_name = "analyze_standard"
    _emit_progress(state, node_name, 0.37, "Running comprehensive analysis...")
    
    df = state.get("dataframe")
    schema = state.get("schema_profile")
    
    if df is None or schema is None:
        return _create_error_state(
            state, node_name,
            "Data or schema not available for analysis",
            "DATA_MISSING",
            "Please re-run the analysis from the beginning.",
        )
    
    warnings = list(state.get("warnings", []))
    
    # Run full Standard analysis
    _emit_progress(state, node_name, 0.40, "Running Quick mode analysis...")
    
    try:
        standard_results = run_standard_analysis(df)
    except Exception as e:
        return _create_error_state(
            state, node_name,
            f"Standard analysis failed: {str(e)}",
            "ANALYSIS_FAILED",
            "Try using Quick mode for faster, simpler analysis.",
        )
    
    # Extract warnings from results
    warnings.extend(standard_results.get("warnings", []))
    
    _emit_progress(state, node_name, 0.50, "Analyzing distributions...")
    
    # Extract individual components
    num_summary = standard_results.get("numeric_summary", [])
    cat_summary = standard_results.get("categorical_summary", [])
    kpis = standard_results.get("auto_kpis", [])
    distributions = standard_results.get("distributions", [])
    temporal = standard_results.get("temporal_analysis")
    segmentation = standard_results.get("segmentation", [])
    correlations = standard_results.get("correlations")
    anomalies = standard_results.get("anomalies")
    
    _emit_progress(state, node_name, 0.55, "Analyzing trends and correlations...")
    
    # Check if we have meaningful results
    has_results = any([
        num_summary,
        cat_summary,
        kpis,
        distributions,
        temporal,
        segmentation,
        correlations,
        anomalies,
    ])
    
    if not has_results:
        return _create_error_state(
            state, node_name,
            "No statistical analysis could be performed",
            "ANALYSIS_FAILED",
            "The data may lack suitable columns for Standard analysis. Try Quick mode.",
        )
    
    _emit_progress(state, node_name, 0.65, "Standard analysis complete", "complete")
    
    # Build progress message
    components = []
    if distributions:
        components.append(f"{len(distributions)} distributions")
    if temporal:
        components.append("temporal trends")
    if segmentation:
        components.append(f"{len(segmentation)} segments")
    if correlations and correlations.get("significant_pairs"):
        components.append(f"{len(correlations['significant_pairs'])} correlations")
    if anomalies and anomalies.get("column_outliers"):
        components.append(f"{len(anomalies['column_outliers'])} anomaly reports")
    
    progress_msg = f"Analyzed: {', '.join(components)}" if components else "Standard analysis complete"
    
    return {
        "standard_analysis": sanitize_dict_for_json(standard_results),
        "numeric_summary": sanitize_dict_for_json(num_summary),
        "categorical_summary": sanitize_dict_for_json(cat_summary),
        "auto_kpis": sanitize_dict_for_json(kpis),
        "distributions": sanitize_dict_for_json(distributions) if distributions else None,
        "temporal_analysis": sanitize_dict_for_json(temporal) if temporal else None,
        "segmentation": sanitize_dict_for_json(segmentation) if segmentation else None,
        "correlations": sanitize_dict_for_json(correlations) if correlations else None,
        "anomalies": sanitize_dict_for_json(anomalies) if anomalies else None,
        "warnings": warnings,
        "current_node": node_name,
        "progress": 0.65,
        "progress_message": progress_msg,
    }


# =============================================================================
# NODE: SYNTHESIZE STANDARD INSIGHTS
# =============================================================================

def synthesize_standard_insights_node(state: dict) -> dict:
    """
    Transform Standard mode results into business insights.
    
    Uses local LLM (Ollama/tinyllama) to:
    - Rank insights by business importance
    - Add contextual interpretations
    - Generate executive summary
    
    Falls back to template-based synthesis if LLM unavailable.
    
    Input state:
        - standard_analysis: dict (from analyze_standard_node)
        - auto_kpis: list[dict]
        - distributions: list[dict]
        - temporal_analysis: dict | None
        - segmentation: list[dict]
        - correlations: dict | None
        - anomalies: dict | None
        - data_quality: dict
        - resolved_goal: str
        
    Output state updates:
        - insights: list[dict]
        - executive_summary: str
        - ui_payload: dict
        - current_node: str
        - progress: float
    """
    node_name = "synthesize_standard_insights"
    _emit_progress(state, node_name, 0.67, "Generating insights...")
    
    # Extract all analysis components
    kpis = state.get("auto_kpis", [])
    num_summary = state.get("numeric_summary", [])
    cat_summary = state.get("categorical_summary", [])
    distributions = state.get("distributions", [])
    temporal = state.get("temporal_analysis")
    segmentation = state.get("segmentation", [])
    correlations = state.get("correlations")
    anomalies = state.get("anomalies")
    quality = state.get("data_quality", {})
    schema = state.get("schema_profile", {})
    goal = state.get("resolved_goal", "General analysis")
    warnings = list(state.get("warnings", []))
    
    insights = []
    llm_available = False
    llm = None
    
    # Try to initialize LLM for enhanced synthesis
    _emit_progress(state, node_name, 0.70, "Initializing AI assistant...")
    try:
        from config.llm_config import get_llm, OllamaConnectionError
        llm = get_llm(temperature=0.6)  # Auto-selects Groq or Ollama
        
        # Quick availability check
        if llm and llm.is_available():
            llm_available = True
        else:
            warnings.append("LLM unavailable - using template-based insights")
    except Exception as e:
        warnings.append(f"LLM initialization skipped: {str(e)}")
    
    # FIRST: Try to directly answer the user's question if they have one
    if llm_available and goal and goal != "General analysis":
        _emit_progress(state, node_name, 0.71, "Answering your question...")
        goal_answer = _answer_user_question(
            llm=llm,
            goal=goal,
            kpis=kpis,
            num_summary=num_summary,
            cat_summary=cat_summary,
            segmentation=segmentation,
            schema=schema,
        )
        if goal_answer:
            insights.append(goal_answer)
    
    # Generate insights from each analysis component
    _emit_progress(state, node_name, 0.73, "Analyzing key metrics...")
    
    # 1. KPI insights (use LLM directly if available)
    for kpi in kpis[:5]:
        insight = _generate_kpi_insight(kpi, llm=llm if llm_available else None, warnings=warnings)
        if insight:
            insights.append(insight)
    
    # 2. Distribution insights (Standard mode specific)
    _emit_progress(state, node_name, 0.76, "Analyzing distributions...")
    for dist in (distributions or [])[:3]:
        insight = _generate_distribution_insight(dist)
        if insight:
            if llm_available:
                insight = _enhance_insight_with_llm(llm, insight, goal)
            insights.append(insight)
    
    # 3. Temporal insights (Standard mode specific)
    _emit_progress(state, node_name, 0.79, "Analyzing trends...")
    if temporal:
        trend_insights = _generate_temporal_insights(temporal)
        for insight in trend_insights[:3]:
            if llm_available:
                insight = _enhance_insight_with_llm(llm, insight, goal)
            insights.append(insight)
    
    # 4. Segmentation insights (Standard mode specific)
    _emit_progress(state, node_name, 0.82, "Analyzing segments...")
    for seg in (segmentation or [])[:2]:
        insight = _generate_segmentation_insight(seg)
        if insight:
            if llm_available:
                insight = _enhance_insight_with_llm(llm, insight, goal)
            insights.append(insight)
    
    # 5. Correlation insights (Standard mode specific)
    _emit_progress(state, node_name, 0.85, "Analyzing correlations...")
    if correlations:
        corr_insights = _generate_correlation_insights(correlations)
        for insight in corr_insights[:3]:
            if llm_available:
                insight = _enhance_insight_with_llm(llm, insight, goal)
            insights.append(insight)
    
    # 6. Anomaly insights (Standard mode specific)
    _emit_progress(state, node_name, 0.88, "Analyzing anomalies...")
    if anomalies:
        anomaly_insights = _generate_anomaly_insights(anomalies)
        for insight in anomaly_insights[:3]:
            if llm_available:
                insight = _enhance_insight_with_llm(llm, insight, goal)
            insights.append(insight)
    
    # 7. Data quality insight
    if quality.get("overall_score", 1.0) < 0.8:
        insights.append(_generate_quality_insight(quality))
    
    # Rank insights by importance
    _emit_progress(state, node_name, 0.90, "Ranking insights...")
    insights = _rank_insights(insights, llm if llm_available else None, goal)
    
    # Generate executive summary
    _emit_progress(state, node_name, 0.92, "Writing executive summary...")
    if llm_available:
        executive_summary = _generate_llm_executive_summary(
            llm, schema, quality, kpis, insights, goal
        )
    else:
        executive_summary = _generate_standard_executive_summary(
            schema, quality, kpis, temporal, correlations, anomalies, goal
        )
    
    # Build UI payload
    _emit_progress(state, node_name, 0.94, "Preparing report...")
    ui_payload = _build_standard_ui_payload(state, insights, executive_summary)
    
    _emit_progress(state, node_name, 0.95, "Insights ready", "complete")
    
    return {
        "insights": sanitize_dict_for_json(insights),
        "executive_summary": executive_summary,
        "ui_payload": sanitize_dict_for_json(ui_payload),
        "warnings": warnings,
        "current_node": node_name,
        "progress": 0.95,
        "progress_message": f"Generated {len(insights)} insights" + (" (AI-enhanced)" if llm_available else ""),
    }


# =============================================================================
# STANDARD MODE INSIGHT GENERATORS
# =============================================================================

def _generate_distribution_insight(dist: dict) -> dict | None:
    """Generate insight from distribution analysis."""
    col = dist.get("column", "")
    dist_type = dist.get("distribution_type", "")
    is_normal = dist.get("is_normal", False)
    quartiles = dist.get("quartiles", {})
    iqr = dist.get("iqr", 0)
    
    if not col:
        return None
    
    q1 = quartiles.get("q1", 0)
    q3 = quartiles.get("q3", 0)
    
    if is_normal:
        severity = "info"
        body = f"`{col}` follows a **normal distribution**. The middle 50% of values fall between {q1:.2f} and {q3:.2f}."
    elif dist_type in ["right-skewed", "left-skewed"]:
        severity = "warning"
        body = f"`{col}` is **{dist_type}**. The middle 50% of values (IQR: {iqr:.2f}) may not represent the typical value well."
    else:
        severity = "info"
        body = f"`{col}` has a **{dist_type}** distribution. IQR: {iqr:.2f}."
    
    return {
        "title": f"Distribution: {col}",
        "body": body,
        "severity": severity,
        "category": "distribution",
        "supporting_data": dist,
    }


def _generate_temporal_insights(temporal: dict) -> list[dict]:
    """Generate insights from temporal analysis."""
    insights = []
    
    trends = temporal.get("trends", [])
    seasonality = temporal.get("seasonality", {})
    date_range = temporal.get("date_range", {})
    
    # Trend insights
    for trend in trends[:2]:
        col = trend.get("metric_column", "")
        direction = trend.get("direction", "stable")
        slope_pct = trend.get("slope_pct", 0)
        pop = trend.get("period_over_period", {})
        
        if direction == "increasing":
            severity = "positive"
            body = f"**{col}** is trending **up** at {abs(slope_pct):.1f}% per period."
        elif direction == "decreasing":
            severity = "warning"
            body = f"**{col}** is trending **down** at {abs(slope_pct):.1f}% per period."
        else:
            severity = "info"
            body = f"**{col}** is **stable** with no significant trend."
        
        if pop:
            change_pct = pop.get("change_pct", 0)
            body += f" Latest period change: {change_pct:+.1f}%."
        
        insights.append({
            "title": f"Trend: {col}",
            "body": body,
            "severity": severity,
            "category": "trend",
            "supporting_data": trend,
        })
    
    # Seasonality insight
    if seasonality.get("detected"):
        pattern = seasonality.get("pattern", "Seasonal pattern detected")
        cv = seasonality.get("coefficient_of_variation", 0)
        
        insights.append({
            "title": "Seasonality Detected",
            "body": f"**{pattern}**. Seasonal variation coefficient: {cv:.1f}%. Consider this in forecasting.",
            "severity": "info",
            "category": "seasonality",
            "supporting_data": seasonality,
        })
    
    return insights


def _generate_segmentation_insight(seg: dict) -> dict | None:
    """Generate insight from segmentation analysis."""
    segment_col = seg.get("segment_column", "")
    kpi_col = seg.get("kpi_column", "")
    insights_data = seg.get("insights", {})
    
    if not segment_col or not kpi_col:
        return None
    
    top = insights_data.get("top_performer", {})
    bottom = insights_data.get("bottom_performer", {})
    spread = insights_data.get("performance_spread")
    
    top_val = top.get("value", "")
    top_kpi = top.get("kpi_value", 0)
    bottom_val = bottom.get("value", "")
    bottom_kpi = bottom.get("kpi_value", 0)
    
    if spread and spread > 2:
        severity = "warning"
        body = f"**{top_val}** leads in `{kpi_col}` ({top_kpi:,.0f}) — **{spread:.1f}x** higher than **{bottom_val}** ({bottom_kpi:,.0f}). Large performance gap warrants investigation."
    else:
        severity = "info"
        body = f"**{top_val}** leads `{kpi_col}` ({top_kpi:,.0f}), while **{bottom_val}** trails ({bottom_kpi:,.0f})."
    
    return {
        "title": f"Segment Performance: {segment_col}",
        "body": body,
        "severity": severity,
        "category": "segmentation",
        "supporting_data": seg,
    }


def _generate_correlation_insights(correlations: dict) -> list[dict]:
    """Generate insights from correlation analysis."""
    insights = []
    
    top_corr = correlations.get("top_correlations", [])
    multicollinearity = correlations.get("multicollinearity_warning", False)
    
    for corr in top_corr[:2]:
        col_a = corr.get("column_a", "")
        col_b = corr.get("column_b", "")
        r = corr.get("correlation", 0)
        strength = corr.get("strength", "")
        
        direction = "positive" if r > 0 else "negative"
        
        if abs(r) >= 0.7:
            severity = "info"
            body = f"**Strong {direction} correlation** (r={r:.2f}) between `{col_a}` and `{col_b}`. Changes in one likely reflect in the other."
        else:
            severity = "info"
            body = f"**Moderate {direction} correlation** (r={r:.2f}) between `{col_a}` and `{col_b}`."
        
        insights.append({
            "title": f"Correlation: {col_a} ↔ {col_b}",
            "body": body,
            "severity": severity,
            "category": "correlation",
            "supporting_data": corr,
        })
    
    if multicollinearity:
        insights.append({
            "title": "⚠️ Multicollinearity Warning",
            "body": "Some variables are **highly correlated (r > 0.9)**. This may cause issues in predictive modeling. Consider removing redundant features.",
            "severity": "warning",
            "category": "correlation",
            "supporting_data": {"multicollinearity": True},
        })
    
    return insights


def _generate_anomaly_insights(anomalies: dict) -> list[dict]:
    """Generate insights from anomaly detection."""
    insights = []
    
    summary = anomalies.get("outlier_summary", {})
    column_outliers = anomalies.get("column_outliers", [])
    suspicious = anomalies.get("suspicious_values", [])
    
    total_outliers = summary.get("total_outliers", 0)
    outlier_pct = summary.get("outlier_pct", 0)
    
    # Overall outlier summary
    if total_outliers > 0:
        if outlier_pct > 5:
            severity = "warning"
            body = f"**{total_outliers:,} outliers** detected ({outlier_pct:.1f}% of data). This is unusually high — verify data quality."
        else:
            severity = "info"
            body = f"**{total_outliers:,} outliers** detected ({outlier_pct:.1f}% of data). Review for data entry errors or genuine extremes."
        
        insights.append({
            "title": "Outlier Summary",
            "body": body,
            "severity": severity,
            "category": "anomaly",
            "supporting_data": summary,
        })
    
    # Top column with outliers
    if column_outliers:
        top_outlier = max(column_outliers, key=lambda x: x.get("outlier_count", 0))
        col = top_outlier.get("column", "")
        count = top_outlier.get("outlier_count", 0)
        direction = top_outlier.get("direction", "both")
        
        dir_text = {"high": "above normal", "low": "below normal", "both": "above and below normal"}
        
        insights.append({
            "title": f"Outliers: {col}",
            "body": f"`{col}` has **{count} outliers** ({dir_text.get(direction, 'extreme values')}). Verify these are valid data points.",
            "severity": "warning",
            "category": "anomaly",
            "supporting_data": top_outlier,
        })
    
    # Suspicious values
    for susp in suspicious[:1]:
        col = susp.get("column", "")
        issue = susp.get("issue", "")
        count = susp.get("count", 0)
        
        issue_text = {
            "unexpected_zeros": f"**{count} zero values** in `{col}` — unexpected for this type of metric.",
            "unexpected_negatives": f"**{count} negative values** in `{col}` — may indicate returns, corrections, or errors.",
        }
        
        insights.append({
            "title": f"Data Alert: {col}",
            "body": issue_text.get(issue, f"Suspicious values found in `{col}`."),
            "severity": "warning",
            "category": "anomaly",
            "supporting_data": susp,
        })
    
    return insights


# =============================================================================
# GOAL-SPECIFIC ANSWER GENERATION
# =============================================================================

def _answer_user_question(
    llm,
    goal: str,
    kpis: list,
    num_summary: list,
    cat_summary: list,
    segmentation: list,
    schema: dict,
) -> dict | None:
    """
    Directly answer the user's specific question using LLM and data.
    
    This is the key function that makes the analysis respond to user goals.
    """
    try:
        # Build comprehensive data context for LLM
        data_context = _build_data_context_for_question(
            kpis, num_summary, cat_summary, segmentation, schema
        )
        
        prompt = f"""Based on this data analysis, answer the user's question directly and specifically.

USER'S QUESTION: {goal}

DATA AVAILABLE:
{data_context}

Instructions:
1. Answer the question DIRECTLY using the data provided
2. Include specific numbers and values from the data
3. If the question asks "which is highest/lowest/best", identify and name the specific item
4. Be concise but include the key data points
5. If the data doesn't contain information to answer the question, say so

ANSWER:"""

        response = llm.generate(
            prompt=prompt,
            system_prompt="You are a data analyst answering specific questions about a dataset. Always be specific and include numbers from the data. If asked which is highest, actually name the specific item.",
            temperature=0.3,  # Lower temperature for factual answers
            max_tokens=300,
        )
        
        if response and len(response) > 20:
            return {
                "title": f"📌 Answer: {_truncate_goal(goal)}",
                "body": response.strip(),
                "severity": "info",
                "category": "goal_answer",
                "supporting_data": {"goal": goal},
                "priority_boost": 100,  # Ensure this appears first
            }
    except Exception as e:
        pass  # Fall back to regular insights
    
    return None


def _truncate_goal(goal: str, max_len: int = 50) -> str:
    """Truncate goal for display in title."""
    if len(goal) <= max_len:
        return goal
    return goal[:max_len-3] + "..."


def _build_data_context_for_question(
    kpis: list,
    num_summary: list,
    cat_summary: list,
    segmentation: list,
    schema: dict,
) -> str:
    """Build a comprehensive data summary for answering questions."""
    context_parts = []
    
    # Dataset overview
    row_count = schema.get("row_count", 0)
    col_count = schema.get("col_count", 0)
    context_parts.append(f"Dataset: {row_count:,} records, {col_count} columns")
    
    # Key metrics
    if kpis:
        kpi_strs = []
        for kpi in kpis[:10]:
            name = kpi.get("name", "")
            val = kpi.get("formatted_value", kpi.get("value", ""))
            kpi_strs.append(f"  - {name}: {val}")
        context_parts.append("KEY METRICS:\n" + "\n".join(kpi_strs))
    
    # Numeric columns summary
    if num_summary:
        num_strs = []
        for ns in num_summary[:8]:
            col = ns.get("column", "")
            total = ns.get("sum", ns.get("mean", 0) * row_count) if row_count else 0
            mean = ns.get("mean", 0)
            max_val = ns.get("max", 0)
            min_val = ns.get("min", 0)
            num_strs.append(f"  - {col}: Total={total:,.2f}, Avg={mean:,.2f}, Max={max_val:,.2f}, Min={min_val:,.2f}")
        context_parts.append("NUMERIC COLUMNS:\n" + "\n".join(num_strs))
    
    # Categorical breakdown (critical for "which product" questions)
    if cat_summary:
        cat_strs = []
        for cs in cat_summary[:5]:
            col = cs.get("column", "")
            unique = cs.get("unique_count", 0)
            top_values = cs.get("top_values", [])
            if top_values:
                top_items = [f"{tv['value']} ({tv['count']} records, {tv['pct']:.1f}%)" for tv in top_values[:5]]
                cat_strs.append(f"  - {col} ({unique} unique): {', '.join(top_items)}")
        if cat_strs:
            context_parts.append("CATEGORICAL BREAKDOWN:\n" + "\n".join(cat_strs))
    
    # Segmentation (shows aggregates by category - great for "which X has highest Y")
    if segmentation:
        seg_strs = []
        for seg in segmentation[:3]:
            segment_col = seg.get("segment_column", "")
            kpi_col = seg.get("kpi_column", "")
            segments = seg.get("segments", [])
            if segments:
                seg_items = []
                for s in sorted(segments, key=lambda x: x.get("sum", 0), reverse=True)[:5]:
                    name = s.get("name", "")
                    total = s.get("sum", 0)
                    seg_items.append(f"{name}={total:,.2f}")
                seg_strs.append(f"  - {kpi_col} by {segment_col}: {', '.join(seg_items)}")
        if seg_strs:
            context_parts.append("PERFORMANCE BY SEGMENT:\n" + "\n".join(seg_strs))
    
    return "\n\n".join(context_parts)


# =============================================================================
# LLM ENHANCEMENT HELPERS
# =============================================================================

def _enhance_insight_with_llm(llm, insight: dict, goal: str, warnings: list = None) -> dict:
    """
    Enhance an insight with LLM-generated business interpretation.
    
    LLM is used ONLY for interpretation, NOT for computing numbers.
    Falls back gracefully on any error.
    """
    try:
        statistical_finding = insight.get("body", "")
        
        # Skip if body is too short
        if len(statistical_finding) < 20:
            return insight
        
        # Generate business interpretation
        enhanced_body = llm.generate_insight(
            statistical_finding=statistical_finding,
            context=goal,
        )
        
        # Only use if we got a meaningful response
        if enhanced_body and len(enhanced_body) > 20:
            insight["body"] = enhanced_body
            insight["llm_enhanced"] = True
        
    except Exception as e:
        # Log warning but fall back to original insight
        if warnings is not None:
            warnings.append(f"LLM enhancement failed: {str(e)[:100]}")
    
    return insight


def _rank_insights(insights: list[dict], llm, goal: str) -> list[dict]:
    """
    Rank insights by business importance.
    
    Priority order (without LLM):
    1. Warnings (severity=warning)
    2. Trends (category=trend)
    3. Correlations (category=correlation)
    4. Segmentation (category=segmentation)
    5. Others
    """
    # Define priority scores
    severity_scores = {"warning": 100, "positive": 80, "info": 50}
    category_scores = {
        "anomaly": 90,
        "trend": 85,
        "correlation": 70,
        "segmentation": 65,
        "distribution": 50,
        "kpi": 45,
        "quality": 40,
    }
    
    def score_insight(insight: dict) -> float:
        severity = insight.get("severity", "info")
        category = insight.get("category", "")
        
        return severity_scores.get(severity, 50) + category_scores.get(category, 30)
    
    # Sort by score descending
    ranked = sorted(insights, key=score_insight, reverse=True)
    
    # Limit to top 15 insights
    return ranked[:15]


def _generate_llm_executive_summary(
    llm,
    schema: dict,
    quality: dict,
    kpis: list,
    insights: list,
    goal: str,
) -> str:
    """Generate executive summary using LLM."""
    try:
        # Build context for LLM
        row_count = schema.get("row_count", 0)
        col_count = schema.get("col_count", 0)
        quality_score = quality.get("overall_score", 0) * 100
        
        # Top 3 insight titles
        top_insights = [i.get("title", "") for i in insights[:3]]
        
        # Top KPI
        top_kpi = kpis[0] if kpis else None
        kpi_str = f"{top_kpi['name']}: {top_kpi['formatted_value']}" if top_kpi else "No key metrics detected"
        
        prompt = f"""Write a 2-3 sentence executive summary for this data analysis:

Dataset: {row_count:,} records, {col_count} columns
Data Quality: {quality_score:.0f}%
Key Metric: {kpi_str}
Top Findings: {', '.join(top_insights)}
Analysis Goal: {goal}

Be concise and business-focused. Start with the most important finding."""

        summary = llm.generate(
            prompt=prompt,
            system_prompt="You are a business analyst writing executive summaries. Be concise and data-driven.",
            temperature=0.5,
            max_tokens=200,
        )
        
        if summary and len(summary) > 30:
            return summary.strip()
        
    except Exception:
        pass
    
    # Fallback to template
    return _generate_standard_executive_summary(
        schema, quality, kpis, None, None, None, goal
    )


def _generate_standard_executive_summary(
    schema: dict,
    quality: dict,
    kpis: list,
    temporal: dict | None,
    correlations: dict | None,
    anomalies: dict | None,
    goal: str,
) -> str:
    """Generate template-based executive summary for Standard mode."""
    row_count = schema.get("row_count", 0)
    col_count = schema.get("col_count", 0)
    quality_score = quality.get("overall_score", 0) * 100
    
    type_summary = schema.get("type_summary", {})
    num_cols = type_summary.get("numeric", 0)
    cat_cols = type_summary.get("categorical", 0)
    
    # Top KPI
    top_kpi = kpis[0] if kpis else None
    top_kpi_str = f"Key metric: {top_kpi['name']} = {top_kpi['formatted_value']}. " if top_kpi else ""
    
    # Trend summary
    trend_str = ""
    if temporal and temporal.get("trends"):
        trend = temporal["trends"][0]
        direction = trend.get("direction", "stable")
        col = trend.get("metric_column", "")
        if direction != "stable":
            trend_str = f"Trend: {col} is {direction}. "
    
    # Anomaly summary
    anomaly_str = ""
    if anomalies:
        total = anomalies.get("outlier_summary", {}).get("total_outliers", 0)
        if total > 0:
            anomaly_str = f"Detected {total:,} outliers. "
    
    summary = (
        f"Comprehensive analysis of **{row_count:,} records** across **{col_count} columns** "
        f"({num_cols} numeric, {cat_cols} categorical). "
        f"Data quality: **{quality_score:.0f}%**. "
        f"{top_kpi_str}"
        f"{trend_str}"
        f"{anomaly_str}"
        f"Goal: {goal[:80]}."
    )
    
    return summary


def _build_standard_ui_payload(state: dict, insights: list, summary: str) -> dict:
    """Build Streamlit-ready UI payload for Standard mode."""
    return {
        "filename": state.get("filename", "data.csv"),
        "row_count": state.get("row_count", 0),
        "col_count": state.get("col_count", 0),
        "schema_profile": state.get("schema_profile"),
        "data_quality": state.get("data_quality"),
        "kpis": state.get("auto_kpis", []),
        "numeric_summary": state.get("numeric_summary", []),
        "categorical_summary": state.get("categorical_summary", []),
        "distributions": state.get("distributions"),
        "temporal_analysis": state.get("temporal_analysis"),
        "segmentation": state.get("segmentation"),
        "correlations": state.get("correlations"),
        "anomalies": state.get("anomalies"),
        "insights": insights,
        "executive_summary": summary,
        "warnings": state.get("warnings", []),
        "goal": state.get("resolved_goal"),
        "goal_source": state.get("goal_source"),
        "analysis_depth": "standard",
        "dashboard_plan": state.get("dashboard_plan", []),
    }


# =============================================================================
# NODE: PLAN VISUALS (Shared by Quick & Standard modes)
# =============================================================================

def plan_visuals_node(state: dict) -> dict:
    """
    Reasoning node: Decide which insights deserve visualization.
    
    This node does NOT generate charts. It analyzes insights and statistical
    outputs to create a structured visualization plan that the frontend
    can execute.
    
    Decision criteria:
    - Insight has supporting numerical data
    - Visualization would add clarity beyond text
    - Data columns exist and are appropriate for chart type
    - Quality over quantity (max 4 charts)
    
    Input state:
        - insights: list[dict] (ranked, validated)
        - numeric_summary: list[dict]
        - categorical_summary: list[dict]
        - schema_profile: dict
        - analysis_mode: str ("quick" or "standard")
        - resolved_goal: str | None
        - distributions: list[dict] | None (Standard mode)
        - temporal_analysis: dict | None (Standard mode)
        - correlations: dict | None (Standard mode)
        - anomalies: dict | None (Standard mode)
        
    Output state updates:
        - dashboard_plan: list[dict] (max 4 visualization specs)
        - current_node: str
        - progress: float
    """
    node_name = "plan_visuals"
    _emit_progress(state, node_name, 0.96, "Planning visualizations...")
    
    # Extract inputs (defensive)
    insights = state.get("insights", [])
    numeric_summary = state.get("numeric_summary", [])
    categorical_summary = state.get("categorical_summary", [])
    schema = state.get("schema_profile", {})
    analysis_mode = state.get("analysis_mode", "quick")
    goal = state.get("resolved_goal", "")
    
    # Standard mode extras
    distributions = state.get("distributions", [])
    temporal = state.get("temporal_analysis")
    correlations = state.get("correlations")
    anomalies = state.get("anomalies")
    
    warnings = list(state.get("warnings", []))
    
    # Get available columns by type
    columns = schema.get("columns", [])
    numeric_cols = [c["name"] for c in columns if c.get("inferred_type") == "numeric"]
    categorical_cols = [c["name"] for c in columns if c.get("inferred_type") == "categorical"]
    datetime_cols = [c["name"] for c in columns if c.get("inferred_type") == "temporal"]
    
    # Graceful degradation: if no insights or no numeric columns, return empty plan
    if not insights or not numeric_cols:
        _emit_progress(state, node_name, 0.98, "No visualizations needed", "complete")
        return {
            "dashboard_plan": [],
            "current_node": node_name,
            "progress": 0.98,
            "progress_message": "No suitable data for visualization",
        }
    
    # =========================================================================
    # VISUALIZATION PLANNING LOGIC
    # =========================================================================
    
    candidates = []
    
    # 1. Scan insights for visualization opportunities
    for idx, insight in enumerate(insights):
        viz_spec = _evaluate_insight_for_visualization(
            insight=insight,
            insight_idx=idx,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
        )
        if viz_spec:
            candidates.append(viz_spec)
    
    # 2. Standard mode: Add specialized visualizations
    if analysis_mode == "standard":
        # Temporal trend chart
        if temporal and temporal.get("trends"):
            trend_viz = _plan_temporal_visualization(temporal, datetime_cols, numeric_cols)
            if trend_viz:
                candidates.append(trend_viz)
        
        # Correlation scatter plot
        if correlations and correlations.get("top_correlations"):
            corr_viz = _plan_correlation_visualization(correlations, numeric_cols)
            if corr_viz:
                candidates.append(corr_viz)
        
        # Distribution histogram
        if distributions:
            dist_viz = _plan_distribution_visualization(distributions, numeric_cols)
            if dist_viz:
                candidates.append(dist_viz)
    
    # 3. Rank and select top 4 visualizations
    ranked = _rank_visualizations(candidates, goal)
    dashboard_plan = ranked[:4]
    
    # Add position/order to each visualization
    for i, viz in enumerate(dashboard_plan):
        viz["position"] = i + 1
    
    _emit_progress(state, node_name, 0.98, f"Planned {len(dashboard_plan)} visualizations", "complete")
    
    return {
        "dashboard_plan": sanitize_dict_for_json(dashboard_plan),
        "current_node": node_name,
        "progress": 0.98,
        "progress_message": f"Planned {len(dashboard_plan)} visualizations",
    }


# =============================================================================
# VISUALIZATION PLANNING HELPERS
# =============================================================================

def _evaluate_insight_for_visualization(
    insight: dict,
    insight_idx: int,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
    numeric_summary: list[dict],
    categorical_summary: list[dict],
) -> dict | None:
    """
    Evaluate whether an insight should be visualized.
    
    Returns a visualization spec if appropriate, None otherwise.
    """
    category = insight.get("category", "")
    severity = insight.get("severity", "info")
    title = insight.get("title", "")
    supporting_data = insight.get("supporting_data", {})
    
    # Skip low-value categories for visualization
    if category in ["quality", "info"]:
        return None
    
    # KPI insights → Bar chart comparing top KPIs
    if category == "kpi" and numeric_summary:
        # Find top 3-5 numeric columns by variance/range
        top_metrics = [ns.get("column") for ns in numeric_summary[:5] if ns.get("column")]
        if len(top_metrics) >= 2:
            return {
                "chart_type": "bar",
                "title": "Key Metrics Overview",
                "x_column": "_metric_name",  # Special: will be metric names
                "y_column": "_metric_value",  # Special: will be aggregated values
                "aggregation": "mean",
                "insight_ref": insight_idx,
                "priority": "high" if severity == "warning" else "medium",
                "rationale": "Compare key business metrics at a glance",
                "data_columns": top_metrics,
            }
    
    # Segmentation insights → Bar chart by category
    if category == "segmentation":
        segment_col = supporting_data.get("segment_column", "")
        kpi_col = supporting_data.get("kpi_column", "")
        
        if segment_col in categorical_cols and kpi_col in numeric_cols:
            return {
                "chart_type": "bar",
                "title": f"{kpi_col} by {segment_col}",
                "x_column": segment_col,
                "y_column": kpi_col,
                "aggregation": "sum",
                "insight_ref": insight_idx,
                "priority": "high",
                "rationale": f"Visualize performance differences across {segment_col} segments",
                "data_columns": [segment_col, kpi_col],
            }
    
    # Trend insights → Line chart
    if category == "trend":
        metric_col = supporting_data.get("metric_column", "")
        date_col = supporting_data.get("date_column", "")
        
        if not date_col and datetime_cols:
            date_col = datetime_cols[0]
        
        if date_col and metric_col in numeric_cols:
            return {
                "chart_type": "line",
                "title": f"{metric_col} Over Time",
                "x_column": date_col,
                "y_column": metric_col,
                "aggregation": "sum",
                "insight_ref": insight_idx,
                "priority": "high",
                "rationale": f"Show temporal pattern in {metric_col}",
                "data_columns": [date_col, metric_col],
            }
    
    # Correlation insights → Scatter plot
    if category == "correlation":
        col_a = supporting_data.get("column_a", "")
        col_b = supporting_data.get("column_b", "")
        
        if col_a in numeric_cols and col_b in numeric_cols:
            return {
                "chart_type": "scatter",
                "title": f"{col_a} vs {col_b}",
                "x_column": col_a,
                "y_column": col_b,
                "aggregation": "none",
                "insight_ref": insight_idx,
                "priority": "medium",
                "rationale": f"Visualize relationship between {col_a} and {col_b}",
                "data_columns": [col_a, col_b],
            }
    
    # Distribution insights → Histogram
    if category == "distribution":
        col = supporting_data.get("column", "")
        
        if col in numeric_cols:
            return {
                "chart_type": "histogram",
                "title": f"Distribution of {col}",
                "x_column": col,
                "y_column": None,
                "aggregation": "none",
                "insight_ref": insight_idx,
                "priority": "medium",
                "rationale": f"Show value distribution and identify skewness in {col}",
                "data_columns": [col],
            }
    
    # Anomaly insights → Could be bar showing outlier counts by column
    if category == "anomaly" and supporting_data.get("column_outliers"):
        return {
            "chart_type": "bar",
            "title": "Outliers by Column",
            "x_column": "_column_name",
            "y_column": "_outlier_count",
            "aggregation": "none",
            "insight_ref": insight_idx,
            "priority": "medium",
            "rationale": "Highlight which columns have data quality issues",
            "data_columns": [],  # Will be derived from anomaly data
        }
    
    return None


def _plan_temporal_visualization(
    temporal: dict,
    datetime_cols: list[str],
    numeric_cols: list[str],
) -> dict | None:
    """Plan a line chart for temporal trends."""
    trends = temporal.get("trends", [])
    if not trends or not datetime_cols:
        return None
    
    # Pick the strongest trend
    best_trend = max(trends, key=lambda t: abs(t.get("slope_pct", 0)))
    metric_col = best_trend.get("metric_column", "")
    date_col = best_trend.get("date_column", datetime_cols[0])
    direction = best_trend.get("direction", "stable")
    
    if metric_col not in numeric_cols:
        return None
    
    return {
        "chart_type": "line",
        "title": f"{metric_col} Trend ({direction.title()})",
        "x_column": date_col,
        "y_column": metric_col,
        "aggregation": "sum",
        "insight_ref": None,  # Not tied to specific insight
        "priority": "high",
        "rationale": f"Primary trend: {metric_col} is {direction}",
        "data_columns": [date_col, metric_col],
    }


def _plan_correlation_visualization(
    correlations: dict,
    numeric_cols: list[str],
) -> dict | None:
    """Plan a scatter plot for strongest correlation."""
    top_corr = correlations.get("top_correlations", [])
    if not top_corr:
        return None
    
    # Pick strongest correlation
    best = top_corr[0]
    col_a = best.get("column_a", "")
    col_b = best.get("column_b", "")
    r = best.get("correlation", 0)
    
    if col_a not in numeric_cols or col_b not in numeric_cols:
        return None
    
    strength = "Strong" if abs(r) >= 0.7 else "Moderate"
    direction = "Positive" if r > 0 else "Negative"
    
    return {
        "chart_type": "scatter",
        "title": f"{col_a} vs {col_b}",
        "x_column": col_a,
        "y_column": col_b,
        "aggregation": "none",
        "insight_ref": None,
        "priority": "high" if abs(r) >= 0.7 else "medium",
        "rationale": f"{strength} {direction.lower()} correlation (r={r:.2f})",
        "data_columns": [col_a, col_b],
    }


def _plan_distribution_visualization(
    distributions: list[dict],
    numeric_cols: list[str],
) -> dict | None:
    """Plan a histogram for most interesting distribution."""
    if not distributions:
        return None
    
    # Prefer non-normal distributions (more interesting to visualize)
    non_normal = [d for d in distributions if not d.get("is_normal", True)]
    target = non_normal[0] if non_normal else distributions[0]
    
    col = target.get("column", "")
    dist_type = target.get("distribution_type", "unknown")
    
    if col not in numeric_cols:
        return None
    
    return {
        "chart_type": "histogram",
        "title": f"Distribution of {col}",
        "x_column": col,
        "y_column": None,
        "aggregation": "none",
        "insight_ref": None,
        "priority": "medium",
        "rationale": f"{col} shows {dist_type} distribution pattern",
        "data_columns": [col],
    }


def _rank_visualizations(candidates: list[dict], goal: str) -> list[dict]:
    """
    Rank visualization candidates by value.
    
    Ranking factors:
    1. Priority (high > medium)
    2. Chart type diversity (prefer variety)
    3. Goal alignment (if goal mentions specific metrics)
    """
    if not candidates:
        return []
    
    # Priority scores
    priority_scores = {"high": 100, "medium": 50}
    
    # Track chart types seen for diversity scoring
    chart_type_counts = {}
    
    def score(viz: dict) -> float:
        s = priority_scores.get(viz.get("priority", "medium"), 50)
        
        # Diversity bonus: first of each chart type gets +20
        chart_type = viz.get("chart_type", "")
        if chart_type not in chart_type_counts:
            s += 20
            chart_type_counts[chart_type] = 0
        
        # Goal alignment bonus
        if goal:
            goal_lower = goal.lower()
            for col in viz.get("data_columns", []):
                if col and col.lower() in goal_lower:
                    s += 30
                    break
        
        return s
    
    # Sort by score descending
    ranked = sorted(candidates, key=score, reverse=True)
    
    # Ensure chart type diversity in top 4 (max 2 of same type)
    final = []
    types_used = {}
    
    for viz in ranked:
        chart_type = viz.get("chart_type", "")
        
        # Allow max 2 of same chart type
        if types_used.get(chart_type, 0) >= 2:
            continue
        
        final.append(viz)
        types_used[chart_type] = types_used.get(chart_type, 0) + 1
        
        if len(final) >= 4:
            break
    
    return final


# =============================================================================
# NODE REGISTRY
# =============================================================================

NODE_REGISTRY = {
    "ingest_data": ingest_data_node,
    "profile_data": profile_data_node,
    "resolve_goal": resolve_goal_node,
    "analyze_quick": analyze_quick_node,
    "synthesize_insights": synthesize_insights_node,
    "handle_error": handle_error_node,
    # Standard mode nodes
    "analyze_standard": analyze_standard_node,
    "synthesize_standard_insights": synthesize_standard_insights_node,
    # Visualization planning (shared)
    "plan_visuals": plan_visuals_node,
}
