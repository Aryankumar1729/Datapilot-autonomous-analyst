# validators.py — Input sanitization & validation
# File type checks, column schema validation, user input guards
"""
validators.py — Input Sanitization & Validation

Production implementation for:
- File type validation
- User input sanitization
- Goal validation against schema
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

ALLOWED_EXTENSIONS = {".csv"}
MAX_GOAL_LENGTH = 500
MIN_GOAL_LENGTH = 3

# Patterns for extracting column references from user goals
COLUMN_REF_PATTERN = re.compile(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"")


# =============================================================================
# FILE VALIDATION
# =============================================================================

def validate_file_extension(filename: str) -> tuple[bool, str | None]:
    """
    Validate that file has an allowed extension.
    
    Returns:
        (is_valid, error_message)
    """
    if not filename:
        return False, "No filename provided"
    
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, None


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str | None]:
    """
    Validate that a DataFrame is suitable for analysis.
    
    Returns:
        (is_valid, error_message)
    """
    if df is None:
        return False, "No data provided"
    
    if not isinstance(df, pd.DataFrame):
        return False, "Data is not a valid DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    
    if len(df) == 0:
        return False, "DataFrame has no rows"
    
    # Check for minimum viable data
    if len(df) < 2:
        return False, "DataFrame must have at least 2 rows for analysis"
    
    if len(df.columns) < 1:
        return False, "DataFrame must have at least 1 column"
    
    return True, None


# =============================================================================
# GOAL VALIDATION
# =============================================================================

def sanitize_user_goal(goal: str | None) -> str | None:
    """
    Sanitize user-provided analysis goal.
    
    Returns:
        Cleaned goal string or None if invalid/empty
    """
    if goal is None:
        return None
    
    if not isinstance(goal, str):
        return None
    
    # Strip whitespace
    goal = goal.strip()
    
    # Check length
    if len(goal) < MIN_GOAL_LENGTH:
        return None
    
    if len(goal) > MAX_GOAL_LENGTH:
        goal = goal[:MAX_GOAL_LENGTH]
    
    # Remove potentially dangerous characters (basic sanitization)
    goal = re.sub(r"[<>{}[\]\\]", "", goal)
    
    return goal if goal else None


def validate_goal_against_schema(
    goal: str,
    schema_profile: dict,
) -> tuple[bool, float, list[str]]:
    """
    Validate that a user goal references valid columns.
    
    Args:
        goal: User-provided analysis goal
        schema_profile: Output from profile_schema()
        
    Returns:
        (is_valid, confidence, matched_columns)
    """
    if not goal or not schema_profile:
        return False, 0.0, []
    
    column_names = {
        c["name"].lower() 
        for c in schema_profile.get("columns", [])
    }
    
    # Extract quoted column references from goal
    matches = COLUMN_REF_PATTERN.findall(goal)
    explicit_refs = [m[0] or m[1] or m[2] for m in matches if any(m)]
    
    # Also check for column names mentioned directly (case-insensitive)
    goal_lower = goal.lower()
    mentioned_columns = [
        name for name in column_names 
        if name in goal_lower
    ]
    
    all_refs = set(r.lower() for r in explicit_refs) | set(mentioned_columns)
    matched = [ref for ref in all_refs if ref in column_names]
    
    # Calculate confidence
    if not all_refs:
        # No specific columns mentioned - generic goal
        confidence = 0.6
        is_valid = True
    elif len(matched) == len(all_refs):
        # All referenced columns exist
        confidence = 0.95
        is_valid = True
    elif len(matched) > 0:
        # Some columns matched
        confidence = 0.7
        is_valid = True
    else:
        # No matches - references invalid columns
        confidence = 0.3
        is_valid = False
    
    return is_valid, confidence, matched


# =============================================================================
# INPUT SANITIZATION
# =============================================================================

def sanitize_analysis_depth(depth: str | None) -> str:
    """
    Sanitize analysis depth input.
    
    Returns:
        One of: "quick", "standard", "deep"
    """
    valid_depths = {"quick", "standard", "deep"}
    
    if not depth:
        return "standard"
    
    depth_lower = str(depth).lower().strip()
    
    if depth_lower in valid_depths:
        return depth_lower
    
    return "standard"


def sanitize_dict_for_json(obj: Any) -> Any:
    """
    Recursively sanitize a dict/list for JSON serialization.
    Handles numpy types, NaN, Inf, etc.
    """
    import numpy as np
    
    if obj is None:
        return None
    
    if isinstance(obj, dict):
        return {k: sanitize_dict_for_json(v) for k, v in obj.items()}
    
    if isinstance(obj, list):
        return [sanitize_dict_for_json(v) for v in obj]
    
    if isinstance(obj, (np.integer,)):
        return int(obj)
    
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    
    if isinstance(obj, np.ndarray):
        return sanitize_dict_for_json(obj.tolist())
    
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):  # NaN check
            return None
        return obj
    
    return obj
