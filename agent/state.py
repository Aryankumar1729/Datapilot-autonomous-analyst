# state.py — Shared AgentState schema
# Pydantic/TypedDict definition for state passed between nodes
"""
state.py — Agent State Schema

Defines the TypedDict structure for state passed between LangGraph nodes.
This schema is FINAL and LOCKED per project specification.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all agent nodes.
    
    All fields are optional (total=False) to support partial updates.
    """
    
    # =========================================================================
    # INPUT LAYER
    # =========================================================================
    raw_file: bytes | None  # Raw uploaded file bytes
    filename: str | None  # Original filename
    user_goal: str | None  # User-provided analysis goal (optional)
    analysis_depth: str  # "quick" | "standard" | "deep"
    
    # =========================================================================
    # DATA LAYER
    # =========================================================================
    dataframe: pd.DataFrame | None  # Parsed DataFrame
    schema_profile: dict | None  # Output from profile_schema()
    row_count: int
    col_count: int
    
    # =========================================================================
    # QUALITY LAYER
    # =========================================================================
    data_quality: dict | None  # Output from scan_data_quality()
    
    # =========================================================================
    # GOAL LAYER
    # =========================================================================
    resolved_goal: str | None  # Final analysis goal (user or inferred)
    goal_source: str | None  # "user" | "inferred"
    goal_confidence: float  # 0.0 - 1.0
    
    # =========================================================================
    # ANALYSIS LAYER (Quick Mode)
    # =========================================================================
    numeric_summary: list[dict] | None
    categorical_summary: list[dict] | None
    auto_kpis: list[dict] | None
    
    # =========================================================================
    # SYNTHESIS LAYER
    # =========================================================================
    insights: list[dict] | None  # Generated business insights
    executive_summary: str | None  # One-paragraph summary
    warnings: list[str]  # Analysis warnings
    
    # =========================================================================
    # OUTPUT LAYER
    # =========================================================================
    ui_payload: dict | None  # Streamlit-ready structured output
    dashboard_plan: list[dict] | None  # Visualization plan from plan_visuals_node
    
    # =========================================================================
    # CONTROL LAYER
    # =========================================================================
    current_node: str | None  # Current executing node
    analysis_mode: str | None  # "quick" | "standard" (runtime mode)
    progress: float  # 0.0 - 1.0
    progress_message: str | None  # Human-readable progress
    
    # =========================================================================
    # ERROR LAYER
    # =========================================================================
    error: str | None  # Error message if any
    error_type: str | None  # Error classification
    failed_node: str | None  # Node that failed
    partial_results: bool  # Whether partial results are available
    recovery_hint: str | None  # User-facing recovery suggestion
    
    # =========================================================================
    # CALLBACKS (not persisted)
    # =========================================================================
    progress_callback: Callable[[dict], None] | None


def create_initial_state(
    raw_file: bytes | None = None,
    filename: str | None = None,
    user_goal: str | None = None,
    analysis_depth: str = "quick",
    progress_callback: Callable[[dict], None] | None = None,
) -> AgentState:
    """
    Create a fresh AgentState with default values.
    
    Args:
        raw_file: Raw uploaded file bytes
        filename: Original filename
        user_goal: Optional user-provided analysis goal
        analysis_depth: One of "quick", "standard", "deep"
        progress_callback: Optional callback for progress updates
        
    Returns:
        Initialized AgentState dict
    """
    return AgentState(
        # Input
        raw_file=raw_file,
        filename=filename,
        user_goal=user_goal,
        analysis_depth=analysis_depth,
        
        # Data
        dataframe=None,
        schema_profile=None,
        row_count=0,
        col_count=0,
        
        # Quality
        data_quality=None,
        
        # Goal
        resolved_goal=None,
        goal_source=None,
        goal_confidence=0.0,
        
        # Analysis
        numeric_summary=None,
        categorical_summary=None,
        auto_kpis=None,
        
        # Synthesis
        insights=None,
        executive_summary=None,
        warnings=[],
        
        # Output
        ui_payload=None,
        dashboard_plan=None,
        
        # Control
        current_node=None,
        analysis_mode=analysis_depth,  # Copy depth to mode for node access
        progress=0.0,
        progress_message=None,
        
        # Error
        error=None,
        error_type=None,
        failed_node=None,
        partial_results=False,
        recovery_hint=None,
        
        # Callbacks
        progress_callback=progress_callback,
    )
