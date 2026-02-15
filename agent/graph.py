# graph.py — LangGraph workflow definition
# Defines state machine, node edges, and conditional routing
"""
graph.py — LangGraph Workflow Definition

Production implementation of the Autonomous Data Analyst agent workflow.
Wires nodes into a single-agent graph with error routing.

Flow:
    START → ingest → profile → resolve_goal → analyze_quick → synthesize → END
                ↓         ↓            ↓              ↓             ↓
              [ERROR] → [ERROR] → [ERROR] → [ERROR] → [ERROR] → handle_error → END

Any node that sets state["error"] routes to handle_error_node.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph

from agent.state import AgentState, create_initial_state
from agent.nodes import (
    ingest_data_node,
    profile_data_node,
    resolve_goal_node,
    analyze_quick_node,
    synthesize_insights_node,
    handle_error_node,
    # Standard mode nodes
    analyze_standard_node,
    synthesize_standard_insights_node,
    # Visualization planning
    plan_visuals_node,
)


# =============================================================================
# CONDITIONAL ROUTING
# =============================================================================

def route_after_node(state: AgentState) -> Literal["continue", "error"]:
    """
    Conditional router: check if error occurred, route accordingly.
    
    Returns:
        "error" if state has error, "continue" otherwise
    """
    if state.get("error"):
        return "error"
    return "continue"


def route_after_resolve_goal(state: AgentState) -> Literal["quick", "standard", "error"]:
    """
    Mode-aware router after resolve_goal node.
    
    Routes to appropriate analysis path based on analysis_mode:
    - "standard" → analyze_standard → synthesize_standard_insights → END
    - default    → analyze_quick → synthesize_insights → END
    
    Returns:
        "error" if error, "standard" for Standard mode, "quick" otherwise
    """
    if state.get("error"):
        return "error"
    
    mode = state.get("analysis_mode", "quick")
    if mode == "standard":
        return "standard"
    return "quick"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph workflow for Quick mode analysis.
    
    Returns:
        Compiled StateGraph ready for invocation
    """
    # Initialize graph with state schema
    workflow = StateGraph(AgentState)
    
    # =========================================================================
    # ADD NODES
    # =========================================================================
    
    workflow.add_node("ingest_data", ingest_data_node)
    workflow.add_node("profile_data", profile_data_node)
    workflow.add_node("resolve_goal", resolve_goal_node)
    workflow.add_node("analyze_quick", analyze_quick_node)
    workflow.add_node("synthesize_insights", synthesize_insights_node)
    workflow.add_node("handle_error", handle_error_node)
    # Standard mode nodes
    workflow.add_node("analyze_standard", analyze_standard_node)
    workflow.add_node("synthesize_standard_insights", synthesize_standard_insights_node)
    # Visualization planning (shared by both modes)
    workflow.add_node("plan_visuals", plan_visuals_node)
    
    # =========================================================================
    # DEFINE EDGES (Happy Path + Error Routing)
    # =========================================================================
    
    # START → ingest_data
    workflow.add_edge(START, "ingest_data")
    
    # ingest_data → profile_data OR handle_error
    workflow.add_conditional_edges(
        "ingest_data",
        route_after_node,
        {
            "continue": "profile_data",
            "error": "handle_error",
        },
    )
    
    # profile_data → resolve_goal OR handle_error
    workflow.add_conditional_edges(
        "profile_data",
        route_after_node,
        {
            "continue": "resolve_goal",
            "error": "handle_error",
        },
    )
    
    # resolve_goal → analyze_quick OR analyze_standard OR handle_error
    workflow.add_conditional_edges(
        "resolve_goal",
        route_after_resolve_goal,
        {
            "quick": "analyze_quick",
            "standard": "analyze_standard",
            "error": "handle_error",
        },
    )
    
    # analyze_quick → synthesize_insights OR handle_error
    workflow.add_conditional_edges(
        "analyze_quick",
        route_after_node,
        {
            "continue": "synthesize_insights",
            "error": "handle_error",
        },
    )
    
    # synthesize_insights → plan_visuals OR handle_error
    workflow.add_conditional_edges(
        "synthesize_insights",
        route_after_node,
        {
            "continue": "plan_visuals",
            "error": "handle_error",
        },
    )
    
    # =========================================================================
    # STANDARD MODE PATH
    # =========================================================================
    
    # analyze_standard → synthesize_standard_insights OR handle_error
    workflow.add_conditional_edges(
        "analyze_standard",
        route_after_node,
        {
            "continue": "synthesize_standard_insights",
            "error": "handle_error",
        },
    )
    
    # synthesize_standard_insights → plan_visuals OR handle_error
    workflow.add_conditional_edges(
        "synthesize_standard_insights",
        route_after_node,
        {
            "continue": "plan_visuals",
            "error": "handle_error",
        },
    )
    
    # =========================================================================
    # VISUALIZATION PLANNING (Shared terminal path)
    # =========================================================================
    
    # plan_visuals → END OR handle_error
    workflow.add_conditional_edges(
        "plan_visuals",
        route_after_node,
        {
            "continue": END,
            "error": "handle_error",
        },
    )
    
    # handle_error → END (terminal node)
    workflow.add_edge("handle_error", END)
    
    return workflow


def compile_agent_graph():
    """
    Build and compile the agent graph.
    
    Returns:
        Compiled graph ready for .invoke() or .stream()
    """
    workflow = build_agent_graph()
    return workflow.compile()


# =============================================================================
# GRAPH EXECUTION
# =============================================================================

# Compiled graph singleton (lazy initialization)
_compiled_graph = None


def get_compiled_graph():
    """
    Get or create the compiled graph singleton.
    
    Returns:
        Compiled StateGraph
    """
    global _compiled_graph
    # Always recompile during development to pick up changes
    _compiled_graph = compile_agent_graph()
    return _compiled_graph


def run_analysis(
    raw_file: bytes,
    filename: str,
    user_goal: str | None = None,
    analysis_depth: str = "quick",
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Run the complete analysis workflow.
    
    This is the main entry point for executing the agent.
    
    Args:
        raw_file: Raw CSV file bytes
        filename: Original filename
        user_goal: Optional user-provided analysis goal
        analysis_depth: Analysis depth ("quick", "standard", "deep")
        progress_callback: Optional callback for progress updates
        
    Returns:
        Final AgentState dict with ui_payload containing results or error
        
    Example:
        result = run_analysis(
            raw_file=uploaded_file.read(),
            filename="sales.csv",
            user_goal="Analyze revenue trends",
            progress_callback=lambda p: st.progress(p["progress"])
        )
        
        if result.get("ui_payload", {}).get("is_error"):
            st.error(result["ui_payload"]["error_message"])
        else:
            display_results(result["ui_payload"])
    """
    # Create initial state
    initial_state = create_initial_state(
        raw_file=raw_file,
        filename=filename,
        user_goal=user_goal,
        analysis_depth=analysis_depth,
        progress_callback=progress_callback,
    )
    
    # Get compiled graph
    graph = get_compiled_graph()
    
    # Execute graph
    final_state = graph.invoke(initial_state)
    
    return final_state


def stream_analysis(
    raw_file: bytes,
    filename: str,
    user_goal: str | None = None,
    analysis_depth: str = "quick",
    progress_callback: Callable[[dict], None] | None = None,
):
    """
    Stream the analysis workflow, yielding state after each node.
    
    Useful for real-time progress updates in Streamlit.
    
    Args:
        raw_file: Raw CSV file bytes
        filename: Original filename
        user_goal: Optional user-provided analysis goal
        analysis_depth: Analysis depth ("quick", "standard", "deep")
        progress_callback: Optional callback for progress updates
        
    Yields:
        Tuple of (node_name, state_snapshot) after each node execution
        
    Example:
        for node_name, state in stream_analysis(file_bytes, "data.csv"):
            st.write(f"Completed: {node_name}")
            if state.get("progress"):
                progress_bar.progress(state["progress"])
    """
    # Create initial state
    initial_state = create_initial_state(
        raw_file=raw_file,
        filename=filename,
        user_goal=user_goal,
        analysis_depth=analysis_depth,
        progress_callback=progress_callback,
    )
    
    # Get compiled graph
    graph = get_compiled_graph()
    
    # Track accumulated state
    accumulated_state = dict(initial_state)
    
    # Stream execution
    for event in graph.stream(initial_state):
        # event is a dict with node_name as key and state update as value
        for node_name, state_update in event.items():
            # Merge update into accumulated state
            accumulated_state.update(state_update)
            yield node_name, accumulated_state


# =============================================================================
# GRAPH VISUALIZATION (Development Only)
# =============================================================================

def get_graph_mermaid() -> str:
    """
    Get Mermaid diagram representation of the graph.
    
    Useful for documentation and debugging.
    
    Returns:
        Mermaid diagram string
    """
    graph = get_compiled_graph()
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        # Fallback manual diagram
        return """
graph TD
    START --> ingest_data
    ingest_data -->|success| profile_data
    ingest_data -->|error| handle_error
    profile_data -->|success| resolve_goal
    profile_data -->|error| handle_error
    resolve_goal -->|quick| analyze_quick
    resolve_goal -->|standard| analyze_standard
    resolve_goal -->|error| handle_error
    analyze_quick -->|success| synthesize_insights
    analyze_quick -->|error| handle_error
    synthesize_insights -->|success| END
    synthesize_insights -->|error| handle_error
    analyze_standard -->|success| synthesize_standard_insights
    analyze_standard -->|error| handle_error
    synthesize_standard_insights -->|success| END
    synthesize_standard_insights -->|error| handle_error
    handle_error --> END
"""
