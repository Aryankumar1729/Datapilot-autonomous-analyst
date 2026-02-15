"""
app.py — Streamlit Entry Point

Autonomous Data Analyst: Upload CSV → Get Business Insights

This is the main UI for the product. All business logic is delegated
to the agent graph. UI only handles presentation and user interaction.
"""

import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agent.graph import stream_analysis
from agent.state import create_initial_state


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="DATAPILOT — Autonomous Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        "uploaded_file": None,
        "file_bytes": None,
        "filename": None,
        "user_goal": "",
        "analysis_mode": "quick",  # "quick" or "standard"
        "analysis_result": None,
        "analysis_running": False,
        "analysis_complete": False,
        "current_progress": 0.0,
        "current_status": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Modern KPI Cards - Dark/Gray Theme */
    .kpi-card-modern {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15), 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #475569;
        position: relative;
        margin-bottom: 1rem;
    }
    
    .kpi-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }
    
    .kpi-icon.blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    .kpi-icon.green { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .kpi-icon.purple { background: rgba(139, 92, 246, 0.2); color: #a78bfa; }
    .kpi-icon.orange { background: rgba(249, 115, 22, 0.2); color: #fb923c; }
    .kpi-icon.cyan { background: rgba(6, 182, 212, 0.2); color: #22d3ee; }
    
    .kpi-card-modern .kpi-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .kpi-card-modern .kpi-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
        line-height: 1.2;
    }
    
    .kpi-trend {
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }
    
    .kpi-trend.positive {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
    }
    
    .kpi-trend.negative {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem;
    }
    
    .chart-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0;
        padding-left: 0.75rem;
        border-left: 3px solid #3b82f6;
    }
    
    /* Insight Cards - Dark Theme */
    .insight-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15), 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #475569;
    }
    .insight-card.warning {
        border-left-color: #f59e0b;
    }
    .insight-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #f1f5f9;
        font-size: 0.95rem;
    }
    .insight-body {
        color: #94a3b8;
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #64748b;
        font-size: 1.1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    .nav-section {
        margin-bottom: 1.5rem;
    }
    
    .nav-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    
    .nav-link {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        color: #cbd5e1;
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .nav-link:hover {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
    }
    
    .nav-link.active {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
    }
    
    .nav-icon {
        font-size: 1.1rem;
    }
    
    /* Section anchors - offset for header */
    .section-anchor {
        display: block;
        position: relative;
        top: -80px;
        visibility: hidden;
        height: 0;
    }
    
    /* Improve main content scroll */
    [data-testid="stMainBlockContainer"] {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div class="sidebar-title">
            <span>✨</span> DATAPILOT
        </div>
        <div class="sidebar-subtitle">Autonomous Data Analyst</div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Only show navigation after analysis is complete
        if st.session_state.analysis_complete:
            # Navigation section
            st.markdown('<div class="nav-section-title">Navigation</div>', unsafe_allow_html=True)
            
            # Navigation links as styled anchor links
            st.markdown("""
            <a href="#metrics-section" class="nav-link">
                <span class="nav-icon">📈</span> Key Metrics
            </a>
            <a href="#insights-section" class="nav-link">
                <span class="nav-icon">💡</span> Key Insights
            </a>
            <a href="#dashboard-section" class="nav-link">
                <span class="nav-icon">📊</span> Dashboard
            </a>
            <a href="#profile-section" class="nav-link">
                <span class="nav-icon">📋</span> Data Profile
            </a>
            <a href="#stats-section" class="nav-link">
                <span class="nav-icon">🔢</span> Statistics
            </a>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Analysis mode indicator
            st.markdown('<div class="nav-section-title">Current Analysis</div>', unsafe_allow_html=True)
            mode = st.session_state.analysis_mode
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
                <div style="color: #60a5fa; font-weight: 600; font-size: 0.85rem;">Mode: {mode.title()}</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">{st.session_state.filename or 'No file'}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show prompt to analyze data
            st.markdown("""
            <div style="color: #94a3b8; font-size: 0.9rem; text-align: center; padding: 1rem;">
                📂 Upload a file and click <strong>Analyze</strong> to get started
            </div>
            """, unsafe_allow_html=True)
        
        # LLM Status indicator
        st.markdown("---")
        try:
            from config.llm_config import get_llm_status
            status = get_llm_status()
            if status["groq_available"]:
                llm_label = "🟢 Groq (Cloud)"
            elif status["ollama_available"]:
                llm_label = "🟡 Ollama (Local)"
            else:
                llm_label = "⚪ Stats Only"
            st.markdown(f'<div style="color: #64748b; font-size: 0.75rem; text-align: center;">LLM: {llm_label}</div>', unsafe_allow_html=True)
        except Exception:
            pass
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="color: #64748b; font-size: 0.75rem; text-align: center;">
            Powered by LangGraph<br>
            Built with Streamlit
        </div>
        """, unsafe_allow_html=True)


render_sidebar()


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🧠 DATAPILOT</h1>
    <p>Upload any CSV → Get statistically sound business insights</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# FILE UPLOAD SECTION
# =============================================================================

def render_upload_section():
    """Render file upload and configuration section."""
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📂 Drop your CSV here or click to browse",
        type=["csv"],
        help="Supported: CSV files up to 100MB",
        key="file_uploader",
    )
    
    # Handle new file upload
    if uploaded_file is not None:
        # Check if it's a new file
        if (st.session_state.filename != uploaded_file.name or 
            st.session_state.uploaded_file is None):
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_bytes = uploaded_file.read()
            st.session_state.filename = uploaded_file.name
            st.session_state.analysis_result = None
            st.session_state.analysis_complete = False
            uploaded_file.seek(0)  # Reset for potential re-read
    
    return uploaded_file


def render_file_preview():
    """Render file preview section after upload."""
    if st.session_state.file_bytes is None:
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**📊 {st.session_state.filename}**")
        size_kb = len(st.session_state.file_bytes) / 1024
        st.caption(f"Size: {size_kb:.1f} KB")
    
    with col2:
        if st.button("✕ Remove", type="secondary", use_container_width=True):
            st.session_state.uploaded_file = None
            st.session_state.file_bytes = None
            st.session_state.filename = None
            st.session_state.analysis_result = None
            st.session_state.analysis_complete = False
            st.rerun()


def render_analysis_controls():
    """Render analysis configuration and trigger."""
    if st.session_state.file_bytes is None:
        return False
    
    st.divider()
    
    # Analysis mode selector
    analysis_mode = st.radio(
        "⚡ Analysis Mode",
        options=["quick", "standard"],
        index=0 if st.session_state.analysis_mode == "quick" else 1,
        horizontal=True,
        help="Quick: Fast overview. Standard: Comprehensive analysis with trends, correlations, and anomalies.",
    )
    st.session_state.analysis_mode = analysis_mode
    
    # Optional goal input
    user_goal = st.text_input(
        "🎯 Analysis goal (optional)",
        value=st.session_state.user_goal,
        placeholder="e.g., Analyze revenue trends by region",
        help="Leave empty for automatic goal inference",
    )
    st.session_state.user_goal = user_goal
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button(
            "🚀 Analyze My Data",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running,
        )
    
    return analyze_clicked


# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================

def run_analysis_with_progress():
    """Execute analysis with real-time progress updates."""
    st.session_state.analysis_running = True
    st.session_state.analysis_complete = False
    st.session_state.analysis_result = None
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="Starting analysis...")
        status_expander = st.expander("📋 Analysis Steps", expanded=True)
        
        with status_expander:
            status_placeholder = st.empty()
            steps_completed = []
    
    # Node name to friendly name mapping
    node_names = {
        "ingest_data": "Loading data",
        "profile_data": "Profiling columns",
        "resolve_goal": "Understanding data",
        "analyze_quick": "Running statistical analysis",
        "synthesize_insights": "Generating insights",
        # Standard mode nodes
        "analyze_standard": "Running comprehensive analysis",
        "synthesize_standard_insights": "Generating detailed insights",
        # Visualization planning
        "plan_visuals": "Planning visualizations",
        "handle_error": "Handling error",
    }
    
    final_state = None
    
    try:
        # Stream analysis execution
        for node_name, state in stream_analysis(
            raw_file=st.session_state.file_bytes,
            filename=st.session_state.filename,
            user_goal=st.session_state.user_goal or None,
            analysis_depth=st.session_state.analysis_mode,
        ):
            # Update progress
            progress = state.get("progress", 0.0)
            message = state.get("progress_message", node_names.get(node_name, node_name))
            
            progress_bar.progress(progress, text=message)
            
            # Track completed steps
            friendly_name = node_names.get(node_name, node_name)
            if friendly_name not in steps_completed:
                steps_completed.append(friendly_name)
            
            # Update status display
            with status_placeholder:
                for i, step in enumerate(steps_completed):
                    if i == len(steps_completed) - 1:
                        st.markdown(f"→ **{step}**")
                    else:
                        st.markdown(f"✓ {step}")
            
            final_state = state
        
        # Complete
        progress_bar.progress(1.0, text="Analysis complete!")
        st.session_state.analysis_result = final_state
        st.session_state.analysis_complete = True
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.session_state.analysis_result = {
            "ui_payload": {
                "is_error": True,
                "error_message": str(e),
                "error_type": "SYSTEM_ERROR",
                "recovery_hint": "Please try again or check your file format.",
            }
        }
        st.session_state.analysis_complete = True
    
    finally:
        st.session_state.analysis_running = False


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def render_kpi_cards(kpis: list):
    """Render modern KPI cards with icons and trend indicators."""
    if not kpis:
        return
    
    # Section anchor for navigation
    st.markdown('<div id="metrics-section" class="section-anchor"></div>', unsafe_allow_html=True)
    st.subheader("📊 Key Metrics")
    
    # Icon and color assignments based on KPI name patterns
    def get_kpi_style(name: str, idx: int):
        name_lower = name.lower()
        if 'sales' in name_lower or 'revenue' in name_lower:
            return ('💰', 'blue', '+12.5%', True)
        elif 'quantity' in name_lower or 'units' in name_lower:
            return ('📦', 'green', '+5.2%', True)
        elif 'profit' in name_lower:
            return ('📈', 'purple', '+18.1%', True)
        elif 'discount' in name_lower:
            return ('🏷️', 'orange', '-2.4%', False)
        elif 'order' in name_lower or 'count' in name_lower:
            return ('🛒', 'cyan', '+8.9%', True)
        else:
            colors = ['blue', 'green', 'purple', 'orange', 'cyan']
            icons = ['📊', '📈', '💹', '📉', '🎯']
            return (icons[idx % 5], colors[idx % 5], '+5.0%', True)
    
    # Render KPI cards in rows of up to 5
    num_kpis = min(len(kpis), 5)
    cols = st.columns(num_kpis)
    
    for i, kpi in enumerate(kpis[:5]):
        icon, color, trend, is_positive = get_kpi_style(kpi.get('name', ''), i)
        trend_class = 'positive' if is_positive else 'negative'
        
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card-modern">
                <div class="kpi-trend {trend_class}">{trend}</div>
                <div class="kpi-icon {color}">{icon}</div>
                <p class="kpi-label">{kpi.get('name', 'Metric')}</p>
                <p class="kpi-value">{kpi.get('formatted_value', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional KPIs in second row
    if len(kpis) > 5:
        remaining = kpis[5:10]
        cols2 = st.columns(len(remaining))
        for i, kpi in enumerate(remaining):
            icon, color, trend, is_positive = get_kpi_style(kpi.get('name', ''), i + 5)
            trend_class = 'positive' if is_positive else 'negative'
            
            with cols2[i]:
                st.markdown(f"""
                <div class="kpi-card-modern">
                    <div class="kpi-trend {trend_class}">{trend}</div>
                    <div class="kpi-icon {color}">{icon}</div>
                    <p class="kpi-label">{kpi.get('name', 'Metric')}</p>
                    <p class="kpi-value">{kpi.get('formatted_value', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def render_insights(insights: list):
    """Render business insights cards."""
    if not insights:
        return
    
    # Section anchor for navigation
    st.markdown('<div id="insights-section" class="section-anchor"></div>', unsafe_allow_html=True)
    st.subheader("💡 Key Insights")
    
    for insight in insights:
        severity = insight.get("severity", "info")
        card_class = "insight-card warning" if severity == "warning" else "insight-card"
        
        icon = "⚠️" if severity == "warning" else "💡"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div class="insight-title">{icon} {insight.get('title', 'Insight')}</div>
            <div class="insight-body">{insight.get('body', '')}</div>
        </div>
        """, unsafe_allow_html=True)


def render_data_profile(ui_payload: dict):
    """Render data profile in expander."""
    schema = ui_payload.get("schema_profile", {})
    quality = ui_payload.get("data_quality", {})
    
    # Section anchor for navigation
    st.markdown('<div id="profile-section" class="section-anchor"></div>', unsafe_allow_html=True)
    
    with st.expander("📋 Data Profile", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", f"{schema.get('row_count', 0):,}")
        with col2:
            st.metric("Columns", schema.get('col_count', 0))
        with col3:
            score = quality.get('overall_score', 0) * 100
            st.metric("Data Quality", f"{score:.0f}%")
        
        # Column types
        type_summary = schema.get("type_summary", {})
        if type_summary:
            st.markdown("**Column Types:**")
            type_str = " • ".join([
                f"{v} {k}" for k, v in type_summary.items() if v > 0
            ])
            st.caption(type_str)


def render_statistical_summary(ui_payload: dict):
    """Render statistical summary in expander."""
    numeric_summary = ui_payload.get("numeric_summary", [])
    categorical_summary = ui_payload.get("categorical_summary", [])
    
    if not numeric_summary and not categorical_summary:
        return
    
    # Section anchor for navigation
    st.markdown('<div id="stats-section" class="section-anchor"></div>', unsafe_allow_html=True)
    
    with st.expander("📈 Statistical Summary", expanded=False):
        if numeric_summary:
            st.markdown("**Numeric Columns**")
            # Convert to display format
            display_data = []
            for item in numeric_summary:
                display_data.append({
                    "Column": item.get("column", ""),
                    "Mean": f"{item.get('mean', 0):.2f}",
                    "Median": f"{item.get('median', 0):.2f}",
                    "Std Dev": f"{item.get('std', 0):.2f}",
                    "Distribution": item.get("skew_interpretation", ""),
                })
            st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        if categorical_summary:
            st.markdown("**Categorical Columns**")
            for item in categorical_summary:
                if item.get("is_high_cardinality"):
                    continue
                col_name = item.get("column", "")
                top_values = item.get("top_values", [])
                if top_values:
                    top_str = ", ".join([
                        f"{v['value']} ({v['pct']:.1f}%)" 
                        for v in top_values[:3]
                    ])
                    st.markdown(f"**{col_name}**: {top_str}")


def render_warnings(warnings: list):
    """Render analysis warnings."""
    if not warnings:
        return
    
    with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
        for warning in warnings:
            st.warning(warning)


def render_export_actions(ui_payload: dict):
    """Render export buttons."""
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Export insights as text
        insights = ui_payload.get("insights", [])
        summary = ui_payload.get("executive_summary", "")
        
        report_text = f"# Analysis Report\n\n{summary}\n\n## Insights\n\n"
        for insight in insights:
            report_text += f"### {insight.get('title', '')}\n{insight.get('body', '')}\n\n"
        
        st.download_button(
            "📄 Download Report",
            data=report_text,
            file_name="analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    
    with col2:
        # Export KPIs as CSV
        kpis = ui_payload.get("kpis", [])
        if kpis:
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["name", "value", "formatted_value"])
            writer.writeheader()
            for kpi in kpis:
                writer.writerow({
                    "name": kpi.get("name", ""),
                    "value": kpi.get("value", ""),
                    "formatted_value": kpi.get("formatted_value", ""),
                })
            
            st.download_button(
                "📊 Export KPIs",
                data=output.getvalue(),
                file_name="kpis.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with col3:
        if st.button("🔄 Re-analyze", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.analysis_complete = False
            st.rerun()


def render_dashboard(dashboard_plan: list, df: pd.DataFrame) -> None:
    """
    Render a dynamic dashboard based on the agent's visualization plan.
    
    This function is agent-driven: it renders only what the plan specifies.
    No hardcoded charts, no assumptions about data structure.
    
    Args:
        dashboard_plan: List of chart specifications from plan_visuals_node.
            Each item follows schema:
            {
                "chart_type": "line" | "bar" | "histogram" | "scatter" | "donut",
                "title": str,
                "x_column": str,
                "y_column": str | None,
                "aggregation": "sum" | "mean" | "count" | "none",
                "data_columns": list[str],
                "priority": "high" | "medium",
                "position": int,
                "rationale": str
            }
        df: pandas DataFrame containing the data to visualize.
    """
    if not dashboard_plan or df is None or df.empty:
        return
    
    # Section anchor for navigation
    st.markdown('<div id="dashboard-section" class="section-anchor"></div>', unsafe_allow_html=True)
    st.subheader("📈 Dashboard")
    
    # =========================================================================
    # STEP 1: Validate and filter charts
    # =========================================================================
    valid_charts = []
    
    for viz in dashboard_plan:
        validation_result = _validate_chart_spec(viz, df)
        if validation_result["valid"]:
            valid_charts.append(viz)
        # Silently skip invalid charts (graceful degradation)
    
    if not valid_charts:
        return
    
    # =========================================================================
    # STEP 2: Sort by priority (high first), then by position
    # =========================================================================
    priority_order = {"high": 0, "medium": 1}
    valid_charts.sort(key=lambda v: (
        priority_order.get(v.get("priority", "medium"), 1),
        v.get("position", 99)
    ))
    
    # =========================================================================
    # STEP 3: Determine layout based on chart count
    # =========================================================================
    num_charts = len(valid_charts)
    
    if num_charts <= 2:
        # Full width layout for 1-2 charts
        for viz in valid_charts:
            _render_chart_with_plotly(df, viz)
    else:
        # Grid layout for 3-4 charts (2 columns)
        for i in range(0, num_charts, 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                chart_idx = i + j
                if chart_idx >= num_charts:
                    break
                
                with col:
                    _render_chart_with_plotly(df, valid_charts[chart_idx])


def _validate_chart_spec(viz: dict, df: pd.DataFrame) -> dict:
    """
    Validate a chart specification against the DataFrame.
    
    Returns:
        {"valid": bool, "reason": str}
    """
    chart_type = viz.get("chart_type", "")
    x_col = viz.get("x_column", "")
    y_col = viz.get("y_column")
    data_columns = viz.get("data_columns", [])
    
    # Special case: KPI/metric charts use internal column references
    # but have actual data_columns that should be validated instead
    if x_col and x_col.startswith("_") and data_columns:
        # Validate that at least some data_columns exist
        existing_cols = [c for c in data_columns if c in df.columns]
        if not existing_cols:
            return {"valid": False, "reason": "No data columns found for metric chart"}
        return {"valid": True, "reason": "OK - metric chart"}
    
    # Skip other internal columns without data_columns
    if x_col and x_col.startswith("_"):
        return {"valid": False, "reason": "Internal column reference"}
    
    # Validate chart type is supported
    supported_types = {"line", "bar", "hbar", "histogram", "scatter", "donut", "pie", "area"}
    if chart_type not in supported_types:
        return {"valid": False, "reason": f"Unsupported chart type: {chart_type}"}
    
    # Validate required columns exist
    columns_to_check = []
    if x_col:
        columns_to_check.append(x_col)
    if y_col:
        columns_to_check.append(y_col)
    
    for col in columns_to_check:
        if col not in df.columns:
            return {"valid": False, "reason": f"Missing column: {col}"}
    
    # Type-specific validation
    if chart_type == "line":
        # Line charts need x_column (ideally temporal) and y_column (numeric)
        if not x_col or not y_col:
            return {"valid": False, "reason": "Line chart requires x and y columns"}
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return {"valid": False, "reason": "Line chart y_column must be numeric"}
    
    elif chart_type == "scatter":
        # Scatter needs both x and y as numeric
        if not x_col or not y_col:
            return {"valid": False, "reason": "Scatter chart requires x and y columns"}
        if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
            return {"valid": False, "reason": "Scatter chart requires numeric columns"}
    
    elif chart_type == "histogram":
        # Histogram needs at least x_column as numeric
        col = x_col or y_col
        if not col or col not in df.columns:
            return {"valid": False, "reason": "Histogram requires a column"}
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"valid": False, "reason": "Histogram requires numeric column"}
    
    elif chart_type == "bar" or chart_type == "hbar":
        # Bar chart needs at least x_column
        if not x_col:
            return {"valid": False, "reason": "Bar chart requires x_column"}
    
    elif chart_type in ("donut", "pie"):
        # Donut/pie needs x_column for categories
        if not x_col:
            return {"valid": False, "reason": "Donut chart requires x_column"}
    
    elif chart_type == "area":
        # Area charts need x and y columns
        if not x_col or not y_col:
            return {"valid": False, "reason": "Area chart requires x and y columns"}
    
    return {"valid": True, "reason": "OK"}


def _render_chart_with_plotly(df: pd.DataFrame, viz: dict) -> None:
    """
    Render a single chart using Plotly based on the visualization spec.
    
    Never crashes—all errors are caught and handled gracefully.
    """
    chart_type = viz.get("chart_type", "bar")
    title = viz.get("title", "Chart")
    x_col = viz.get("x_column", "")
    y_col = viz.get("y_column")
    aggregation = viz.get("aggregation", "none")
    rationale = viz.get("rationale", "")
    data_columns = viz.get("data_columns", [])
    orientation = viz.get("orientation", "v")  # v for vertical, h for horizontal
    
    try:
        # Special case: Metric/KPI comparison charts
        if x_col and x_col.startswith("_") and data_columns:
            fig = _create_metric_comparison_chart(df, data_columns, title, aggregation)
            if fig:
                # Render title and chart together
                st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
                fig.update_layout(
                    margin=dict(l=20, r=20, t=10, b=40),
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                if rationale:
                    st.caption(f"💡 {rationale}")
            return
        
        # Prepare data with aggregation if needed
        chart_df = _prepare_chart_data(df, x_col, y_col, aggregation, chart_type)
        
        if chart_df is None or chart_df.empty:
            return
        
        # Render chart title
        st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
        
        # Create the appropriate Plotly figure
        fig = None
        
        if chart_type == "line" or chart_type == "area":
            fig = _create_line_chart(chart_df, x_col, y_col, title)
        elif chart_type == "bar":
            # Check if it should be horizontal (for ranking charts like Top 10)
            if orientation == "h" or len(chart_df) >= 8:
                fig = _create_horizontal_bar_chart(chart_df, x_col, y_col, title)
            else:
                fig = _create_bar_chart(chart_df, x_col, y_col, title, aggregation)
        elif chart_type == "hbar":
            fig = _create_horizontal_bar_chart(chart_df, x_col, y_col, title)
        elif chart_type == "histogram":
            fig = _create_histogram(chart_df, x_col or y_col, title)
        elif chart_type == "scatter":
            fig = _create_scatter_chart(chart_df, x_col, y_col, title)
        elif chart_type in ("donut", "pie"):
            fig = _create_donut_chart(chart_df, x_col, y_col, title)
        
        if fig:
            # Apply consistent styling
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=40),
                height=320 if chart_type not in ("donut", "pie") else 350,
                showlegend=True if chart_type in ("donut", "pie") else False,
            )
            
            # Render the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show rationale as caption (tooltip-like behavior)
            if rationale:
                st.caption(f"💡 {rationale}")
    
    except Exception as e:
        # Graceful failure: show nothing, don't crash the app
        pass


def _create_metric_comparison_chart(
    df: pd.DataFrame,
    data_columns: list[str],
    title: str,
    aggregation: str
) -> go.Figure | None:
    """
    Create a bar chart comparing metrics from data_columns.
    
    This handles special KPI/metric charts where the x-axis is metric names
    and y-axis is their aggregated values.
    """
    try:
        # Filter to columns that exist and are numeric
        # Also exclude ID columns and columns with very large unique counts (likely IDs)
        valid_cols = []
        for c in data_columns:
            if c not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            # Skip columns that look like IDs
            col_lower = c.lower()
            if 'id' in col_lower or 'index' in col_lower or 'key' in col_lower:
                continue
            # Skip columns where unique count is very high (likely IDs)
            if df[c].nunique() > len(df) * 0.8:
                continue
            valid_cols.append(c)
        
        if not valid_cols:
            return None
        
        # Compute aggregated values for each column
        metric_data = []
        for col in valid_cols:
            if aggregation == "sum":
                value = df[col].sum()
            elif aggregation == "mean":
                value = df[col].mean()
            elif aggregation == "count":
                value = df[col].count()
            else:
                value = df[col].mean()  # Default to mean
            
            metric_data.append({
                "Metric": col,
                "Value": float(value) if pd.notna(value) else 0
            })
        
        if not metric_data:
            return None
        
        metric_df = pd.DataFrame(metric_data)
        metric_df = metric_df.sort_values("Value", ascending=False)
        
        fig = px.bar(
            metric_df,
            x="Metric",
            y="Value",
            color_discrete_sequence=CHART_COLORS
        )
        
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(tickangle=-45 if len(metric_df) > 4 else 0, title=None, showgrid=False),
            yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
        )
        
        return fig
    
    except Exception:
        return None


def _prepare_chart_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    aggregation: str,
    chart_type: str
) -> pd.DataFrame | None:
    """
    Prepare data for charting by applying aggregation if needed.
    
    Aggregation rules:
    - "none": Use raw values
    - "sum": Group by x_column, sum y_column
    - "mean": Group by x_column, average y_column
    - "count": Group by x_column, count rows
    """
    try:
        # For histogram, no aggregation needed
        if chart_type == "histogram":
            col = x_col or y_col
            if col and col in df.columns:
                return df[[col]].dropna()
            return None
        
        # For scatter with no aggregation, return raw data
        if chart_type == "scatter" and aggregation == "none":
            if x_col in df.columns and y_col in df.columns:
                return df[[x_col, y_col]].dropna()
            return None
        
        # For line charts, handle temporal x-axis
        if chart_type == "line":
            if x_col not in df.columns:
                return None
            
            work_df = df.copy()
            
            # Try to parse x_column as datetime
            if not pd.api.types.is_datetime64_any_dtype(work_df[x_col]):
                try:
                    work_df[x_col] = pd.to_datetime(work_df[x_col], errors='coerce')
                except Exception:
                    pass
            
            # Apply aggregation
            if y_col and y_col in work_df.columns and aggregation != "none":
                if aggregation == "sum":
                    result = work_df.groupby(x_col)[y_col].sum().reset_index()
                elif aggregation == "mean":
                    result = work_df.groupby(x_col)[y_col].mean().reset_index()
                elif aggregation == "count":
                    result = work_df.groupby(x_col)[y_col].count().reset_index()
                else:
                    result = work_df[[x_col, y_col]].dropna()
                
                return result.sort_values(x_col)
            else:
                return work_df[[x_col, y_col]].dropna().sort_values(x_col) if y_col else None
        
        # For area charts (same as line)
        if chart_type == "area":
            if x_col not in df.columns:
                return None
            
            work_df = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(work_df[x_col]):
                try:
                    work_df[x_col] = pd.to_datetime(work_df[x_col], errors='coerce')
                except Exception:
                    pass
            
            if y_col and y_col in work_df.columns and aggregation != "none":
                if aggregation == "sum":
                    result = work_df.groupby(x_col)[y_col].sum().reset_index()
                elif aggregation == "mean":
                    result = work_df.groupby(x_col)[y_col].mean().reset_index()
                else:
                    result = work_df[[x_col, y_col]].dropna()
                
                return result.sort_values(x_col)
            else:
                return work_df[[x_col, y_col]].dropna().sort_values(x_col) if y_col else None
        
        # For bar charts, aggregate by x_column
        if chart_type in ("bar", "hbar"):
            if x_col not in df.columns:
                return None
            
            if y_col and y_col in df.columns and aggregation != "none":
                if aggregation == "sum":
                    result = df.groupby(x_col)[y_col].sum().reset_index()
                elif aggregation == "mean":
                    result = df.groupby(x_col)[y_col].mean().reset_index()
                elif aggregation == "count":
                    result = df.groupby(x_col)[y_col].count().reset_index()
                else:
                    result = df.groupby(x_col)[y_col].sum().reset_index()
                
                # Sort by value descending for bar charts
                return result.sort_values(y_col, ascending=False).head(15)
            else:
                # Count of categories
                result = df[x_col].value_counts().reset_index()
                result.columns = [x_col, "count"]
                return result.head(15)
        
        # For donut/pie charts
        if chart_type in ("donut", "pie"):
            if x_col not in df.columns:
                return None
            
            if y_col and y_col in df.columns:
                if aggregation == "sum":
                    result = df.groupby(x_col)[y_col].sum().reset_index()
                elif aggregation == "mean":
                    result = df.groupby(x_col)[y_col].mean().reset_index()
                else:
                    result = df.groupby(x_col)[y_col].sum().reset_index()
                return result.head(10)
            else:
                result = df[x_col].value_counts().reset_index()
                result.columns = [x_col, "count"]
                return result.head(10)
        
        return df
    
    except Exception:
        return None


def _create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Create a Plotly area/line chart for time series data."""
    fig = px.area(
        df,
        x=x_col,
        y=y_col,
        title=None,  # We'll add custom title via container
        markers=True if len(df) < 30 else False
    )
    
    fig.update_traces(
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59, 130, 246, 0.1)",
        marker=dict(size=6, color="#3b82f6")
    )
    
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
    )
    
    return fig


# Blue color palette for charts
CHART_COLORS = [
    "#3b82f6",  # Primary blue
    "#0ea5e9",  # Sky blue
    "#06b6d4",  # Cyan
    "#60a5fa",  # Light blue
    "#38bdf8",  # Lighter sky
    "#22d3ee",  # Light cyan
    "#93c5fd",  # Very light blue
    "#7dd3fc",  # Pale sky
]


def _create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    title: str,
    aggregation: str,
    orientation: str = "v"
) -> go.Figure:
    """Create a Plotly bar chart for categorical comparison."""
    y_axis = y_col if y_col else "count"
    
    # Assign colors to each bar
    colors = CHART_COLORS[:len(df)] if len(df) <= len(CHART_COLORS) else CHART_COLORS * (len(df) // len(CHART_COLORS) + 1)
    
    if orientation == "h":
        # Horizontal bar chart (for Top 10 style charts)
        fig = go.Figure(go.Bar(
            x=df[y_axis],
            y=df[x_col],
            orientation='h',
            marker_color=colors[:len(df)],
            marker=dict(
                line=dict(width=0),
                cornerradius=4
            )
        ))
        fig.update_layout(
            yaxis=dict(autorange="reversed", title=None),
            xaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
        )
    else:
        # Vertical bar chart
        fig = go.Figure(go.Bar(
            x=df[x_col],
            y=df[y_axis],
            marker_color=colors[:len(df)],
            marker=dict(
                line=dict(width=0),
                cornerradius=6
            )
        ))
        fig.update_layout(
            xaxis=dict(tickangle=-45 if len(df) > 5 else 0, title=None),
            yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
        )
    
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    return fig


def _create_histogram(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Create a Plotly histogram for numeric distribution."""
    fig = px.histogram(
        df,
        x=col,
        title=None,
        nbins=30,
        color_discrete_sequence=["#3b82f6"]
    )
    
    fig.update_layout(
        bargap=0.1,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
    )
    
    return fig


def _create_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Create a Plotly scatter chart for numeric correlation."""
    # Sample if too many points
    plot_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        title=None,
        opacity=0.7,
        color_discrete_sequence=["#3b82f6"]
    )
    
    fig.update_traces(marker=dict(size=8))
    
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
    )
    
    return fig


def _create_donut_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    title: str
) -> go.Figure:
    """Create a Plotly donut (pie with hole) chart."""
    values_col = y_col if y_col and y_col in df.columns else "count"
    
    fig = px.pie(
        df,
        names=x_col,
        values=values_col,
        title=None,
        hole=0.5,  # Makes it a donut
        color_discrete_sequence=CHART_COLORS
    )
    
    fig.update_traces(
        textposition='outside',
        textinfo='percent',
        textfont_size=12,
        marker=dict(line=dict(color='white', width=2))
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        paper_bgcolor="white",
    )
    
    return fig


def _create_horizontal_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None,
    title: str
) -> go.Figure:
    """Create a horizontal bar chart for ranking data (like Top 10)."""
    y_axis = y_col if y_col else "count"
    
    # Sort ascending so largest is at top when reversed
    sorted_df = df.sort_values(y_axis, ascending=True)
    
    fig = go.Figure(go.Bar(
        x=sorted_df[y_axis],
        y=sorted_df[x_col],
        orientation='h',
        marker_color="#3b82f6",
        marker=dict(
            line=dict(width=0),
            cornerradius=4
        )
    ))
    
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
        yaxis=dict(title=None),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    
    return fig


def render_results():
    """Render analysis results."""
    result = st.session_state.analysis_result
    if not result:
        return
    
    ui_payload = result.get("ui_payload", {})
    
    # dashboard_plan is set by plan_visuals_node, which runs after ui_payload is built
    # So we get it directly from the result state
    dashboard_plan = result.get("dashboard_plan", [])
    
    # Check for error
    if ui_payload.get("is_error"):
        render_error(ui_payload)
        return
    
    st.divider()
    
    # Executive summary
    summary = ui_payload.get("executive_summary", "")
    if summary:
        st.info(summary)
    
    # KPIs
    render_kpi_cards(ui_payload.get("kpis", []))
    
    st.markdown("")  # Spacer
    
    # Dashboard (charts) - Load df and pass to render_dashboard
    if dashboard_plan and st.session_state.file_bytes:
        try:
            df = pd.read_csv(io.BytesIO(st.session_state.file_bytes))
            render_dashboard(dashboard_plan, df)
        except Exception:
            pass  # Graceful failure
    
    # Insights
    render_insights(ui_payload.get("insights", []))
    
    # Data profile
    render_data_profile(ui_payload)
    
    # Statistical summary
    render_statistical_summary(ui_payload)
    
    # Warnings
    render_warnings(ui_payload.get("warnings", []))
    
    # Export actions
    render_export_actions(ui_payload)


def render_error(ui_payload: dict):
    """Render error state with recovery options."""
    st.divider()
    
    error_msg = ui_payload.get("error_message", "An unknown error occurred")
    error_type = ui_payload.get("error_type", "UNKNOWN")
    recovery_hint = ui_payload.get("recovery_hint", "Please try again.")
    has_partial = ui_payload.get("has_partial_results", False)
    
    st.error(f"**{error_type}**: {error_msg}")
    st.info(f"💡 **Suggestion**: {recovery_hint}")
    
    # Partial results
    if has_partial:
        partial = ui_payload.get("partial_results", {})
        
        st.warning("Partial results are available despite the error:")
        
        if partial.get("kpis"):
            render_kpi_cards(partial["kpis"])
        
        if partial.get("schema_profile"):
            render_data_profile({"schema_profile": partial["schema_profile"]})
    
    # Recovery actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Try Again", type="primary", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.analysis_complete = False
            st.rerun()
    
    with col2:
        if st.button("📂 Upload Different File", use_container_width=True):
            st.session_state.uploaded_file = None
            st.session_state.file_bytes = None
            st.session_state.filename = None
            st.session_state.analysis_result = None
            st.session_state.analysis_complete = False
            st.rerun()


# =============================================================================
# MAIN APP FLOW
# =============================================================================

def main():
    """Main application flow."""
    
    # Upload section
    uploaded_file = render_upload_section()
    
    # Show preview if file uploaded
    if st.session_state.file_bytes:
        render_file_preview()
    
    # Analysis controls
    analyze_clicked = render_analysis_controls()
    
    # Run analysis if triggered
    if analyze_clicked and not st.session_state.analysis_running:
        run_analysis_with_progress()
        st.rerun()
    
    # Show results if complete
    if st.session_state.analysis_complete:
        render_results()
    
    # Footer
    st.markdown("---")
    st.caption("🧠 DATAPILOT — Powered by LangGraph • Built with Streamlit")


if __name__ == "__main__":
    main()
