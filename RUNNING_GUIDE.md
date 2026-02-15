# DATAPILOT — Execution & Validation Guide

## 1. Local Setup

### 1.1 Create Virtual Environment

```bash
cd /Users/aryankumar/Ai_Project

# Create venv
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Verify
which python  # Should show .venv/bin/python
```

### 1.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Start Ollama (Optional but Recommended)

```bash
# Install Ollama if not already installed
brew install ollama

# Start Ollama server (runs on localhost:11434)
ollama serve &

# Pull TinyLLaMA model (first time only, ~600MB)
ollama pull tinyllama

# Verify model is available
ollama list
```

> **Note**: The app works without Ollama — insights will use template-based fallback.

### 1.4 Launch Streamlit App

```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 2. Validation Checklist

### 2.1 Landing Page (Before Upload)

| Check | Expected |
|-------|----------|
| Header | "🧠 DATAPILOT" with subtitle |
| File uploader | Visible with "Drop your CSV here" text |
| Analyze button | Not visible (appears after upload) |
| No errors | Console should be clean |

### 2.2 After CSV Upload

| Check | Expected |
|-------|----------|
| File info | Filename and size displayed |
| Remove button | "✕ Remove" appears |
| Mode selector | Radio: `quick` / `standard` (quick selected by default) |
| Goal input | Text field visible |
| Analyze button | "🚀 Analyze My Data" enabled |

### 2.3 During Analysis (Progress Updates)

**Quick Mode steps:**
1. Loading data
2. Profiling columns
3. Understanding data
4. Running statistical analysis
5. Generating insights
6. Analysis complete!

**Standard Mode steps:**
1. Loading data
2. Profiling columns
3. Understanding data
4. Running comprehensive analysis
5. Generating detailed insights
6. Analysis complete!

Progress bar should advance smoothly from 0% → 100%.

### 2.4 Expected Outputs

#### Quick Mode

| Section | Content |
|---------|---------|
| Key Metrics | 3-5 auto-detected KPIs with values |
| Insights | 5-8 insight cards (info/warning severity) |
| Executive Summary | 1-2 sentence overview |
| Data Quality | Quality score badge |

#### Standard Mode (Additional)

| Section | Content |
|---------|---------|
| Distributions | Normality/skewness analysis |
| Trends | Temporal trend direction (if date column) |
| Correlations | Top correlated pairs with r-values |
| Anomalies | Outlier counts and warnings |
| Segments | Category performance comparison |
| More insights | 10-15 insight cards |

---

## 3. Sample Test CSV

Save as `test_sales.csv`:

```csv
date,region,product,revenue,units_sold,discount_pct
2025-01-01,North,Widget A,1500.00,30,5
2025-01-01,South,Widget B,2200.50,44,10
2025-01-02,North,Widget A,1350.00,27,5
2025-01-02,East,Widget C,3100.00,62,15
2025-01-03,South,Widget A,1800.00,36,8
2025-01-03,North,Widget B,2500.00,50,12
2025-01-04,East,Widget A,1650.00,33,5
2025-01-04,West,Widget C,2900.00,58,10
2025-01-05,North,Widget B,2100.00,42,10
2025-01-05,South,Widget A,1400.00,28,5
2025-01-06,West,Widget B,2800.00,56,15
2025-01-06,East,Widget C,3300.00,66,12
2025-01-07,North,Widget A,1550.00,31,5
2025-01-07,South,Widget C,2750.00,55,10
2025-01-08,East,Widget B,2400.00,48,8
2025-01-08,West,Widget A,1200.00,24,20
2025-01-09,North,Widget C,3050.00,61,10
2025-01-09,South,Widget B,2650.00,53,12
2025-01-10,West,Widget A,1700.00,34,5
2025-01-10,East,Widget B,150.00,3,50
```

**Test coverage:**
- ✅ Date column → temporal trends
- ✅ Categorical columns (region, product) → segmentation
- ✅ Numeric columns (revenue, units_sold, discount_pct) → statistics
- ✅ Outlier row (last row: low revenue, high discount) → anomaly detection
- ✅ Correlations (revenue ↔ units_sold)

---

## 4. Common Failure Cases

### 4.1 Ollama Not Running

**Symptom:**
- Insights appear but lack "AI-enhanced" label
- Warnings in output: "Local LLM unavailable - using template-based insights"

**Fix:**
```bash
ollama serve
ollama pull tinyllama
```

**Verification:**
```bash
curl http://localhost:11434/api/tags
# Should return JSON with tinyllama in models list
```

### 4.2 CSV Missing Numeric Columns

**Symptom:**
- Warning: "No numeric columns found"
- KPIs section empty or minimal
- Quick mode: "No data suitable for statistical analysis"

**Fix:** Ensure CSV has at least one numeric column (int or float).

### 4.3 CSV Missing Date Columns

**Symptom:**
- Standard mode: "temporal_analysis: null" in results
- No trend insights generated

**Expected behavior:** App continues — temporal analysis is optional.

### 4.4 Streamlit Rerun Issues

**Symptom:**
- App reruns mid-analysis
- Progress bar resets
- "Analysis running" stuck

**Cause:** Clicking UI elements during streaming.

**Fix:** Wait for analysis to complete before interacting.

### 4.5 Large File Timeout

**Symptom:**
- Analysis hangs on large files (>50k rows)

**Fix:** Test with smaller samples first. Production deployment needs async handling.

---

## 5. Verification Procedures

### 5.1 Confirm LangGraph Streaming is Working

**Test:**
1. Upload CSV and click Analyze
2. Watch progress bar

**Expected:**
- Progress bar updates incrementally (not jumping 0→100)
- Steps appear one-by-one in "Analysis Steps" expander
- Console shows no errors

**Debug (optional):**
Add to `app.py` temporarily:
```python
# Inside the stream loop
print(f"[STREAM] Node: {node_name}, Progress: {state.get('progress')}")
```

### 5.2 Confirm Standard Mode Routing is Active

**Test:**
1. Upload CSV
2. Select `standard` radio option
3. Click Analyze

**Expected:**
- Progress shows "Running comprehensive analysis" (not "Running statistical analysis")
- Progress shows "Generating detailed insights" (not "Generating insights")
- Results include: distributions, correlations, anomalies, segmentation
- Insight count: 10-15 (vs 5-8 in Quick mode)

**Verify in state (debug):**
Check `st.session_state.analysis_result` contains:
```python
{
    "distributions": [...],      # Present in Standard only
    "correlations": {...},       # Present in Standard only
    "anomalies": {...},          # Present in Standard only
    "ui_payload": {
        "analysis_depth": "standard"  # Confirms mode
    }
}
```

### 5.3 Confirm LLM Fallback is Functioning

**Test A: With Ollama running**
1. Start Ollama: `ollama serve`
2. Run Standard mode analysis
3. Check insight cards

**Expected:**
- Progress message ends with "(AI-enhanced)"
- Insight bodies have richer, more contextual language

**Test B: Without Ollama**
1. Stop Ollama: `pkill ollama`
2. Run Standard mode analysis
3. Check insight cards

**Expected:**
- Warning appears: "Local LLM unavailable - using template-based insights"
- Progress message ends WITHOUT "(AI-enhanced)"
- Insights still appear (template-based)
- No crash or error

---

## 6. Quick Test Sequence

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run app
cd /Users/aryankumar/Ai_Project
source .venv/bin/activate
streamlit run app.py

# In browser:
# 1. Upload test_sales.csv
# 2. Run Quick mode → verify KPIs + ~5 insights
# 3. Run Standard mode → verify distributions, correlations, anomalies
# 4. Stop Ollama (pkill ollama)
# 5. Run Standard mode again → verify fallback warning + insights still appear
```

---

## 7. Expected Console Output (Clean Run)

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

No tracebacks. No warnings (unless Ollama is down).

---

## 8. Troubleshooting Commands

```bash
# Check Python version (need 3.10+)
python --version

# Check installed packages
pip list | grep -E "streamlit|langgraph|langchain|pandas"

# Check Ollama status
curl -s http://localhost:11434/api/tags | python -m json.tool

# Check port 8501 availability
lsof -i :8501

# Kill stuck Streamlit
pkill -f streamlit

# Reset session (in browser)
# Add ?reset=1 to URL or hard refresh (Cmd+Shift+R)
```

---

**Last updated:** 3 February 2026
