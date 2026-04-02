import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="PM Intel — Live Dashboard", layout="wide")
st.title("PM Intel — Live Metrics Dashboard")
st.caption("Powered by Google Sheets · All cloud runs tracked in real time")

# ---------------------------
# Load from Google Sheets
# ---------------------------

@st.cache_data(ttl=30)
def load_sheets_data():
    try:
        from sheets_logger import read_all_runs
        rows = read_all_runs()
        return rows
    except Exception as e:
        st.error("Could not load from Google Sheets: " + str(e))
        return []

with st.spinner("Loading live data from Google Sheets..."):
    rows = load_sheets_data()

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

if not rows:
    st.warning("No runs logged yet. Run a query on the live app first.")
    st.stop()

df = pd.DataFrame(rows)
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " · Auto-refreshes every 30 seconds")
st.divider()

# ---------------------------
# Top metrics
# ---------------------------

total = len(df)
scores = pd.to_numeric(df["Critic Score"], errors="coerce").dropna()
latencies = pd.to_numeric(df["Latency"], errors="coerce").dropna()
cache_hits = (df["Cache Status"] == "hit").sum()
excellent = (df.get("Quality Label", df.get("Quality", "")) == "Excellent").sum()
strong = (df.get("Quality Label", df.get("Quality", "")) == "Strong").sum()
fb_loops = (df["Feedback Loop"] == "Yes").sum()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total runs", total)
col2.metric("Avg critic score", str(round(scores.mean(), 1)) + "/10" if len(scores) > 0 else "—")
col3.metric("Avg latency", str(round(latencies.mean(), 1)) + "s" if len(latencies) > 0 else "—")
col4.metric("Excellent reports", str(excellent), str(round(excellent/total*100, 1)) + "%")
col5.metric("Cache hits", str(cache_hits), str(round(cache_hits/total*100, 1)) + "%")
col6.metric("Feedback loops", str(fb_loops), str(round(fb_loops/total*100, 1)) + "%")

st.divider()

# ---------------------------
# Charts
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Critic Score Distribution")
    if len(scores) > 0:
        score_counts = scores.astype(int).value_counts().sort_index()
        df_scores = pd.DataFrame({
            "Score": score_counts.index,
            "Count": score_counts.values
        })
        st.bar_chart(df_scores.set_index("Score"))
        st.caption(
            "Min: " + str(int(scores.min())) +
            " · Max: " + str(int(scores.max())) +
            " · Median: " + str(int(scores.median())) +
            " · Target: ≥7"
        )
    else:
        st.info("No scores yet")

with col2:
    st.subheader("Latency per Run")
    if len(latencies) > 0:
        df_lat = pd.DataFrame({
            "Run": range(1, len(latencies)+1),
            "Latency": latencies.values
        })
        st.line_chart(df_lat.set_index("Run"))
        p95_idx = int(len(latencies) * 0.95)
        sorted_lat = sorted(latencies.values)
        p95 = sorted_lat[p95_idx] if p95_idx < len(sorted_lat) else sorted_lat[-1]
        st.caption(
            "Avg: " + str(round(latencies.mean(), 1)) + "s" +
            " · p95: " + str(round(p95, 1)) + "s" +
            " · Target: <120s"
        )
    else:
        st.info("No latency data yet")

st.divider()

# ---------------------------
# Intent + Quality breakdown
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Queries by Intent")
    intent_counts = df["Intent"].value_counts()
    if len(intent_counts) > 0:
        df_intent = pd.DataFrame({
            "Intent": intent_counts.index,
            "Count": intent_counts.values
        })
        st.bar_chart(df_intent.set_index("Intent"))

with col2:
    st.subheader("Quality Distribution")
    quality_counts = df.get("Quality Label", df.get("Quality", "")).value_counts()
    if len(quality_counts) > 0:
        col_a, col_b, col_c, col_d = st.columns(4)
        labels = ["Excellent", "Strong", "Good", "Fair"]
        cols = [col_a, col_b, col_c, col_d]
        colors = ["success", "info", "warning", "error"]
        for label, col in zip(labels, cols):
            count = quality_counts.get(label, 0)
            pct = round(count/total*100, 1)
            col.metric(label, str(count), str(pct) + "%")

st.divider()

# ---------------------------
# Cache performance
# ---------------------------

st.subheader("Cache Performance")
cache_counts = df["Cache Status"].value_counts()
total_cache = len(df)
col1, col2, col3 = st.columns(3)
col1.metric("Cache hits",    str(cache_counts.get("hit",0)),     str(round(cache_counts.get("hit",0)/total_cache*100,1)) + "%")
col2.metric("Hybrid",        str(cache_counts.get("context",0)), str(round(cache_counts.get("context",0)/total_cache*100,1)) + "%")
col3.metric("Fresh search",  str(cache_counts.get("miss",0)),    str(round(cache_counts.get("miss",0)/total_cache*100,1)) + "%")

st.divider()

# ---------------------------
# Recent runs table
# ---------------------------

st.subheader("All Runs")
display_cols = ["Timestamp", "Query", "Intent", "Critic Score", "Latency", "Cache Status", "Quality Label", "Feedback Loop"]
available = [c for c in display_cols if c in df.columns]
df_display = df[available].copy()
df_display = df_display.iloc[::-1].reset_index(drop=True)

st.dataframe(df_display, use_container_width=True)

st.divider()

# ---------------------------
# PRD targets
# ---------------------------

st.subheader("PRD Targets vs Actuals")
median_score = scores.median() if len(scores) > 0 else None
avg_lat = latencies.mean() if len(latencies) > 0 else None
sorted_lat = sorted(latencies.values) if len(latencies) > 0 else []
p95_val = sorted_lat[int(len(sorted_lat)*0.95)] if len(sorted_lat) > 1 else None
hall_rate = round((df["Hallucination Flagged"] == "Yes").sum() / total * 100, 1)

prd = {
    "Metric": ["Critic score median", "Latency p95", "Hallucination flag rate"],
    "Target": ["≥ 7/10", "< 120s", "< 10%"],
    "Actual": [
        str(round(median_score,1)) + "/10" if median_score else "—",
        str(round(p95_val,1)) + "s" if p95_val else "—",
        str(hall_rate) + "%"
    ],
    "Status": [
        "✓ Met" if median_score and median_score >= 7 else "✗ Below",
        "✓ Met" if p95_val and p95_val < 120 else "✗ Below",
        "✓ Met" if hall_rate <= 10 else "✗ Above"
    ]
}
st.table(prd)

st.divider()
st.caption(
    "Data source: Google Sheets · "
    "Sheet ID: 1Qfrt7afqBYDiv5THOtBvHCC-puomjWK_Dm1MpFWTzJU · "
    "PM Intel · " + str(total) + " total runs"
)
