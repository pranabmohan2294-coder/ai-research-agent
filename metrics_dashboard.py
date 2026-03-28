import os
from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import json
from datetime import datetime
from metrics_logger import load_metrics, get_summary_stats
from chroma_manager import get_chroma_stats

st.set_page_config(page_title="PM Intel — Metrics Dashboard", layout="wide")

st.title("PM Intel — Metrics Dashboard")
st.caption("Live metrics across all research runs · Day 25")

metrics = load_metrics()
stats   = get_summary_stats()
chroma  = get_chroma_stats()

if not metrics:
    st.warning("No runs logged yet. Run some queries first.")
    st.stop()

# ---------------------------
# Top metrics row
# ---------------------------
st.divider()
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Total runs", stats.get("total_runs", 0))

avg_score = stats.get("avg_critic_score")
col2.metric("Avg critic score",
    f"{avg_score}/10" if avg_score else "—",
    delta="target ≥7" if avg_score and avg_score < 7 else None,
    delta_color="inverse" if avg_score and avg_score < 7 else "normal"
)

avg_latency = stats.get("avg_latency")
col3.metric("Avg latency",
    f"{avg_latency}s" if avg_latency else "—",
    delta="target <120s" if avg_latency and avg_latency > 120 else None,
    delta_color="inverse"
)

approval = stats.get("approval_rate")
col4.metric("Approval rate",
    f"{approval}%" if approval else "—",
    delta="target >85%" if approval and approval < 85 else None,
    delta_color="inverse"
)

hall_rate = stats.get("hallucination_flag_rate")
col5.metric("Hallucination flags",
    f"{hall_rate}%" if hall_rate is not None else "—",
    delta="target <10%" if hall_rate and hall_rate > 10 else None,
    delta_color="inverse"
)

col6.metric("Stored in Chroma", chroma.get("total_stored", 0))

st.divider()

# ---------------------------
# PRD targets vs actuals
# ---------------------------
st.subheader("PRD Targets vs Actuals")

prd_data = {
    "Metric": [
        "Critic score median",
        "Latency p95",
        "Approval rate",
        "Hallucination flag rate"
    ],
    "Target": ["≥ 7/10", "< 120s", "> 85%", "< 10%"],
    "Actual": [
        f"{stats.get('median_critic_score', '—')}/10" if stats.get('median_critic_score') else "—",
        f"{stats.get('latency_p95', '—')}s" if stats.get('latency_p95') else "—",
        f"{stats.get('approval_rate', '—')}%" if stats.get('approval_rate') else "—",
        f"{stats.get('hallucination_flag_rate', '—')}%" if stats.get('hallucination_flag_rate') is not None else "—"
    ],
    "Status": []
}

# Compute status
scores_ok   = stats.get('median_critic_score', 0) >= 7 if stats.get('median_critic_score') else None
latency_ok  = stats.get('latency_p95', 999) < 120 if stats.get('latency_p95') else None
approval_ok = stats.get('approval_rate', 0) >= 85 if stats.get('approval_rate') else None
hall_ok     = stats.get('hallucination_flag_rate', 100) <= 10 if stats.get('hallucination_flag_rate') is not None else None

for ok in [scores_ok, latency_ok, approval_ok, hall_ok]:
    if ok is None:
        prd_data["Status"].append("—")
    elif ok:
        prd_data["Status"].append("✓ Met")
    else:
        prd_data["Status"].append("✗ Below target")

st.table(prd_data)

st.divider()

# ---------------------------
# Charts row
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Critic Score Distribution")
    scores = [r.get("critic_score", -1) for r in metrics if r.get("critic_score", -1) > 0]
    if scores:
        score_counts = {i: scores.count(i) for i in range(1, 11)}
        chart_data = {"Score": list(score_counts.keys()),
                      "Count": list(score_counts.values())}
        import pandas as pd
        df_scores = pd.DataFrame(chart_data)
        st.bar_chart(df_scores.set_index("Score"))
        st.caption(f"Min: {min(scores)} · Max: {max(scores)} · Median: {sorted(scores)[len(scores)//2]}")
    else:
        st.info("No scores logged yet")

with col2:
    st.subheader("Latency per Run")
    latencies = [(r.get("run_id",""), r.get("total_latency", 0))
                 for r in metrics if r.get("total_latency", 0) > 0]
    if latencies:
        import pandas as pd
        df_lat = pd.DataFrame(latencies, columns=["Run", "Latency (s)"])
        st.line_chart(df_lat.set_index("Run"))
        p95 = stats.get("latency_p95")
        st.caption(f"Avg: {stats.get('avg_latency')}s · p95: {p95}s · Target: <120s")
    else:
        st.info("No latency data yet")

st.divider()

# ---------------------------
# Intent breakdown + Cache
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Queries by Intent")
    intent_counts = {}
    for r in metrics:
        intent = r.get("intent", "unknown") or "unknown"
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    if intent_counts:
        import pandas as pd
        df_intent = pd.DataFrame({
            "Intent": list(intent_counts.keys()),
            "Count":  list(intent_counts.values())
        })
        st.bar_chart(df_intent.set_index("Intent"))
    else:
        st.info("No intent data yet")

with col2:
    st.subheader("Cache Performance")
    cache_counts = {"hit": 0, "context": 0, "miss": 0}
    for r in metrics:
        cs = r.get("cache_status", "miss") or "miss"
        if cs in cache_counts:
            cache_counts[cs] += 1
    total_with_cache = sum(cache_counts.values())
    if total_with_cache > 0:
        hit_rate     = round(cache_counts["hit"]     / total_with_cache * 100, 1)
        context_rate = round(cache_counts["context"] / total_with_cache * 100, 1)
        miss_rate    = round(cache_counts["miss"]    / total_with_cache * 100, 1)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cache hits",    f"{hit_rate}%",     f"{cache_counts['hit']} runs")
        col_b.metric("Hybrid",        f"{context_rate}%", f"{cache_counts['context']} runs")
        col_c.metric("Fresh search",  f"{miss_rate}%",    f"{cache_counts['miss']} runs")
        st.caption("Cache hit = web search skipped · Hybrid = cache + fresh search")
    else:
        st.info("No cache data yet — run more queries")

st.divider()

# ---------------------------
# Pipeline type breakdown
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pipeline Usage")
    pipeline_counts = {}
    for r in metrics:
        pt = r.get("pipeline_type", "research") or "research"
        pipeline_counts[pt] = pipeline_counts.get(pt, 0) + 1
    for pt, count in pipeline_counts.items():
        pct = round(count / len(metrics) * 100, 1)
        st.metric(pt.title(), f"{count} runs", f"{pct}%")

with col2:
    st.subheader("Feedback Loop Usage")
    fb_used  = sum(1 for r in metrics if r.get("feedback_loop_used"))
    fb_rate  = round(fb_used / len(metrics) * 100, 1) if metrics else 0
    entity_warnings = sum(1 for r in metrics if not r.get("entity_valid", True))

    st.metric("Feedback loops triggered", fb_used, f"{fb_rate}% of runs")
    st.metric("Entity validation warnings", entity_warnings,
              f"{round(entity_warnings/len(metrics)*100,1)}% of runs" if metrics else "0%")
    hall_flags = sum(1 for r in metrics if r.get("hallucination_flagged"))
    st.metric("Hallucination flags", hall_flags,
              f"{round(hall_flags/len(metrics)*100,1)}% of runs" if metrics else "0%")

st.divider()

# ---------------------------
# Recent runs table
# ---------------------------
st.subheader("Recent Runs")

recent = metrics[-10:][::-1]
for r in recent:
    score  = r.get("critic_score", -1)
    cs     = r.get("cache_status", "miss") or "miss"
    ts     = r.get("timestamp","")[:16].replace("T"," ") if r.get("timestamp") else "—"

    cache_icon = "✓" if cs=="hit" else ("~" if cs=="context" else "○")

    col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 1])
    col1.markdown(f"**{r.get('query','—')[:60]}**")
    col2.markdown(f"`{r.get('intent','—')[:12]}`")

    if score >= 7:
        col3.success(f"{score}/10")
    elif score >= 5:
        col3.warning(f"{score}/10")
    elif score > 0:
        col3.error(f"{score}/10")
    else:
        col3.markdown("—")

    col4.markdown(f"`{r.get('total_latency',0):.0f}s`")
    col5.markdown(f"`{cache_icon} {cs}`")

st.divider()

# ---------------------------
# Cost estimate
# ---------------------------
st.subheader("Cost Estimate")
total_tokens = sum(r.get("total_tokens", 0) for r in metrics)
col1, col2, col3 = st.columns(3)
col1.metric("Total tokens (estimated)", f"{total_tokens:,}")
col2.metric("Cost — llama3.2 (current)", "$0.00", "fully local")
col3.metric("Cost — Claude Sonnet (V2)",
    f"${round(total_tokens * 0.000003 + total_tokens * 0.000015, 2):.2f}",
    "estimated at $3/$15 per 1M tokens")

st.caption("Token counts are word-based estimates (~20% off actual tokeniser counts)")

st.divider()
st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(metrics)} total runs · Data from metrics_log.json")
