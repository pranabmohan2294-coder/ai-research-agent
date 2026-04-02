import os
from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from metrics_logger import load_metrics, get_summary_stats
from chroma_manager import get_chroma_stats

st.set_page_config(page_title="PM Intel — Metrics Dashboard", layout="wide")
st.title("PM Intel — Metrics Dashboard")
st.caption("Live metrics · NVIDIA llama-3.1-8b · DuckDuckGo · Day 29")

metrics  = load_metrics()
stats    = get_summary_stats()

try:
    chroma = get_chroma_stats()
except Exception:
    chroma = {"total_stored": 0}

try:
    from agent_comms_logger import load_comms, get_comms_stats
    comms       = load_comms()
    comms_stats = get_comms_stats()
except Exception:
    comms       = []
    comms_stats = {"total_handoffs": 0}

if not metrics:
    st.warning("No runs logged yet.")
    st.stop()

# ---------------------------
# Top metrics
# ---------------------------
st.divider()
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Total runs", stats.get("total_runs", 0))

avg_score = stats.get("avg_critic_score")
col2.metric("Avg critic score",
    str(avg_score) + "/10" if avg_score else "—",
    delta="below target" if avg_score and avg_score < 7 else ("on target" if avg_score else None),
    delta_color="inverse" if avg_score and avg_score < 7 else "normal"
)

avg_latency = stats.get("avg_latency")
col3.metric("Avg latency",
    str(avg_latency) + "s" if avg_latency else "—",
    delta="below target" if avg_latency and avg_latency > 120 else None,
    delta_color="inverse"
)

approval = stats.get("approval_rate")
col4.metric("Approval rate",
    str(approval) + "%" if approval else "—"
)

hall_rate = stats.get("hallucination_flag_rate")
col5.metric("Hallucination flags",
    str(hall_rate) + "%" if hall_rate is not None else "—",
    delta="above target" if hall_rate and hall_rate > 10 else None,
    delta_color="inverse"
)

col6.metric("Agent handoffs", comms_stats.get("total_handoffs", 0))

st.divider()

# ---------------------------
# PRD targets vs actuals
# ---------------------------
st.subheader("PRD Targets vs Actuals")

median_score = stats.get("median_critic_score")
p95          = stats.get("latency_p95")

prd = {
    "Metric": [
        "Critic score median",
        "Latency p95",
        "Approval rate",
        "Hallucination flag rate"
    ],
    "Target": ["≥ 7/10", "< 120s", "> 85%", "< 10%"],
    "Actual": [
        str(median_score) + "/10" if median_score else "—",
        str(p95) + "s"           if p95          else "—",
        str(approval) + "%"      if approval      else "—",
        str(hall_rate) + "%"     if hall_rate is not None else "—"
    ],
    "Status": []
}

for ok in [
    (median_score or 0) >= 7,
    (p95 or 999) < 120,
    (approval or 0) >= 85,
    (hall_rate or 100) <= 10
]:
    prd["Status"].append("✓ Met" if ok else "✗ Below target")

st.table(prd)
st.divider()

# ---------------------------
# Charts row 1 — scores + latency
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Critic Score Distribution")
    scores = [r.get("critic_score",-1) for r in metrics if r.get("critic_score",-1) > 0]
    if scores:
        score_counts = {i: scores.count(i) for i in range(1, 11)}
        df = pd.DataFrame({
            "Score": list(score_counts.keys()),
            "Count": list(score_counts.values())
        })
        st.bar_chart(df.set_index("Score"))
        st.caption(
            "Min: " + str(min(scores)) +
            " · Max: " + str(max(scores)) +
            " · Median: " + str(sorted(scores)[len(scores)//2]) +
            " · Target: ≥7"
        )
    else:
        st.info("No scores yet")

with col2:
    st.subheader("Latency per Run")
    latencies = [(r.get("run_id",""), round(r.get("total_latency",0),1))
                 for r in metrics if r.get("total_latency",0) > 0]
    if latencies:
        df_lat = pd.DataFrame(latencies, columns=["Run","Latency (s)"])
        st.line_chart(df_lat.set_index("Run"))
        st.caption(
            "Avg: " + str(avg_latency) + "s" +
            " · p95: " + str(p95) + "s" +
            " · Target: <120s"
        )
    else:
        st.info("No latency data yet")

st.divider()

# ---------------------------
# Charts row 2 — cache + intent
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cache Performance")
    cache_counts = {"hit": 0, "context": 0, "miss": 0}
    for r in metrics:
        cs = r.get("cache_status","miss") or "miss"
        if cs in cache_counts:
            cache_counts[cs] += 1
    total = sum(cache_counts.values())
    if total > 0:
        hit_rate     = round(cache_counts["hit"]     / total * 100, 1)
        context_rate = round(cache_counts["context"] / total * 100, 1)
        miss_rate    = round(cache_counts["miss"]    / total * 100, 1)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cache hits",   str(hit_rate) + "%",     str(cache_counts["hit"]) + " runs")
        col_b.metric("Hybrid",       str(context_rate) + "%", str(cache_counts["context"]) + " runs")
        col_c.metric("Fresh search", str(miss_rate) + "%",    str(cache_counts["miss"]) + " runs")
        st.caption("Chroma library: " + str(chroma.get("total_stored",0)) + " runs stored")

        # Cache trend chart
        cache_trend = []
        for r in metrics:
            cs = r.get("cache_status","miss") or "miss"
            cache_trend.append({
                "Run":    r.get("run_id",""),
                "Hit":    1 if cs == "hit" else 0,
                "Hybrid": 1 if cs == "context" else 0,
                "Miss":   1 if cs == "miss" else 0
            })
        if len(cache_trend) > 2:
            df_cache = pd.DataFrame(cache_trend).set_index("Run")
            st.area_chart(df_cache)
    else:
        st.info("No cache data yet")

with col2:
    st.subheader("Queries by Intent")
    intent_counts = {}
    for r in metrics:
        intent = r.get("intent","unknown") or "unknown"
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    if intent_counts:
        df_intent = pd.DataFrame({
            "Intent": list(intent_counts.keys()),
            "Count":  list(intent_counts.values())
        })
        st.bar_chart(df_intent.set_index("Intent"))
        st.caption("Most used: " + max(intent_counts, key=intent_counts.get))
    else:
        st.info("No intent data yet")

st.divider()

# ---------------------------
# Agent communications
# ---------------------------
st.subheader("Agent Communications")
if comms_stats.get("total_handoffs", 0) > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total handoffs",    comms_stats["total_handoffs"])
    col2.metric("Avg input tokens",  comms_stats.get("avg_input_tokens","—"))
    col3.metric("Avg output tokens", comms_stats.get("avg_output_tokens","—"))

    if comms_stats.get("handoff_breakdown"):
        st.caption("Handoff breakdown:")
        cols = st.columns(len(comms_stats["handoff_breakdown"]))
        for i, (k, v) in enumerate(comms_stats["handoff_breakdown"].items()):
            cols[i].metric(k, v)

    # Token flow per agent
    if comms:
        agent_tokens = {}
        for c in comms:
            agent = c.get("from_agent","unknown")
            tokens = c.get("output",{}).get("tokens",0)
            if agent not in agent_tokens:
                agent_tokens[agent] = []
            agent_tokens[agent].append(tokens)
        avg_tokens = {a: round(sum(t)/len(t),1) for a, t in agent_tokens.items()}
        df_tokens = pd.DataFrame({
            "Agent":       list(avg_tokens.keys()),
            "Avg Tokens":  list(avg_tokens.values())
        })
        st.caption("Avg output tokens per agent:")
        st.bar_chart(df_tokens.set_index("Agent"))
else:
    st.info("No agent communications logged yet")

st.divider()

# ---------------------------
# Pipeline breakdown
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pipeline Usage")
    pipeline_counts = {}
    for r in metrics:
        pt = r.get("pipeline_type","research") or "research"
        pipeline_counts[pt] = pipeline_counts.get(pt, 0) + 1
    for pt, count in pipeline_counts.items():
        pct = round(count / len(metrics) * 100, 1)
        st.metric(pt.title(), str(count) + " runs", str(pct) + "%")

with col2:
    st.subheader("Quality Signals")
    fb_used      = sum(1 for r in metrics if r.get("feedback_loop_used"))
    hall_flags   = sum(1 for r in metrics if r.get("hallucination_flagged"))
    entity_warns = sum(1 for r in metrics if not r.get("entity_valid", True))
    total        = len(metrics)

    st.metric("Feedback loops triggered",   str(fb_used),
              str(round(fb_used/total*100,1)) + "% of runs")
    st.metric("Hallucination flags",        str(hall_flags),
              str(round(hall_flags/total*100,1)) + "% of runs")
    st.metric("Entity validation warnings", str(entity_warns),
              str(round(entity_warns/total*100,1)) + "% of runs")

st.divider()

# ---------------------------
# Recent runs table
# ---------------------------
st.subheader("Recent Runs")
recent = metrics[-10:][::-1]
for r in recent:
    score = r.get("critic_score", -1)
    cs    = r.get("cache_status","miss") or "miss"
    cache_icon = "✓" if cs=="hit" else ("~" if cs=="context" else "○")

    col1, col2, col3, col4, col5 = st.columns([4,1,1,1,1])
    col1.markdown("**" + (r.get("query","—")[:55] or "—") + "**")
    col2.markdown("`" + (r.get("intent","—") or "—")[:12] + "`")

    if score >= 7:
        col3.success(str(score) + "/10")
    elif score >= 5:
        col3.warning(str(score) + "/10")
    elif score > 0:
        col3.error(str(score) + "/10")
    else:
        col3.markdown("—")

    col4.markdown("`" + str(round(r.get("total_latency",0))) + "s`")
    col5.markdown("`" + cache_icon + " " + cs + "`")

st.divider()

# ---------------------------
# Cost estimate
# ---------------------------
st.subheader("Cost Estimate")
total_tokens = sum(r.get("total_tokens",0) for r in metrics)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total tokens (est)",      str(f"{total_tokens:,}"))
col2.metric("Current cost",            "$0.00",      "NVIDIA free tier")
col3.metric("Claude Sonnet V2 est",
    "$" + str(round(total_tokens * 0.000015, 2)),
    "at $15/1M output tokens")
col4.metric("Claude Haiku V2 est",
    "$" + str(round(total_tokens * 0.0000013, 4)),
    "at $1.25/1M output tokens")

st.caption(
    "Model: NVIDIA meta/llama-3.1-8b-instruct · "
    "Token counts are word-based estimates (~20% off actual) · "
    "V2 upgrade path: Claude Sonnet for writer/critic, Haiku for researcher/analyst"
)

st.divider()
st.caption(
    "Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M") +
    " · " + str(len(metrics)) + " total runs · " +
    "PM Intel Day 29"
)
