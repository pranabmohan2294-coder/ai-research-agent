import streamlit as st
import time
from typing import TypedDict
from langchain_ollama import ChatOllama

class ResearchState(TypedDict):
    topic: str
    research: str
    summary: str
    human_approved: bool

llm = ChatOllama(model="llama3.2", temperature=0)

def summariser(state: ResearchState) -> ResearchState:
    prompt = f"""You are a summarisation agent. Take this research and produce a structured PM summary.

Research:
{state['research']}

Structure your response with these exact headers:

## Key Takeaways
- Bullet 1
- Bullet 2
- Bullet 3

## Most Important Fact
One sentence — the single most critical piece of information.

## What a PM Should Know
2-3 sentences on the product management implications of this topic.

## Open Question Worth Investigating
One specific question that deserves further research.

Be concise, precise, and PM-focused."""
    response = llm.invoke(prompt)
    return {**state, "summary": response.content}

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("AI Research Agent")
st.caption("Sequential 2-agent pipeline · LangGraph + Ollama llama3.2 · Day 18")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:

    st.header("State Inspector")
    st.caption("Live view of pipeline state at each step")
    if "state" in st.session_state and st.session_state.state:
        s = st.session_state.state
        st.markdown("**topic**")
        st.code(s.get("topic", "—"))
        st.markdown("**human_approved**")
        st.code(str(s.get("human_approved", False)))
        st.markdown("**research** (chars)")
        st.code(str(len(s.get("research", ""))))
        st.markdown("**summary** (chars)")
        st.code(str(len(s.get("summary", ""))))
    else:
        st.info("Run a topic to see state")

    st.divider()

    st.header("Agent Performance")
    st.caption("Metrics updated after each agent run")

    perf = st.session_state.get("performance", {})

    if perf:
        # Agent 1 metrics
        if perf.get("agent1_latency"):
            st.markdown("**Agent 1 — Researcher**")
            col1, col2 = st.columns(2)
            col1.metric("Latency", f"{perf['agent1_latency']:.1f}s")
            col2.metric("Tokens", str(perf.get("agent1_tokens", "—")))
            st.markdown("**Data source**")
            st.info("llama3.2 training data\nKnowledge cutoff: early 2024\nNo live web search")
            st.markdown("**Accuracy note**")
            st.warning("Not grounded in real-time data. Facts may be outdated. Web search tool added in Day 19.")

        st.divider()

        # Agent 2 metrics
        if perf.get("agent2_latency"):
            st.markdown("**Agent 2 — Summariser**")
            col1, col2 = st.columns(2)
            col1.metric("Latency", f"{perf['agent2_latency']:.1f}s")
            col2.metric("Tokens", str(perf.get("agent2_tokens", "—")))
            st.markdown("**Input source**")
            st.success("Agent 1 research output\nPassed via shared state")

        st.divider()

        # Pipeline totals
        if perf.get("agent1_latency") and perf.get("agent2_latency"):
            st.markdown("**Pipeline Total**")
            total_latency = perf.get("agent1_latency", 0) + perf.get("agent2_latency", 0)
            total_tokens = perf.get("agent1_tokens", 0) + perf.get("agent2_tokens", 0)
            col1, col2 = st.columns(2)
            col1.metric("Total time", f"{total_latency:.1f}s")
            col2.metric("Total tokens", str(total_tokens))
            st.markdown("**Resources used**")
            st.markdown("""
- Model: `llama3.2` (local)
- Runtime: Ollama
- Orchestration: LangGraph
- Cost: $0.00 (fully local)
            """)
    else:
        st.info("Run a topic to see performance metrics")

# ---------------------------
# Session state init
# ---------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "state" not in st.session_state:
    st.session_state.state = {}
if "performance" not in st.session_state:
    st.session_state.performance = {}

# ── STAGE 1: Input ──────────────────────────────────────────
if st.session_state.stage == "input":
    st.subheader("Step 1 — Enter a research topic")
    topic = st.text_input("Topic", placeholder="e.g. RAG pipelines in production AI products")

    if st.button("Start Research", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            st.session_state.state = {
                "topic": topic,
                "research": "",
                "summary": "",
                "human_approved": False
            }
            st.session_state.performance = {}
            st.session_state.stage = "researching"
            st.rerun()

# ── STAGE 2: Streaming Research ─────────────────────────────
elif st.session_state.stage == "researching":
    topic = st.session_state.state["topic"]
    st.subheader("Step 2 — Agent 1: Researcher")
    st.info(f"Researching: **{topic}**")

    prompt = f"""You are a research agent. Research the following topic thoroughly.

Topic: {topic}

Structure your response with these exact headers:

## Key Facts and Background
Provide 4-5 sentences of core background.

## Current State and Recent Developments
What is happening right now in this space? Latest trends and developments.

## Main Challenges and Considerations
What are the key problems, risks, or things to watch out for?

## Key Players and Stakeholders
Who are the main companies, people, or groups involved?

## Data and Numbers
Include any relevant statistics, market data, or quantitative facts.

## Summary
2-3 sentence wrap-up of the most important points.

Write comprehensively. Do not truncate any section. Minimum 500 words total."""

    research_placeholder = st.empty()
    full_research = ""
    token_count = 0
    start_time = time.time()

    with st.spinner("Agent 1 is researching..."):
        for chunk in llm.stream(prompt):
            full_research += chunk.content
            token_count += len(chunk.content.split())
            research_placeholder.markdown(full_research)

    latency = round(time.time() - start_time, 1)

    st.session_state.state["research"] = full_research
    st.session_state.performance["agent1_latency"] = latency
    st.session_state.performance["agent1_tokens"] = token_count
    st.session_state.stage = "checkpoint"
    st.rerun()

# ── STAGE 3: Human Checkpoint ───────────────────────────────
elif st.session_state.stage == "checkpoint":
    st.subheader("Step 3 — Human Checkpoint")
    st.success("Agent 1 finished. Review the full research below before Agent 2 runs.")

    state = st.session_state.state
    perf = st.session_state.performance

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Research length", f"{len(state['research'])} chars")
    col2.metric("Word count", f"{len(state['research'].split())} words")
    col3.metric("Agent 1 latency", f"{perf.get('agent1_latency', 0):.1f}s")
    col4.metric("Agent 1 tokens", str(perf.get("agent1_tokens", "—")))

    st.divider()

    with st.expander("Full Research from Agent 1 — click to expand/collapse", expanded=True):
        st.markdown(state["research"])

    st.divider()
    st.markdown("**Approve this research? Agent 2 will summarise it.**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve — run Agent 2", type="primary", use_container_width=True):
            st.session_state.state["human_approved"] = True
            st.session_state.stage = "summarising"
            st.rerun()
    with col2:
        if st.button("Reject — stop pipeline", use_container_width=True):
            st.session_state.state["human_approved"] = False
            st.session_state.stage = "rejected"
            st.rerun()

# ── STAGE 4: Summarising ────────────────────────────────────
elif st.session_state.stage == "summarising":
    st.subheader("Step 4 — Agent 2: Summariser")
    st.info("Summarising research into PM-focused report...")

    summary_placeholder = st.empty()
    full_summary = ""
    token_count = 0
    start_time = time.time()

    summary_prompt = f"""You are a summarisation agent. Take this research and produce a structured PM summary.

Research:
{st.session_state.state['research']}

Structure your response with these exact headers:

## Key Takeaways
- Bullet 1
- Bullet 2
- Bullet 3

## Most Important Fact
One sentence — the single most critical piece of information.

## What a PM Should Know
2-3 sentences on the product management implications of this topic.

## Open Question Worth Investigating
One specific question that deserves further research.

Be concise, precise, and PM-focused."""

    with st.spinner("Agent 2 is summarising..."):
        for chunk in llm.stream(summary_prompt):
            full_summary += chunk.content
            token_count += len(chunk.content.split())
            summary_placeholder.markdown(full_summary)

    latency = round(time.time() - start_time, 1)

    st.session_state.state["summary"] = full_summary
    st.session_state.performance["agent2_latency"] = latency
    st.session_state.performance["agent2_tokens"] = token_count
    st.session_state.stage = "done"
    st.rerun()

# ── STAGE 5: Done ───────────────────────────────────────────
elif st.session_state.stage == "done":
    st.subheader("Pipeline Complete")
    state = st.session_state.state
    perf = st.session_state.performance

    total_latency = perf.get("agent1_latency", 0) + perf.get("agent2_latency", 0)
    total_tokens = perf.get("agent1_tokens", 0) + perf.get("agent2_tokens", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total latency", f"{total_latency:.1f}s")
    col2.metric("Total tokens", str(total_tokens))
    col3.metric("Research", f"{len(state['research'])} chars")
    col4.metric("Status", "Approved")

    st.divider()

    tab1, tab2 = st.tabs(["Summary (Agent 2)", "Full Research (Agent 1)"])

    with tab1:
        st.markdown(state["summary"])

    with tab2:
        with st.expander("Full Research — click to expand/collapse", expanded=False):
            st.markdown(state["research"])

    st.divider()
    if st.button("Run new topic", type="primary"):
        st.session_state.stage = "input"
        st.session_state.state = {}
        st.session_state.performance = {}
        st.rerun()

# ── STAGE 6: Rejected ───────────────────────────────────────
elif st.session_state.stage == "rejected":
    st.error("Pipeline stopped at human checkpoint.")
    st.markdown("Research was rejected. Agent 2 did not run.")

    with st.expander("Rejected research — click to expand", expanded=False):
        st.markdown(st.session_state.state.get("research", ""))

    if st.button("Try again", type="primary"):
        st.session_state.stage = "input"
        st.session_state.state = {}
        st.session_state.performance = {}
        st.rerun()
