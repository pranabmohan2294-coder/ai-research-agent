import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ai-research-agent'

import streamlit as st
import time
import json
import asyncio
import concurrent.futures
from typing import TypedDict, List, Dict, Optional
from langchain_ollama import ChatOllama
from ddgs import DDGS

# ---------------------------
# State
# ---------------------------

class OrchestratorState(TypedDict):
    query: str
    intent: str
    execution_mode: str
    agent_outputs: Dict[str, str]
    completed_agents: List[str]
    next_agent: str
    final_report: str
    gaps: List[str]
    feedback_loop_used: bool
    iteration: int
    done: bool

# ---------------------------
# LLM
# ---------------------------

llm = ChatOllama(model="llama3.2", temperature=0)

# ---------------------------
# Read vs Write classification
# ---------------------------

READ_AGENTS  = {"web_researcher", "data_analyst", "parallel_researcher"}
WRITE_AGENTS = {"writer", "critic"}

def requires_human_checkpoint(agent: str) -> bool:
    return agent in WRITE_AGENTS

# ---------------------------
# Web search
# ---------------------------

def web_search(query: str, max_results: int = 10) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query, max_results=max_results, region='wt-wt'
            ))
        if not results:
            return f"No results for: {query}"
        out = ""
        for i, r in enumerate(results, 1):
            out += f"[Result {i}]\nTitle: {r.get('title','')}\n"
            out += f"URL: {r.get('href','')}\n"
            out += f"Content: {r.get('body','')}\n\n"
        return out
    except Exception as e:
        return f"Search failed: {e}"

# ---------------------------
# Orchestrator — intent + plan
# ---------------------------

def classify_intent(query: str) -> dict:
    prompt = f"""Analyse this user query and return a JSON plan.

Query: "{query}"

Decide:
1. intent: one of "competitive_analysis", "market_research", "job_research", "comparison", "general_research"
2. execution_mode: "sequential" if each step needs the previous step's output, "parallel" if multiple independent searches needed
3. needs_data_analyst: true if query needs numbers/stats/market sizing, false otherwise
4. search_queries: list of 1-3 specific search queries to run (for parallel mode, list multiple)
5. output_format: one of "competitive_report", "market_report", "comparison_table", "research_report"

Return ONLY valid JSON like this:
{{
  "intent": "competitive_analysis",
  "execution_mode": "sequential",
  "needs_data_analyst": true,
  "search_queries": ["AI coding assistants market share 2025", "Copilot vs Cursor vs Tabnine comparison"],
  "output_format": "competitive_report"
}}"""

    response = llm.invoke(prompt)
    text = response.content.strip()
    try:
        if "```" in text:
            text = text.split("```")[1].replace("json","").strip()
        return json.loads(text)
    except Exception:
        return {
            "intent": "general_research",
            "execution_mode": "sequential",
            "needs_data_analyst": False,
            "search_queries": [query],
            "output_format": "research_report"
        }

def decide_next_agent(state: OrchestratorState) -> dict:
    completed  = state.get("completed_agents", [])
    intent     = state.get("intent", "general_research")
    mode       = state.get("execution_mode", "sequential")
    needs_data = st.session_state.get("plan", {}).get("needs_data_analyst", False)
    gaps       = state.get("gaps", [])
    fb_used    = state.get("feedback_loop_used", False)

    # Feedback loop — critic found gaps and loop not yet used
    if gaps and not fb_used and "web_researcher" in completed and "writer" in completed:
        return {
            "next_agent": "gap_researcher",
            "reason": f"Critic found gaps: {', '.join(gaps[:2])}. Running targeted search."
        }

    # Standard sequence
    if "web_researcher" not in completed and mode == "sequential":
        return {"next_agent": "web_researcher", "reason": "Starting with live web search to ground all findings."}

    if mode == "parallel" and "parallel_researcher" not in completed:
        return {"next_agent": "parallel_researcher", "reason": "Running parallel searches for multiple sources."}

    if needs_data and "data_analyst" not in completed and \
       ("web_researcher" in completed or "parallel_researcher" in completed):
        return {"next_agent": "data_analyst", "reason": "Research gathered — extracting numbers and structured data."}

    if "writer" not in completed:
        return {"next_agent": "writer", "reason": "Sufficient research gathered — synthesising final report."}

    if "critic" not in completed:
        return {"next_agent": "critic", "reason": "Report written — reviewing for gaps and quality."}

    return {"next_agent": "DONE", "reason": "All agents complete."}

# ---------------------------
# Agent: Web Researcher
# ---------------------------

def run_web_researcher(state: OrchestratorState, placeholder,
                       custom_queries: List[str] = None) -> str:
    query    = state["query"]
    fmt      = st.session_state.get("plan", {}).get("output_format", "research_report")
    queries  = custom_queries or st.session_state.get("plan", {}).get(
        "search_queries", [query]
    )

    all_results = ""
    for i, q in enumerate(queries, 1):
        placeholder.info(f"Searching ({i}/{len(queries)}): {q}")
        all_results += f"\n=== SEARCH: {q} ===\n"
        all_results += web_search(q, max_results=8)

    placeholder.info(f"Found {all_results.count('[Result')} total results. Synthesising...")

    if fmt == "competitive_report":
        format_instructions = """
Structure your response EXACTLY as follows:

## Market Overview
2-3 sentences on the overall market size, growth rate, and trajectory.
Include specific numbers if found in results.

## Key Players (list ALL companies/tools mentioned in results)
For each player found:
**[Company/Tool Name]**
- What it is: one sentence
- Key features: 2-3 specific features mentioned
- Target users: who uses it
- Pricing: exact pricing if mentioned, otherwise "not found"
- Market position: leader / challenger / niche
- Source: URL

## Competitive Differentiators
What separates the top players from each other?

## Market Share and Adoption
Any data on market share, user numbers, or adoption rates found in results.
If not found, state explicitly: "No market share data found in sources."

## Recent Developments
Latest news, funding, launches from the results.

## Sources
All URLs used."""
    elif fmt == "comparison_table":
        format_instructions = """
Structure your response EXACTLY as follows:

## Head-to-Head Comparison
Create a detailed comparison table.

## Option A — Deep Dive
Specific strengths with evidence from results.

## Option B — Deep Dive
Specific strengths with evidence from results.

## Use Case Fit
Which option for which scenario.

## Sources
All URLs used."""
    else:
        format_instructions = """
Structure with these sections:
## Key Findings
## Companies and Tools Mentioned
## Data and Numbers Found
## Recent Developments
## Sources"""

    prompt = f"""You are a web research agent with REAL search results below.

CRITICAL RULES:
- Use ONLY the information in the search results
- Do NOT generate from memory or training data
- If data is not in the results, write "not found in sources"
- Reference results by number e.g. [Result 3] says...
- List every company, tool, and product mentioned by name

Original query: {query}

REAL SEARCH RESULTS:
{all_results}

{format_instructions}

Write at least 400 words. Be specific — name every tool, company, and metric you find."""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output

# ---------------------------
# Agent: Parallel Researcher
# ---------------------------

def run_parallel_researcher(state: OrchestratorState, placeholder) -> str:
    queries = st.session_state.get("plan", {}).get("search_queries", [state["query"]])
    placeholder.info(f"Running {len(queries)} parallel searches...")

    all_results = {}
    for i, q in enumerate(queries, 1):
        placeholder.info(f"Search {i}/{len(queries)}: {q}")
        all_results[q] = web_search(q, max_results=6)

    combined = ""
    for q, results in all_results.items():
        combined += f"\n\n=== SOURCE: {q} ===\n{results}"

    prompt = f"""You are a research agent. Synthesise these parallel search results.

Query: {state['query']}

Results from multiple searches:
{combined}

For each search source, extract:
- Key findings specific to that source
- Unique information not found in other sources
- Any contradictions between sources

Then provide a unified synthesis combining all sources."""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output

# ---------------------------
# Agent: Gap Researcher
# ---------------------------

def run_gap_researcher(state: OrchestratorState, placeholder) -> str:
    gaps  = state.get("gaps", [])
    query = state["query"]

    gap_queries = [f"{query} {gap}" for gap in gaps[:2]]
    placeholder.info(f"Filling {len(gap_queries)} gaps identified by critic...")

    return run_web_researcher(
        state, placeholder, custom_queries=gap_queries
    )

# ---------------------------
# Agent: Data Analyst
# ---------------------------

def run_data_analyst(state: OrchestratorState, placeholder) -> str:
    query    = state["query"]
    research = state.get("agent_outputs", {}).get(
        "web_researcher",
        state.get("agent_outputs", {}).get("parallel_researcher", "No research.")
    )

    prompt = f"""You are a data analyst. Extract ALL quantitative data from this research.

Query: {query}

Research:
{research}

## Numbers and Statistics Table
| Metric | Value | Source |
|---|---|---|
Extract EVERY number, percentage, market size, growth rate, user count, price.
If no numbers exist, write "No quantitative data found in sources."

## Market Sizing
Any TAM, SAM, SOM or market size figures.

## Growth Trends
Any YoY growth, CAGR, or trend data.

## Competitive Numbers
User counts, market share percentages, revenue figures per company.

## Data Quality Rating
Rate each data point: High (cited source) / Medium (estimated) / Low (unclear source)

## What's Missing
Critical numbers not found that would strengthen this analysis."""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output

# ---------------------------
# Agent: Writer
# ---------------------------

def run_writer(state: OrchestratorState, placeholder) -> str:
    query   = state["query"]
    outputs = state.get("agent_outputs", {})
    fmt     = st.session_state.get("plan", {}).get("output_format", "research_report")

    all_research = ""
    for agent, content in outputs.items():
        all_research += f"\n\n=== {agent.upper()} OUTPUT ===\n{content}"

    if fmt == "competitive_report":
        structure = """
Write a professional competitive intelligence report:

## Executive Summary (3 sentences — be specific, name the top players)

## Market Landscape
Size, growth rate, key trends. Use numbers from research.

## Competitive Map
| Player | Category | Key Strength | Weakness | Price |
|---|---|---|---|---|
Include ALL players found in research.

## Top 5 Players — Deep Dive
For each: positioning, differentiators, target customer, recent news.

## Market Share and Adoption
All data found. If none: state clearly and explain implications.

## Whitespace and Opportunities
Specific gaps in the market not served by current players.

## Risks and Threats
Key risks for players in this space.

## PM Recommendations
5 specific, actionable recommendations with reasoning.

## Sources
All URLs cited."""
    elif fmt == "comparison_table":
        structure = """
Write a definitive comparison report:

## One-Line Verdict (which to choose and why)

## Detailed Comparison Table
At least 10 dimensions.

## When to Choose Option A
3 specific scenarios with reasoning.

## When to Choose Option B
3 specific scenarios with reasoning.

## Hidden Considerations
What most comparisons miss.

## PM Recommendation
Clear decision framework."""
    else:
        structure = """
## Executive Summary
## Key Findings (numbered, specific)
## Analysis and Implications
## Data and Evidence
## Recommendations
## Sources"""

    prompt = f"""You are a senior analyst writing a final report.

Query: {query}

All Research:
{all_research}

{structure}

CRITICAL:
- Name every company, tool, and product specifically
- Include every number and statistic from the research
- If data is missing, say so explicitly — do not invent
- Be specific, not generic
- Minimum 500 words"""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output

# ---------------------------
# Agent: Critic
# ---------------------------

def run_critic(state: OrchestratorState, placeholder) -> str:
    query   = state["query"]
    report  = state.get("agent_outputs", {}).get("writer", "No report.")
    fb_used = state.get("feedback_loop_used", False)

    prompt = f"""You are a senior editor and critic reviewing this report.

Original query: {query}

Report:
{report}

## What Works Well
3 specific strengths with examples from the report.

## Critical Gaps (be specific)
List exactly what data or analysis is missing.
Format each gap as a searchable query e.g. "Cursor pricing 2025" or "GitHub Copilot market share".

## Fact Check
Flag any claims that appear to be hallucinated or lack source citations.

## Improved Executive Summary
Rewrite the executive summary to be sharper and more specific.

## Gap Search Queries
{'List 2 specific search queries to fill the most critical gaps.' if not fb_used else 'No further searches needed — feedback loop already used.'}

## Final Verdict
Is this ready to share with a stakeholder? Score 1-10 and explain."""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)

    # Extract gap queries from output
    gaps = []
    if not fb_used:
        lines = output.split("\n")
        in_gaps = False
        for line in lines:
            if "Gap Search Queries" in line:
                in_gaps = True
                continue
            if in_gaps and line.strip().startswith(("- ", "* ", "1.", "2.")):
                gap = line.strip().lstrip("-*0123456789. ")
                if len(gap) > 5:
                    gaps.append(gap)
            elif in_gaps and line.startswith("##"):
                break

    state["gaps"] = gaps[:2]
    st.session_state.state = state
    return output

# ---------------------------
# Agent registry
# ---------------------------

AGENT_FUNCTIONS = {
    "web_researcher":    run_web_researcher,
    "parallel_researcher": run_parallel_researcher,
    "gap_researcher":    run_gap_researcher,
    "data_analyst":      run_data_analyst,
    "writer":            run_writer,
    "critic":            run_critic,
}

AGENT_LABELS = {
    "web_researcher":      "Web Researcher",
    "parallel_researcher": "Parallel Researcher",
    "gap_researcher":      "Gap Researcher",
    "data_analyst":        "Data Analyst",
    "writer":              "Writer",
    "critic":              "Critic / Reviewer",
}

AGENT_ICONS = {
    "web_researcher":      "🔍",
    "parallel_researcher": "🔍🔍",
    "gap_researcher":      "🔎",
    "data_analyst":        "📊",
    "writer":              "✍️",
    "critic":              "🔬",
}

AGENT_TYPE = {
    "web_researcher":      "read",
    "parallel_researcher": "read",
    "gap_researcher":      "read",
    "data_analyst":        "read",
    "writer":              "write",
    "critic":              "write",
}

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Multi-Agent Orchestrator", layout="wide")
st.title("Multi-Agent Orchestrator")
st.caption(
    "Orchestrator decides agents + order · "
    "Human checkpoint only on write ops · "
    "Critic feeds back to researcher · Day 20"
)

# Sidebar
with st.sidebar:
    st.header("Pipeline State")
    state = st.session_state.get("state", {})

    if state:
        plan = st.session_state.get("plan", {})
        st.markdown("**Query**")
        st.code(state.get("query","—")[:50])
        st.markdown("**Intent**")
        st.code(plan.get("intent","—"))
        st.markdown("**Mode**")
        st.code(plan.get("execution_mode","—"))
        st.markdown("**Format**")
        st.code(plan.get("output_format","—"))
        st.markdown("**Feedback loop**")
        st.code("Used" if state.get("feedback_loop_used") else "Available")
        st.markdown("**Gaps found**")
        gaps = state.get("gaps",[])
        if gaps:
            for g in gaps:
                st.warning(g)
        else:
            st.code("none yet")

        st.divider()
        st.markdown("**Completed agents**")
        for a in state.get("completed_agents",[]):
            t = AGENT_TYPE.get(a,"read")
            if t == "write":
                st.success(f"{AGENT_ICONS.get(a,'')} {AGENT_LABELS.get(a,a)} (write)")
            else:
                st.info(f"{AGENT_ICONS.get(a,'')} {AGENT_LABELS.get(a,a)} (read)")
    else:
        st.info("Run a query to see state")

    st.divider()
    perf = st.session_state.get("performance", {})
    if perf:
        st.header("Performance")
        for agent, m in perf.items():
            st.markdown(
                f"**{AGENT_ICONS.get(agent,'')} {AGENT_LABELS.get(agent,agent)}**  "
                f"`{m.get('latency',0):.1f}s` · `{m.get('tokens',0)} tokens`"
            )
        total_l = sum(m.get("latency",0) for m in perf.values())
        total_t = sum(m.get("tokens",0) for m in perf.values())
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Total", f"{total_l:.1f}s")
        col2.metric("Tokens", str(total_t))
        st.markdown("- llama3.2 local · DuckDuckGo · $0.00")

    st.divider()
    st.success("LangSmith tracing on")
    st.markdown("[View traces →](https://smith.langchain.com)")

# Session init
for key, default in [
    ("stage","input"), ("state",{}),
    ("performance",{}), ("agent_log",[]), ("plan",{})
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── STAGE 1: Input ──────────────────────────────────────────
if st.session_state.stage == "input":
    st.subheader("What do you want to research?")
    st.caption(
        "The orchestrator analyses your query, decides which agents are needed, "
        "whether to run sequentially or in parallel, and only asks for your "
        "approval on write operations."
    )

    col1, col2 = st.columns(2)
    col1.info("**Read agents** (auto)\n🔍 Web Researcher\n📊 Data Analyst")
    col2.warning("**Write agents** (need approval)\n✍️ Writer\n🔬 Critic")

    st.divider()

    examples = [
        "Analyse the competitive landscape for AI coding assistants in 2025",
        "Research the job market for senior product managers in India 2025",
        "Compare LangGraph vs CrewAI for building production agents",
        "What are the latest developments in RAG pipeline optimisation 2025",
    ]

    st.markdown("**Try these:**")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i%2].button(ex, use_container_width=True):
            st.session_state.prefill = ex
            st.rerun()

    prefill = st.session_state.get("prefill","")
    query = st.text_input(
        "Your query",
        value=prefill,
        placeholder="Ask anything — competitive analysis, market research, comparisons..."
    )

    if st.button("Analyse and Start", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Orchestrator analysing query..."):
                plan = classify_intent(query)
            st.session_state.plan = plan
            st.session_state.state = OrchestratorState(
                query=query,
                intent=plan.get("intent","general_research"),
                execution_mode=plan.get("execution_mode","sequential"),
                agent_outputs={},
                completed_agents=[],
                next_agent="",
                final_report="",
                gaps=[],
                feedback_loop_used=False,
                iteration=0,
                done=False
            )
            st.session_state.performance = {}
            st.session_state.agent_log = []
            if "prefill" in st.session_state:
                del st.session_state.prefill
            st.session_state.stage = "orchestrating"
            st.rerun()

# ── STAGE 2: Orchestrator decides ───────────────────────────
elif st.session_state.stage == "orchestrating":
    state = st.session_state.state
    plan  = st.session_state.plan

    st.subheader("Orchestrator planning next step...")

    col1, col2, col3 = st.columns(3)
    col1.metric("Intent", plan.get("intent","—"))
    col2.metric("Mode", plan.get("execution_mode","—"))
    col3.metric("Iteration", str(state.get("iteration",0)))

    decision = decide_next_agent(state)
    next_agent = decision["next_agent"]
    reason     = decision["reason"]

    if next_agent == "DONE":
        state["done"] = True
        st.session_state.state = state
        st.session_state.stage = "done"
    else:
        state["next_agent"] = next_agent
        state["iteration"]  = state.get("iteration",0) + 1
        st.session_state.state = state

        agent_type = AGENT_TYPE.get(next_agent,"read")
        if agent_type == "write":
            st.session_state.stage = "write_checkpoint"
        else:
            st.session_state.stage = "running_agent"

        st.info(
            f"**Next:** {AGENT_ICONS.get(next_agent,'')} "
            f"{AGENT_LABELS.get(next_agent)} ({agent_type})\n\n_{reason}_"
        )
    st.rerun()

# ── STAGE 3a: Write checkpoint (only for write agents) ──────
elif st.session_state.stage == "write_checkpoint":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])

    st.subheader(
        f"Write Operation — "
        f"{AGENT_ICONS.get(next_agent,'')} {AGENT_LABELS.get(next_agent)}"
    )
    st.warning(
        f"**{AGENT_LABELS.get(next_agent)}** is a write operation — "
        f"it will produce content. Do you want to proceed?"
    )

    st.markdown(f"**Completed so far:** {', '.join([AGENT_LABELS.get(a,a) for a in completed]) or 'none'}")
    st.markdown(f"**Reason:** {state.get('plan','')}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            f"Run {AGENT_LABELS.get(next_agent)}",
            type="primary", use_container_width=True
        ):
            st.session_state.stage = "running_agent"
            st.rerun()
    with col2:
        if st.button("Skip this agent", use_container_width=True):
            state["completed_agents"].append(next_agent)
            state["agent_outputs"][next_agent] = "Skipped by user."
            st.session_state.state = state
            st.session_state.stage = "orchestrating"
            st.rerun()

# ── STAGE 3b: Run agent ─────────────────────────────────────
elif st.session_state.stage == "running_agent":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])
    agent_type = AGENT_TYPE.get(next_agent,"read")

    n_done  = len(completed)
    n_total = 4
    st.progress(n_done/n_total, text=f"{n_done}/{n_total} agents complete")

    st.subheader(
        f"{AGENT_ICONS.get(next_agent,'')} "
        f"{AGENT_LABELS.get(next_agent)} is working..."
    )
    st.caption(f"Type: {agent_type} · No checkpoint needed for read agents")

    placeholder = st.empty()
    start = time.time()

    agent_fn = AGENT_FUNCTIONS.get(next_agent)
    output   = agent_fn(state, placeholder) if agent_fn else "Unknown agent."

    latency     = round(time.time()-start, 1)
    token_count = len(output.split())

    state["agent_outputs"][next_agent] = output
    state["completed_agents"].append(next_agent)

    if next_agent == "gap_researcher":
        state["feedback_loop_used"] = True
        existing = state["agent_outputs"].get("web_researcher","")
        state["agent_outputs"]["web_researcher"] = existing + "\n\n=== GAP FILL ===\n" + output

    if next_agent in ("writer","critic"):
        state["final_report"] = output

    st.session_state.state = state
    st.session_state.performance[next_agent] = {
        "latency": latency, "tokens": token_count
    }
    st.session_state.agent_log.append({
        "agent": next_agent, "output": output, "latency": latency
    })

    if agent_type == "write":
        st.session_state.stage = "review_output"
    else:
        st.session_state.stage = "orchestrating"
    st.rerun()

# ── STAGE 4: Review output (write agents only) ───────────────
elif st.session_state.stage == "review_output":
    state      = st.session_state.state
    last_agent = state["completed_agents"][-1]
    last_out   = state["agent_outputs"].get(last_agent,"")
    perf       = st.session_state.performance.get(last_agent,{})
    gaps       = state.get("gaps",[])

    st.subheader(
        f"Review — {AGENT_ICONS.get(last_agent,'')} "
        f"{AGENT_LABELS.get(last_agent)} output"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{perf.get('latency',0):.1f}s")
    col2.metric("Tokens", str(perf.get("tokens","—")))
    col3.metric("Gaps found", str(len(gaps)))

    if gaps:
        st.warning(f"Critic found {len(gaps)} gap(s): {' · '.join(gaps)}")
        if not state.get("feedback_loop_used"):
            st.info("The orchestrator will trigger a targeted re-search to fill these gaps.")

    with st.expander(
        f"{AGENT_ICONS.get(last_agent,'')} {AGENT_LABELS.get(last_agent)} output",
        expanded=True
    ):
        st.markdown(last_out)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Approve — continue pipeline",
            type="primary", use_container_width=True
        ):
            st.session_state.stage = "orchestrating"
            st.rerun()
    with col2:
        if st.button(
            "Stop — use this as final report",
            use_container_width=True
        ):
            state["final_report"] = last_out
            state["done"] = True
            st.session_state.state = state
            st.session_state.stage = "done"
            st.rerun()

# ── STAGE 5: Done ───────────────────────────────────────────
elif st.session_state.stage == "done":
    state        = st.session_state.state
    perf         = st.session_state.performance
    agents_used  = state.get("completed_agents",[])
    total_l      = sum(m.get("latency",0) for m in perf.values())
    total_t      = sum(m.get("tokens",0) for m in perf.values())

    st.subheader("Pipeline Complete")
    st.success(f"Query: {state['query']}")
    st.progress(1.0, text="All agents complete")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agents run",    str(len(agents_used)))
    col2.metric("Total latency", f"{total_l:.1f}s")
    col3.metric("Total tokens",  str(total_t))
    col4.metric("Cost",          "$0.00")

    fb = "Yes" if state.get("feedback_loop_used") else "No"
    st.info(
        f"Intent: `{state.get('intent')}` · "
        f"Mode: `{state.get('execution_mode')}` · "
        f"Feedback loop: `{fb}`"
    )
    st.markdown("Traces → [View in LangSmith](https://smith.langchain.com)")
    st.divider()

    final = state.get("final_report","No final report.")
    tabs  = st.tabs(
        ["Final Report"] +
        [f"{AGENT_ICONS.get(a,'')} {AGENT_LABELS.get(a,a)}" for a in agents_used]
    )

    with tabs[0]:
        st.markdown(final)
        st.download_button(
            "Download report as markdown",
            data=final,
            file_name=f"report_{state['query'][:30].replace(' ','_')}.md",
            mime="text/markdown"
        )

    for i, agent in enumerate(agents_used):
        with tabs[i+1]:
            st.markdown(state["agent_outputs"].get(agent,"No output."))

    st.divider()
    with st.expander("Execution log", expanded=False):
        for entry in st.session_state.get("agent_log",[]):
            st.markdown(
                f"**{AGENT_ICONS.get(entry['agent'],'')} "
                f"{AGENT_LABELS.get(entry['agent'])}** — {entry['latency']}s"
            )
            st.caption(entry["output"][:400]+"...")
            st.divider()

    if st.button("Run new query", type="primary"):
        for k in ["stage","state","performance","agent_log","plan"]:
            st.session_state[k] = {"stage":"input"}.get(k, [] if k=="agent_log" else {})
        st.session_state.stage = "input"
        st.rerun()
