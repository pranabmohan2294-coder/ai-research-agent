import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ai-research-agent'

import streamlit as st
import time
import json
from typing import TypedDict, List, Dict
from langchain_ollama import ChatOllama
from ddgs import DDGS
from metrics_logger import save_run, extract_critic_score, get_summary_stats
from chroma_manager import check_cache, store_run, get_chroma_stats

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
    entity_valid: bool
    data_available: bool
    pipeline_type: str
    critic_score: int
    cache_status: str
    cache_context: str

# ---------------------------
# LLM
# ---------------------------

llm = ChatOllama(model="llama3.2", temperature=0)

# ---------------------------
# Read vs Write
# ---------------------------

READ_AGENTS  = {"web_researcher", "data_analyst", "gap_researcher"}
WRITE_AGENTS = {"writer", "critic"}

# ---------------------------
# Fictional entity detection
# ---------------------------

FICTIONAL_INDICATORS = [
    "fictional", "fake", "made up", "imaginary",
    "hypothetical", "does not exist", "not real",
    "pretend", "invented", "test company", "example company"
]

def validate_entity(query: str) -> dict:
    if any(w in query.lower() for w in FICTIONAL_INDICATORS):
        return {"valid": False,
                "reason": "Query contains fictional indicator."}
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3, region='wt-wt'))
        if not results:
            return {"valid": False,
                    "reason": f"No web results for '{query}'."}
        return {"valid": True, "reason": "Entity found"}
    except Exception as e:
        return {"valid": True, "reason": f"Validation skipped: {e}"}

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
            return f"NO_RESULTS: No web results for: {query}"
        out = ""
        for i, r in enumerate(results, 1):
            out += f"[Result {i}]\nTitle: {r.get('title','')}\n"
            out += f"URL: {r.get('href','')}\n"
            out += f"Content: {r.get('body','')}\n\n"
        return out
    except Exception as e:
        return f"SEARCH_ERROR: {e}"

# ---------------------------
# Intent classification
# ---------------------------

LIFESTYLE_INTENTS = [
    "recipe", "how to make", "ingredients", "cook",
    "places to visit", "best places", "where to go",
    "what to eat", "best food", "weather", "things to buy",
    "how to", "steps to", "guide to", "tutorial",
    "best restaurants", "travel to", "visit in"
]

def detect_pipeline_type(query: str) -> str:
    q = query.lower()
    if any(w in q for w in LIFESTYLE_INTENTS):
        return "lifestyle"
    return "research"

def detect_lifestyle_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["recipe","how to make","cook","ingredients"]):
        return "recipe"
    if any(w in q for w in ["places to visit","best places","where to go","travel","visit in"]):
        return "places"
    if any(w in q for w in ["what to eat","best food","restaurant","cuisine"]):
        return "food"
    if any(w in q for w in ["how to","steps to","guide to","tutorial"]):
        return "howto"
    if any(w in q for w in ["buy","shop","purchase","things to get"]):
        return "shopping"
    return "general"

def classify_research_intent(query: str) -> dict:
    prompt = f"""Analyse this research query and return JSON.

Query: "{query}"

Return ONLY valid JSON:
{{
  "intent": "competitive_analysis" | "market_research" | "job_research" | "comparison" | "general_research",
  "execution_mode": "sequential",
  "needs_data_analyst": true | false,
  "search_queries": ["{query} detailed analysis 2025", "{query} latest developments"],
  "output_format": "competitive_report" | "comparison_table" | "market_report" | "research_report",
  "plain_english_summary": "one sentence: what I understood"
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
            "output_format": "research_report",
            "plain_english_summary": f"Research about: {query}"
        }

def decide_next_research_agent(state: OrchestratorState) -> dict:
    completed   = state.get("completed_agents", [])
    needs_data  = st.session_state.get("plan", {}).get("needs_data_analyst", False)
    gaps        = state.get("gaps", [])
    fb_used     = state.get("feedback_loop_used", False)
    cache_status = state.get("cache_status", "miss")

    # If cache hit — skip web researcher entirely
    if cache_status == "hit" and "web_researcher" not in completed:
        completed_copy = list(completed) + ["web_researcher"]
        state["completed_agents"] = completed_copy
        st.session_state.state = state

    if gaps and not fb_used and "web_researcher" in completed and "writer" in completed:
        return {"next_agent": "gap_researcher",
                "reason": f"Critic found gaps: {', '.join(gaps[:2])}."}
    if "web_researcher" not in completed:
        return {"next_agent": "web_researcher",
                "reason": "Starting with live web search."}
    if needs_data and "data_analyst" not in completed:
        return {"next_agent": "data_analyst",
                "reason": "Extracting numbers and structured data."}
    if "writer" not in completed:
        return {"next_agent": "writer",
                "reason": "Synthesising final report."}
    if "critic" not in completed:
        return {"next_agent": "critic",
                "reason": "Reviewing for gaps and quality."}
    return {"next_agent": "DONE", "reason": "Pipeline complete."}

# ---------------------------
# Lifestyle prompt templates
# ---------------------------

def get_lifestyle_research_prompt(query: str, intent: str, results: str) -> str:
    if intent == "recipe":
        return f"""Extract recipe information from search results.
Query: {query}
Results: {results}

## Overview
## Ingredients (with quantities)
## Method — Step by Step
## Tips and Variations
## Serving Suggestions

Base on search results only."""
    elif intent == "places":
        return f"""Extract place recommendations from search results.
Query: {query}
Results: {results}

## Overview
## Top Places to Visit
For each: name · what it is · why visit · best time · must see

## Best Time to Visit Overall
## Quick Travel Tips"""
    elif intent == "food":
        return f"""Extract food recommendations from search results.
Query: {query}
Results: {results}

## Overview
## Must-Try Items (by category)
## Healthy Options
## Tips"""
    elif intent == "howto":
        return f"""Extract step-by-step guidance from search results.
Query: {query}
Results: {results}

## What You Need
## Step-by-Step Guide
## Tips for Success
## Common Mistakes"""
    else:
        return f"""Answer this query using search results.
Query: {query}
Results: {results}

## Key Findings
## Details
## Recommendations
## Sources"""

def get_lifestyle_summary_prompt(query: str, intent: str, research: str) -> str:
    if intent == "recipe":
        return f"""Summarise into a clean recipe card.
Research: {research}

## The Recipe at a Glance
## Ingredients (quick list with quantities)
## Method (numbered steps — concise but complete)
## The One Tip That Makes It"""
    elif intent == "places":
        return f"""Summarise into a quick travel reference.
Research: {research}

## Top 5 Highlights
## Best Time to Visit
## Don't Miss
## Quick Tips (3)"""
    else:
        return f"""Summarise concisely.
Research: {research}

## Key Takeaways (3 bullets)
## Most Important Point
## What to Do Next"""

# ---------------------------
# Research agents
# ---------------------------

def run_web_researcher(state: OrchestratorState, placeholder,
                       custom_queries: List[str] = None) -> str:
    query        = state["query"]
    fmt          = st.session_state.get("plan",{}).get("output_format","research_report")
    queries      = custom_queries or st.session_state.get(
        "plan",{}).get("search_queries",[query])
    cache_context = state.get("cache_context","")

    all_results = ""
    for i, q in enumerate(queries, 1):
        placeholder.info(f"Searching ({i}/{len(queries)}): {q}")
        all_results += f"\n=== SEARCH: {q} ===\n{web_search(q, max_results=8)}"

    if "NO_RESULTS" in all_results and all_results.count("[Result") < 2:
        msg = "⚠️ Insufficient web data found. Report may be incomplete."
        placeholder.warning(msg)
        return msg

    placeholder.info(f"Found {all_results.count('[Result')} results. Synthesising...")

    # Include cache context if available (Zone 2 hybrid)
    cache_section = ""
    if cache_context:
        cache_section = f"""
PREVIOUSLY RESEARCHED CONTEXT (from cache — use as background):
{cache_context[:2000]}

Prioritise fresh results for recent developments.
Use cached context for background and established facts.
Flag contradictions between old and new findings.
"""

    if fmt == "competitive_report":
        fmt_instructions = """
## Market Overview (numbers if found)
## Key Players (ALL mentioned)
For each: name · features · pricing (exact or "not found") · position · source
## Market Share (only if found — otherwise state missing)
## Recent Developments
## Sources"""
    elif fmt == "comparison_table":
        fmt_instructions = """
## Comparison Table (8+ dimensions, "not found" for missing)
## Option A Strengths (with evidence)
## Option B Strengths (with evidence)
## Data Gaps
## Sources"""
    else:
        fmt_instructions = """
## Key Findings
## Players and Tools Mentioned
## Numbers Found
## Recent Developments
## What Was Not Found
## Sources"""

    prompt = f"""You are a web research agent.

RULES:
- Use ONLY the search results and cached context below
- Write "not found in sources" for missing data
- Reference results by number e.g. [Result 3]
- Name every company and tool found

Query: {query}
{cache_section}
FRESH SEARCH RESULTS:
{all_results}

{fmt_instructions}

Minimum 400 words."""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output

def run_gap_researcher(state: OrchestratorState, placeholder) -> str:
    gaps = state.get("gaps", [])
    gap_queries = [f"{state['query']} {g}" for g in gaps[:2]]
    placeholder.info(f"Filling {len(gap_queries)} gaps...")
    return run_web_researcher(state, placeholder, custom_queries=gap_queries)

def run_data_analyst(state: OrchestratorState, placeholder) -> str:
    query    = state["query"]
    research = state.get("agent_outputs",{}).get("web_researcher","No research.")

    prompt = f"""Data analyst. Extract ALL quantitative data.
Query: {query}
Research: {research}

## Numbers Table
| Metric | Value | Source |
If none: "No quantitative data found."

## Market Sizing
## Growth Trends
## Competitive Numbers
## Data Quality (High/Medium/Low)
## What's Missing"""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output

def run_writer(state: OrchestratorState, placeholder) -> str:
    query   = state["query"]
    outputs = state.get("agent_outputs", {})
    fmt     = st.session_state.get("plan",{}).get("output_format","research_report")

    all_research = "\n\n".join(
        f"=== {k.upper()} ===\n{v}" for k, v in outputs.items()
    )

    if fmt == "competitive_report":
        structure = """
## Executive Summary (name top players, state if data unavailable)
## Competitive Map
| Player | Category | Strength | Weakness | Price |
Include ALL. Use "not found" for missing.
## Top Players Deep Dive
## Market Share (only if found)
## Whitespace and Opportunities
## Risks
## PM Recommendations (5 specific, evidence-based)
## Sources"""
    elif fmt == "comparison_table":
        structure = """
## Verdict (only declare winner if evidence supports it)
## Comparison Table (10+ dimensions)
## When to Choose Option A (3 scenarios)
## When to Choose Option B (3 scenarios)
## Data Gaps
## PM Decision Framework"""
    else:
        structure = """
## Executive Summary
## Key Findings (evidence-based)
## Analysis
## Data and Evidence
## What Was Not Found
## Recommendations
## Sources"""

    prompt = f"""Senior analyst writing final report.
Query: {query}
Research: {all_research}

{structure}

Rules: Name every company. Include all numbers.
"not found in sources" for missing data.
No winners without evidence. 500+ words."""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output

def run_critic(state: OrchestratorState, placeholder) -> str:
    query   = state["query"]
    report  = state.get("agent_outputs",{}).get("writer","No report.")
    fb_used = state.get("feedback_loop_used", False)

    prompt = f"""Senior editor reviewing this report.
Query: {query}
Report: {report}

## What Works Well (3 strengths)
## Critical Gaps (as searchable queries)
## Hallucination Check
## Improved Executive Summary
## Gap Search Queries
{'2 specific queries to fill gaps.' if not fb_used else 'Feedback loop used.'}
## Final Verdict
Score X/10. Ready to share? What one change would help most?"""

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)

    gaps = []
    if not fb_used:
        in_gaps = False
        for line in output.split("\n"):
            if "Gap Search Queries" in line:
                in_gaps = True
                continue
            if in_gaps and line.strip().startswith(("-","*","1.","2.")):
                g = line.strip().lstrip("-*0123456789. ")
                if len(g) > 5:
                    gaps.append(g)
            elif in_gaps and line.startswith("##"):
                break

    score = extract_critic_score(output)
    state["gaps"]         = gaps[:2]
    state["critic_score"] = score
    st.session_state.state = state
    return output

# ---------------------------
# Agent registry
# ---------------------------

AGENT_FUNCTIONS = {
    "web_researcher": run_web_researcher,
    "gap_researcher": run_gap_researcher,
    "data_analyst":   run_data_analyst,
    "writer":         run_writer,
    "critic":         run_critic,
}
AGENT_LABELS = {
    "web_researcher":"Web Researcher","gap_researcher":"Gap Researcher",
    "data_analyst":"Data Analyst","writer":"Writer","critic":"Critic",
}
AGENT_ICONS = {
    "web_researcher":"🔍","gap_researcher":"🔎",
    "data_analyst":"📊","writer":"✍️","critic":"🔬",
}
AGENT_TYPE = {
    "web_researcher":"read","gap_researcher":"read",
    "data_analyst":"read","writer":"write","critic":"write",
}

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("AI Research Agent")
st.caption("One system · Any query · Chroma persistence · Metrics logged · Day 24")

with st.sidebar:
    st.header("Pipeline State")
    state = st.session_state.get("state", {})
    if state:
        plan = st.session_state.get("plan", {})
        st.markdown("**Query**")
        st.code(state.get("query","—")[:50])
        st.markdown("**Pipeline**")
        pt = state.get("pipeline_type","—")
        st.info(f"{'🎯 Lifestyle' if pt=='lifestyle' else '🔬 Research'}")
        if pt == "research":
            st.markdown("**Intent**")
            st.code(plan.get("intent","—"))
            st.markdown("**Format**")
            st.code(plan.get("output_format","—"))

        # Cache status
        cs = state.get("cache_status","")
        if cs:
            st.markdown("**Cache**")
            if cs == "hit":
                st.success("✓ Cache hit — web search skipped")
            elif cs == "context":
                st.info("~ Context retrieved — hybrid search")
            else:
                st.warning("○ Cache miss — fresh search")

        st.markdown("**Entity valid**")
        if state.get("entity_valid", True):
            st.success("Validated")
        else:
            st.error("Unverified")

        if state.get("critic_score",-1) > 0:
            st.markdown("**Critic score**")
            score = state.get("critic_score")
            fn = st.success if score >= 7 else (st.warning if score >= 5 else st.error)
            fn(f"{score}/10")

        st.markdown("**Completed agents**")
        for a in state.get("completed_agents",[]):
            t = AGENT_TYPE.get(a,"read")
            (st.success if t=="write" else st.info)(
                f"{AGENT_ICONS.get(a,'')} {AGENT_LABELS.get(a,a)}"
            )
    else:
        st.info("Run a query to see state")

    st.divider()

    perf = st.session_state.get("performance", {})
    if perf:
        st.header("Performance")
        for agent, m in perf.items():
            st.markdown(
                f"**{AGENT_ICONS.get(agent,'')} {AGENT_LABELS.get(agent,agent)}**  "
                f"`{m.get('latency',0):.1f}s` · `{m.get('tokens',0)} tok`"
            )
        tl = sum(m.get("latency",0) for m in perf.values())
        tt = sum(m.get("tokens",0) for m in perf.values())
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Total", f"{tl:.1f}s")
        col2.metric("Tokens", str(tt))

    st.divider()

    # Chroma stats
    st.header("Research Library")
    try:
        cs = get_chroma_stats()
        st.metric("Stored runs", cs.get("total_stored", 0))
        if cs.get("avg_quality_score"):
            st.metric("Avg quality", f"{cs['avg_quality_score']}/10")
        if cs.get("intents_breakdown"):
            st.caption("Topics stored:")
            for intent, count in cs["intents_breakdown"].items():
                st.caption(f"  {intent}: {count}")
    except Exception as e:
        st.caption(f"Library unavailable: {e}")

    st.divider()

    # Cumulative stats
    st.header("Cumulative Stats")
    try:
        stats = get_summary_stats()
        if stats.get("total_runs", 0) > 0:
            st.metric("Total runs", stats["total_runs"])
            if stats.get("avg_critic_score"):
                st.metric("Avg score", f"{stats['avg_critic_score']}/10")
            if stats.get("avg_latency"):
                st.metric("Avg latency", f"{stats['avg_latency']}s")
            if stats.get("approval_rate"):
                st.metric("Approval rate", f"{stats['approval_rate']}%")
        else:
            st.info("No runs yet")
    except Exception as e:
        st.caption(f"Stats unavailable: {e}")

    st.divider()
    st.success("LangSmith on")
    st.markdown("[Traces →](https://smith.langchain.com)")

# Session init
for key, default in [
    ("stage","input"),("state",{}),
    ("performance",{}),("agent_log",[]),("plan",{}),
    ("writer_approved",None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── STAGE 1: Input ───────────────────────────────────────────
if st.session_state.stage == "input":
    st.subheader("Ask anything")
    st.caption("Recipes · Travel · Research · Competitive analysis · Comparisons")

    examples = [
        "Analyse the competitive landscape for AI coding assistants in 2025",
        "What is the best recipe for butter chicken",
        "Compare LangGraph vs CrewAI for production agents",
        "Best places to visit in Rajasthan in December",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i%2].button(ex, use_container_width=True):
            st.session_state.prefill = ex
            st.rerun()

    prefill = st.session_state.get("prefill","")
    query = st.text_input("Your query", value=prefill,
                          placeholder="Ask anything...")

    if st.button("Start", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            pipeline_type = detect_pipeline_type(query)
            entity_check  = validate_entity(query)

            # Check Chroma cache
            with st.spinner("Checking research library..."):
                cache_result = check_cache(query)

            if pipeline_type == "lifestyle":
                lifestyle_intent = detect_lifestyle_intent(query)
                plan = {
                    "intent": lifestyle_intent,
                    "pipeline_type": "lifestyle",
                    "plain_english_summary": f"Finding {lifestyle_intent} info for: {query}"
                }
            else:
                with st.spinner("Classifying intent..."):
                    plan = classify_research_intent(query)
                plan["pipeline_type"] = "research"

            st.session_state.plan = plan
            st.session_state.writer_approved = None
            st.session_state.state = OrchestratorState(
                query=query,
                intent=plan.get("intent","general"),
                execution_mode=plan.get("execution_mode","sequential"),
                agent_outputs={},
                completed_agents=[],
                next_agent="",
                final_report="",
                gaps=[],
                feedback_loop_used=False,
                iteration=0,
                done=False,
                entity_valid=entity_check["valid"],
                data_available=True,
                pipeline_type=pipeline_type,
                critic_score=-1,
                cache_status=cache_result["status"],
                cache_context=cache_result.get("context","") or ""
            )
            st.session_state.performance = {}
            st.session_state.agent_log = []
            if "prefill" in st.session_state:
                del st.session_state.prefill
            st.session_state.stage = "confirm_intent"
            st.rerun()

# ── STAGE 2: Confirm intent ──────────────────────────────────
elif st.session_state.stage == "confirm_intent":
    state = st.session_state.state
    plan  = st.session_state.plan
    pt    = state.get("pipeline_type","research")
    cs    = state.get("cache_status","miss")

    st.subheader("Confirm — is this what you meant?")

    if not state.get("entity_valid"):
        st.warning("⚠️ Query subject could not be verified. Results may be unreliable.")

    # Show cache status
    if cs == "hit":
        st.success(
            "✓ Found in research library — previous research will be used. "
            "Web search skipped. Instant results."
        )
    elif cs == "context":
        st.info(
            "~ Related research found in library — will be used as context. "
            "Fresh web search will also run to supplement."
        )

    if pt == "lifestyle":
        st.info(
            f"**🔍 Web Researcher** will search then "
            f"**✍️ Writer** will produce a direct answer.\n\n"
            f"Intent: **{plan.get('intent','').replace('_',' ').title()}**"
        )
    else:
        if cs == "hit":
            st.info(
                "Pipeline: **Cache → ✍️ Writer → 🔬 Critic**\n\n"
                "Web search skipped — using stored research."
            )
        else:
            st.info(
                "Pipeline: **🔍 Web Researcher → 📊 Data Analyst (if needed) "
                "→ ✍️ Writer → 🔬 Critic**\n\n"
                f"Intent: **{plan.get('intent','').replace('_',' ').title()}** · "
                f"Format: **{plan.get('output_format','').replace('_',' ').title()}**"
            )
        st.caption(f"Understood: {plan.get('plain_english_summary','')}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes — start", type="primary", use_container_width=True):
            st.session_state.stage = (
                "lifestyle_search" if pt == "lifestyle"
                else "orchestrating"
            )
            st.rerun()
    with col2:
        if st.button("No — rephrase", use_container_width=True):
            st.session_state.stage = "input"
            st.session_state.state = {}
            st.session_state.plan  = {}
            st.rerun()

# ── LIFESTYLE: Web search ────────────────────────────────────
elif st.session_state.stage == "lifestyle_search":
    state  = st.session_state.state
    query  = state["query"]
    intent = state.get("intent","general")

    st.subheader("🔍 Web Researcher is searching...")
    placeholder = st.empty()
    start = time.time()

    search_results = web_search(query, max_results=8)
    placeholder.info(f"Found {search_results.count('[Result')} results. Synthesising...")

    prompt   = get_lifestyle_research_prompt(query, intent, search_results)
    research = ""
    for chunk in llm.stream(prompt):
        research += chunk.content
        placeholder.markdown(research)

    latency = round(time.time()-start, 1)
    state["agent_outputs"]["web_researcher"] = research
    state["completed_agents"].append("web_researcher")
    st.session_state.state = state
    st.session_state.performance["web_researcher"] = {
        "latency": latency, "tokens": len(research.split())
    }
    st.session_state.stage = "lifestyle_write"
    st.rerun()

# ── LIFESTYLE: Write checkpoint ──────────────────────────────
elif st.session_state.stage == "lifestyle_write":
    state    = st.session_state.state
    research = state["agent_outputs"].get("web_researcher","")

    st.subheader("✍️ Writer — approve to generate final answer")
    with st.expander("Research gathered", expanded=True):
        st.markdown(research)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve", type="primary", use_container_width=True):
            st.session_state.writer_approved = True
            st.session_state.stage = "lifestyle_summarise"
            st.rerun()
    with col2:
        if st.button("Use research as final", use_container_width=True):
            st.session_state.writer_approved = True
            state["final_report"] = research
            state["completed_agents"].append("writer")
            st.session_state.state = state
            st.session_state.stage = "done"
            st.rerun()

# ── LIFESTYLE: Summarise ─────────────────────────────────────
elif st.session_state.stage == "lifestyle_summarise":
    state    = st.session_state.state
    query    = state["query"]
    intent   = state.get("intent","general")
    research = state["agent_outputs"].get("web_researcher","")

    st.subheader("✍️ Writer generating answer...")
    placeholder = st.empty()
    start   = time.time()
    prompt  = get_lifestyle_summary_prompt(query, intent, research)
    summary = ""
    for chunk in llm.stream(prompt):
        summary += chunk.content
        placeholder.markdown(summary)

    latency = round(time.time()-start, 1)
    state["agent_outputs"]["writer"] = summary
    state["completed_agents"].append("writer")
    state["final_report"] = summary
    st.session_state.state = state
    st.session_state.performance["writer"] = {
        "latency": latency, "tokens": len(summary.split())
    }
    st.session_state.stage = "done"
    st.rerun()

# ── RESEARCH: Orchestrator ───────────────────────────────────
elif st.session_state.stage == "orchestrating":
    state     = st.session_state.state
    completed = state.get("completed_agents",[])
    n = len(completed)
    st.progress(min(n/4,1.0), text=f"{n} agents complete")

    decision   = decide_next_research_agent(state)
    next_agent = decision["next_agent"]

    if next_agent == "DONE":
        state["done"] = True
        st.session_state.state = state
        st.session_state.stage = "done"
    else:
        state["next_agent"] = next_agent
        state["iteration"]  = state.get("iteration",0)+1
        st.session_state.state = state
        agent_type = AGENT_TYPE.get(next_agent,"read")
        st.session_state.stage = (
            "write_checkpoint" if agent_type=="write"
            else "running_agent"
        )
        st.info(
            f"**Next:** {AGENT_ICONS.get(next_agent,'')} "
            f"{AGENT_LABELS.get(next_agent)} ({agent_type})\n\n"
            f"_{decision['reason']}_"
        )
    st.rerun()

# ── RESEARCH: Write checkpoint ───────────────────────────────
elif st.session_state.stage == "write_checkpoint":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])

    st.subheader(f"Write Operation — {AGENT_ICONS.get(next_agent,'')} {AGENT_LABELS.get(next_agent)}")
    st.warning(f"**{AGENT_LABELS.get(next_agent)}** produces content. Approve to proceed.")
    st.caption(f"Completed: {', '.join([AGENT_LABELS.get(a,a) for a in completed]) or 'none'}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Run {AGENT_LABELS.get(next_agent)}",
                     type="primary", use_container_width=True):
            st.session_state.stage = "running_agent"
            st.rerun()
    with col2:
        if st.button("Skip", use_container_width=True):
            state["completed_agents"].append(next_agent)
            state["agent_outputs"][next_agent] = "Skipped."
            st.session_state.state = state
            st.session_state.stage = "orchestrating"
            st.rerun()

# ── RESEARCH: Run agent ──────────────────────────────────────
elif st.session_state.stage == "running_agent":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])
    agent_type = AGENT_TYPE.get(next_agent,"read")
    n = len(completed)

    st.progress(min(n/4,1.0), text=f"{n}/4 complete")
    st.subheader(f"{AGENT_ICONS.get(next_agent,'')} {AGENT_LABELS.get(next_agent)} is working...")

    placeholder = st.empty()
    start    = time.time()
    agent_fn = AGENT_FUNCTIONS.get(next_agent)
    output   = agent_fn(state, placeholder) if agent_fn else "Unknown agent."
    latency  = round(time.time()-start, 1)

    state["agent_outputs"][next_agent] = output
    state["completed_agents"].append(next_agent)

    if next_agent == "gap_researcher":
        state["feedback_loop_used"] = True
        existing = state["agent_outputs"].get("web_researcher","")
        state["agent_outputs"]["web_researcher"] = (
            existing + "\n\n=== GAP FILL ===\n" + output
        )

    if next_agent in ("writer","critic"):
        state["final_report"] = output

    st.session_state.state = state
    st.session_state.performance[next_agent] = {
        "latency": latency, "tokens": len(output.split())
    }
    st.session_state.agent_log.append({
        "agent": next_agent, "output": output, "latency": latency
    })
    st.session_state.stage = (
        "review_output" if agent_type=="write"
        else "orchestrating"
    )
    st.rerun()

# ── RESEARCH: Review output ──────────────────────────────────
elif st.session_state.stage == "review_output":
    state      = st.session_state.state
    last_agent = state["completed_agents"][-1]
    last_out   = state["agent_outputs"].get(last_agent,"")
    perf       = st.session_state.performance.get(last_agent,{})
    gaps       = state.get("gaps",[])
    score      = state.get("critic_score",-1)

    st.subheader(f"Review — {AGENT_ICONS.get(last_agent,'')} {AGENT_LABELS.get(last_agent)} output")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latency", f"{perf.get('latency',0):.1f}s")
    col2.metric("Tokens",  str(perf.get("tokens","—")))
    col3.metric("Gaps",    str(len(gaps)))
    if score > 0:
        col4.metric("Score", f"{score}/10")

    if score > 0 and score < 6:
        st.error(f"⚠️ Critic scored {score}/10 — below quality threshold.")
    elif score >= 7:
        st.success(f"✓ Critic scored {score}/10")

    if gaps and not state.get("feedback_loop_used"):
        st.warning(f"Gaps: {' · '.join(gaps)} — orchestrator will trigger re-search.")

    with st.expander(
        f"{AGENT_ICONS.get(last_agent,'')} {AGENT_LABELS.get(last_agent)} output",
        expanded=True
    ):
        st.markdown(last_out)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve — continue",
                     type="primary", use_container_width=True):
            if last_agent == "writer":
                st.session_state.writer_approved = True
            st.session_state.stage = "orchestrating"
            st.rerun()
    with col2:
        if st.button("Stop — use as final", use_container_width=True):
            if last_agent == "writer":
                st.session_state.writer_approved = True
            state["final_report"] = last_out
            state["done"] = True
            st.session_state.state = state
            st.session_state.stage = "done"
            st.rerun()

# ── DONE ─────────────────────────────────────────────────────
elif st.session_state.stage == "done":
    state       = st.session_state.state
    perf        = st.session_state.performance
    agents_used = state.get("completed_agents",[])
    tl = sum(m.get("latency",0) for m in perf.values())
    tt = sum(m.get("tokens",0) for m in perf.values())

    st.subheader("Done")
    st.success(f"Query: {state['query']}")
    st.progress(1.0, text="Complete")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agents",  str(len(agents_used)))
    col2.metric("Latency", f"{tl:.1f}s")
    col3.metric("Tokens",  str(tt))
    col4.metric("Cost",    "$0.00")

    score = state.get("critic_score",-1)
    cs    = state.get("cache_status","miss")
    pt    = state.get("pipeline_type","research")

    st.info(
        f"Pipeline: `{'Lifestyle' if pt=='lifestyle' else 'Research'}` · "
        f"Cache: `{cs}` · "
        f"Score: `{score}/10` · "
        f"Feedback loop: `{'Yes' if state.get('feedback_loop_used') else 'No'}`"
    )

    # Store in Chroma
    try:
        final_report = state.get("final_report","")
        if final_report and len(final_report) > 100:
            run_id = f"run_{int(time.time())}"
            stored = store_run(
                run_id       = run_id,
                query        = state.get("query",""),
                research_output = final_report,
                intent       = state.get("intent",""),
                critic_score = score,
                pipeline_type = pt
            )
            if stored:
                st.caption(f"✓ Stored in research library")
    except Exception as e:
        st.caption(f"Library storage skipped: {e}")

    # Log to metrics
    try:
        logged_id = save_run({
            "query":              state.get("query"),
            "intent":             state.get("intent"),
            "pipeline_type":      pt,
            "entity_valid":       state.get("entity_valid", True),
            "agents_used":        agents_used,
            "total_latency":      tl,
            "total_tokens":       tt,
            "critic_score":       score,
            "writer_approved":    st.session_state.get("writer_approved"),
            "hallucination_flagged": score > 0 and score < 6,
            "feedback_loop_used": state.get("feedback_loop_used", False),
            "cache_status":       cs,
        })
        st.caption(f"Run logged: {logged_id}")
    except Exception as e:
        st.caption(f"Logging skipped: {e}")

    st.markdown("Traces → [View in LangSmith](https://smith.langchain.com)")
    st.divider()

    final = state.get("final_report","No report generated.")
    tabs  = st.tabs(
        ["Final Answer"] +
        [f"{AGENT_ICONS.get(a,'')} {AGENT_LABELS.get(a,a)}" for a in agents_used]
    )

    with tabs[0]:
        st.markdown(final)
        st.download_button(
            "Download report",
            data=final,
            file_name=f"report_{state['query'][:30].replace(' ','_')}.md",
            mime="text/markdown"
        )

    for i, agent in enumerate(agents_used):
        with tabs[i+1]:
            st.markdown(state["agent_outputs"].get(agent,"No output."))

    with st.expander("Execution log", expanded=False):
        for entry in st.session_state.get("agent_log",[]):
            st.markdown(
                f"**{AGENT_ICONS.get(entry['agent'],'')} "
                f"{AGENT_LABELS.get(entry['agent'])}** — {entry['latency']}s"
            )
            st.caption(entry["output"][:300]+"...")
            st.divider()

    if st.button("New query", type="primary"):
        for k in ["stage","state","performance","agent_log","plan"]:
            st.session_state[k] = (
                "input" if k=="stage" else [] if k=="agent_log" else {}
            )
        st.session_state.writer_approved = None
        st.rerun()
