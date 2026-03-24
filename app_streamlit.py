import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ai-research-agent'

import streamlit as st
import time
from typing import TypedDict
from langchain_ollama import ChatOllama
from ddgs import DDGS

class ResearchState(TypedDict):
    topic: str
    search_results: str
    research: str
    summary: str
    human_approved: bool

llm = ChatOllama(model="llama3.2", temperature=0)

# ---------------------------
# PM keyword classifier
# ---------------------------
PM_KEYWORDS = [
    "product", "saas", "startup", "business", "strategy", "market",
    "revenue", "growth", "retention", "churn", "api", "platform",
    "ai", "ml", "llm", "rag", "agent", "software", "app", "tech",
    "engineering", "data", "analytics", "metrics", "kpi", "roadmap",
    "launch", "b2b", "b2c", "enterprise", "pricing", "competition",
    "user", "customer", "ux", "design", "feature", "sprint", "agile",
    "investment", "funding", "valuation", "ecommerce", "fintech"
]

def is_pm_relevant(topic: str) -> bool:
    return any(kw in topic.lower() for kw in PM_KEYWORDS)

# ---------------------------
# Intent classifier
# ---------------------------
def detect_intent(topic: str) -> str:
    t = topic.lower()
    if any(w in t for w in ["places to visit", "best places", "where to go",
                             "destinations", "travel to", "visit in", "tourist"]):
        return "places"
    elif any(w in t for w in ["what to eat", "food", "cuisine", "restaurant",
                               "dish", "recipe", "best food"]):
        return "food"
    elif any(w in t for w in ["buy", "shop", "purchase", "gift",
                               "product to buy", "things to get"]):
        return "shopping"
    elif any(w in t for w in ["how to", "steps to", "guide to",
                               "tutorial", "learn"]):
        return "howto"
    elif any(w in t for w in ["vs", "versus", "compare", "difference between",
                               "which is better"]):
        return "compare"
    elif any(w in t for w in ["news", "latest", "recent", "update",
                               "happening", "current"]):
        return "news"
    else:
        return "general"

def get_research_prompt(topic: str, search_results: str, intent: str) -> str:
    base = f"Topic: {topic}\n\nWeb Search Results:\n{search_results}\n\n"

    if intent == "places":
        return base + """You are a travel research agent. Extract and compile a comprehensive list of places from the search results.

Structure your response exactly as follows:

## Overview
2-3 sentences about the destination overall.

## Top Places to Visit (numbered list)
For each place provide:
**[Number]. [Place Name]**
- What it is: one sentence description
- Why visit: what makes it special
- Best time to visit: specific months or season
- Must see: one specific landmark or experience

List as many places as the search results support — aim for 20-30 if data allows.

## Best Time to Visit Overall
Which months or seasons are ideal and why.

## Quick Tips
3-5 practical travel tips for this destination.

Base everything on the search results. Do not invent places."""

    elif intent == "food":
        return base + """You are a food research agent. Extract and compile food recommendations from the search results.

Structure your response exactly as follows:

## Overview
2-3 sentences about the food culture.

## Must-Try Dishes (by category)
Group dishes into categories like Breakfast, Street Food, Main Course, Desserts, Drinks.
For each dish:
**[Dish Name]**
- What it is: one sentence
- Where to find it: region or type of place
- Why try it: what makes it special

## Healthy Options
Specifically highlight nutritious or diet-friendly choices.

## What to Avoid
Any common food safety considerations.

Base everything on the search results."""

    elif intent == "shopping":
        return base + """You are a shopping research agent. Compile buying recommendations from the search results.

Structure your response exactly as follows:

## Overview
2-3 sentences about shopping in this context.

## What to Buy (by category)
Organise by room or category (e.g. Living Room, Kitchen, Bedroom, Office, or Gift, Personal, Home).
For each item:
**[Item Name]**
- What it is: one sentence
- Why buy it: value or quality
- Price range: if available from search results
- Where to buy: online or offline

## Best Value Picks
Top 3-5 items that offer the best value.

Base everything on the search results."""

    elif intent == "howto":
        return base + """You are a how-to research agent. Compile a clear step-by-step guide from the search results.

Structure your response exactly as follows:

## What You Need
Tools, materials, or prerequisites.

## Step-by-Step Guide
Number each step clearly.
For each step:
**Step [N]: [Step Title]**
- What to do: clear instruction
- Why: reason or tip
- Common mistake: what to avoid

## Tips for Success
3-5 pro tips from the search results.

## Common Mistakes
What most people get wrong.

Base everything on the search results."""

    elif intent == "compare":
        return base + """You are a comparison research agent. Build a clear comparison from the search results.

Structure your response exactly as follows:

## Overview
What is being compared and why it matters.

## Head-to-Head Comparison
| Dimension | Option A | Option B |
|---|---|---|
Fill with at least 8 meaningful dimensions.

## Where Option A Wins
3-5 specific advantages with evidence from search results.

## Where Option B Wins
3-5 specific advantages with evidence from search results.

## Verdict
Who should choose which option and why.

Base everything on the search results."""

    elif intent == "news":
        return base + """You are a news research agent. Compile the latest developments from the search results.

Structure your response exactly as follows:

## Latest Developments (chronological, newest first)
For each development:
**[Date or timeframe] — [Headline]**
- What happened: 2-3 sentences
- Why it matters: significance
- Source: publication name

## Key Themes
What patterns or trends emerge across these developments.

## What to Watch
What to follow next in this space.

Base everything on the search results."""

    else:
        return base + """You are a research agent. Synthesise the search results into a comprehensive report.

Structure your response with these exact headers:

## Key Facts and Background
4-5 sentences of core background.

## Current State and Recent Developments
What is happening now, citing specific findings.

## Main Challenges and Considerations
Key problems, risks, or things to watch out for.

## Key Players and Stakeholders
Main companies, people, or groups involved.

## Data and Numbers
Relevant statistics from the search results.

## Sources
List URLs from the search results.

## Summary
2-3 sentence wrap-up.

Base everything on the search results. Minimum 400 words."""

def web_search(query: str, max_results: int = 8) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region='wt-wt'))
        if not results:
            return "No search results found."
        formatted = ""
        for i, r in enumerate(results, 1):
            formatted += f"[Result {i}]\n"
            formatted += f"Title: {r.get('title', 'No title')}\n"
            formatted += f"URL: {r.get('href', 'No URL')}\n"
            formatted += f"Summary: {r.get('body', 'No content')}\n\n"
        return formatted
    except Exception as e:
        return f"Search failed: {e}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("AI Research Agent")
st.caption("Sequential 2-agent pipeline · Intent-aware · Web Search + LangSmith · Day 19")

with st.sidebar:
    st.header("State Inspector")
    if "state" in st.session_state and st.session_state.state:
        s = st.session_state.state
        st.markdown("**topic**")
        st.code(s.get("topic", "—"))
        st.markdown("**intent detected**")
        st.code(st.session_state.get("intent", "—"))
        st.markdown("**search results** (chars)")
        st.code(str(len(s.get("search_results", ""))))
        st.markdown("**research** (chars)")
        st.code(str(len(s.get("research", ""))))
        st.markdown("**summary** (chars)")
        st.code(str(len(s.get("summary", ""))))
        st.markdown("**human_approved**")
        st.code(str(s.get("human_approved", False)))
        st.markdown("**PM relevant**")
        st.code(str(st.session_state.get("pm_relevant", False)))
    else:
        st.info("Run a topic to see state")

    st.divider()

    st.header("Agent Performance")
    perf = st.session_state.get("performance", {})

    if perf:
        if perf.get("search_latency"):
            st.markdown("**Web Search**")
            col1, col2 = st.columns(2)
            col1.metric("Latency", f"{perf['search_latency']:.1f}s")
            col2.metric("Results", str(perf.get("search_results_count", "—")))
            st.success("Live web data · Grounded")

        st.divider()

        if perf.get("agent1_latency"):
            st.markdown("**Agent 1 — Researcher**")
            col1, col2 = st.columns(2)
            col1.metric("Latency", f"{perf['agent1_latency']:.1f}s")
            col2.metric("Tokens", str(perf.get("agent1_tokens", "—")))

        st.divider()

        if perf.get("agent2_latency"):
            st.markdown("**Agent 2 — Summariser**")
            col1, col2 = st.columns(2)
            col1.metric("Latency", f"{perf['agent2_latency']:.1f}s")
            col2.metric("Tokens", str(perf.get("agent2_tokens", "—")))

        st.divider()

        if perf.get("agent1_latency") and perf.get("agent2_latency"):
            st.markdown("**Pipeline Total**")
            total_latency = (perf.get("search_latency", 0) +
                           perf.get("agent1_latency", 0) +
                           perf.get("agent2_latency", 0))
            total_tokens = (perf.get("agent1_tokens", 0) +
                          perf.get("agent2_tokens", 0))
            col1, col2 = st.columns(2)
            col1.metric("Total time", f"{total_latency:.1f}s")
            col2.metric("Total tokens", str(total_tokens))
            st.markdown("""
**Resources**
- Search: DuckDuckGo
- Model: llama3.2 (local)
- Cost: $0.00
            """)

        st.divider()
        st.success("LangSmith tracing on")
        st.markdown("[View traces →](https://smith.langchain.com)")
    else:
        st.info("Run a topic to see metrics")

# Session state init
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "state" not in st.session_state:
    st.session_state.state = {}
if "performance" not in st.session_state:
    st.session_state.performance = {}
if "intent" not in st.session_state:
    st.session_state.intent = "general"
if "pm_relevant" not in st.session_state:
    st.session_state.pm_relevant = False

# Intent badge colours
INTENT_COLORS = {
    "places": "🗺️ Travel",
    "food": "🍽️ Food",
    "shopping": "🛍️ Shopping",
    "howto": "📋 How-to",
    "compare": "⚖️ Compare",
    "news": "📰 News",
    "general": "🔍 Research"
}

# ── STAGE 1: Input ──────────────────────────────────────────
if st.session_state.stage == "input":
    st.subheader("Step 1 — Enter your question")
    st.caption("Ask anything — the agent detects intent and formats output accordingly")

    topic = st.text_input(
        "Question or topic",
        placeholder="e.g. Best places to visit in India · What to eat in Tokyo · RAG pipelines in 2025"
    )

    if topic:
        intent = detect_intent(topic)
        pm = is_pm_relevant(topic)
        st.info(f"Detected intent: **{INTENT_COLORS.get(intent, intent)}**  ·  PM relevant: **{'Yes' if pm else 'No'}**")

    if st.button("Start Research", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            intent = detect_intent(topic)
            pm = is_pm_relevant(topic)
            st.session_state.intent = intent
            st.session_state.pm_relevant = pm
            st.session_state.state = {
                "topic": topic,
                "search_results": "",
                "research": "",
                "summary": "",
                "human_approved": False
            }
            st.session_state.performance = {}
            st.session_state.stage = "searching"
            st.rerun()

# ── STAGE 2: Web Search ─────────────────────────────────────
elif st.session_state.stage == "searching":
    topic = st.session_state.state["topic"]
    intent = st.session_state.intent
    st.subheader("Step 2 — Web Search")
    st.info(f"Intent: **{INTENT_COLORS.get(intent)}** · Searching for: **{topic}**")

    start_time = time.time()
    with st.spinner("Searching DuckDuckGo..."):
        results = web_search(topic, max_results=8)

    latency = round(time.time() - start_time, 1)
    result_count = results.count("[Result")

    st.session_state.state["search_results"] = results
    st.session_state.performance["search_latency"] = latency
    st.session_state.performance["search_results_count"] = result_count

    st.success(f"Found {result_count} results in {latency}s")
    with st.expander("Raw search results", expanded=False):
        st.text(results)

    st.session_state.stage = "researching"
    st.rerun()

# ── STAGE 3: Agent 1 Research ───────────────────────────────
elif st.session_state.stage == "researching":
    topic = st.session_state.state["topic"]
    search_results = st.session_state.state["search_results"]
    intent = st.session_state.intent

    st.subheader("Step 3 — Agent 1: Researcher")
    st.info(f"Intent: **{INTENT_COLORS.get(intent)}** · Compiling results for: **{topic}**")

    prompt = get_research_prompt(topic, search_results, intent)

    research_placeholder = st.empty()
    full_research = ""
    token_count = 0
    start_time = time.time()

    with st.spinner("Agent 1 is working..."):
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

# ── STAGE 4: Human Checkpoint ───────────────────────────────
elif st.session_state.stage == "checkpoint":
    st.subheader("Step 4 — Human Checkpoint")
    st.success("Agent 1 finished. Review before Agent 2 summarises.")

    state = st.session_state.state
    perf = st.session_state.performance

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Intent", INTENT_COLORS.get(st.session_state.intent, "—"))
    col2.metric("Research", f"{len(state['research'])} chars")
    col3.metric("Latency", f"{perf.get('agent1_latency', 0):.1f}s")
    col4.metric("Tokens", str(perf.get("agent1_tokens", "—")))

    st.divider()

    with st.expander("Raw search results", expanded=False):
        st.text(state["search_results"])

    with st.expander("Full Research from Agent 1", expanded=True):
        st.markdown(state["research"])

    st.divider()
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

# ── STAGE 5: Agent 2 Summarise ──────────────────────────────
elif st.session_state.stage == "summarising":
    st.subheader("Step 5 — Agent 2: Summariser")
    st.info("Generating summary...")

    topic = st.session_state.state["topic"]
    pm_relevant = st.session_state.pm_relevant
    intent = st.session_state.intent

    pm_section = """
## What a PM Should Know
2-3 sentences on the product management implications of this topic.""" if pm_relevant else ""

    if intent == "places":
        summary_prompt = f"""Summarise this travel research into a quick reference card.

Research:
{st.session_state.state['research']}

## Top 5 Highlights
The 5 most compelling places from the research with one-line reasons.

## Best Time to Visit
One clear recommendation.

## Budget Snapshot
Rough cost indication if available in the research.

## One Thing Not to Miss
The single most unmissable experience.
{pm_section}"""

    elif intent == "food":
        summary_prompt = f"""Summarise this food research into a quick reference.

Research:
{st.session_state.state['research']}

## Top 5 Must-Try Items
Most essential dishes with one-line descriptions.

## Healthiest Options
Best choices for health-conscious visitors.

## One Dish That Defines This Place
The single most iconic food item.
{pm_section}"""

    elif intent == "compare":
        summary_prompt = f"""Summarise this comparison research.

Research:
{st.session_state.state['research']}

## The Verdict in One Line
A single sentence conclusion.

## Choose Option A if...
3 specific scenarios.

## Choose Option B if...
3 specific scenarios.

## The Deciding Factor
The single most important dimension.
{pm_section}"""

    else:
        summary_prompt = f"""Summarise this research concisely.

Research:
{st.session_state.state['research']}

## Key Takeaways
- Bullet 1
- Bullet 2
- Bullet 3

## Most Important Fact
One sentence.

## Open Question Worth Investigating
One specific question.
{pm_section}"""

    summary_placeholder = st.empty()
    full_summary = ""
    token_count = 0
    start_time = time.time()

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

# ── STAGE 6: Done ───────────────────────────────────────────
elif st.session_state.stage == "done":
    st.subheader("Pipeline Complete")
    state = st.session_state.state
    perf = st.session_state.performance

    total_latency = (perf.get("search_latency", 0) +
                    perf.get("agent1_latency", 0) +
                    perf.get("agent2_latency", 0))
    total_tokens = perf.get("agent1_tokens", 0) + perf.get("agent2_tokens", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total latency", f"{total_latency:.1f}s")
    col2.metric("Total tokens", str(total_tokens))
    col3.metric("Intent", INTENT_COLORS.get(st.session_state.intent, "—"))
    col4.metric("PM section", "Yes" if st.session_state.pm_relevant else "No")

    st.info("Traces → [View in LangSmith](https://smith.langchain.com)")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Summary (Agent 2)", "Full Research (Agent 1)", "Raw Search"])

    with tab1:
        st.markdown(state["summary"])

    with tab2:
        with st.expander("Full Research", expanded=False):
            st.markdown(state["research"])

    with tab3:
        with st.expander("Raw DuckDuckGo results", expanded=False):
            st.text(state["search_results"])

    st.divider()
    if st.button("Run new topic", type="primary"):
        st.session_state.stage = "input"
        st.session_state.state = {}
        st.session_state.performance = {}
        st.session_state.intent = "general"
        st.session_state.pm_relevant = False
        st.rerun()

# ── STAGE 7: Rejected ───────────────────────────────────────
elif st.session_state.stage == "rejected":
    st.error("Pipeline stopped at human checkpoint.")
    with st.expander("Rejected research", expanded=False):
        st.markdown(st.session_state.state.get("research", ""))
    if st.button("Try again", type="primary"):
        st.session_state.stage = "input"
        st.session_state.state = {}
        st.session_state.performance = {}
        st.session_state.intent = "general"
        st.session_state.pm_relevant = False
        st.rerun()
