import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ai-research-agent'

import streamlit as st
import time
import json
import re
from typing import TypedDict, List, Dict
from openai import OpenAI as NvidiaClient
from ddgs import DDGS
from metrics_logger import save_run, extract_critic_score, get_summary_stats
from chroma_manager import check_cache, store_run, get_chroma_stats
from agent_comms_logger import log_handoff, get_comms_stats

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

nvidia_client = NvidiaClient(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key=os.getenv('NVIDIA_API_KEY')
)
NVIDIA_MODEL = 'meta/llama-3.1-8b-instruct'

class NvidiaLLM:
    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        completion = nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{'role': 'user', 'content': text}],
            temperature=0.2,
            max_tokens=2048,
            stream=False
        )
        class R:
            content = completion.choices[0].message.content or ''
        return R()

    def stream(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        completion = nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{'role': 'user', 'content': text}],
            temperature=0.2,
            max_tokens=2048,
            stream=True
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                class C:
                    content = chunk.choices[0].delta.content
                yield C()

llm = NvidiaLLM()

# ---------------------------
# Meta system message
# ---------------------------

META_SYSTEM = (
    "You are a specialist agent in PM Intel, an AI competitive intelligence "
    "system for product managers.\n\n"
    "Rules that apply to ALL agents:\n"
    "1. Never invent facts, companies, or data not in your inputs\n"
    "2. Cite sources for every specific claim using [Source: URL]\n"
    "3. Write 'not found in sources' when data is unavailable\n"
    "4. Stay within the geographic and topical scope of the original query\n"
    "5. Another agent will review your output — accuracy over completeness\n"
    "6. The final reader is a product manager making real business decisions\n"
)

AGENT_PERSONAS = {
    "web_researcher": (
        "Your role: retrieve and synthesise factual information from "
        "search results only. Do not broaden the query scope."
    ),
    "data_analyst": (
        "Your role: extract and structure quantitative data from research. "
        "If no numbers exist — say so explicitly. Never estimate."
    ),
    "writer": (
        "Your role: synthesise research into a structured PM report. "
        "Every claim must cite a source. The Critic reviews your output next."
    ),
    "critic": (
        "Your role: quality-control the Writer's report. "
        "You are the last agent before the human sees the output. "
        "Flag exact claims that lack citations or seem invented."
    ),
    "gap_researcher": (
        "Your role: fill specific data gaps identified by the Critic. "
        "Search only for the missing information."
    )
}

def build_prompt(agent_role, query, core_prompt):
    persona = AGENT_PERSONAS.get(agent_role, "")
    return (
        "[SYSTEM]\n" + META_SYSTEM + "\n"
        "[YOUR ROLE]\n" + persona + "\n\n"
        "[QUERY CONTEXT]\n" + query + "\n\n"
        "[TASK]\n" + core_prompt
    )

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

def validate_entity(query):
    if any(w in query.lower() for w in FICTIONAL_INDICATORS):
        return {"valid": False, "reason": "Query contains fictional indicator."}
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3, region='wt-wt'))
        if not results:
            return {"valid": False, "reason": "No web results found."}
        return {"valid": True, "reason": "Entity found"}
    except Exception as e:
        return {"valid": True, "reason": "Validation skipped: " + str(e)}

# ---------------------------
# Web search
# ---------------------------

def web_search(query, max_results=10):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region='wt-wt'))
        if not results:
            return "NO_RESULTS: No web results for: " + query
        out = ""
        for i, r in enumerate(results, 1):
            out += "[Result " + str(i) + "]\n"
            out += "Title: " + r.get('title','') + "\n"
            out += "URL: " + r.get('href','') + "\n"
            out += "Content: " + r.get('body','') + "\n\n"
        return out
    except Exception as e:
        return "SEARCH_ERROR: " + str(e)

# ---------------------------
# Geographic scope extractor
# ---------------------------

def extract_geographic_scope(query):
    place_pattern = re.findall(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', query)
    time_words = {
        "december","january","february","march","april","may","june",
        "july","august","september","october","november","winter",
        "summer","monsoon","spring","autumn","fall"
    }
    places = [p for p in place_pattern if p.lower() not in time_words]
    if places:
        return places[0]
    if " in " in query.lower():
        return query.split(" in ")[0].strip()
    return query

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

def detect_pipeline_type(query):
    q = query.lower()
    if any(w in q for w in LIFESTYLE_INTENTS):
        return "lifestyle"
    return "research"

def detect_lifestyle_intent(query):
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

def classify_research_intent(query):
    prompt = (
        'Analyse this research query and return JSON.\n\n'
        'Query: "' + query + '"\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "intent": "competitive_analysis" | "market_research" | "job_research" | "comparison" | "general_research",\n'
        '  "execution_mode": "sequential",\n'
        '  "needs_data_analyst": true | false,\n'
        '  "search_queries": ["specific query with entity names 2025", "second specific query"],\n'
        '  "output_format": "competitive_report" | "comparison_table" | "market_report" | "research_report",\n'
        '  "plain_english_summary": "one sentence what I understood",\n'
        '  "key_entities": ["company1", "product1"]\n'
        '}'
    )
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
            "search_queries": [query, query + " 2025 analysis"],
            "output_format": "research_report",
            "plain_english_summary": "Research about: " + query,
            "key_entities": []
        }

def decide_next_research_agent(state):
    completed    = state.get("completed_agents", [])
    needs_data   = st.session_state.get("plan", {}).get("needs_data_analyst", False)
    gaps         = state.get("gaps", [])
    fb_used      = state.get("feedback_loop_used", False)
    cache_status = state.get("cache_status", "miss")

    if cache_status == "hit" and "web_researcher" not in completed:
        state["completed_agents"] = list(completed) + ["web_researcher"]
        st.session_state.state = state

    completed = state.get("completed_agents", [])

    if gaps and not fb_used and "web_researcher" in completed and "writer" in completed:
        return {"next_agent": "gap_researcher",
                "reason": "Critic found gaps: " + ", ".join(gaps[:2])}
    if "web_researcher" not in completed:
        return {"next_agent": "web_researcher", "reason": "Starting with live web search."}
    if needs_data and "data_analyst" not in completed:
        return {"next_agent": "data_analyst", "reason": "Extracting numbers and structured data."}
    if "writer" not in completed:
        return {"next_agent": "writer", "reason": "Synthesising final report."}
    if "critic" not in completed:
        return {"next_agent": "critic", "reason": "Reviewing for gaps and quality."}
    return {"next_agent": "DONE", "reason": "Pipeline complete."}

# ---------------------------
# Lifestyle prompts — geographic constraint enforced
# ---------------------------

def get_lifestyle_research_prompt(query, intent, results):
    if intent == "places":
        geo = extract_geographic_scope(query)
        core = (
            "You are a travel research agent with strict geographic accuracy rules.\n\n"
            "CRITICAL RULES:\n"
            "1. The query asks about: " + geo + "\n"
            "2. ONLY list places that are INSIDE " + geo + "\n"
            "3. ONLY list places explicitly mentioned in the search results\n"
            "4. Do NOT add places from training data or general knowledge\n"
            "5. If a result mentions a place OUTSIDE " + geo + " — IGNORE it\n\n"
            "Query: " + query + "\n"
            "Geographic scope: " + geo + " ONLY\n\n"
            "Search Results:\n" + results + "\n\n"
            "## Overview of " + geo + "\n"
            "2-3 sentences based on results.\n\n"
            "## Places to Visit in " + geo + "\n"
            "ONLY places found in results AND within " + geo + ":\n"
            "**[Place Name]** — [Result X]\n"
            "- What it is\n"
            "- Why visit\n"
            "- Best time if mentioned\n\n"
            "## Places Excluded\n"
            "Any places outside " + geo + " — list and explain why excluded.\n\n"
            "## Best Time to Visit " + geo + "\n"
            "Only if in results.\n\n"
            "## Travel Tips\n"
            "Only from results.\n\n"
            "## Sources\n"
            "All URLs."
        )
        return build_prompt("web_researcher", query, core)

    elif intent == "recipe":
        core = (
            "Extract ONLY recipe information from search results. "
            "Do not add generic advice not in results.\n\n"
            "Query: " + query + "\n"
            "Search Results:\n" + results + "\n\n"
            "## Overview\n"
            "## Ingredients (with quantities from results)\n"
            "## Method — Step by Step\n"
            "## Tips (only from results)\n"
            "## Sources"
        )
        return build_prompt("web_researcher", query, core)

    elif intent == "food":
        core = (
            "Extract food recommendations from search results only.\n\n"
            "Query: " + query + "\n"
            "Results:\n" + results + "\n\n"
            "## Overview\n"
            "## Must-Try Items (from results only)\n"
            "## Tips\n"
            "## Sources"
        )
        return build_prompt("web_researcher", query, core)

    elif intent == "howto":
        core = (
            "Extract step-by-step guidance from search results only.\n\n"
            "Query: " + query + "\n"
            "Results:\n" + results + "\n\n"
            "## Prerequisites\n"
            "## Steps (numbered, from results only)\n"
            "## Tips\n"
            "## Common Mistakes\n"
            "## Sources"
        )
        return build_prompt("web_researcher", query, core)

    else:
        core = (
            "Answer using ONLY search results.\n\n"
            "Query: " + query + "\n"
            "Results:\n" + results + "\n\n"
            "## Key Findings\n"
            "## Details\n"
            "## Recommendations\n"
            "## Sources"
        )
        return build_prompt("web_researcher", query, core)


def get_lifestyle_summary_prompt(query, intent, research):
    if intent == "places":
        geo = extract_geographic_scope(query)
        core = (
            "Create a clean travel reference card. "
            "All places must be in: " + geo + "\n\n"
            "Research:\n" + research + "\n\n"
            "## " + geo + " — Quick Overview\n"
            "2 sentences.\n\n"
            "## Top Places in " + geo + "\n"
            "Numbered list. Each: name + one-line reason.\n"
            "ONLY places confirmed in " + geo + ".\n\n"
            "## Best Time to Visit\n\n"
            "## 3 Quick Tips for " + geo
        )
        return build_prompt("writer", query, core)

    elif intent == "recipe":
        core = (
            "Create a clean recipe card.\n\n"
            "Research:\n" + research + "\n\n"
            "## The Dish\n"
            "## Ingredients\n"
            "## Method (numbered steps)\n"
            "## Key Tip"
        )
        return build_prompt("writer", query, core)

    else:
        core = (
            "Summarise concisely.\n\n"
            "Research:\n" + research + "\n\n"
            "## Key Takeaways (3 bullets)\n"
            "## Most Important Point\n"
            "## What to Do Next"
        )
        return build_prompt("writer", query, core)

# ---------------------------
# Research agents
# ---------------------------

def run_web_researcher(state, placeholder, custom_queries=None):
    query         = state["query"]
    fmt           = st.session_state.get("plan",{}).get("output_format","research_report")
    key_entities  = st.session_state.get("plan",{}).get("key_entities",[])
    queries       = custom_queries or st.session_state.get("plan",{}).get("search_queries",[query])
    cache_context = state.get("cache_context","")

    all_results = ""
    for i, q in enumerate(queries, 1):
        placeholder.info("Searching (" + str(i) + "/" + str(len(queries)) + "): " + q)
        all_results += "\n=== SEARCH: " + q + " ===\n" + web_search(q, max_results=8)

    if "NO_RESULTS" in all_results and all_results.count("[Result") < 2:
        msg = "Insufficient web data found. Report may be incomplete."
        placeholder.warning(msg)
        return msg

    placeholder.info("Found " + str(all_results.count("[Result")) + " results. Synthesising...")

    cache_section = ""
    if cache_context:
        cache_section = (
            "\nPREVIOUSLY RESEARCHED CONTEXT (background only):\n"
            + cache_context[:1500]
            + "\n---\n"
        )

    entities_str = ", ".join(key_entities) if key_entities else query

    if fmt == "competitive_report":
        relevance = (
            "RELEVANCE FILTER:\n"
            "Query: " + query + "\n"
            "Focus on: " + entities_str + "\n\n"
            "## Market Overview\n"
            "Specific numbers only. If not found: 'Market size data not found in sources.'\n\n"
            "## Key Players Found in Search Results\n"
            "ONLY companies explicitly mentioned. For each:\n"
            "**[Name]** — [Result X]\n"
            "- What it does\n"
            "- Key differentiator\n"
            "- Pricing (exact or 'not found in sources')\n"
            "- Source URL\n\n"
            "Do NOT invent companies.\n\n"
            "## Market Share\n"
            "Only if explicitly in results. Otherwise: 'No market share data found.'\n\n"
            "## Recent Developments\n\n"
            "## What Was NOT Found\n\n"
            "## Sources"
        )
    elif fmt == "comparison_table":
        parts = [e.strip() for e in entities_str.split(',')]
        opt_a = parts[0] if len(parts) > 0 else "Option A"
        opt_b = parts[1] if len(parts) > 1 else "Option B"
        relevance = (
            "RELEVANCE FILTER:\n"
            "Focus exclusively on: " + entities_str + "\n\n"
            "## Comparison Overview\n\n"
            "## Head-to-Head Data\n"
            "| Dimension | " + opt_a + " | " + opt_b + " |\n"
            "|---|---|---|\n"
            "Only include dimensions with actual data. Mark missing as 'not found in sources'.\n\n"
            "## What Was NOT Found\n\n"
            "## Sources"
        )
    else:
        relevance = (
            "Focus on answering: " + query + "\n\n"
            "## Key Findings (cited by source)\n"
            "## Specific Data Found\n"
            "## What Was NOT Found\n"
            "## Sources"
        )

    core = (
        "GOLDEN RULES:\n"
        "1. Cite [Result N] for every fact\n"
        "2. Write 'not found in sources' for missing data\n"
        "3. Skip off-topic results entirely\n"
        "4. Name every specific company and tool found\n\n"
        "Query: " + query + "\n"
        + cache_section +
        "SEARCH RESULTS:\n" + all_results + "\n\n"
        + relevance + "\n\n"
        "Minimum 400 words."
    )
    prompt = build_prompt("web_researcher", query, core)

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output


def run_gap_researcher(state, placeholder):
    gaps = state.get("gaps", [])
    gap_queries = [state['query'] + " " + g for g in gaps[:2]]
    placeholder.info("Filling " + str(len(gap_queries)) + " gaps...")
    return run_web_researcher(state, placeholder, custom_queries=gap_queries)


def run_data_analyst(state, placeholder):
    query    = state["query"]
    research = state.get("agent_outputs",{}).get("web_researcher","No research.")

    core = (
        "Extract ALL quantitative data from research.\n"
        "Only include numbers explicitly stated.\n\n"
        "Query: " + query + "\n"
        "Research:\n" + research + "\n\n"
        "## Numbers Table\n"
        "| Metric | Value | Source | Confidence |\n"
        "If none: 'No quantitative data found.'\n\n"
        "## Market Sizing\n"
        "## Growth Trends\n"
        "## Competitive Numbers\n"
        "## Critical Missing Data"
    )
    prompt = build_prompt("data_analyst", query, core)

    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        placeholder.markdown(output)
    return output


def run_writer(state, placeholder):
    query   = state["query"]
    outputs = state.get("agent_outputs", {})
    fmt     = st.session_state.get("plan",{}).get("output_format","research_report")

    all_research = "\n\n".join(
        "=== " + k.upper() + " ===\n" + v for k, v in outputs.items()
    )

    if fmt == "competitive_report":
        structure = (
            "## Executive Summary\n"
            "3 sentences. Name top 3 players. One specific number with source.\n\n"
            "## Competitive Map\n"
            "| Player | Segment | Key Strength | Key Weakness | Pricing |\n"
            "ALL players from research. Cite every row.\n\n"
            "## Top Players Deep Dive\n"
            "For each top 3-4 players:\n"
            "**[Name]** — positioning\n"
            "- Differentiator [Source: URL]\n"
            "- Target customer\n"
            "- Pricing (exact or 'not found')\n"
            "- Recent news [Source: URL]\n\n"
            "## Market Share\n"
            "ONLY if found with citation. Otherwise: 'Not found in sources.'\n\n"
            "## Whitespace and Opportunities\n"
            "3 gaps grounded in research.\n\n"
            "## PM Recommendations\n"
            "Exactly 5. Each must follow:\n"
            "**[Recommendation]**\n"
            "Evidence: [finding + source]\n"
            "Action: [specific action this week]\n"
            "Risk if ignored: [consequence]\n\n"
            "## Sources"
        )
    elif fmt == "comparison_table":
        structure = (
            "## Verdict\n"
            "Only declare winner if evidence supports it.\n\n"
            "## Comparison Table\n"
            "10+ dimensions. Cite sources. 'not found' for missing.\n\n"
            "## When to Choose Option A (3 scenarios)\n"
            "## When to Choose Option B (3 scenarios)\n"
            "## Data Gaps\n"
            "## PM Decision Framework"
        )
    else:
        structure = (
            "## Executive Summary\n"
            "## Key Findings (each with source)\n"
            "## Analysis\n"
            "## Data and Evidence\n"
            "## What Was Not Found\n"
            "## Recommendations\n"
            "## Sources"
        )

    draft_core = (
        "CITATION RULE: Every claim must include its source inline.\n"
        "Format: 'X has Y users [Source: URL]'\n"
        "NO GENERIC STATEMENTS — every sentence must be specific.\n\n"
        "Query: " + query + "\n\n"
        "Research:\n" + all_research + "\n\n"
        + structure + "\n\n"
        "Minimum 600 words. Cite every claim."
    )
    draft_prompt = build_prompt("writer", query, draft_core)

    placeholder.info("Writer drafting report...")
    draft = ""
    for chunk in llm.stream(draft_prompt):
        draft += chunk.content
        placeholder.markdown(draft)

    # Self-correction step
    placeholder.info("Writer self-reviewing...")
    check_core = (
        "Review your own report against this checklist:\n"
        "1. Does every factual claim have a source citation? Fix uncited claims.\n"
        "2. Are PM recommendations specific with evidence + action + risk?\n"
        "3. Does the executive summary name companies and include a number?\n"
        "4. Is market share cited or explicitly stated as unavailable?\n"
        "5. Are there invented facts not in the research? Remove them.\n\n"
        "Your draft:\n" + draft + "\n\n"
        "Return improved report only. No preamble."
    )
    check_prompt = build_prompt("writer", query, check_core)

    improved = ""
    for chunk in llm.stream(check_prompt):
        improved += chunk.content
        placeholder.markdown(improved)

    return improved


def run_critic(state, placeholder):
    query   = state["query"]
    report  = state.get("agent_outputs",{}).get("writer","No report.")
    fb_used = state.get("feedback_loop_used", False)

    gap_instruction = (
        "2 specific search queries to fill the most critical gaps."
        if not fb_used else
        "Feedback loop already used."
    )

    core = (
        "Review this competitive intelligence report.\n\n"
        "Original query: " + query + "\n\n"
        "Report:\n" + report + "\n\n"
        "## Query Alignment Check\n"
        "Does every piece of content directly answer: '" + query + "'?\n"
        "Flag any content outside the geographic or topical scope.\n\n"
        "## What Works Well\n"
        "3 specific strengths.\n\n"
        "## Critical Gaps\n"
        "3 weaknesses. Each as a searchable query.\n\n"
        "## Hallucination Check\n"
        "Flag claims without citations or that appear invented.\n\n"
        "## Improved Executive Summary\n\n"
        "## Gap Search Queries\n"
        + gap_instruction + "\n\n"
        "## Final Verdict\n"
        "Score X/10. What single change would most improve this report?"
    )
    prompt = build_prompt("critic", query, core)

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

st.set_page_config(
    page_title="PM Intel — Competitive Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.product-badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    color: #60a5fa;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 8px;
}
.hero-title { font-size: 2rem; font-weight: 700; margin: 0; padding: 0; }
.hero-sub { color: rgba(255,255,255,0.6); font-size: 0.95rem; margin-top: 4px; }
.query-hint { font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("### PM Intel")
    st.caption("Competitive intelligence for product managers")
    st.divider()

    state = st.session_state.get("state", {})
    if state:
        plan = st.session_state.get("plan", {})
        st.markdown("**Current query**")
        st.code(state.get("query","—")[:45], language=None)
        pt = state.get("pipeline_type","—")
        st.markdown("**Pipeline**")
        st.info("🎯 Lifestyle" if pt=="lifestyle" else "🔬 Research")
        if pt == "research":
            st.markdown("**Intent**")
            st.code(plan.get("intent","—").replace("_"," "), language=None)
            st.markdown("**Format**")
            st.code(plan.get("output_format","—").replace("_"," "), language=None)
        cs = state.get("cache_status","")
        if cs:
            st.markdown("**Cache**")
            if cs == "hit":
                st.success("✓ Cache hit")
            elif cs == "context":
                st.info("~ Hybrid")
            else:
                st.warning("○ Fresh search")
        if not state.get("entity_valid", True):
            st.error("⚠️ Entity unverified")
        if state.get("critic_score",-1) > 0:
            score = state.get("critic_score")
            st.markdown("**Critic score**")
            if score >= 7:
                st.success(str(score) + "/10")
            elif score >= 5:
                st.warning(str(score) + "/10")
            else:
                st.error(str(score) + "/10")
        completed = state.get("completed_agents",[])
        if completed:
            st.markdown("**Agents**")
            for a in completed:
                t = AGENT_TYPE.get(a,"read")
                fn = st.success if t=="write" else st.info
                fn(AGENT_ICONS.get(a,"") + " " + AGENT_LABELS.get(a,a))
    else:
        st.info("Run a query to see pipeline state")

    st.divider()
    perf = st.session_state.get("performance", {})
    if perf:
        st.markdown("**Performance**")
        for agent, m in perf.items():
            st.markdown(
                AGENT_ICONS.get(agent,"") + " **" + AGENT_LABELS.get(agent,agent) + "**  "
                + "`" + str(round(m.get('latency',0),1)) + "s`"
                + " · `" + str(m.get('tokens',0)) + " tok`"
            )
        tl = sum(m.get("latency",0) for m in perf.values())
        tt = sum(m.get("tokens",0) for m in perf.values())
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Total", str(round(tl,1)) + "s")
        col2.metric("Tokens", str(tt))

    st.divider()

    # Agent comms stats
    st.markdown("**Agent Communications**")
    try:
        comms = get_comms_stats()
        if comms.get("total_handoffs", 0) > 0:
            st.metric("Total handoffs", comms["total_handoffs"])
            st.metric("Avg input tokens", comms.get("avg_input_tokens","—"))
            if comms.get("handoff_breakdown"):
                for k, v in comms["handoff_breakdown"].items():
                    st.caption(k + ": " + str(v))
        else:
            st.caption("No handoffs logged yet")
    except Exception:
        st.caption("No handoffs yet")

    st.divider()
    try:
        cs_stats  = get_chroma_stats()
        run_stats = get_summary_stats()
        st.markdown("**Research Library**")
        st.metric("Stored runs", cs_stats.get("total_stored",0))
        if run_stats.get("total_runs",0) > 0:
            st.metric("Total runs", run_stats["total_runs"])
            if run_stats.get("avg_critic_score"):
                st.metric("Avg score", str(run_stats['avg_critic_score']) + "/10")
    except Exception:
        pass

    st.divider()
    st.success("LangSmith tracing on")
    st.markdown("[View traces →](https://smith.langchain.com)")
    st.divider()
    st.caption("PM Intel · Day 27")

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

    st.markdown("""
<div style="padding:1rem 0 0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:1.5rem;">
    <div class="product-badge">AI-Powered Research</div>
    <div class="hero-title">PM Intel</div>
    <div class="hero-sub">Competitive intelligence for product managers · Research any market in minutes</div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.info("🔍 **Search**\nLive web research")
    col2.info("📊 **Analyse**\nExtract key data")
    col3.info("✍️ **Synthesise**\nStructured report")
    col4.info("🔬 **Review**\nCritic scores output")

    st.divider()
    st.markdown("#### What do you want to research?")

    examples = {
        "🏆 Competitive": [
            "Competitive landscape for AI coding assistants in 2025",
            "CRM tools for SMBs in India — competitive analysis",
        ],
        "⚖️ Compare": [
            "Compare LangGraph vs CrewAI for production agents",
            "Notion vs Linear for startup product teams",
        ],
        "📈 Market": [
            "Indian fintech market key players 2025",
            "No-code app builders market landscape 2025",
        ],
        "🌍 Lifestyle": [
            "Best places to visit in Goa in December",
            "Best butter chicken recipe",
        ]
    }

    tabs = st.tabs(list(examples.keys()))
    for tab, (category, queries) in zip(tabs, examples.items()):
        with tab:
            cols = st.columns(2)
            for i, q in enumerate(queries):
                if cols[i].button(q, use_container_width=True, key="ex_" + q[:20]):
                    st.session_state.prefill = q
                    st.rerun()

    st.markdown("")
    prefill = st.session_state.get("prefill","")
    query = st.text_input(
        "Or type your own query",
        value=prefill,
        placeholder="e.g. Competitive landscape for project management tools in India 2025",
        label_visibility="collapsed"
    )
    st.markdown(
        '<div class="query-hint">Works for competitive analysis, comparisons, market research, recipes, travel, and more</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1,4])
    with col1:
        start = st.button("Research →", type="primary", use_container_width=True)

    if start:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            pipeline_type = detect_pipeline_type(query)
            entity_check  = validate_entity(query)

            with st.spinner("Checking research library and classifying intent..."):
                cache_result = check_cache(query)
                if pipeline_type == "lifestyle":
                    lifestyle_intent = detect_lifestyle_intent(query)
                    plan = {
                        "intent": lifestyle_intent,
                        "pipeline_type": "lifestyle",
                        "plain_english_summary": "Finding " + lifestyle_intent + " info for: " + query,
                        "key_entities": []
                    }
                else:
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

# ── STAGE 2: Confirm ─────────────────────────────────────────
elif st.session_state.stage == "confirm_intent":
    state = st.session_state.state
    plan  = st.session_state.plan
    pt    = state.get("pipeline_type","research")
    cs    = state.get("cache_status","miss")

    st.subheader("Confirm query interpretation")

    if not state.get("entity_valid"):
        st.warning("⚠️ Query subject could not be verified. Results may be unreliable.")
    if cs == "hit":
        st.success("✓ Found in research library — web search skipped.")
    elif cs == "context":
        st.info("~ Related research found — used as background context.")

    if pt == "lifestyle":
        geo = extract_geographic_scope(state.get("query","")) if plan.get("intent")=="places" else ""
        geo_note = ("\n\n**Geographic scope:** " + geo + " only") if geo else ""
        st.info(
            "**Pipeline:** 🔍 Web Researcher → ✍️ Writer\n\n"
            "**Intent:** " + plan.get("intent","").replace("_"," ").title()
            + "\n\n**Understood:** " + plan.get("plain_english_summary","")
            + geo_note
        )
    else:
        pipeline_desc = (
            "Cache → ✍️ Writer → 🔬 Critic" if cs=="hit"
            else "🔍 Web Researcher → 📊 Data Analyst (if needed) → ✍️ Writer → 🔬 Critic"
        )
        st.info(
            "**Pipeline:** " + pipeline_desc + "\n\n"
            "**Intent:** " + plan.get("intent","").replace("_"," ").title()
            + " · **Format:** " + plan.get("output_format","").replace("_"," ").title()
            + "\n\n**Understood:** " + plan.get("plain_english_summary","")
        )
        if plan.get("key_entities"):
            st.caption("Key entities: " + ", ".join(plan["key_entities"]))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes — start research", type="primary", use_container_width=True):
            st.session_state.stage = "lifestyle_search" if pt=="lifestyle" else "orchestrating"
            st.rerun()
    with col2:
        if st.button("No — rephrase query", use_container_width=True):
            st.session_state.stage = "input"
            st.session_state.state = {}
            st.session_state.plan  = {}
            st.rerun()

# ── LIFESTYLE: Search ─────────────────────────────────────────
elif st.session_state.stage == "lifestyle_search":
    state  = st.session_state.state
    query  = state["query"]
    intent = state.get("intent","general")

    if intent == "places":
        geo = extract_geographic_scope(query)
        st.subheader("🔍 Searching for places in " + geo + "...")
    else:
        st.subheader("🔍 Researching: " + query)

    placeholder = st.empty()
    start = time.time()
    search_results = web_search(query, max_results=8)
    placeholder.info("Found " + str(search_results.count("[Result")) + " results. Synthesising...")

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

# ── LIFESTYLE: Write checkpoint ───────────────────────────────
elif st.session_state.stage == "lifestyle_write":
    state    = st.session_state.state
    research = state["agent_outputs"].get("web_researcher","")

    st.subheader("✍️ Review research before generating final answer")
    st.caption("Write operation — requires your approval")

    with st.expander("Research gathered", expanded=True):
        st.markdown(research)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve — generate answer", type="primary", use_container_width=True):
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

# ── LIFESTYLE: Summarise ──────────────────────────────────────
elif st.session_state.stage == "lifestyle_summarise":
    state    = st.session_state.state
    query    = state["query"]
    intent   = state.get("intent","general")
    research = state["agent_outputs"].get("web_researcher","")

    st.subheader("✍️ Generating answer...")
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

# ── RESEARCH: Orchestrator ────────────────────────────────────
elif st.session_state.stage == "orchestrating":
    state     = st.session_state.state
    completed = state.get("completed_agents",[])
    n = len(completed)
    st.progress(min(n/4,1.0), text=str(n) + " agents complete")

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
        st.session_state.stage = "write_checkpoint" if agent_type=="write" else "running_agent"
        st.info(
            "**Next:** " + AGENT_ICONS.get(next_agent,"")
            + " " + AGENT_LABELS.get(next_agent,"")
            + " (" + agent_type + ")\n\n_" + decision["reason"] + "_"
        )
    st.rerun()

# ── RESEARCH: Write checkpoint ────────────────────────────────
elif st.session_state.stage == "write_checkpoint":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])

    st.subheader("Write Operation — " + AGENT_ICONS.get(next_agent,"") + " " + AGENT_LABELS.get(next_agent,""))
    st.warning("**" + AGENT_LABELS.get(next_agent,"") + "** will produce content. Approve to proceed.")
    st.caption("Completed: " + (", ".join([AGENT_LABELS.get(a,a) for a in completed]) or "none"))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run " + AGENT_LABELS.get(next_agent,""), type="primary", use_container_width=True):
            st.session_state.stage = "running_agent"
            st.rerun()
    with col2:
        if st.button("Skip", use_container_width=True):
            state["completed_agents"].append(next_agent)
            state["agent_outputs"][next_agent] = "Skipped."
            st.session_state.state = state
            st.session_state.stage = "orchestrating"
            st.rerun()

# ── RESEARCH: Run agent ───────────────────────────────────────
elif st.session_state.stage == "running_agent":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])
    agent_type = AGENT_TYPE.get(next_agent,"read")
    n = len(completed)

    st.progress(min(n/4,1.0), text=str(n) + "/4 complete")
    st.subheader(AGENT_ICONS.get(next_agent,"") + " " + AGENT_LABELS.get(next_agent,"") + " is working...")
    if next_agent == "writer":
        st.caption("Draft → self-review → improved report")

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
        state["agent_outputs"]["web_researcher"] = existing + "\n\n=== GAP FILL ===\n" + output

    if next_agent in ("writer","critic"):
        state["final_report"] = output

    st.session_state.state = state
    st.session_state.performance[next_agent] = {
        "latency": latency, "tokens": len(output.split())
    }
    st.session_state.agent_log.append({
        "agent": next_agent, "output": output, "latency": latency
    })

    # Log agent communication
    try:
        input_text = ""
        if next_agent == "data_analyst":
            input_text = state.get("agent_outputs",{}).get("web_researcher","")
        elif next_agent == "writer":
            input_text = "\n".join(state.get("agent_outputs",{}).values())
        elif next_agent == "critic":
            input_text = state.get("agent_outputs",{}).get("writer","")
        elif next_agent == "gap_researcher":
            input_text = str(state.get("gaps",[]))

        log_handoff(
            run_id         = "run_" + str(int(time.time())),
            from_agent     = next_agent,
            to_agent       = "orchestrator",
            query          = state.get("query",""),
            input_text     = input_text,
            output_text    = output,
            state_snapshot = {
                "completed_agents":   state.get("completed_agents",[]),
                "cache_status":       state.get("cache_status",""),
                "critic_score":       state.get("critic_score",-1),
                "gaps":               state.get("gaps",[]),
                "feedback_loop_used": state.get("feedback_loop_used",False),
            }
        )
    except Exception:
        pass

    st.session_state.stage = "review_output" if agent_type=="write" else "orchestrating"
    st.rerun()

# ── RESEARCH: Review output ───────────────────────────────────
elif st.session_state.stage == "review_output":
    state      = st.session_state.state
    last_agent = state["completed_agents"][-1]
    last_out   = state["agent_outputs"].get(last_agent,"")
    perf       = st.session_state.performance.get(last_agent,{})
    gaps       = state.get("gaps",[])
    score      = state.get("critic_score",-1)

    st.subheader("Review — " + AGENT_ICONS.get(last_agent,"") + " " + AGENT_LABELS.get(last_agent,"") + " output")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latency", str(round(perf.get('latency',0),1)) + "s")
    col2.metric("Tokens",  str(perf.get("tokens","—")))
    col3.metric("Gaps",    str(len(gaps)))
    if score > 0:
        if score >= 7:
            col4.success("Score: " + str(score) + "/10")
        elif score >= 5:
            col4.warning("Score: " + str(score) + "/10")
        else:
            col4.error("Score: " + str(score) + "/10")

    if score > 0 and score < 6:
        st.error("⚠️ Critic scored " + str(score) + "/10 — below quality threshold.")
    elif score >= 7:
        st.success("✓ Report quality: " + str(score) + "/10 — meets target.")

    if gaps and not state.get("feedback_loop_used"):
        st.warning("Gaps identified: " + " · ".join(gaps) + " — orchestrator will re-search.")

    with st.expander(
        AGENT_ICONS.get(last_agent,"") + " " + AGENT_LABELS.get(last_agent,"") + " output",
        expanded=True
    ):
        st.markdown(last_out)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve — continue", type="primary", use_container_width=True):
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

# ── DONE ──────────────────────────────────────────────────────
elif st.session_state.stage == "done":
    state       = st.session_state.state
    perf        = st.session_state.performance
    agents_used = state.get("completed_agents",[])
    tl    = sum(m.get("latency",0) for m in perf.values())
    tt    = sum(m.get("tokens",0) for m in perf.values())
    score = state.get("critic_score",-1)
    cs    = state.get("cache_status","miss")
    pt    = state.get("pipeline_type","research")

    st.subheader("Research Complete")
    st.success("**" + state["query"] + "**")
    st.progress(1.0, text="All agents complete")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Agents",  str(len(agents_used)))
    col2.metric("Latency", str(round(tl,1)) + "s")
    col3.metric("Tokens",  str(tt))
    col4.metric("Cost",    "$0.00")
    if score > 0:
        col5.metric("Quality", str(score) + "/10",
                    delta="✓ target met" if score >= 7 else "below target")

    st.caption(
        "Pipeline: " + pt.title()
        + " · Cache: " + cs
        + " · Feedback loop: " + ("Yes" if state.get("feedback_loop_used") else "No")
        + " · Traces: [LangSmith](https://smith.langchain.com)"
    )

    # Store in Chroma
    try:
        final_report = state.get("final_report","")
        if final_report and len(final_report) > 100:
            run_id = "run_" + str(int(time.time()))
            stored = store_run(
                run_id=run_id,
                query=state.get("query",""),
                research_output=final_report,
                intent=state.get("intent",""),
                critic_score=score,
                pipeline_type=pt
            )
            if stored:
                st.caption("✓ Stored in research library · " + run_id)
    except Exception as e:
        st.caption("Library storage skipped: " + str(e))

    # Log metrics
    try:
        logged_id = save_run({
            "query":               state.get("query"),
            "intent":              state.get("intent"),
            "pipeline_type":       pt,
            "entity_valid":        state.get("entity_valid", True),
            "agents_used":         agents_used,
            "total_latency":       tl,
            "total_tokens":        tt,
            "critic_score":        score,
            "writer_approved":     st.session_state.get("writer_approved"),
            "hallucination_flagged": score > 0 and score < 6,
            "feedback_loop_used":  state.get("feedback_loop_used", False),
            "cache_status":        cs,
        })
        st.caption("Run logged: " + logged_id)
    except Exception as e:
        st.caption("Logging skipped: " + str(e))

    st.divider()

    final = state.get("final_report","No report generated.")
    tab_labels = ["Final Report"] + [
        AGENT_ICONS.get(a,"") + " " + AGENT_LABELS.get(a,a) for a in agents_used
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.markdown(final)
        col1, col2 = st.columns([1,4])
        with col1:
            st.download_button(
                "Download report",
                data=(
                    "# PM Intel Report\n\n"
                    "**Query:** " + state.get("query","") + "\n"
                    "**Date:** " + time.strftime("%Y-%m-%d") + "\n"
                    "**Quality score:** " + str(score) + "/10\n\n"
                    "---\n\n" + final + "\n\n---\n"
                    "*Generated by PM Intel · Verify critical facts before business use.*"
                ),
                file_name="pm-intel-" + state.get("query","report")[:30].replace(" ","-") + ".md",
                mime="text/markdown"
            )

    for i, agent in enumerate(agents_used):
        with tabs[i+1]:
            st.markdown(state["agent_outputs"].get(agent,"No output."))

    with st.expander("Execution log", expanded=False):
        for entry in st.session_state.get("agent_log",[]):
            st.markdown(
                "**" + AGENT_ICONS.get(entry['agent'],"") + " "
                + AGENT_LABELS.get(entry['agent'],"") + "** — "
                + str(entry['latency']) + "s"
            )
            st.caption(entry["output"][:300]+"...")
            st.divider()

    st.divider()
    if st.button("New research query", type="primary"):
        for k in ["stage","state","performance","agent_log","plan"]:
            st.session_state[k] = (
                "input" if k=="stage" else [] if k=="agent_log" else {}
            )
        st.session_state.writer_approved = None
        st.rerun()
