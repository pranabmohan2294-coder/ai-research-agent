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
from openai import OpenAI as GroqClient
from ddgs import DDGS
from metrics_logger import save_run, extract_critic_score, get_summary_stats
from chroma_manager import check_cache, store_run, get_chroma_stats, is_available as chroma_available
from agent_comms_logger import log_handoff, get_comms_stats
try:
    from sheets_logger import log_run_to_sheets
    SHEETS_AVAILABLE = True
except Exception:
    SHEETS_AVAILABLE = False

# ---------------------------
# Mode detection
# ---------------------------

def is_dev_mode():
    try:
        params = st.query_params
        return params.get("mode", "") == "dev"
    except Exception:
        return False

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

NVIDIA_MODEL = 'llama-3.1-8b-instant'

def get_groq_client():
    import streamlit as st
    key = os.getenv('GROQ_API_KEY') or st.secrets.get('GROQ_API_KEY', '')
    return GroqClient(
        base_url='https://api.groq.com/openai/v1',
        api_key=key
    )

class NvidiaLLM:
    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                completion = get_groq_client().chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[{'role': 'user', 'content': text}],
                    temperature=0.2,
                    max_tokens=1200,
                    stream=False,
                    timeout=120
                )
                result = completion.choices[0].message.content or ''
                if result.strip():
                    class R:
                        content = result
                    return R()
                else:
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    class R:
                        content = ''
                    return R()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                class R:
                    content = ''
                return R()

    def stream(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                completion = get_groq_client().chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[{'role': 'user', 'content': text}],
                    temperature=0.2,
                    max_tokens=1200,
                    stream=True,
                    timeout=120
                )
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        class C:
                            content = chunk.choices[0].delta.content
                        yield C()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                class ErrChunk:
                    content = "[Connection error — please try again]"
                yield ErrChunk()
                return

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
# Entity validation
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
    return "general"

def classify_research_intent(query):
    prompt = (
        'Analyse this research query and return JSON.\n\n'
        'Query: "' + query + '"\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "intent": "competitive_analysis" | "market_research" | "job_research" | "comparison" | "general_research",\n'
        '  "execution_mode": "sequential",\n'
        '  "needs_data_analyst": false,\n'
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

def extract_key_sections(text, max_chars=4000):
    """Extract most relevant sections — drop sources and gaps to reduce token count."""
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    key_lines = []
    chars = 0
    skip_sections = [
        "## Sources",
        "## What Was NOT Found",
        "## What Was Not Found",
        "## Data Gaps",
        "## Places Excluded",
    ]
    skipping = False
    for line in lines:
        if any(s in line for s in skip_sections):
            skipping = True
        if line.startswith("##") and not any(s in line for s in skip_sections):
            skipping = False
        if not skipping:
            key_lines.append(line)
            chars += len(line)
            if chars > max_chars:
                key_lines.append("\n[Research truncated — full version in researcher tab]")
                break
    return "\n".join(key_lines)

def decide_next_agent(state):
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
        return {"next_agent": "gap_researcher", "reason": "Filling gaps identified in review."}
    if "web_researcher" not in completed:
        return {"next_agent": "web_researcher", "reason": "Searching the web."}
    if needs_data and "data_analyst" not in completed:
        return {"next_agent": "data_analyst", "reason": "Extracting data."}
    if "writer" not in completed:
        return {"next_agent": "writer", "reason": "Writing report."}
    if "critic" not in completed:
        return {"next_agent": "critic", "reason": "Reviewing quality."}
    return {"next_agent": "DONE", "reason": "Complete."}

# ---------------------------
# Public step mapping
# ---------------------------

def get_public_step(agent_name):
    if agent_name in ("web_researcher", "gap_researcher"):
        return "finding"
    elif agent_name in ("data_analyst", "writer"):
        return "generating"
    elif agent_name == "critic":
        return "optimising"
    return "finding"

def get_step_label(step):
    return {
        "finding":    "🔍 Finding",
        "generating": "✍️ Generating",
        "optimising": "🔬 Optimising"
    }.get(step, step)

def get_step_description(step):
    return {
        "finding":    "Searching the web and gathering intelligence from multiple sources",
        "generating": "Analysing data and writing your structured research report",
        "optimising": "Reviewing quality, checking facts, and filling any gaps"
    }.get(step, "")

def get_score_label(score):
    if score >= 8:
        return "Excellent"
    elif score >= 7:
        return "Strong"
    elif score >= 5:
        return "Good"
    else:
        return "Fair"

# ---------------------------
# Lifestyle prompts
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
            "4. Do NOT add places from training data\n"
            "5. If a result mentions a place OUTSIDE " + geo + " — IGNORE it\n\n"
            "Query: " + query + "\n"
            "Geographic scope: " + geo + " ONLY\n\n"
            "Search Results:\n" + results + "\n\n"
            "## Overview of " + geo + "\n"
            "## Places to Visit in " + geo + "\n"
            "ONLY places found in results AND within " + geo + "\n"
            "## Places Excluded (outside " + geo + ")\n"
            "## Best Time to Visit\n"
            "## Travel Tips\n"
            "## Sources"
        )
        return build_prompt("web_researcher", query, core)
    elif intent == "recipe":
        core = (
            "Extract ONLY recipe information from search results.\n\n"
            "Query: " + query + "\n"
            "Results:\n" + results + "\n\n"
            "## Overview\n## Ingredients\n## Method\n## Tips\n## Sources"
        )
        return build_prompt("web_researcher", query, core)
    else:
        core = (
            "Answer using ONLY search results.\n\n"
            "Query: " + query + "\n"
            "Results:\n" + results + "\n\n"
            "## Key Findings\n## Details\n## Recommendations\n## Sources"
        )
        return build_prompt("web_researcher", query, core)

def get_lifestyle_summary_prompt(query, intent, research):
    if intent == "places":
        geo = extract_geographic_scope(query)
        core = (
            "Create a clean travel reference. All places must be in: " + geo + "\n\n"
            "Research:\n" + research + "\n\n"
            "## " + geo + " — Overview\n"
            "## Top Places in " + geo + " (numbered, one-line each)\n"
            "## Best Time to Visit\n"
            "## 3 Quick Tips"
        )
        return build_prompt("writer", query, core)
    elif intent == "recipe":
        core = (
            "Create a clean recipe card.\n\n"
            "Research:\n" + research + "\n\n"
            "## The Dish\n## Ingredients\n## Method\n## Key Tip"
        )
        return build_prompt("writer", query, core)
    else:
        core = (
            "Summarise concisely.\n\nResearch:\n" + research + "\n\n"
            "## Key Takeaways\n## Most Important Point\n## What to Do Next"
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
    dev           = is_dev_mode()

    all_results = ""
    for i, q in enumerate(queries, 1):
        if dev:
            placeholder.info("Searching (" + str(i) + "/" + str(len(queries)) + "): " + q)
        else:
            placeholder.info("Searching " + str(i) + " of " + str(len(queries)) + " sources...")
        all_results += "\n=== SEARCH: " + q + " ===\n" + web_search(q, max_results=8)

    count = all_results.count("[Result")
    if dev:
        placeholder.info("Found " + str(count) + " results. Synthesising...")
    else:
        placeholder.info("Found " + str(count) + " sources. Analysing...")

    cache_section = ""
    if cache_context:
        cache_section = "\nPREVIOUSLY RESEARCHED CONTEXT:\n" + cache_context[:1500] + "\n---\n"

    entities_str = ", ".join(key_entities) if key_entities else query

    if fmt == "competitive_report":
        relevance = (
            "RELEVANCE FILTER — focus on: " + entities_str + "\n\n"
            "## Market Overview (numbers only, cite sources)\n"
            "## Key Players Found\n"
            "For each: **[Name]** — what it does · differentiator · pricing · source URL\n"
            "DO NOT invent companies.\n"
            "## Market Share (only if found)\n"
            "## Recent Developments\n"
            "## What Was NOT Found\n"
            "## Sources"
        )
    elif fmt == "comparison_table":
        parts = [e.strip() for e in entities_str.split(',')]
        opt_a = parts[0] if len(parts) > 0 else "Option A"
        opt_b = parts[1] if len(parts) > 1 else "Option B"
        relevance = (
            "Focus on: " + entities_str + "\n\n"
            "## Comparison Overview\n"
            "## Head-to-Head\n"
            "| Dimension | " + opt_a + " | " + opt_b + " |\n"
            "## What Was NOT Found\n## Sources"
        )
    else:
        relevance = (
            "## Key Findings (cited)\n"
            "## Data Found\n"
            "## What Was NOT Found\n"
            "## Sources"
        )

    core = (
        "RULES: Cite [Result N] for every fact. "
        "'not found in sources' for missing data. Skip off-topic results.\n\n"
        "Query: " + query + "\n"
        + cache_section
        + "SEARCH RESULTS:\n" + all_results + "\n\n"
        + relevance + "\nMinimum 400 words."
    )
    prompt = build_prompt("web_researcher", query, core)
    output = ""
    if dev:
        for chunk in llm.stream(prompt):
            output += chunk.content
            placeholder.markdown(output)
    else:
        placeholder.info("Analysing sources...")
        result = llm.invoke(prompt)
        output = result.content
        placeholder.success("Sources analysed — " + str(count) + " results processed")
    return output


def run_gap_researcher(state, placeholder):
    gaps = state.get("gaps", [])
    gap_queries = [state['query'] + " " + g for g in gaps[:2]]
    if is_dev_mode():
        placeholder.info("Filling " + str(len(gap_queries)) + " gaps...")
    else:
        placeholder.info("Finding additional information to strengthen the report...")
    return run_web_researcher(state, placeholder, custom_queries=gap_queries)


def run_data_analyst(state, placeholder):
    query    = state["query"]
    research = state.get("agent_outputs",{}).get("web_researcher","No research.")
    dev      = is_dev_mode()

    if not dev:
        placeholder.info("Extracting data and statistics...")

    core = (
        "Extract ALL quantitative data. Only include numbers explicitly stated.\n\n"
        "Query: " + query + "\n"
        "Research:\n" + research + "\n\n"
        "## Numbers Table\n"
        "| Metric | Value | Source | Confidence |\n"
        "If none: 'No quantitative data found.'\n\n"
        "## Market Sizing\n## Growth Trends\n## Competitive Numbers\n## Missing Data"
    )
    prompt = build_prompt("data_analyst", query, core)
    output = ""
    for chunk in llm.stream(prompt):
        output += chunk.content
        if dev:
            placeholder.markdown(output)
    if not dev:
        placeholder.success("Data extracted")
    return output


def run_writer(state, placeholder):
    query   = state["query"]
    outputs = state.get("agent_outputs", {})
    fmt     = st.session_state.get("plan",{}).get("output_format","research_report")
    dev     = is_dev_mode()

    all_research = "\n\n".join(
        "=== " + k.upper() + " ===\n" + v for k, v in outputs.items()
    )

    if fmt == "competitive_report":
        structure = (
            "## Executive Summary\n3 sentences. Top 3 players. One number with source.\n\n"
            "## Competitive Map\n"
            "| Player | Segment | Strength | Weakness | Pricing |\n"
            "ALL players. Cite every row.\n\n"
            "## Top Players Deep Dive\n"
            "For each top 3: name · differentiator [source] · pricing · recent news\n\n"
            "## Market Share\nOnly if found. Otherwise: 'Not found in sources.'\n\n"
            "## Whitespace and Opportunities\n3 gaps from research.\n\n"
            "## PM Recommendations\n"
            "5 recommendations. Each:\n"
            "**[Title]** — Evidence: [finding+source] — Action: [specific] — Risk: [consequence]\n\n"
            "## Sources"
        )
    elif fmt == "comparison_table":
        structure = (
            "## Verdict\nOnly if evidence supports it.\n\n"
            "## Comparison Table\n10+ dimensions. Cite sources.\n\n"
            "## When to Choose Option A (3 scenarios)\n"
            "## When to Choose Option B (3 scenarios)\n"
            "## Data Gaps\n## PM Decision Framework"
        )
    else:
        structure = (
            "## Executive Summary\n## Key Findings (each cited)\n"
            "## Analysis\n## Evidence\n## What Was Not Found\n"
            "## Recommendations\n## Sources"
        )

    if dev:
        placeholder.info("Writer drafting report...")
    else:
        placeholder.info("Writing your research report...")

    draft_core = (
        "CITATION RULE: Every claim must include source inline: 'X [Source: URL]'\n"
        "NO GENERIC STATEMENTS — every sentence must be specific.\n\n"
        "Query: " + query + "\nResearch:\n" + all_research + "\n\n"
        + structure + "\nMinimum 600 words."
    )
    draft_prompt = build_prompt("writer", query, draft_core)
    if dev:
        placeholder.info("Writer drafting report...")
        draft = ""
        for chunk in llm.stream(draft_prompt):
            draft += chunk.content
            placeholder.markdown(draft)
        placeholder.info("Writer self-reviewing...")
    else:
        placeholder.info("Writing your research report...")
        draft = llm.invoke(draft_prompt).content
        placeholder.info("Refining report quality...")

    check_core = (
        "Review your report:\n"
        "1. Every claim cited? Fix uncited.\n"
        "2. PM recommendations specific? Rewrite if generic.\n"
        "3. Executive summary names companies + number?\n"
        "4. Market share cited or stated unavailable?\n"
        "5. Invented facts? Remove.\n\n"
        "Draft:\n" + draft + "\n\nReturn improved report only."
    )
    check_prompt = build_prompt("writer", query, check_core)

    if dev:
        improved = ""
        for chunk in llm.stream(check_prompt):
            improved += chunk.content
            placeholder.markdown(improved)
    else:
        improved = llm.invoke(check_prompt).content
        placeholder.success("Report ready for your review")
    return improved


def run_critic(state, placeholder):
    query   = state["query"]
    report  = state.get("agent_outputs",{}).get("writer","No report.")
    fb_used = state.get("feedback_loop_used", False)
    dev     = is_dev_mode()

    if not dev:
        placeholder.info("Reviewing report quality and checking facts...")

    gap_instruction = (
        "2 specific search queries to fill gaps."
        if not fb_used else "Feedback loop already used."
    )

    core = (
        "Review this competitive intelligence report.\n\n"
        "Query: " + query + "\nReport:\n" + report + "\n\n"
        "## Query Alignment Check\n"
        "Does every piece of content answer: '" + query + "'? Flag off-topic content.\n\n"
        "## What Works Well\n3 strengths.\n\n"
        "## Critical Gaps\n3 weaknesses as searchable queries.\n\n"
        "## Hallucination Check\nFlag uncited or invented claims.\n\n"
        "## Improved Executive Summary\n\n"
        "## Gap Search Queries\n" + gap_instruction + "\n\n"
        "## Final Verdict\nScore X/10. One improvement suggestion."
    )
    prompt = build_prompt("critic", query, core)
    if dev:
        output = ""
        for chunk in llm.stream(prompt):
            output += chunk.content
            placeholder.markdown(output)
    else:
        placeholder.info("Reviewing report quality and checking facts...")
        output = llm.invoke(prompt).content
        placeholder.success("Review complete")

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

    if not dev:
        label = get_score_label(score) if score > 0 else "Complete"
        placeholder.success("Review complete — Report quality: " + label)
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
# Page config
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
.step-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.step-active {
    border-color: #2563EB;
    background: rgba(37,99,235,0.1);
}
.step-done {
    border-color: #16a34a;
    background: rgba(22,163,74,0.08);
}
.score-excellent { color: #4ade80; font-weight: 600; }
.score-strong { color: #86efac; font-weight: 600; }
.score-good { color: #fbbf24; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar — public vs dev
# ---------------------------

with st.sidebar:
    st.markdown("### PM Intel")
    st.caption("AI research for product managers")
    st.divider()

    dev = is_dev_mode()

    if dev:
        st.warning("Developer mode")
        state = st.session_state.get("state", {})
        if state:
            plan = st.session_state.get("plan", {})
            st.markdown("**Query**")
            st.code(state.get("query","—")[:45], language=None)
            pt = state.get("pipeline_type","—")
            st.info("🎯 Lifestyle" if pt=="lifestyle" else "🔬 Research")
            if pt == "research":
                st.code(plan.get("intent","—").replace("_"," "), language=None)
                st.code(plan.get("output_format","—").replace("_"," "), language=None)
            cs = state.get("cache_status","")
            if cs:
                if cs == "hit":
                    st.success("✓ Cache hit")
                elif cs == "context":
                    st.info("~ Hybrid")
                else:
                    st.warning("○ Fresh search")
            if state.get("critic_score",-1) > 0:
                score = state.get("critic_score")
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
            col1, col2 = st.columns(2)
            col1.metric("Total", str(round(tl,1)) + "s")
            col2.metric("Tokens", str(tt))
        st.divider()
        try:
            comms = get_comms_stats()
            if comms.get("total_handoffs", 0) > 0:
                st.markdown("**Agent Comms**")
                st.metric("Handoffs", comms["total_handoffs"])
                for k, v in comms.get("handoff_breakdown",{}).items():
                    st.caption(k + ": " + str(v))
        except Exception:
            pass
        st.divider()
        try:
            run_stats = get_summary_stats()
            if run_stats.get("total_runs",0) > 0:
                st.markdown("**Cumulative Stats**")
                st.metric("Total runs", run_stats["total_runs"])
                if run_stats.get("avg_critic_score"):
                    st.metric("Avg score", str(run_stats['avg_critic_score']) + "/10")
                if run_stats.get("avg_latency"):
                    st.metric("Avg latency", str(run_stats['avg_latency']) + "s")
        except Exception:
            pass
        st.divider()
        st.success("LangSmith on")
        st.markdown("[View traces →](https://smith.langchain.com)")

    else:
        # Public sidebar — 3 steps only
        current_step = st.session_state.get("public_step", None)
        steps = [
            ("finding",    "🔍 Finding",    "Gathering intelligence"),
            ("generating", "✍️ Generating",  "Writing your report"),
            ("optimising", "🔬 Optimising",  "Reviewing quality"),
        ]
        completed_steps = st.session_state.get("completed_steps", [])

        for step_id, step_label, step_desc in steps:
            if step_id in completed_steps:
                st.success(step_label + " ✓")
            elif step_id == current_step:
                st.info(step_label + " ⟳")
            else:
                st.caption("○ " + step_label)

        st.divider()
        queries_used = st.session_state.get("queries_used", 0)
        remaining = max(0, 2 - queries_used)
        if remaining > 0:
            st.metric("Queries remaining", str(remaining) + " / 2")
        else:
            st.error("Query limit reached")
            st.caption("Contact pranab.mohan2294@gmail.com for full access")

        st.divider()
        st.caption("PM Intel · AI-powered competitive research")
        st.caption("Built by Pranab Mohan")
        st.markdown("[GitHub →](https://github.com/pranabmohan2294-coder/ai-research-agent)")

# ---------------------------
# Session init
# ---------------------------

for key, default in [
    ("stage","input"),("state",{}),
    ("performance",{}),("agent_log",[]),("plan",{}),
    ("writer_approved",None),("queries_used",0),
    ("completed_steps",[]),("public_step",None),
    ("generating_output",""),("finding_output","")
]:
    if key not in st.session_state:
        st.session_state[key] = default

dev = is_dev_mode()

# ---------------------------
# STAGE: Input
# ---------------------------

if st.session_state.stage == "input":

    st.markdown("""
<div style="padding:1rem 0 0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:1.5rem;">
    <div class="product-badge">AI-Powered Research</div>
    <div class="hero-title">PM Intel</div>
    <div class="hero-sub">Competitive intelligence for product managers · Research any market in minutes</div>
</div>
""", unsafe_allow_html=True)

    if not dev:
        col1, col2, col3 = st.columns(3)
        col1.info("🔍 **Finding**\nSearches the web for real-time intelligence")
        col2.info("✍️ **Generating**\nWrites a structured PM research report")
        col3.info("🔬 **Optimising**\nReviews quality and fills gaps")
        st.divider()
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.info("🔍 **Search**\nLive web research")
        col2.info("📊 **Analyse**\nExtract key data")
        col3.info("✍️ **Synthesise**\nStructured report")
        col4.info("🔬 **Review**\nCritic scores output")
        st.divider()

    # Query limit check — public only
    if not dev:
        queries_used = st.session_state.get("queries_used", 0)
        if queries_used >= 2:
            st.error("You have used your 2 free queries for this session.")
            st.info("For full access contact **pranab.mohan2294@gmail.com**")
            st.stop()

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
        placeholder="e.g. Competitive landscape for AI tools in India 2025",
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

            progress = st.empty()
            progress.info("Step 1 of 4 — Checking research library...")
            try:
                cache_result = check_cache(query) if chroma_available() else {"status": "miss", "context": None}
            except Exception:
                cache_result = {"status": "miss", "context": None}

            if pipeline_type == "lifestyle":
                progress.info("Step 2 of 4 — Detecting query type...")
                lifestyle_intent = detect_lifestyle_intent(query)
                plan = {
                    "intent": lifestyle_intent,
                    "pipeline_type": "lifestyle",
                    "plain_english_summary": "Finding " + lifestyle_intent + " info for: " + query,
                    "key_entities": []
                }
            else:
                progress.info("Step 2 of 4 — Classifying research intent...")
                plan = classify_research_intent(query)
                plan["pipeline_type"] = "research"
            progress.info("Step 3 of 4 — Building search plan...")
            time.sleep(0.3)
            progress.info("Step 4 of 4 — Starting pipeline...")
            time.sleep(0.3)
            progress.success("✓ Ready — launching research")
            time.sleep(0.4)
            progress.empty()

            st.session_state.plan = plan
            st.session_state.writer_approved = None
            st.session_state.completed_steps = []
            st.session_state.public_step = "finding"
            st.session_state.generating_output = ""
            st.session_state.finding_output = ""
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

            if not dev:
                st.session_state.queries_used = st.session_state.get("queries_used", 0) + 1

            st.session_state.stage = (
                "lifestyle_search" if pipeline_type=="lifestyle"
                else "orchestrating"
            )
            st.rerun()

# ---------------------------
# LIFESTYLE: Search
# ---------------------------

elif st.session_state.stage == "lifestyle_search":
    state  = st.session_state.state
    query  = state["query"]
    intent = state.get("intent","general")

    if not dev:
        st.session_state.public_step = "finding"
        st.subheader("🔍 Finding")
        st.caption(get_step_description("finding"))
    else:
        geo = extract_geographic_scope(query) if intent=="places" else ""
        st.subheader("🔍 " + ("Searching in " + geo if geo else query))

    placeholder = st.empty()
    start = time.time()
    search_results = web_search(query, max_results=8)

    prompt   = get_lifestyle_research_prompt(query, intent, search_results)
    research = ""
    for chunk in llm.stream(prompt):
        research += chunk.content
        if dev:
            placeholder.markdown(research)

    if not dev:
        placeholder.success("Sources gathered")

    latency = round(time.time()-start, 1)
    state["agent_outputs"]["web_researcher"] = research
    state["completed_agents"].append("web_researcher")
    st.session_state.state = state
    st.session_state.performance["web_researcher"] = {"latency": latency, "tokens": len(research.split())}
    st.session_state.finding_output = research

    if not dev:
        completed = list(st.session_state.get("completed_steps", []))
        if "finding" not in completed:
            completed.append("finding")
        st.session_state.completed_steps = completed
        st.session_state.public_step = "generating"

    st.session_state.stage = "lifestyle_write"
    st.rerun()

# ---------------------------
# LIFESTYLE: Write checkpoint
# ---------------------------

elif st.session_state.stage == "lifestyle_write":
    state    = st.session_state.state
    research = state["agent_outputs"].get("web_researcher","")

    if not dev:
        st.subheader("✍️ Generating")
        st.caption("Your report is ready to generate. Review the research gathered below.")
        with st.expander("View sources gathered", expanded=False):
            st.markdown(research)
        st.info("We found information from multiple sources. Ready to generate your answer?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Report →", type="primary", use_container_width=True):
                st.session_state.writer_approved = True
                st.session_state.stage = "lifestyle_summarise"
                st.rerun()
        with col2:
            if st.button("Use raw research", use_container_width=True):
                st.session_state.writer_approved = True
                state["final_report"] = research
                state["completed_agents"].append("writer")
                st.session_state.state = state
                st.session_state.stage = "done"
                st.rerun()
    else:
        st.subheader("✍️ Review before generating")
        with st.expander("Research gathered", expanded=True):
            st.markdown(research)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve — generate", type="primary", use_container_width=True):
                st.session_state.writer_approved = True
                st.session_state.stage = "lifestyle_summarise"
                st.rerun()
        with col2:
            if st.button("Use as final", use_container_width=True):
                st.session_state.writer_approved = True
                state["final_report"] = research
                state["completed_agents"].append("writer")
                st.session_state.state = state
                st.session_state.stage = "done"
                st.rerun()

# ---------------------------
# LIFESTYLE: Summarise
# ---------------------------

elif st.session_state.stage == "lifestyle_summarise":
    state    = st.session_state.state
    query    = state["query"]
    intent   = state.get("intent","general")
    research = state["agent_outputs"].get("web_researcher","")

    if not dev:
        st.session_state.public_step = "generating"
        st.subheader("✍️ Generating")
        st.caption(get_step_description("generating"))

    placeholder = st.empty()
    start   = time.time()
    prompt  = get_lifestyle_summary_prompt(query, intent, research)
    summary = ""
    for chunk in llm.stream(prompt):
        summary += chunk.content
        if dev:
            placeholder.markdown(summary)

    if not dev:
        placeholder.success("Report generated")

    latency = round(time.time()-start, 1)
    state["agent_outputs"]["writer"] = summary
    state["completed_agents"].append("writer")
    state["final_report"] = summary
    st.session_state.state = state
    st.session_state.performance["writer"] = {"latency": latency, "tokens": len(summary.split())}

    if not dev:
        completed = list(st.session_state.get("completed_steps", []))
        if "generating" not in completed:
            completed.append("generating")
        st.session_state.completed_steps = completed

    st.session_state.stage = "done"
    st.rerun()

# ---------------------------
# RESEARCH: Orchestrator
# ---------------------------

elif st.session_state.stage == "orchestrating":
    state     = st.session_state.state
    completed = state.get("completed_agents",[])

    if dev:
        n = len(completed)
        st.progress(min(n/4,1.0), text=str(n) + " agents complete")

    decision   = decide_next_agent(state)
    next_agent = decision["next_agent"]

    if next_agent == "DONE":
        state["done"] = True
        st.session_state.state = state
        st.session_state.stage = "done"
    else:
        state["next_agent"] = next_agent
        state["iteration"]  = state.get("iteration",0)+1
        st.session_state.state = state
        agent_type  = AGENT_TYPE.get(next_agent,"read")
        public_step = get_public_step(next_agent)
        st.session_state.public_step = public_step

        if not dev and agent_type == "write" and next_agent == "writer":
            st.session_state.stage = "public_write_checkpoint"
        elif dev and agent_type == "write":
            st.session_state.stage = "write_checkpoint"
        else:
            st.session_state.stage = "running_agent"

        if dev:
            st.info(
                "**Next:** " + AGENT_ICONS.get(next_agent,"")
                + " " + AGENT_LABELS.get(next_agent,"")
                + " (" + agent_type + ")\n\n_" + decision["reason"] + "_"
            )
    st.rerun()

# ---------------------------
# PUBLIC: Write checkpoint
# ---------------------------

elif st.session_state.stage == "public_write_checkpoint":
    state = st.session_state.state

    st.subheader("✍️ Generating")
    st.caption("Research complete. Ready to generate your report?")

    finding_out = st.session_state.get("finding_output","")
    if finding_out:
        with st.expander("View research gathered", expanded=False):
            st.markdown(finding_out[:2000] + ("..." if len(finding_out) > 2000 else ""))

    cache_status = state.get("cache_status","miss")
    if cache_status == "hit":
        st.success("✓ Found previous research on this topic — using cached intelligence")
    else:
        sources_count = str(state.get("agent_outputs",{}).get("web_researcher","").count("[Result"))
        st.info("Research gathered from multiple sources. Click Generate to create your report.")

    col1, col2 = st.columns([2,3])
    with col1:
        if st.button("Generate Report →", type="primary", use_container_width=True):
            st.session_state.writer_approved = True
            st.session_state.stage = "running_agent"
            st.rerun()

# ---------------------------
# DEV: Write checkpoint
# ---------------------------

elif st.session_state.stage == "write_checkpoint":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])

    st.subheader("Write Operation — " + AGENT_ICONS.get(next_agent,"") + " " + AGENT_LABELS.get(next_agent,""))
    st.warning("**" + AGENT_LABELS.get(next_agent,"") + "** will produce content. Approve to proceed.")

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

# ---------------------------
# Run agent
# ---------------------------

elif st.session_state.stage == "running_agent":
    state      = st.session_state.state
    next_agent = state["next_agent"]
    completed  = state.get("completed_agents",[])
    agent_type = AGENT_TYPE.get(next_agent,"read")
    public_step = get_public_step(next_agent)

    if dev:
        n = len(completed)
        st.progress(min(n/4,1.0), text=str(n) + "/4 complete")
        st.subheader(AGENT_ICONS.get(next_agent,"") + " " + AGENT_LABELS.get(next_agent,"") + " is working...")
        if next_agent == "writer":
            st.caption("Draft → self-review → improved report")
    else:
        st.session_state.public_step = public_step
        st.subheader(get_step_label(public_step))
        st.caption(get_step_description(public_step))
        agent_display = {
            'web_researcher': 'Searching the web...',
            'gap_researcher': 'Finding additional sources...',
            'data_analyst':   'Extracting data and statistics...',
            'writer':         'Writing and refining your report...',
            'critic':         'Reviewing report quality...'
        }
        st.info(agent_display.get(next_agent, 'Working...'))

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

    if next_agent == "writer":
        state["final_report"] = output
    elif next_agent == "critic":
        state["critic_output"] = output

    if next_agent == "web_researcher":
        st.session_state.finding_output = output

    if not dev:
        completed_steps = list(st.session_state.get("completed_steps", []))
        if public_step not in completed_steps:
            completed_steps.append(public_step)
        st.session_state.completed_steps = completed_steps

    st.session_state.state = state
    st.session_state.performance[next_agent] = {"latency": latency, "tokens": len(output.split())}
    st.session_state.agent_log.append({"agent": next_agent, "output": output, "latency": latency})

    # Log handoff
    try:
        input_text = ""
        if next_agent == "data_analyst":
            input_text = state.get("agent_outputs",{}).get("web_researcher","")
        elif next_agent == "writer":
            input_text = "\n".join(state.get("agent_outputs",{}).values())
        elif next_agent == "critic":
            input_text = state.get("agent_outputs",{}).get("writer","")
        log_handoff(
            run_id="run_" + str(int(time.time())),
            from_agent=next_agent, to_agent="orchestrator",
            query=state.get("query",""), input_text=input_text, output_text=output,
            state_snapshot={
                "completed_agents": state.get("completed_agents",[]),
                "cache_status": state.get("cache_status",""),
                "critic_score": state.get("critic_score",-1),
                "gaps": state.get("gaps",[]),
                "feedback_loop_used": state.get("feedback_loop_used",False),
            }
        )
    except Exception:
        pass

    if dev and agent_type == "write":
        st.session_state.stage = "review_output"
    else:
        st.session_state.stage = "orchestrating"
    st.rerun()

# ---------------------------
# DEV: Review output
# ---------------------------

elif st.session_state.stage == "review_output":
    state      = st.session_state.state
    last_agent = state["completed_agents"][-1]
    last_out   = state["agent_outputs"].get(last_agent,"")
    perf       = st.session_state.performance.get(last_agent,{})
    gaps       = state.get("gaps",[])
    score      = state.get("critic_score",-1)

    st.subheader("Review — " + AGENT_ICONS.get(last_agent,"") + " " + AGENT_LABELS.get(last_agent,""))

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

    if score >= 7:
        st.success("✓ Quality: " + str(score) + "/10 — meets target.")
    elif score > 0 and score < 6:
        st.error("⚠️ Quality: " + str(score) + "/10 — below threshold.")

    if gaps and not state.get("feedback_loop_used"):
        st.warning("Gaps: " + " · ".join(gaps))

    with st.expander(AGENT_ICONS.get(last_agent,"") + " " + AGENT_LABELS.get(last_agent,"") + " output", expanded=True):
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

# ---------------------------
# DONE
# ---------------------------

elif st.session_state.stage == "done":
    state       = st.session_state.state
    perf        = st.session_state.performance
    agents_used = state.get("completed_agents",[])
    tl    = sum(m.get("latency",0) for m in perf.values())
    tt    = sum(m.get("tokens",0) for m in perf.values())
    score = state.get("critic_score",-1)
    cs    = state.get("cache_status","miss")
    pt    = state.get("pipeline_type","research")

    if not dev:
        # Mark optimising complete
        completed_steps = list(st.session_state.get("completed_steps", []))
        if "optimising" not in completed_steps and "critic" in agents_used:
            completed_steps.append("optimising")
        st.session_state.completed_steps = completed_steps
        st.session_state.public_step = None

        st.subheader("Your Research Report")
        if score > 0:
            label = get_score_label(score)
            color = "score-excellent" if score >= 8 else ("score-strong" if score >= 7 else "score-good")
            st.markdown(
                "Report quality: <span class='" + color + "'>" + label + "</span> · "
                "Research date: " + time.strftime("%B %d, %Y"),
                unsafe_allow_html=True
            )
        st.caption("AI-generated research · Verify critical facts before business decisions")
        st.divider()
    else:
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
            "Cache: " + cs + " · "
            "Feedback loop: " + ("Yes" if state.get("feedback_loop_used") else "No") + " · "
            "[LangSmith](https://smith.langchain.com)"
        )

    # Store in Chroma
    try:
        final_report = state.get("final_report","")
        if final_report and len(final_report) > 100 and chroma_available():
            run_id = "run_" + str(int(time.time()))
            store_run(
                run_id=run_id, query=state.get("query",""),
                research_output=final_report, intent=state.get("intent",""),
                critic_score=score, pipeline_type=pt
            )
    except Exception:
        pass

    # Log metrics — guard against duplicate logging on re-render
    if not st.session_state.get("run_logged", False):
        run_data = {
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
            "run_id":              "run_" + str(int(time.time())),
        }
        try:
            save_run(run_data)
        except Exception:
            pass
        try:
            if SHEETS_AVAILABLE:
                log_run_to_sheets(run_data)
        except Exception:
            pass
        st.session_state.run_logged = True

    final = state.get("final_report","No report generated.")

    if not dev:
        # Clean public output — just the report
        st.markdown(final)
        st.divider()
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            st.download_button(
                "Download Report",
                data=(
                    "# PM Intel Research Report\n\n"
                    "**Query:** " + state.get("query","") + "\n"
                    "**Date:** " + time.strftime("%Y-%m-%d") + "\n"
                    "**Quality:** " + (get_score_label(score) if score > 0 else "—") + "\n\n"
                    "---\n\n" + final + "\n\n---\n"
                    "*Generated by PM Intel · AI-powered competitive intelligence · "
                    "Verify critical facts before business use.*"
                ),
                file_name="pm-intel-" + state.get("query","report")[:30].replace(" ","-") + ".md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            if st.button("New Research", use_container_width=True):
                for k in ["stage","state","performance","agent_log","plan",
                          "completed_steps","public_step","generating_output","finding_output"]:
                    st.session_state[k] = (
                        "input" if k=="stage" else
                        [] if k in ["agent_log","completed_steps"] else
                        "" if k in ["generating_output","finding_output"] else
                        {} if k in ["state","performance","plan"] else
                        None
                    )
                st.session_state.writer_approved = None
                st.rerun()

        queries_used = st.session_state.get("queries_used", 0)
        remaining = max(0, 2 - queries_used)
        if remaining > 0:
            st.caption(str(remaining) + " free " + ("query" if remaining==1 else "queries") + " remaining this session")
        else:
            st.info("You have used your 2 free queries. Contact **pranab.mohan2294@gmail.com** for full access.")

    else:
        # Dev: full tabbed view
        tab_labels = ["Final Report"] + [
            AGENT_ICONS.get(a,"") + " " + AGENT_LABELS.get(a,a) for a in agents_used
        ]
        tabs = st.tabs(tab_labels)
        with tabs[0]:
            st.markdown(final)
            st.download_button(
                "Download report",
                data=(
                    "# PM Intel Report\n\n"
                    "**Query:** " + state.get("query","") + "\n"
                    "**Date:** " + time.strftime("%Y-%m-%d") + "\n"
                    "**Score:** " + str(score) + "/10\n\n---\n\n"
                    + final + "\n\n---\n*Verify facts before business use.*"
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
                    "**" + AGENT_ICONS.get(entry['agent'],"")
                    + " " + AGENT_LABELS.get(entry['agent'],"")
                    + "** — " + str(entry['latency']) + "s"
                )
                st.caption(entry["output"][:300]+"...")
                st.divider()

        st.divider()
        if st.button("New research query", type="primary"):
            for k in ["stage","state","performance","agent_log","plan",
                      "completed_steps","public_step","generating_output","finding_output"]:
                st.session_state[k] = (
                    "input" if k=="stage" else
                    [] if k in ["agent_log","completed_steps"] else
                    "" if k in ["generating_output","finding_output"] else
                    {} if k in ["state","performance","plan"] else
                    None
                )
            st.session_state.writer_approved = None
            st.rerun()
