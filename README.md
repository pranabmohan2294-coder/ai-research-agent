# PM Intel — AI Competitive Intelligence for Product Managers

**Live app: https://pm-research-agent.streamlit.app/**

> Built as a 30-day AI PM learning sprint capstone. Integrates Vector DB, RAG, multi-agent orchestration, and observability into a single production-grade product.

A production-grade multi-agent AI research pipeline that handles any query type — competitive analysis, market research, comparisons, recipes, travel — through a single entry point. Built as a 30-day AI PM learning sprint capstone.

---

## Live Architecture
```
User query
      ↓
Chroma cache check — has this been researched before?
      ↓
Query Router — lifestyle vs research intent
      ↓
Entity Validation — warns on fictional/unverifiable subjects
      ↓
Intent Confirmation — shows interpretation before any agent runs
      ↓
┌─────────────────────────────────────────────────────┐
│  LIFESTYLE PIPELINE (recipes, travel, food, how-to) │
│  🔍 Web Researcher → ✍️ Writer                       │
│  Geographic constraint enforced per query           │
└─────────────────────────────────────────────────────┘
                    OR
┌─────────────────────────────────────────────────────┐
│  RESEARCH PIPELINE (competitive, market, comparison)│
│  🔍 Web Researcher (read — auto)                    │
│  📊 Data Analyst (read — auto, optional)            │
│  ✍️ Writer (write — human approval required)         │
│  🔬 Critic (write — human approval required)        │
│  🔎 Gap Researcher (read — auto, critic-triggered)  │
└─────────────────────────────────────────────────────┘
      ↓
Results stored in Chroma (persistent research library)
      ↓
Agent handoffs logged to agent_comms_log.json
      ↓
Metrics logged to metrics_log.json
      ↓
LangSmith traces every LLM call
      ↓
Final Report + Download
```

---

## Meta System Message — Shared Agent Identity

Every agent receives a shared system message before its task prompt via `build_prompt()`. This enforces consistent behaviour across the entire pipeline.
```
[SYSTEM — applies to ALL agents]
1. Never invent facts not in your inputs
2. Cite sources for every specific claim
3. Write "not found in sources" when data is unavailable
4. Stay within the geographic and topical scope of the query
5. Another agent will review your output — accuracy over completeness
6. The final reader is a PM making real business decisions

[YOUR ROLE — per agent]
Researcher: retrieve and synthesise from search results only
Analyst: extract quantitative data — never estimate
Writer: synthesise into PM report — every claim must cite a source
Critic: quality-control — flag exact claims that lack citations
Gap Researcher: fill specific gaps identified by Critic only
```

---

## Agent Communications Logger

Every agent handoff is logged to `agent_comms_log.json` with full audit trail.
```python
{
  "run_id": "run_1775068713",
  "from_agent": "writer",
  "to_agent": "orchestrator",
  "query": "Competitive landscape for AI coding assistants",
  "input":  { "tokens": 450, "preview": "..." },
  "output": { "tokens": 820, "preview": "..." },
  "state": {
    "completed_agents": ["web_researcher", "writer"],
    "critic_score": -1,
    "gaps": [],
    "cache_status": "miss"
  }
}
```

View stats:
```bash
python3 agent_comms_logger.py
```

---

## Chroma Persistence — Research Library

Every completed run is stored in Chroma as a vector. Before web search, the system checks for similar past research.

| Zone | Distance | Age | Quality | Behaviour |
|---|---|---|---|---|
| Zone 1 — Cache hit | < 0.15 | Fresh < 14d | Score ≥ 5 | Skip web search — instant |
| Zone 2 — Hybrid | 0.15–0.40 | Any | Any | Cache as context + fresh search |
| Zone 3 — Miss | > 0.40 | Any | Any | Full pipeline — store after |

---

## Human-in-the-Loop — Read vs Write

| Agent | Type | Checkpoint | Why |
|---|---|---|---|
| Web Researcher | Read | No | Gathering info — reversible |
| Data Analyst | Read | No | Processing info — reversible |
| Gap Researcher | Read | No | Filling gaps — reversible |
| Writer | Write | Yes | Produces content — irreversible |
| Critic | Write | Yes | Modifies report — irreversible |

---

## Writer Self-Correction

The Writer runs two LLM calls per report:
```
Draft report generated
      ↓
Self-review against 5-point checklist:
1. Every claim has a source citation?
2. PM recommendations specific with evidence + action + risk?
3. Executive summary names companies and includes a number?
4. Market share cited or explicitly stated unavailable?
5. Any invented facts not in research? Remove them.
      ↓
Improved report returned to Critic
```

---

## Intent-Aware Output Formats

| Intent | Output format | Example query |
|---|---|---|
| competitive_analysis | Competitive map + player deep-dives + PM recommendations | "AI coding assistants 2025" |
| comparison | Head-to-head table + use case fit + decision framework | "LangGraph vs CrewAI" |
| market_research | Sizing + growth + opportunities + risks | "Indian fintech market" |
| general_research | Findings + analysis + evidence + recommendations | "RAG optimisation" |
| recipe | Ingredients + method + tips | "butter chicken recipe" |
| places | Geographic-constrained place list + tips | "best places in Goa" |
| food | Categorised dishes + healthy options | "what to eat in Tokyo" |
| howto | Step-by-step guide + common mistakes | "how to set up RAG" |

---

## Metrics Tracked

| Metric | Target | Tracked in |
|---|---|---|
| Critic score median | ≥ 7/10 | metrics_log.json |
| Latency p95 | < 120s | metrics_log.json |
| Approval rate | > 85% | metrics_log.json |
| Hallucination flag rate | < 10% | metrics_log.json |
| Cache hit rate | tracked | metrics_log.json |
| Agent handoff tokens | tracked | agent_comms_log.json |

View live dashboard:
```bash
python3 -m streamlit run metrics_dashboard.py
```

---

## PRD Summary

### Model card
- Primary: llama3.2 via Ollama (local, free, 3B params)
- Hallucination rate: ~20-30% on unknown entities
- Intended use: research and synthesis of publicly available information
- Not for: legal, financial, medical decisions

### Evaluation criteria
- Critic score median > 7/10
- Latency p95 < 120s
- Approval rate > 85%
- Hallucination flag rate < 10%

### Fallback design
- Search unavailable → Chroma cached research (implemented)
- Insufficient data → honest "not found" report (implemented)
- Entity not found → warning + confirmation required (implemented)
- Low quality cache → quality gate bypasses cache (implemented)

---

## Responsible AI

- EU AI Act: minimal risk tier — research assistance, no automated decisions about people
- Bias risks: geographic bias (US/EU sources), recency bias, source quality bias
- Guardrails: entity validation, human-in-the-loop on write ops, "not found in sources" enforcement, critic hallucination check
- Known gap: no automated faithfulness scoring — V2 priority
- Every downloaded report includes quality score and verification disclaimer

---

## Tradeoffs — V1 vs V2

| Dimension | Current V1 | Production V2 |
|---|---|---|
| LLM | llama3.2 local (3B) | Claude Sonnet hosted |
| Search | DuckDuckGo snippets | Tavily full-page retrieval |
| Parallelism | Sequential — Ollama queues | True concurrent with hosted LLM |
| State persistence | Streamlit session only | LangGraph checkpointer + DB |
| Error handling | None | Retry + exponential backoff |
| Entity validation | Explicit fictional indicators | Wikipedia + Crunchbase API |
| Cache expiry | 14-day fixed | Configurable per intent |
| Cost tracking | Word estimate | Helicone exact cost per run |
| Agent architecture | Single file | Split agent modules |

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Ollama llama3.2 (local, free) |
| Vector DB | Chroma (persistent, HNSW indexing) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Web search | DuckDuckGo via ddgs (free) |
| UI | Streamlit |
| Observability | LangSmith (free tier) |
| Metrics | Custom JSON logger |
| Agent comms | Custom handoff logger |
| Cost | $0.00 — fully local |

---

## How to Run
```bash
# Install dependencies
pip3 install langchain-ollama langchain-core streamlit ddgs \
             python-dotenv chromadb sentence-transformers

# Set up environment
cp .env.example .env
# Add LangSmith API key to .env

# Confirm Ollama is running
ollama run llama3.2 "say hello"

# Run main app
python3 -m streamlit run orchestrator_pipeline.py

# Run metrics dashboard
python3 -m streamlit run metrics_dashboard.py

# Check agent communications
python3 agent_comms_logger.py

# Check Chroma library stats
python3 chroma_manager.py

# Check run stats
python3 metrics_logger.py
```

---

## File Structure
```
├── orchestrator_pipeline.py  # Main app — unified entry point
├── agent_comms_logger.py     # Agent handoff audit trail
├── chroma_manager.py         # Chroma 3-zone retrieval
├── metrics_logger.py         # Run logging and stats
├── metrics_dashboard.py      # Streamlit metrics dashboard
├── app_streamlit.py          # Day 18-19 legacy
├── agent_pipeline.py         # Day 18 terminal legacy
├── requirements.txt
├── .env                      # API keys (gitignored)
├── .env.example
└── README.md
```

---

## Sprint Roadmap

| Day | Feature | Status |
|---|---|---|
| Day 18 | Sequential 2-agent pipeline + Streamlit UI | ✓ Done |
| Day 19 | LangSmith + live web search + intent detection | ✓ Done |
| Day 20 | Multi-agent orchestrator + critic feedback loop | ✓ Done |
| Day 21 | MVP 3 post-mortem + stress test | ✓ Done |
| Day 22 | AI product strategy — build vs buy, defensibility | ✓ Done |
| Day 23 | AI PRD — model card, eval criteria, fallback design | ✓ Done |
| Day 24 | Chroma persistence + 3-zone retrieval | ✓ Done |
| Day 25 | Metrics dashboard | ✓ Done |
| Day 26 | Responsible AI + red-team report | ✓ Done |
| Day 27 | UI polish + intent-aware researcher + writer self-correction | ✓ Done |
| Day 28 | Meta system message + agent comms logger | ✓ Done |
| Day 29 | Landing page + Loom demo | Next |
| Day 30 | Retrospective + portfolio packaging | Upcoming |

---

## Known Limitations

- llama3.2 hallucinates on unknown entities — entity validation partially mitigates
- DuckDuckGo snippets — market share data often missing
- No true parallelism on local Ollama setup
- No session persistence — browser close loses pipeline progress
- Feedback loop capped at 1
- Deployment requires hosted LLM — architecture is deployment-ready pending API credits

---

## Author

Pranab Mohan
AI Product Manager
github.com/pranabmohan2294-coder
pranab.mohan2294@gmail.com
