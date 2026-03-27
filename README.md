# AI Research Agent — Multi-Agent Orchestrator Pipeline

A production-grade AI research pipeline that handles any query type — competitive
analysis, market research, comparisons, recipes, travel, how-to guides — through
a single entry point. Built as part of a 30-day AI PM learning sprint.

---

## Live Architecture
```
User query
      ↓
Query Router — detects lifestyle vs research intent
      ↓
Entity Validation — warns on fictional/unverifiable subjects
      ↓
Intent Confirmation — shows interpreted intent before any agent runs
      ↓
┌─────────────────────────────────────────────────────┐
│  LIFESTYLE PIPELINE (recipes, travel, food, how-to) │
│  🔍 Web Researcher → ✍️ Writer                       │
│  Intent-aware output formats per query type         │
└─────────────────────────────────────────────────────┘
                    OR
┌─────────────────────────────────────────────────────┐
│  RESEARCH PIPELINE (competitive, market, comparison)│
│  🔍 Web Researcher (read — auto)                    │
│  📊 Data Analyst (read — auto, optional)            │
│  ✍️ Writer (write — human approval required)         │
│  🔬 Critic (write — human approval required)        │
│  🔎 Gap Researcher (read — auto, triggered by critic)│
└─────────────────────────────────────────────────────┘
      ↓
Metrics logged to metrics_log.json
      ↓
LangSmith traces every LLM call
      ↓
Final Report + Download
```

---

## Key Design Decisions

### Human-in-the-loop — Read vs Write only

| Agent | Type | Checkpoint | Why |
|---|---|---|---|
| Web Researcher | Read | No | Gathering info — reversible |
| Data Analyst | Read | No | Processing info — reversible |
| Gap Researcher | Read | No | Filling gaps — reversible |
| Writer | Write | Yes | Produces content — irreversible |
| Critic | Write | Yes | Modifies report — irreversible |

### Intent-aware output formats

| Intent | Output format | Example query |
|---|---|---|
| competitive_analysis | Competitive map + player deep-dives + PM recommendations | "AI coding assistants landscape 2025" |
| comparison | Head-to-head table + use case fit + decision framework | "LangGraph vs CrewAI" |
| market_research | Sizing + growth + opportunities + risks | "Indian fintech market" |
| general_research | Findings + analysis + evidence + recommendations | "RAG pipeline optimisation" |
| recipe | Ingredients + method + tips | "butter chicken recipe" |
| places | Numbered place list + best time + quick tips | "best places Rajasthan" |
| food | Categorised dishes + healthy options | "what to eat in Tokyo" |
| howto | Step-by-step guide + common mistakes | "how to set up RAG" |

### Critic feedback loop
```
Critic reviews Writer output
      ↓
Identifies specific gaps (e.g. "no pricing data found")
      ↓
Gap Researcher runs targeted search to fill gaps
      ↓
Critic output updated
Maximum 1 feedback loop — prevents infinite cycles
```

---

## Metrics — What Gets Tracked

Every completed run logs to `metrics_log.json`:

| Metric | What it measures | Target |
|---|---|---|
| Critic score | Report quality 1-10 | Median ≥ 7 |
| Total latency | End-to-end pipeline time | < 120s |
| Writer approval rate | % of writer outputs approved | > 85% |
| Hallucination flag rate | % of runs with critic score < 6 | < 10% |
| Entity validation rate | % of unverifiable queries flagged | > 90% |
| Feedback loop rate | % of runs that triggered gap research | tracked |

View live cumulative stats in the sidebar while running.

---

## PRD — Key Sections

### Model card
- Primary: llama3.2 via Ollama (local, free, 3B params)
- Known limitation: hallucination rate ~20-30% on unknown entities
- Intended use: research and synthesis of publicly available information
- Not for: legal, financial, medical decisions or private individual research

### Evaluation criteria (Day 23 PRD)
- Retrieval grounding rate > 80%
- Hallucination rate < 10%
- Latency p95 < 120s
- Approval rate > 85%
- Critic score median > 7/10

### Fallback design
- Model unavailable → llama3.2 local fallback (V2)
- Search unavailable → Chroma cached research (V2)
- Insufficient data → honest "not found" report (implemented)
- Entity not found → warning + explicit confirmation required (implemented)

---

## Tradeoffs — Documented

| Dimension | Current (V1) | Production (V2) |
|---|---|---|
| LLM | llama3.2 local (3B) | Claude Sonnet hosted |
| Search | DuckDuckGo snippets | Tavily full-page retrieval |
| Parallelism | Simulated — Ollama queues | True concurrent with hosted LLM |
| State persistence | Streamlit session only | LangGraph checkpointer + DB |
| Token counting | Word estimate (~20% off) | Actual tokeniser |
| Feedback loops | Max 1 | Configurable per use case |
| Error handling | None | Retry + exponential backoff per node |
| Entity validation | Explicit fictional indicators only | Wikipedia + Crunchbase API lookup |
| Cost tracking | Word estimate | Helicone exact cost per run |

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Ollama llama3.2 (local, free) |
| Web search | DuckDuckGo via ddgs (free, no API key) |
| UI | Streamlit |
| Observability | LangSmith (free tier) |
| Metrics | Custom JSON logger (metrics_log.json) |
| State | TypedDict + Streamlit session state |
| Cost | $0.00 — fully local |

---

## How to Run
```bash
# Install dependencies
pip3 install langgraph==0.2.28 langchain-ollama==0.1.3 langchain-core==0.2.43 \
             streamlit ddgs python-dotenv

# Set up environment
cp .env.example .env
# Add your LangSmith API key to .env

# Confirm Ollama is running
ollama run llama3.2 "say hello"

# Run the multi-agent orchestrator (main app)
python3 -m streamlit run orchestrator_pipeline.py

# Check metrics stats
python3 metrics_logger.py

# Run the 2-agent lifestyle pipeline (legacy)
python3 -m streamlit run app_streamlit.py
```

---

## File Structure
```
├── orchestrator_pipeline.py  # Main app — unified entry point
├── metrics_logger.py         # Run logging and stats aggregation
├── metrics_log.json          # Auto-generated — all run data
├── app_streamlit.py          # Day 18-19 — 2-agent pipeline (legacy)
├── agent_pipeline.py         # Day 18 — terminal version (legacy)
├── requirements.txt          # Python dependencies
├── .env                      # API keys (gitignored)
├── .env.example              # Template
└── README.md
```

---

## Sprint Roadmap

| Day | Feature | Status |
|---|---|---|
| Day 18 | Sequential 2-agent pipeline + Streamlit UI | ✓ |
| Day 19 | LangSmith + live web search + intent detection | ✓ |
| Day 20 | Multi-agent orchestrator + critic feedback loop | ✓ |
| Day 21 | MVP 3 post-mortem + stress test + fixes | ✓ |
| Day 22 | AI product strategy — build vs buy, defensibility | ✓ |
| Day 23 | AI PRD — model card, eval criteria, fallback design | ✓ |
| Day 24 | Claude API swap + Chroma persistence | Next |
| Day 25 | Metrics dashboard | Upcoming |
| Day 26 | Responsible AI + red-team report | Upcoming |
| Day 27-28 | Capstone build | Upcoming |
| Day 29 | Launch — landing page + demo | Upcoming |
| Day 30 | Retrospective + portfolio packaging | Upcoming |

---

## Known Limitations

- llama3.2 hallucinates on unknown entities — entity validation partially mitigates
- DuckDuckGo returns short snippets — market share data often missing
- Ollama queues requests — no true parallelism on local setup
- No state persistence — browser close loses all pipeline progress
- Feedback loop capped at 1 — sufficient for V1, configurable in V2
- Search query quality affects output significantly — generic queries return irrelevant results

---

## Author

Pranab Mohan
AI Product Manager — 30-Day Learning Sprint
Week 4: Full-Stack AI Product Capstone
