# AI Research Agent — Multi-Agent Orchestrator Pipeline

A multi-agent research pipeline with an intelligent orchestrator, live web search,
LangSmith observability, and a Streamlit UI.
Part of a 30-day AI PM learning sprint — Week 3: AI Agents.

---

## Architecture
```
User query
      ↓
Orchestrator — classifies intent, decides agents, sequential or parallel
      ↓
Read agents run automatically (no human checkpoint):
  🔍 Web Researcher — live DuckDuckGo search
  📊 Data Analyst — extracts numbers and structured data (optional)
      ↓
Write agents require human approval:
  ✍️ Writer — synthesises into final report
      ↓
Human reviews Writer output
      ↓
  🔬 Critic — reviews report, identifies gaps
      ↓
If gaps found → 🔎 Gap Researcher fills them (max 1 feedback loop)
      ↓
Human reviews Critic output
      ↓
Final Report
```

---

## How the Orchestrator Works

The orchestrator analyses your query before any agent runs and decides:

| Decision | How |
|---|---|
| Intent | competitive_analysis / market_research / job_research / comparison / general_research |
| Execution mode | sequential (each step needs previous output) or parallel (independent sources) |
| Agents needed | which agents are required for this specific query |
| Output format | competitive_report / comparison_table / market_report / research_report |

---

## Human-in-the-Loop — Read vs Write

Human checkpoints only appear for write operations. Read operations run automatically.

| Agent | Type | Checkpoint |
|---|---|---|
| Web Researcher | Read | No — runs automatically |
| Parallel Researcher | Read | No — runs automatically |
| Gap Researcher | Read | No — runs automatically |
| Data Analyst | Read | No — runs automatically |
| Writer | Write | Yes — you approve before it runs |
| Critic | Write | Yes — you review output before continuing |

---

## Critic Feedback Loop

After the Critic runs, it extracts specific gap queries from its review.
The orchestrator triggers a targeted Gap Researcher to fill those gaps.
Maximum 1 feedback loop to prevent infinite cycles.
```
Critic identifies: "No market share data found"
      ↓
Gap Researcher searches: "AI coding assistants market share 2025"
      ↓
Writer updates report with new data
```

---

## Output Formats

The Writer and Researcher use specialised templates per intent:

**competitive_report** — Market overview, competitive map table, top 5 player deep-dives,
market share data, whitespace analysis, PM recommendations

**comparison_table** — Head-to-head table (10+ dimensions), when to choose each,
hidden considerations, PM decision framework

**market_report** — Market sizing, growth trends, key players, opportunities and risks

**research_report** — Key findings, analysis, data and evidence, recommendations

---

## Tradeoffs — Documented

| Dimension | Current | Production upgrade |
|---|---|---|
| LLM | llama3.2 local (3B) | Claude / GPT-4 hosted |
| Search | DuckDuckGo snippets | Tavily full-page retrieval |
| Parallelism | Simulated (Ollama queues) | True concurrent with hosted LLM |
| State persistence | Streamlit session only | LangGraph checkpointer + DB |
| Token counting | Word estimate (~20% off) | Actual tokeniser |
| Feedback loops | Max 1 | Configurable |
| Error handling | None | Retry + fallback per node |
| Intent detection | LLM classifier | LLM + few-shot examples |

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph 0.2.28 |
| LLM | Ollama llama3.2 (local, free) |
| Web search | DuckDuckGo via ddgs (free, no API key) |
| UI | Streamlit |
| Observability | LangSmith (free tier) |
| State | In-context TypedDict + Streamlit session |
| Cost | $0.00 — fully local |

---

## How to Run
```bash
# Install dependencies
pip3 install langgraph==0.2.28 langchain-ollama==0.1.3 langchain-core==0.2.43 streamlit ddgs python-dotenv

# Set up environment
cp .env.example .env
# Add your LangSmith API key to .env

# Confirm Ollama is running
ollama run llama3.2 "say hello"

# Run the 2-agent pipeline (Day 18-19)
python3 -m streamlit run app_streamlit.py

# Run the multi-agent orchestrator (Day 20)
python3 -m streamlit run orchestrator_pipeline.py
```

---

## File Structure
```
├── orchestrator_pipeline.py  # Day 20 — multi-agent orchestrator
├── app_streamlit.py          # Day 18-19 — sequential 2-agent pipeline
├── agent_pipeline.py         # Day 18 — original terminal version
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
├── .env.example              # Template for .env
└── README.md
```

---

## Roadmap

| Day | Feature | Status |
|---|---|---|
| Day 18 | Sequential 2-agent pipeline + Streamlit UI | ✓ Done |
| Day 19 | LangSmith + live web search + intent detection | ✓ Done |
| Day 20 | Multi-agent orchestrator + critic feedback loop | ✓ Done |
| Day 21 | MVP 3 post-mortem + stress test + submission | Next |

---

## Known Limitations

- llama3.2 hallucinates more than larger models — all outputs need review
- DuckDuckGo returns snippets not full articles — market share data often missing
- True parallel execution requires hosted LLM — Ollama queues local requests
- No state persistence — closing browser loses all pipeline progress
- Feedback loop capped at 1 to prevent runaway cost and latency

---

## Author

Pranab Mohan
AI Product Manager
