# AI Research Agent — Sequential Multi-Agent Pipeline

A sequential 2-agent research pipeline with intent detection, live web search,
LangSmith observability, and a Streamlit UI.
Part of a 30-day AI PM learning sprint — Week 3: AI Agents.

---

## Live Demo

Run locally at `http://localhost:8501` after setup.

---

## Architecture
```
User input (topic)
      ↓
Intent Detection (Python classifier — 7 intent types)
      ↓
Web Search (DuckDuckGo — live results, no API key)
      ↓
Agent 1 — Researcher
(synthesises web results into intent-aware structured report)
      ↓
Human Checkpoint
(review full research, approve or reject)
      ↓
Agent 2 — Summariser
(generates intent-aware summary, PM section if relevant)
      ↓
Final output — Summary + Research + Raw Search tabs
      ↓
LangSmith (traces every LLM call with token counts + latency)
```

---

## Intent Detection

The pipeline automatically detects query intent and formats output accordingly.
No configuration needed — works on any topic.

| Intent | Example query | Output format |
|---|---|---|
| Places | "Best places to visit in India" | Numbered list with description, best time, must-see |
| Food | "What to eat in Tokyo" | Categorised dishes by meal type |
| Shopping | "Things to buy for home" | Room-wise or category-wise list |
| How-to | "How to set up a RAG pipeline" | Numbered steps with tips |
| Compare | "LangGraph vs CrewAI" | Head-to-head table with verdict |
| News | "Latest AI news today" | Chronological developments with sources |
| General | "RAG pipelines in production" | 6-section research report |

---

## PM Section — Dynamic Inclusion

Agent 2 includes a "What a PM Should Know" section only when the topic
is relevant to product management, technology, business, AI, or startups.
Detected via Python keyword classifier — not LLM judgment.

PM section appears for: AI, RAG, product, SaaS, startup, analytics, roadmap...
PM section skipped for: travel, food, shopping, lifestyle topics...

---

## Features

- Live web search — DuckDuckGo retrieves real-time results (no API key needed)
- Intent-aware output — 7 different prompt templates, each matching the query type
- Streaming output — both agents write token by token to the screen in real time
- Human-in-the-loop checkpoint — approve or reject research before Agent 2 runs
- Collapsible sections — research expandable/collapsible throughout
- Live state inspector — sidebar shows every state field updating in real time
- Agent performance metrics — latency, token count, data source per agent
- PM keyword classifier — deterministic section inclusion, not LLM judgment
- LangSmith tracing — every LLM call traced with exact tokens and latency
- Pipeline totals — combined latency, tokens, cost ($0.00 fully local)

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph 0.2.28 |
| LLM | Ollama llama3.2 (local, free) |
| Web search | DuckDuckGo via ddgs (free, no API key) |
| UI | Streamlit |
| Observability | LangSmith (free tier) |
| Memory | In-context via TypedDict state |
| Cost | $0.00 — fully local, no paid API keys |

---

## Evaluation — Current State

| Dimension | Status |
|---|---|
| Grounding | Live web search — not training data |
| Hallucination control | Prompt instructs agent to base answers on search results only |
| Faithfulness score | Not yet measured — added in Day 20 |
| Intent accuracy | Keyword-based classifier — deterministic |
| PM relevance detection | Keyword-based classifier — deterministic |

---

## Known Limitations

- News intent returns limited results — DuckDuckGo snippets are brief
- Token counts are word-based estimates, not exact tokeniser counts
- LangSmith tracing requires valid API key in .env file
- No retry logic — if Ollama times out, pipeline fails

---

## Roadmap

| Day | Feature |
|---|---|
| Day 18 | Sequential pipeline + Streamlit UI + human-in-the-loop ✓ |
| Day 19 | LangSmith observability + live web search + intent detection ✓ |
| Day 20 | 3-agent pipeline — Planner + Researcher + Writer |
| Day 21 | MVP 3 submission — post-mortem + stress test |

---

## How to Run
```bash
# 1. Install dependencies
pip3 install langgraph==0.2.28 langchain-ollama==0.1.3 langchain-core==0.2.43 streamlit ddgs

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your LangSmith API key

# 3. Make sure Ollama is running
ollama run llama3.2 "say hello"

# 4. Run the Streamlit app
python3 -m streamlit run app_streamlit.py
```

---

## Environment Variables

Create a `.env` file in the project root:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=ai-research-agent
```

Get a free LangSmith API key at smith.langchain.com.

---

## File Structure
```
├── app_streamlit.py     # Main Streamlit app — full pipeline with UI
├── agent_pipeline.py    # Original terminal version
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed to git)
├── .gitignore           # Excludes .env
└── README.md
```

---

## Author

Pranab Mohan
AI Product Manager — 30-Day Learning Sprint
Week 3: AI Agents and Multi-Agent Orchestration
