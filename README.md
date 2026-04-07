# PM Intel — AI Competitive Intelligence for Product Managers

**Live app: https://pm-research-agent.streamlit.app/**
**Dev mode: https://pm-research-agent.streamlit.app/?mode=dev**

> Built as a 30-day AI PM learning sprint capstone. Integrates Vector DB, RAG, multi-agent orchestration, and observability into a single production-grade product.

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
Results stored in Chroma locally (persistent research library)
      ↓
Every run logged to Google Sheets (live cross-session tracking)
      ↓
Agent handoffs logged to agent_comms_log.json
      ↓
LangSmith traces every LLM call
      ↓
Final Report + Download
```

---

## Public vs Developer Mode

| Feature | Public mode | Dev mode (?mode=dev) |
|---|---|---|
| Pipeline visibility | 3 steps: Finding / Generating / Optimising | Full agent-by-agent view |
| Agent names | Hidden | Visible |
| Token counts | Hidden | Visible per agent |
| LangSmith link | Hidden | Visible |
| Google Sheets link | Hidden | Visible in sidebar |
| Write checkpoints | Automatic | Manual approval |
| Query limit | 2 per session | Unlimited |
| Execution log | Hidden | Visible |

---

## Meta System Message — Shared Agent Identity

Every agent receives a shared system message before its task prompt via `build_prompt()`:
```
Rules for ALL agents:
1. Never invent facts not in your inputs
2. Cite sources for every specific claim
3. Write "not found in sources" when data is unavailable
4. Stay within the geographic and topical scope of the query
5. Another agent will review your output — accuracy over completeness
6. The final reader is a PM making real business decisions
```

Each agent also has a role-specific persona — researcher, analyst, writer, critic, gap researcher.

---

## Agent Communications Logger

Every agent handoff logged to `agent_comms_log.json`:
```json
{
  "run_id": "run_1775068713",
  "from_agent": "writer",
  "to_agent": "orchestrator",
  "query": "Competitive landscape for AI coding assistants",
  "input":  { "tokens": 450, "preview": "..." },
  "output": { "tokens": 820, "preview": "..." },
  "state": { "completed_agents": [...], "critic_score": 7 }
}
```

---

## Google Sheets — Live Run Tracking

Every completed run on the live app logs to Google Sheets permanently:

| Column | Example |
|---|---|
| Timestamp | 2026-04-02 17:18:17 |
| Query | CRM tools for SMBs in India |
| Intent | competitive_analysis |
| Critic Score | 8 |
| Latency | 84s |
| Cache Status | miss |
| Quality Label | Excellent |

View live dashboard:
```bash
python3 -m streamlit run sheets_dashboard.py --server.port 8503
```

---

## Chroma Persistence — Local Research Library

| Zone | Distance | Behaviour |
|---|---|---|
| Zone 1 — Cache hit | < 0.15 | Skip web search — instant |
| Zone 2 — Hybrid | 0.15–0.40 | Cache as context + fresh search |
| Zone 3 — Miss | > 0.40 | Full pipeline — store after |

Age expiry: 14 days. Quality gate: critic score < 5 bypasses cache.

---

## Human-in-the-Loop

| Agent | Type | Checkpoint |
|---|---|---|
| Web Researcher | Read | No — auto |
| Data Analyst | Read | No — auto |
| Gap Researcher | Read | No — auto |
| Writer | Write | Yes — approval required |
| Critic | Write | Yes — review required |

---

## Writer Self-Correction
```
Draft report generated
      ↓
Self-review against 5-point checklist:
1. Every claim has a source citation?
2. PM recommendations specific with evidence + action + risk?
3. Executive summary names companies and includes a number?
4. Market share cited or explicitly stated unavailable?
5. Any invented facts? Remove them.
      ↓
Improved report returned to Critic
```

---

## Metrics Tracked

| Metric | Target | Source |
|---|---|---|
| Critic score median | ≥ 7/10 | Google Sheets + metrics_log.json |
| Latency p95 | < 120s | Google Sheets + metrics_log.json |
| Approval rate | > 85% | metrics_log.json |
| Hallucination flag rate | < 10% | Google Sheets |
| Cache hit rate | tracked | metrics_log.json |
| Agent handoff tokens | tracked | agent_comms_log.json |

---

## Responsible AI

- EU AI Act: minimal risk tier
- Bias risks: geographic bias, recency bias, source quality bias
- Guardrails: entity validation, human-in-the-loop on write ops, critic hallucination check
- Every downloaded report includes quality score and AI disclaimer

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq llama-3.1-8b-instant (hosted, free) |
| Vector DB | Chroma (local, HNSW indexing) |
| Embeddings | all-MiniLM-L6-v2 (local) |
| Web search | DuckDuckGo via ddgs (free) |
| UI | Streamlit |
| Observability | LangSmith (free tier) |
| Run tracking | Google Sheets (cross-session) |
| Local metrics | Custom JSON logger |
| Agent comms | Custom handoff logger |
| Cost | $0.00 — all free tier |

---

## Tradeoffs — V1 vs V2

| Dimension | Current V1 | Production V2 |
|---|---|---|
| LLM | Groq llama-3.1-8b (free) | Claude Sonnet hosted |
| Search | DuckDuckGo snippets | Tavily full-page retrieval |
| State persistence | Streamlit session only | LangGraph checkpointer + DB |
| Error handling | Retry on empty response | Full retry + exponential backoff |
| Entity validation | Explicit fictional indicators | Wikipedia + Crunchbase API |
| Cache | Local Chroma only | Cloud Chroma with persistent disk |
| Cost tracking | Word estimate | Helicone exact cost per run |

---

## How to Run
```bash
# Install dependencies
pip3 install streamlit ddgs python-dotenv openai groq langsmith \
             gspread google-auth langchain-core==0.2.43

# Set up environment
cp .env.example .env
# Add GROQ_API_KEY and LANGCHAIN_API_KEY to .env

# Run main app
python3 -m streamlit run orchestrator_pipeline.py

# Run Google Sheets live dashboard
python3 -m streamlit run sheets_dashboard.py --server.port 8503

# Run local metrics dashboard
python3 -m streamlit run metrics_dashboard.py --server.port 8502

# Check agent communications
python3 agent_comms_logger.py
```

---

## File Structure
```
├── orchestrator_pipeline.py      # Main app — unified entry point
├── orchestrator_pipeline_dev.py  # Dev backup — full version
├── sheets_dashboard.py           # Google Sheets live dashboard
├── metrics_dashboard.py          # Local metrics dashboard
├── sheets_logger.py              # Google Sheets run logger
├── agent_comms_logger.py         # Agent handoff audit trail
├── chroma_manager.py             # Chroma stub (cloud) / full (local)
├── metrics_logger.py             # Local run logging
├── app_streamlit.py              # Day 18-19 legacy
├── agent_pipeline.py             # Day 18 terminal legacy
├── requirements.txt              # Cloud dependencies
├── .env.example                  # API key template
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
| Day 29 | Public UI + Groq LLM + Streamlit Cloud deployment | ✓ Done |
| Day 30 | Retrospective + portfolio packaging | Tomorrow |

---

## Known Limitations

- Groq free tier: 14,400 requests/day — sufficient for portfolio demo
- DuckDuckGo snippets — market share data often missing
- No session persistence — browser close loses pipeline progress
- Chroma only works locally — cloud runs have no cache
- Feedback loop capped at 1

---

## Author

Pranab Mohan
AI Product Manager
pranab.mohan2294@gmail.com

---

## Recent Updates — April 2026

### Search Upgrade — DuckDuckGo → Tavily
- Full article text per result (1900-2200 chars vs 150-200)
- search_depth=advanced extracts complete article body
- DuckDuckGo fallback if Tavily unavailable
- Dramatically improves technical topic research quality

### Smart Entity-Aware Classifier
- Added lookup_entity_context() — quick Tavily lookup before classification
- Two-step classification: entity lookup → informed query generation
- Queries generated from actual web content not model training assumptions
- Correctly identifies new/unknown entities (e.g. TurboQuant as AI compression)
- Limited to 3 targeted queries × 5 results = 15 focused results to LLM

### Known Issue — LLM Grounding
- Researcher LLM occasionally hallucinates on unknown entities
- Grounding instruction added but not fully effective on llama-3.1-8b
- Fix in progress: cross-encoder re-ranking to filter irrelevant results
- V2 upgrade: Claude Haiku as researcher for stronger instruction following

### Pending
- Cross-encoder re-ranking implementation
- Voice chatbot layer (Whisper + ElevenLabs)
- Job Search Engine (separate repo)
