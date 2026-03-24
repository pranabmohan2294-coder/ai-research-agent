# AI Research Agent — Sequential Multi-Agent Pipeline

A sequential 2-agent research pipeline with a Streamlit UI, built with LangGraph and Ollama.
Part of a 30-day AI PM learning sprint — Week 3: AI Agents.

---

## Live Demo

Run locally at `http://localhost:8501` after setup.

---

## Architecture
```
User input (topic)
      ↓
Agent 1 — Researcher
(streams research report in 6 structured sections)
      ↓
Human Checkpoint
(review full research, approve or reject)
      ↓
Agent 2 — Summariser
(streams PM-focused summary in 4 structured sections)
      ↓
Final output — Summary + Full Research tabs
```

---

## Features

- Streaming output — both agents write token by token to the screen in real time
- Structured research — Agent 1 outputs 6 sections: Key Facts, Current State, Challenges, Key Players, Data and Numbers, Summary
- Structured summary — Agent 2 outputs 4 sections: Key Takeaways, Most Important Fact, PM Implications, Open Question
- Human-in-the-loop checkpoint — approve or reject research before Agent 2 runs
- Collapsible research — full text expandable/collapsible at checkpoint and final screen
- Live state inspector — sidebar shows topic, human_approved, research chars, summary chars at every step
- Agent performance metrics — sidebar shows latency, token count, data source, accuracy note per agent
- Pipeline totals — combined latency, tokens, cost ($0.00 — fully local)

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph 0.2.28 |
| LLM | Ollama llama3.2 (local, free) |
| UI | Streamlit |
| Memory | In-context via TypedDict state |
| Tools | None yet — LLM generates from training data |
| Cost | $0.00 — fully local, no API keys |

---

## Known Limitations

- Agent 1 generates from llama3.2 training data — not grounded in live web results
- Knowledge cutoff: early 2024 — recent events may be missing or inaccurate
- No programmatic error handling — human checkpoint is the only safety gate
- Token counts are word-based estimates, not exact tokeniser counts

---

## Roadmap

| Day | Feature |
|---|---|
| Day 18 | Sequential pipeline + Streamlit UI + human-in-the-loop ✓ |
| Day 19 | LangSmith observability — trace every node, measure exact token cost |
| Day 19 | Real web search tool — researcher grounds answers in live data |
| Day 20 | 3-agent pipeline — Planner + Researcher + Writer |
| Day 21 | MVP 3 submission |

---

## How to Run
```bash
# Install dependencies
pip3 install langgraph==0.2.28 langchain-ollama==0.1.3 langchain-core==0.2.43 streamlit

# Make sure Ollama is running with llama3.2
ollama run llama3.2 "say hello"

# Run the Streamlit app
python3 -m streamlit run app_streamlit.py
```

---

## File Structure
```
├── app_streamlit.py     # Streamlit UI — main app
├── agent_pipeline.py    # Terminal version — original pipeline
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Author

Pranab Mohan
AI Product Manager — 30-Day Learning Sprint
Week 3: AI Agents and Multi-Agent Orchestration
