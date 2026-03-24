# AI Research Agent — Sequential Multi-Agent Pipeline

A sequential 2-agent research pipeline built with LangGraph and Ollama.
Part of a 30-day AI PM learning sprint — Week 3: AI Agents.

---

## Architecture
```
User input (topic)
      ↓
Agent 1 — Researcher
(LLM generates research report)
      ↓
Human Checkpoint
(review + approve before proceeding)
      ↓
Agent 2 — Summariser
(LLM synthesises into structured PM summary)
      ↓
Final output
```

---

## Current State — Day 18

Sequential pipeline with in-context state passing and human-in-the-loop checkpoint.

| Component | Detail |
|---|---|
| Orchestration | LangGraph 0.2.28 |
| LLM | Ollama llama3.2 (local, free) |
| Memory | In-context via TypedDict state |
| Tools | None yet — LLM generates from training data |
| Error handling | Human checkpoint only |

**Known limitation:** Researcher agent generates from llama3.2 training data — not grounded in real web results. Real tool calling added in Day 19.

---

## What Each Agent Does

**Agent 1 — Researcher**
Receives topic. Generates a detailed research report using llama3.2. Writes output to shared state.

**Agent 2 — Summariser**
Reads research from shared state. Synthesises into a structured PM-focused summary with key takeaways, most important fact, PM implications, and open questions.

**Human Checkpoint**
Between Agent 1 and Agent 2. Shows research preview. Requires explicit approval before summarisation proceeds. If rejected, pipeline stops.

---

## How to Run
```bash
# Install dependencies
pip3 install langgraph==0.2.28 langchain-ollama==0.1.3 langchain-core==0.2.43

# Make sure Ollama is running with llama3.2
ollama run llama3.2 "say hello"

# Run the pipeline
python3 agent_pipeline.py
```

---

## Roadmap

- Day 18: Sequential pipeline + human-in-the-loop (current)
- Day 19: Add LangSmith observability — trace every node, measure token cost
- Day 19: Add real web search tool — researcher grounds answers in live data
- Day 20: 3-agent pipeline — Planner + Researcher + Writer
- Day 21: MVP 3 submission

---

## Tech Stack

- LangGraph — graph orchestration
- LangChain — LLM abstraction layer
- Ollama — local LLM runtime (llama3.2)
- Python 3.9

---

## Author

Pranab Mohan
AI Product Manager — 30-Day Learning Sprint
Week 3: AI Agents and Multi-Agent Orchestration
