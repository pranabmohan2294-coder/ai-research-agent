from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


# ---------------------------
# State — shared memory
# between both agents
# ---------------------------

class ResearchState(TypedDict):
    topic: str
    research: str
    summary: str
    human_approved: bool


# ---------------------------
# LLM
# ---------------------------

llm = ChatOllama(model="llama3.2", temperature=0)


# ---------------------------
# Agent 1 — Researcher
# ---------------------------

def researcher(state: ResearchState) -> ResearchState:
    print("\n[Agent 1 — Researcher] Starting research...")

    prompt = f"""You are a research agent. Your job is to find detailed, 
factual information about the following topic.

Topic: {state['topic']}

Provide a thorough research report covering:
1. Key facts and background
2. Current state and recent developments  
3. Main challenges or considerations
4. Key players or stakeholders involved

Be specific and factual. Write at least 200 words."""

    response = llm.invoke(prompt)
    research = response.content

    print(f"[Agent 1 — Researcher] Research complete. ({len(research)} chars)")
    return {**state, "research": research}


# ---------------------------
# Human checkpoint
# ---------------------------

def human_checkpoint(state: ResearchState) -> ResearchState:
    print("\n" + "="*60)
    print("HUMAN CHECKPOINT — Review research before summarisation")
    print("="*60)
    print(f"\nTopic: {state['topic']}")
    print(f"\nResearch preview (first 400 chars):")
    print(state['research'][:400] + "...")
    print("\n" + "-"*60)

    approval = input("\nApprove this research? (yes/no): ").strip().lower()

    if approval == "yes":
        print("[Checkpoint] Approved. Proceeding to summariser.")
        return {**state, "human_approved": True}
    else:
        print("[Checkpoint] Rejected. Stopping pipeline.")
        return {**state, "human_approved": False}


# ---------------------------
# Agent 2 — Summariser
# ---------------------------

def summariser(state: ResearchState) -> ResearchState:
    print("\n[Agent 2 — Summariser] Generating summary...")

    prompt = f"""You are a summarisation agent. Your job is to take 
detailed research and produce a clean, structured summary.

Research:
{state['research']}

Write a structured summary with these sections:
1. Key takeaways (3 bullet points)
2. Most important fact
3. What a product manager should know about this topic
4. One open question worth investigating further

Be concise and PM-focused."""

    response = llm.invoke(prompt)
    summary = response.content

    print(f"[Agent 2 — Summariser] Summary complete. ({len(summary)} chars)")
    return {**state, "summary": summary}


# ---------------------------
# Routing — after checkpoint
# ---------------------------

def route_after_checkpoint(state: ResearchState) -> str:
    if state["human_approved"]:
        return "summariser"
    else:
        return END


# ---------------------------
# Build the graph
# ---------------------------

def build_pipeline():
    graph = StateGraph(ResearchState)

    graph.add_node("researcher", researcher)
    graph.add_node("human_checkpoint", human_checkpoint)
    graph.add_node("summariser", summariser)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "human_checkpoint")
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_checkpoint,
        {"summariser": "summariser", END: END}
    )
    graph.add_edge("summariser", END)

    return graph.compile()


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Sequential 2-Agent Research Pipeline")
    print("Researcher → Human Checkpoint → Summariser")
    print("="*60)

    topic = input("\nEnter a research topic: ").strip()

    pipeline = build_pipeline()

    initial_state = ResearchState(
        topic=topic,
        research="",
        summary="",
        human_approved=False
    )

    final_state = pipeline.invoke(initial_state)

    if final_state["human_approved"]:
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(final_state["summary"])
        print("\n" + "="*60)
        print("Pipeline complete.")
        print(f"Research length: {len(final_state['research'])} chars")
        print(f"Summary length:  {len(final_state['summary'])} chars")
    else:
        print("\nPipeline stopped at human checkpoint.")
