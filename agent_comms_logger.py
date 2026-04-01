import json
import os
from datetime import datetime

COMMS_FILE = os.path.join(os.path.dirname(__file__), "agent_comms_log.json")

def load_comms():
    if not os.path.exists(COMMS_FILE):
        return []
    with open(COMMS_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def log_handoff(run_id, from_agent, to_agent,
                query, input_text, output_text, state_snapshot):
    comms = load_comms()
    comms.append({
        "run_id":      run_id,
        "timestamp":   datetime.now().isoformat(),
        "from_agent":  from_agent,
        "to_agent":    to_agent,
        "query":       query,
        "input": {
            "tokens":  len(input_text.split()),
            "preview": input_text[:400]
        },
        "output": {
            "tokens":  len(output_text.split()),
            "preview": output_text[:400]
        },
        "state": {
            "completed_agents": state_snapshot.get("completed_agents", []),
            "cache_status":     state_snapshot.get("cache_status", ""),
            "critic_score":     state_snapshot.get("critic_score", -1),
            "gaps":             state_snapshot.get("gaps", []),
            "feedback_loop":    state_snapshot.get("feedback_loop_used", False),
        }
    })
    with open(COMMS_FILE, "w") as f:
        json.dump(comms, f, indent=2)

def get_run_comms(run_id):
    return [c for c in load_comms() if c.get("run_id") == run_id]

def get_comms_stats():
    comms = load_comms()
    if not comms:
        return {"total_handoffs": 0}
    agents = {}
    for c in comms:
        key = c["from_agent"] + " -> " + c["to_agent"]
        agents[key] = agents.get(key, 0) + 1
    return {
        "total_handoffs":    len(comms),
        "handoff_breakdown": agents,
        "avg_input_tokens":  round(sum(
            c["input"]["tokens"] for c in comms) / len(comms), 1),
        "avg_output_tokens": round(sum(
            c["output"]["tokens"] for c in comms) / len(comms), 1),
    }

if __name__ == "__main__":
    print(json.dumps(get_comms_stats(), indent=2))
