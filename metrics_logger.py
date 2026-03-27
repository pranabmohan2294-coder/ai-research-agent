import json
import os
import re
from datetime import datetime

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics_log.json")

def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return []
    with open(METRICS_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def save_run(run: dict):
    metrics = load_metrics()
    run["timestamp"] = datetime.now().isoformat()
    run["run_id"] = f"run_{len(metrics)+1:04d}"
    metrics.append(run)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    return run["run_id"]

def extract_critic_score(critic_output: str) -> int:
    patterns = [
        r"score[:\s]+(\d+)/10",
        r"(\d+)/10",
        r"score[:\s]+(\d+)",
        r"rate[:\s]+(\d+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, critic_output.lower())
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    return -1

def get_summary_stats():
    metrics = load_metrics()
    if not metrics:
        return {"total_runs": 0}

    total = len(metrics)
    scores = [r.get("critic_score",-1) for r in metrics if r.get("critic_score",-1) > 0]
    latencies = [r.get("total_latency",0) for r in metrics if r.get("total_latency",0) > 0]
    approvals = [r for r in metrics if r.get("writer_approved") is not None]
    approved = [r for r in approvals if r.get("writer_approved")]
    hallucination_flags = [r for r in metrics if r.get("hallucination_flagged")]
    entity_warnings = [r for r in metrics if not r.get("entity_valid", True)]
    feedback_loops = [r for r in metrics if r.get("feedback_loop_used")]

    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)

    return {
        "total_runs": total,
        "avg_critic_score": round(sum(scores)/len(scores), 1) if scores else None,
        "median_critic_score": sorted(scores)[len(scores)//2] if scores else None,
        "latency_p95": sorted_latencies[p95_index] if len(sorted_latencies) > 1 else (sorted_latencies[0] if sorted_latencies else None),
        "avg_latency": round(sum(latencies)/len(latencies), 1) if latencies else None,
        "approval_rate": round(len(approved)/len(approvals)*100, 1) if approvals else None,
        "hallucination_flag_rate": round(len(hallucination_flags)/total*100, 1),
        "entity_warning_rate": round(len(entity_warnings)/total*100, 1),
        "feedback_loop_rate": round(len(feedback_loops)/total*100, 1),
        "runs_with_scores": len(scores),
        "runs_with_latency": len(latencies),
        "recent_runs": metrics[-5:]
    }

if __name__ == "__main__":
    stats = get_summary_stats()
    print(json.dumps(stats, indent=2))
