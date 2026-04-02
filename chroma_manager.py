import os
import json
from datetime import datetime

CHROMA_AVAILABLE = False
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def is_available():
    return False

def check_cache(query):
    return {"status": "miss", "context": None, "reason": "Chroma not available on cloud"}

def store_run(run_id, query, research_output, intent="", critic_score=-1, pipeline_type="research"):
    return False

def get_chroma_stats():
    return {"total_stored": 0}

def get_collection():
    raise RuntimeError("Chroma not available")

if __name__ == "__main__":
    print(json.dumps(get_chroma_stats(), indent=2))
