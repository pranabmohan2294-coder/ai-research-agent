import os
import json
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------

CHROMA_DIR         = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME    = "research_runs"
CACHE_HIT_THRESHOLD  = 0.15   # same topic — skip web search
CONTEXT_THRESHOLD    = 0.40   # related topic — use as context + search
CACHE_MAX_AGE_DAYS   = 14     # older than this = stale
MIN_QUALITY_SCORE    = 5      # critic score below this = don't use as cache

# ---------------------------
# Embedder — same model as Week 1
# ---------------------------

print("[Chroma] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[Chroma] Embedding model ready.")

# ---------------------------
# Chroma client
# ---------------------------

def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# ---------------------------
# Store a research run
# ---------------------------

def store_run(
    run_id: str,
    query: str,
    research_output: str,
    intent: str = "",
    critic_score: int = -1,
    pipeline_type: str = "research"
):
    try:
        collection = get_collection()
        embedding  = embedder.encode([query])[0].tolist()

        collection.add(
            documents  = [research_output],
            embeddings = [embedding],
            metadatas  = [{
                "query":         query,
                "intent":        intent,
                "timestamp":     datetime.now().isoformat(),
                "run_id":        run_id,
                "word_count":    len(research_output.split()),
                "critic_score":  critic_score,
                "pipeline_type": pipeline_type
            }],
            ids=[run_id]
        )
        print(f"[Chroma] Stored run {run_id} — query: '{query[:50]}'")
        return True
    except Exception as e:
        print(f"[Chroma] Store failed: {e}")
        return False

# ---------------------------
# Check cache for a query
# ---------------------------

def check_cache(query: str) -> dict:
    try:
        collection = get_collection()
        count      = collection.count()

        if count == 0:
            return {"status": "miss", "context": None,
                    "reason": "Empty collection"}

        embedding = embedder.encode([query])[0].tolist()
        results   = collection.query(
            query_embeddings = [embedding],
            n_results        = 1,
            include          = ["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return {"status": "miss", "context": None,
                    "reason": "No results returned"}

        distance  = results["distances"][0][0]
        document  = results["documents"][0][0]
        metadata  = results["metadatas"][0][0]
        timestamp = metadata.get("timestamp", "")
        cached_score = metadata.get("critic_score", -1)

        # Age check
        age_days = None
        is_stale = False
        if timestamp:
            try:
                age      = datetime.now() - datetime.fromisoformat(timestamp)
                age_days = age.days
                is_stale = age_days > CACHE_MAX_AGE_DAYS
            except Exception:
                is_stale = True

        # Quality gate
        low_quality = (cached_score > 0 and cached_score < MIN_QUALITY_SCORE)

        base = {
            "context":      document,
            "distance":     round(distance, 4),
            "age_days":     age_days,
            "cached_query": metadata.get("query", ""),
            "cached_score": cached_score,
            "is_stale":     is_stale,
            "low_quality":  low_quality
        }

        # Zone 1 — cache hit
        if distance < CACHE_HIT_THRESHOLD and not is_stale and not low_quality:
            return {**base, "status": "hit",
                    "reason": f"Cache hit (distance={distance:.3f}, age={age_days}d)"}

        # Zone 2 — use as context
        elif distance < CONTEXT_THRESHOLD:
            reason = "stale" if is_stale else \
                     "low quality" if low_quality else \
                     f"partial match (distance={distance:.3f})"
            return {**base, "status": "context",
                    "reason": f"Use as context — {reason}"}

        # Zone 3 — fresh search
        else:
            return {**base, "status": "miss",
                    "reason": f"No relevant cache (distance={distance:.3f})"}

    except Exception as e:
        return {"status": "miss", "context": None,
                "reason": f"Cache check failed: {e}"}

# ---------------------------
# Get all stored runs
# ---------------------------

def get_all_runs(limit: int = 50) -> list:
    try:
        collection = get_collection()
        count      = collection.count()
        if count == 0:
            return []
        results = collection.get(
            limit   = min(limit, count),
            include = ["metadatas"]
        )
        return results["metadatas"]
    except Exception as e:
        print(f"[Chroma] Get all runs failed: {e}")
        return []

# ---------------------------
# Stats
# ---------------------------

def get_chroma_stats() -> dict:
    try:
        collection = get_collection()
        count      = collection.count()
        if count == 0:
            return {"total_stored": 0}

        all_runs   = get_all_runs(limit=count)
        scores     = [r.get("critic_score",-1) for r in all_runs
                      if r.get("critic_score",-1) > 0]
        intents    = {}
        for r in all_runs:
            intent = r.get("intent","unknown")
            intents[intent] = intents.get(intent, 0) + 1

        return {
            "total_stored":      count,
            "avg_quality_score": round(sum(scores)/len(scores),1) if scores else None,
            "intents_breakdown": intents,
            "oldest_run":        min((r.get("timestamp","") for r in all_runs), default=None),
            "newest_run":        max((r.get("timestamp","") for r in all_runs), default=None),
        }
    except Exception as e:
        return {"total_stored": 0, "error": str(e)}

if __name__ == "__main__":
    print("Chroma stats:", json.dumps(get_chroma_stats(), indent=2))
