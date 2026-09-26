from fastapi import APIRouter
import asyncio
import json
from collections import Counter

from services.classifier import classify_issue
from config import SPLITS_DIR, EVAL_RESULTS_DIR

router = APIRouter()

CONCURRENCY = 3  # Groq on_demand TPM is the binding limit, not RPM


@router.post("/api/baseline/predict")
async def predict_baseline():
    with open(SPLITS_DIR / "test.jsonl", encoding="utf-8") as f:
        issues = [json.loads(line) for line in f]

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async def run_one(issue):
        nonlocal done
        async with sem:
            result = await classify_issue(issue["title"], issue["body"])
        done += 1
        if done % 25 == 0:
            print(f"  classified {done}/{len(issues)}...", flush=True)
        return {
            "issue_number": issue["number"],
            "label": result["label"],
            "confidence": result["confidence"],
            "status": result["status"],
            "reason": result["reason"],
        }

    predictions = await asyncio.gather(*(run_one(i) for i in issues))

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / "predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    status_counts = Counter(p["status"] for p in predictions)
    return {
        "predictions_count": len(predictions),
        "status_counts": dict(status_counts),
        "saved_to": str(out_path),
    }
