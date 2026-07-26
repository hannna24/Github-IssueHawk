from fastapi import APIRouter
import json
from services.metrics import evaluate
from config import SPLITS_DIR, EVAL_RESULTS_DIR

router = APIRouter()

@router.get("/api/eval/report")
async def get_eval_report():
    predictions = []
    with open(EVAL_RESULTS_DIR / "predictions.jsonl") as f:
        for line in f:
            predictions.append(json.loads(line))
    
    ground_truth = []
    with open(SPLITS_DIR / "test.jsonl") as f:
        for line in f:
            issue = json.loads(line)
            ground_truth.append({
                "issue_number": issue["number"],
                "label": issue["label"]
            })
    
    report = evaluate(predictions, ground_truth)
    return report