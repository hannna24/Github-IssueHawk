from fastapi import APIRouter
import json
from services.classifier import classify_issue
from config import SPLITS_DIR, EVAL_RESULTS_DIR

router = APIRouter()

@router.post("/api/baseline/predict")
async def predict_baseline():
    predictions = []
    with open(SPLITS_DIR / "test.jsonl") as f:
        for line in f:
            issue = json.loads(line)
            result = await classify_issue(issue["title"], issue["body"])
            predictions.append({
                "issue_number": issue["number"],
                "label": result["label"],
                "confidence": result["confidence"]
            })
            if len(predictions) % 25 == 0:
                print(f"  classified {len(predictions)}/353...", flush=True)

            
    
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / "predictions.jsonl"
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    return {"predictions_count": len(predictions), "saved_to": str(out_path)}