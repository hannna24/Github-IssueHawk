import json
from sklearn.metrics import classification_report, confusion_matrix
from config import LABELS, EVAL_RESULTS_DIR
 
def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """predictions/ground_truth: list of {'issue_number': int, 'label': str}"""
    gt_map = {g['issue_number']: g['label'] for g in ground_truth}
    y_true, y_pred = [], []
    for p in predictions:
        if p['issue_number'] in gt_map:
            y_true.append(gt_map[p['issue_number']])
            y_pred.append(p['label'])
 
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
 
    result = {'report': report, 'confusion_matrix': matrix, 'labels': LABELS, 'n': len(y_true)}
 
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_DIR / 'latest.json', 'w') as f:
        json.dump(result, f, indent=2)
 
    return result
