import json
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix
from config import LABELS, EVAL_RESULTS_DIR


def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """predictions: {'issue_number', 'label', 'status'}; ground_truth: {'issue_number', 'label'}

    Only rows with status == 'ok' are scored. Failures are counted and surfaced
    in 'coverage' — never folded into the report as if they were predictions.
    """
    gt_map = {g['issue_number']: g['label'] for g in ground_truth}

    y_true, y_pred = [], []
    scored, failed, unmatched = 0, 0, 0
    status_counts = Counter()

    for p in predictions:
        status = p.get('status', 'ok' if p.get('label') else 'api_error')
        status_counts[status] += 1

        if p['issue_number'] not in gt_map:
            unmatched += 1
            continue
        if status != 'ok' or p.get('label') is None:
            failed += 1
            continue

        y_true.append(gt_map[p['issue_number']])
        y_pred.append(p['label'])
        scored += 1

    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    n_truth = len(gt_map)
    coverage = {
        'ground_truth_rows': n_truth,
        'predictions_received': len(predictions),
        'scored': scored,
        'failed': failed,
        'unmatched_issue_number': unmatched,
        'coverage_rate': round(scored / n_truth, 4) if n_truth else 0.0,
        'status_counts': dict(status_counts),
    }

    warnings = []
    if scored < n_truth:
        warnings.append(
            f'Only {scored}/{n_truth} test issues were scored '
            f'({failed} failed, {unmatched} unmatched). Accuracy below describes '
            f'the scored subset ONLY and is not a complete baseline.'
        )
    if unmatched:
        warnings.append(f'{unmatched} predictions had an issue_number absent from ground truth.')
    coverage['warnings'] = warnings

    result = {
        'report': report,
        'confusion_matrix': matrix,
        'labels': LABELS,
        'n': scored,
        'coverage': coverage,
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_DIR / 'latest.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return result
