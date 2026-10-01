# backend/serving/eval_quantized.py
"""Score the Ollama-served quantized model on the test split.

  ollama create issuehawk -f Modelfile      # once, from this folder
  python eval_quantized.py --limit 5         # sanity check
  python eval_quantized.py                   # full test split

Writes eval/results/predictions_quantized.jsonl and latest_quantized.json.
Compare against the fp16 student (acc 0.9517, macro-F1 0.930) to see what
quantization cost.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LABELS, SPLITS_DIR, EVAL_RESULTS_DIR                      # noqa: E402
from services.prompts import STUDENT_SYSTEM_PROMPT, build_user_prompt        # noqa: E402
from services.metrics import evaluate                                        # noqa: E402

OLLAMA_URL = 'http://localhost:11434/api/chat'


def parse(text: str) -> dict:
    try:
        label = json.loads(text.strip()).get('label')
    except (json.JSONDecodeError, AttributeError):
        return {'label': None, 'status': 'malformed_json', 'raw': text}
    if label not in LABELS:
        return {'label': None, 'status': 'invalid_label', 'raw': text}
    return {'label': label, 'status': 'ok', 'raw': text}


def ask(model: str, issue: dict) -> str:
    resp = requests.post(OLLAMA_URL, json={
        'model': model,
        'stream': False,
        'messages': [
            {'role': 'system', 'content': STUDENT_SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_prompt(issue['title'], issue.get('body'))},
        ],
        'options': {'temperature': 0, 'num_predict': 24},
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()['message']['content']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='issuehawk')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    with open(SPLITS_DIR / 'test.jsonl', encoding='utf-8') as f:
        issues = [json.loads(line) for line in f]
    if args.limit:
        issues = issues[:args.limit]

    predictions, start = [], time.time()
    for n, issue in enumerate(issues, 1):
        try:
            p = parse(ask(args.model, issue))
        except requests.RequestException as e:
            p = {'label': None, 'status': 'api_error', 'raw': str(e)}
        predictions.append({'issue_number': issue['number'], **p})
        if n % 10 == 0 or n == len(issues):
            print(f'  {n}/{len(issues)}  ({(time.time() - start) / n:.1f}s/issue)', flush=True)

    print('status counts:', dict(Counter(p['status'] for p in predictions)))
    for p, issue in list(zip(predictions, issues))[:5]:
        print(f"  #{p['issue_number']} true={issue['label']!r} raw={p['raw']!r}")

    if args.limit:
        print('sanity run only (--limit): not writing result files')
        return

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_DIR / 'predictions_quantized.jsonl', 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    truth = [{'issue_number': x['number'], 'label': x['label']} for x in issues]
    r = evaluate(predictions, truth, save_as='latest_quantized.json')
    print(json.dumps(r['coverage'], indent=2))
    print(f"accuracy {r['report']['accuracy']:.4f}  "
          f"macroF1 {r['report']['macro avg']['f1-score']:.4f}  n={r['n']}")
    for k in r['labels']:
        v = r['report'][k]
        print(f"  {k:16s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  f1 {v['f1-score']:.3f}")


if __name__ == '__main__':
    main()
