# training/predict_student.py
"""Run the merged student model over a split, write predictions, and score them.

  python predict_student.py --model ../checkpoints/merged --limit 5   # sanity check
  python predict_student.py --model ../checkpoints/merged             # full test set

Writes eval/results/predictions_student.jsonl and eval/results/latest_student.json.
The Layer 1 baseline files (predictions.jsonl / latest.json) are never touched.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from config import LABELS, SPLITS_DIR, EVAL_RESULTS_DIR                      # noqa: E402
from services.prompts import STUDENT_SYSTEM_PROMPT, build_user_prompt        # noqa: E402
from services.metrics import evaluate                                        # noqa: E402


def parse(text: str) -> dict:
    """Mirror classifier.py's failure taxonomy so student and baseline are scored alike."""
    try:
        label = json.loads(text.strip()).get('label')
    except (json.JSONDecodeError, AttributeError):
        return {'label': None, 'status': 'malformed_json', 'raw': text}
    if label not in LABELS:
        return {'label': None, 'status': 'invalid_label', 'raw': text}
    return {'label': label, 'status': 'ok', 'raw': text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='merged model folder or HF repo id')
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=None, help='only the first N issues (sanity check)')
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map='auto')
    model.eval()

    with open(SPLITS_DIR / f'{args.split}.jsonl', encoding='utf-8') as f:
        issues = [json.loads(line) for line in f]
    if args.limit:
        issues = issues[:args.limit]

    def prompt(issue):
        messages = [
            {'role': 'system', 'content': STUDENT_SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_prompt(issue['title'], issue.get('body'))},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    predictions = []
    for i in range(0, len(issues), args.batch_size):
        batch = issues[i:i + args.batch_size]
        enc = tok([prompt(x) for x in batch], return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for issue, text in zip(batch, texts):
            predictions.append({'issue_number': issue['number'], **parse(text)})
        print(f'  {len(predictions)}/{len(issues)}', flush=True)

    status = Counter(p['status'] for p in predictions)
    print('status counts:', dict(status))
    for p, issue in list(zip(predictions, issues))[:5]:
        print(f"  #{p['issue_number']} true={issue['label']!r} raw={p['raw']!r}")

    if args.limit:
        print('sanity run only (--limit): not writing result files')
        return

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_DIR / 'predictions_student.jsonl', 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    truth = [{'issue_number': x['number'], 'label': x['label']} for x in issues]
    r = evaluate(predictions, truth, save_as='latest_student.json')
    print(json.dumps(r['coverage'], indent=2))
    print(f"accuracy {r['report']['accuracy']:.4f}  "
          f"macroF1 {r['report']['macro avg']['f1-score']:.4f}  n={r['n']}")


if __name__ == '__main__':
    main()
