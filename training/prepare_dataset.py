# training/prepare_dataset.py
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from datasets import Dataset

# training/ sits beside backend/, so make the app package importable rather
# than duplicating the taxonomy or the prompt text here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from config import LABELS                                    # noqa: E402
from services.prompts import STUDENT_SYSTEM_PROMPT, build_user_prompt  # noqa: E402


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def format_example(issue: dict, tokenizer) -> dict:
    messages = [
        {'role': 'system', 'content': STUDENT_SYSTEM_PROMPT},
        {'role': 'user', 'content': build_user_prompt(issue['title'], issue.get('body'))},
        {'role': 'assistant', 'content': json.dumps({'label': issue['label']})}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': text}


def build_dataset(split_path: str, tokenizer_name: str = 'Qwen/Qwen2.5-3B-Instruct'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    issues = load_jsonl(split_path)

    # Fail loudly on a label the prompt never offers. Silently training on an
    # off-taxonomy target is unrecoverable later -- the model just learns to
    # emit a class the evaluator will always score as wrong.
    bad = Counter(i['label'] for i in issues if i.get('label') not in LABELS)
    if bad:
        raise ValueError(
            f'{sum(bad.values())} example(s) in {split_path} carry labels outside '
            f'LABELS={LABELS}: {dict(bad)}'
        )

    examples = [format_example(i, tokenizer) for i in issues]
    return Dataset.from_list(examples)
