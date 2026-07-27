import json
from transformers import AutoTokenizer
from datasets import Dataset
 
SYSTEM_PROMPT = ('You are an issue triage expert for open-source repositories. '
                  'Classify the issue into EXACTLY ONE label: '
                  'bug, feature-request, documentation, question, duplicate. '
                  'Respond with ONLY JSON: {"label": "<label>"}.')
 
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
 
def format_example(issue: dict, tokenizer) -> dict:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f"ISSUE TITLE: {issue['title']}\nISSUE BODY: {issue['body'][:800]}"},
        {'role': 'assistant', 'content': json.dumps({'label': issue['label']})}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': text}
 
def build_dataset(split_path: str, tokenizer_name: str = 'Qwen/Qwen2.5-3B-Instruct'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    issues = load_jsonl(split_path)
    examples = [format_example(i, tokenizer) for i in issues]
    return Dataset.from_list(examples)
