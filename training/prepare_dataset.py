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
 
