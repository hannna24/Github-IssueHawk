# backend/services/serving_client.py
"""Thin client for the local Ollama model. Mirrors classifier.py's result shape
(label / confidence / status / reason) so callers can treat both the same way."""
import json
import math

import httpx

from config import LABELS, OLLAMA_URL, OLLAMA_MODEL
from services.prompts import STUDENT_SYSTEM_PROMPT, build_user_prompt


def _failure(status: str, reason: str) -> dict:
    return {'label': None, 'confidence': None, 'status': status, 'reason': reason}


def label_confidence(tokens: list[dict], label: str) -> float | None:
    """Probability the model assigned to the label's own tokens.

    The student only emits {"label": "..."}, so there is no self-reported
    confidence to read. Instead, multiply the probabilities of the tokens that
    spell the label out -- the one decision the whole output hinges on.
    """
    text = ''.join(t['token'] for t in tokens)
    start = text.find(label)
    if start < 0:
        return None
    end = start + len(label)
    logprob, pos = 0.0, 0
    for t in tokens:
        tok_start, pos = pos, pos + len(t['token'])
        if tok_start < end and pos > start:
            logprob += t['logprob']
    return round(math.exp(logprob), 4)


async def classify_local(title: str, body: str | None) -> dict:
    payload = {
        'model': OLLAMA_MODEL,
        'stream': False,
        'logprobs': True,
        'messages': [
            {'role': 'system', 'content': STUDENT_SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_prompt(title, body)},
        ],
        'options': {'temperature': 0, 'num_predict': 24},
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return _failure('api_error', f'{type(e).__name__}: {e}')

    raw = data['message']['content']
    try:
        label = json.loads(raw.strip()).get('label')
    except (json.JSONDecodeError, AttributeError):
        return _failure('malformed_json', raw)
    if label not in LABELS:
        return _failure('invalid_label', raw)

    return {
        'label': label,
        'confidence': label_confidence(data.get('logprobs') or [], label),
        'status': 'ok',
        'reason': '',
    }
