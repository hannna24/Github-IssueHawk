# backend/services/classifier.py
import asyncio
import json
import random

from groq import AsyncGroq, APIStatusError, APIConnectionError
from config import GROQ_API_KEY, LLM_MODEL, LABELS

client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = f'''You are an issue triage expert for open-source
repositories. Classify the issue into EXACTLY ONE of these labels:
{', '.join(LABELS)}
Respond with ONLY JSON: {{"label": "<label>", "confidence": <0-1>, "reason": "<one sentence>"}}'''

MAX_ATTEMPTS = 8
BASE_DELAY = 2.0   # seconds; doubled each retry
MAX_DELAY = 75.0   # TPM buckets refill on a 60s window, so allow one full refill


def _retry_after(exc) -> float | None:
    """Seconds the API asked us to wait, if it said."""
    headers = getattr(getattr(exc, 'response', None), 'headers', None) or {}
    for key in ('retry-after', 'x-ratelimit-reset-tokens', 'x-ratelimit-reset-requests'):
        raw = headers.get(key)
        if not raw:
            continue
        try:
            return float(str(raw).rstrip('s'))
        except ValueError:
            continue
    return None


def _failure(status: str, reason: str) -> dict:
    """A non-prediction. label is None so the harness can never score it as
    a real answer — the old code returned 'bug' here and silently poisoned
    every metric with a 26% slug of fake bug predictions."""
    return {'label': None, 'confidence': None, 'status': status, 'reason': reason}


async def classify_issue(title: str, body: str) -> dict:
    user_prompt = f'ISSUE TITLE: {title}\nISSUE BODY: {(body or "")[:800]}'
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt}
                ],
                max_tokens=512,
                temperature=0.1,
                # gpt-oss is a reasoning model: it spends tokens thinking before
                # it emits the JSON. At max_tokens=100 it hit the cap mid-document
                # and the API rejected the call outright (json_validate_failed).
                reasoning_effort='low',
                response_format={'type': 'json_object'}
            )
        except (APIStatusError, APIConnectionError) as e:
            status = getattr(e, 'status_code', None)
            retry_after = _retry_after(e)
            # 429 = rate limit, 5xx = transient upstream. Anything else
            # (401 bad key, 400 bad request) will never succeed on retry.
            if status is not None and status != 429 and status < 500:
                return _failure('api_error', f'{type(e).__name__} {status}: {e}')
            last_error = f'{type(e).__name__} {status}: {e}'
        except Exception as e:
            retry_after = None
            last_error = f'{type(e).__name__}: {e}'
        else:
            try:
                result = json.loads(resp.choices[0].message.content)
            except (json.JSONDecodeError, TypeError, IndexError) as e:
                # Model emitted something unparseable. That's a model failure,
                # not a transport failure — don't burn retries on it.
                return _failure('malformed_json', f'{type(e).__name__}: {e}')

            label = result.get('label')
            if label not in LABELS:
                # The model answered, it just answered off-taxonomy. Distinct
                # from an API error, and worth counting separately.
                return _failure('invalid_label', f'model returned: {label!r}')

            return {
                'label': label,
                'confidence': result.get('confidence'),
                'status': 'ok',
                'reason': result.get('reason', '')
            }

        if attempt < MAX_ATTEMPTS - 1:
            # Groq's 429 carries a Retry-After telling us exactly when the
            # token bucket refills. Blind backoff ignored it and gave up early.
            if retry_after is not None:
                delay = min(retry_after + 1.0, MAX_DELAY)
            else:
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            await asyncio.sleep(delay + random.uniform(0, 1))

    return _failure('api_error', f'exhausted {MAX_ATTEMPTS} attempts; last: {last_error}')
