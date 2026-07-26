# backend/services/classifier.py
import json
from groq import AsyncGroq
from config import GROQ_API_KEY, LLM_MODEL, LABELS

client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = f'''You are an issue triage expert for open-source
repositories. Classify the issue into EXACTLY ONE of these labels:
{', '.join(LABELS)}
Respond with ONLY JSON: {{"label": "<label>", "confidence": <0-1>, "reason": "<one sentence>"}}'''


async def classify_issue(title: str, body: str) -> dict:
    user_prompt = f'ISSUE TITLE: {title}\nISSUE BODY: {body[:800]}'

    try:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ],
            max_tokens=100,
            temperature=0.1,
            response_format={'type': 'json_object'}
        )
        result = json.loads(resp.choices[0].message.content)

        # Guard: model sometimes invents a label outside our taxonomy
        if result.get('label') not in LABELS:
            return {
                'label': 'bug',
                'confidence': 0.0,
                'reason': f"invalid label returned: {result.get('label')}"
            }
        return result

    except Exception as e:
        # Malformed JSON, rate limit, network blip — don't kill the run
        return {'label': 'bug', 'confidence': 0.0, 'reason': f'error: {e}'}