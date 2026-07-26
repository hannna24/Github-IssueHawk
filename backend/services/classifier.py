from groq import AsyncGroq
import json
from config import GROQ_API_KEY, LLM_MODEL, LABELS
 
client = AsyncGroq(api_key=GROQ_API_KEY)
 
SYSTEM_PROMPT = f'''You are an issue triage expert for open-source
repositories. Classify the issue into EXACTLY ONE of these labels:
{', '.join(LABELS)}
Respond with ONLY JSON: {{"label": "<label>", "confidence": <0-1>, "reason": "<one sentence>"}}'''
 
async def classify_issue(title: str, body: str) -> dict:
    user_prompt = f'ISSUE TITLE: {title}\nISSUE BODY: {body[:800]}'
 
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
    return json.loads(resp.choices[0].message.content)
