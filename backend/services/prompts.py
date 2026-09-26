"""Single source of truth for triage prompts.

Both prompts derive their label list from config.LABELS, so a taxonomy change
propagates everywhere instead of rotting in a hardcoded string. That drift is
exactly what happened before: this project's training prompt still advertised
'question' and 'duplicate' long after neither existed in the data.

Two prompts, deliberately:

  TEACHER_SYSTEM_PROMPT - the hosted API path (classifier.py). Asks for
      confidence and a reason, which shadow mode logs alongside the label.

  STUDENT_SYSTEM_PROMPT - the fine-tuned local model. Label only, because
      training data carries a gold label and nothing else; asking a student to
      emit a confidence and a rationale it was never taught would just be
      teaching it to fabricate both.

Each path uses ONE prompt for both training and inference, so neither suffers a
train/serve mismatch.
"""
from config import LABELS

_LABEL_LIST = ', '.join(LABELS)

TEACHER_SYSTEM_PROMPT = f'''You are an issue triage expert for open-source
repositories. Classify the issue into EXACTLY ONE of these labels:
{_LABEL_LIST}
Respond with ONLY JSON: {{"label": "<label>", "confidence": <0-1>, "reason": "<one sentence>"}}'''

STUDENT_SYSTEM_PROMPT = (
    'You are an issue triage expert for open-source repositories. '
    'Classify the issue into EXACTLY ONE label: '
    f'{_LABEL_LIST}. '
    'Respond with ONLY JSON: {"label": "<label>"}.'
)

BODY_CHAR_LIMIT = 800


def build_user_prompt(title: str, body: str | None) -> str:
    """Identical framing for training and inference.

    `body` is None on some GitHub issues, and slicing None raises TypeError --
    that crash was latent in the training path.
    """
    return f'ISSUE TITLE: {title}\nISSUE BODY: {(body or "")[:BODY_CHAR_LIMIT]}'
