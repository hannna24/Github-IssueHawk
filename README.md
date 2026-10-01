# Github IssueHawk

**Fine-tuned issue triage agent that learns from your repo's own labeled history. Runs locally via quantized GGUF + Ollama. Watches real traffic in shadow mode. Measures trust by agreement rate before acting.**

---

## What It Does

IssueHawk automatically classifies GitHub issues and detects duplicates using a model you train on your repo's own labeled data. It runs entirely on your hardware — no external API calls — and operates in shadow mode: it logs what it *would* have decided, never actually posts a label, until the numbers prove it's trustworthy.

### The Triage Pipeline

1. **New issue opens** → GitHub webhook fires
2. **Duplicate check** → Semantic search via ChromaDB against historical issues
3. **Classification** → If not a duplicate, classify into: bug, feature-request, or documentation
4. **Shadow-logged** → Prediction recorded to Postgres with confidence score
5. **Agreement measured** → Once a maintainer labels the issue, compare predictions to actual labels
6. **Trust earned** → After hundreds of predictions with high agreement, consider real actions

### Why This Approach

- **No hallucinations matter yet** — shadow mode means wrong predictions never reach users
- **You own the model** — fine-tuned on your repo's data, running on your hardware
- **Measurable trust** — agreement rate = honest number you can show stakeholders
- **Graceful graduation** — start shadow-logging, move to comments, then auto-label only when metrics justify it

## Results: Layer 1 baseline vs. Layer 2 fine-tuned model

Both models classify the same held-out test split (352 issues: 227 bug, 59 feature-request, 66 documentation; three labels) and every issue is scored for both.

- **Baseline (Layer 1):** `openai/gpt-oss-120b` via the Groq API, zero-shot.
- **Fine-tuned (Layer 2):** Qwen2.5-3B-Instruct + LoRA (r=16, 3 epochs, 1,639 training issues, one free Colab T4), adapter merged into the fp16 base model, greedy decoding. Runs locally.

| Model | Accuracy | Macro-F1 | bug F1 | feature-request F1 | documentation F1 |
|---|---|---|---|---|---|
| Baseline: gpt-oss-120b (zero-shot, hosted) | 0.9602 | 0.9493 | 0.972 | 0.974 | 0.902 |
| Fine-tuned: Qwen2.5-3B + LoRA (local) | 0.9517 | 0.9300 | 0.974 | 0.952 | 0.864 |

The fine-tuned model got 335 of 352 right against the baseline's 338. A three-issue difference on a test set this size is within noise, so the fair reading is that a 3B model fine-tuned on this repo's own history gets close to a 120B hosted model, without API calls, but did not beat it.

Where it falls short: documentation recall (0.773 vs. 0.833). Ten documentation issues were predicted as bug and five as feature-request; the baseline made 10 and 1 of the same mistakes.

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | bug | 0.950 | 0.996 | 0.972 |
| Baseline | feature-request | 0.983 | 0.966 | 0.974 |
| Baseline | documentation | 0.982 | 0.833 | 0.902 |
| Fine-tuned | bug | 0.957 | 0.991 | 0.974 |
| Fine-tuned | feature-request | 0.908 | 1.000 | 0.952 |
| Fine-tuned | documentation | 0.981 | 0.773 | 0.864 |

All 352 test issues produced valid JSON with an allowed label for both models.

## How to Contribute

This project is designed for you to fork and customize. The intent is to learn and own every piece:

1. Fork the repo
2. Follow the Layer 1–4 documentation (see `docs/` folder)
3. Run on your own repo's data
4. Fine-tune if your data supports it
5. Deploy to your server or Render

---


---

Built with PyTorch, PEFT/LoRA, TRL, llama.cpp, LangGraph, FastAPI, PostgreSQL, and Docker.
