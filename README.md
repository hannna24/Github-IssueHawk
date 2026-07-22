# IssueHawk

**Fine-tuned issue triage agent that learns from your repo's own labeled history. Runs locally via quantized GGUF + Ollama. Watches real traffic in shadow mode. Measures trust by agreement rate before acting.**

---

## What It Does

IssueHawk automatically classifies GitHub issues and detects duplicates using a model you train on your repo's own labeled data. It runs entirely on your hardware — no external API calls — and operates in shadow mode: it logs what it *would* have decided, never actually posts a label, until the numbers prove it's trustworthy.

### The Triage Pipeline

1. **New issue opens** → GitHub webhook fires
2. **Duplicate check** → Semantic search via ChromaDB against historical issues
3. **Classification** → If not a duplicate, classify into: bug, feature-request, documentation, question, or duplicate
4. **Shadow-logged** → Prediction recorded to Postgres with confidence score
5. **Agreement measured** → Once a maintainer labels the issue, compare predictions to actual labels
6. **Trust earned** → After hundreds of predictions with high agreement, consider real actions

### Why This Approach

- **No hallucinations matter yet** — shadow mode means wrong predictions never reach users
- **You own the model** — fine-tuned on your repo's data, running on your hardware
- **Measurable trust** — agreement rate = honest number you can show stakeholders
- **Graceful graduation** — start shadow-logging, move to comments, then auto-label only when metrics justify it



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
