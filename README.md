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

---

## Getting Started

### Prerequisites

- Python 3.11+
- GPU (free T4 on Google Colab for fine-tuning) or CPU (for inference only)
- GitHub personal access token (read issues)
- Groq API key (free tier; Layer 1 baseline only)
- PostgreSQL 14+ (for shadow logging)
- Docker (recommended for Layer 4 deployment)

### Quick Start: Layer 1 (Baseline + Evaluation)

Layer 1 proves the concept works on your repo's data without any training.

```bash
git clone https://github.com/YOUR_USERNAME/issuehawk.git
cd issuehawk
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
GITHUB_TOKEN=github_pat_YOUR_TOKEN_HERE
GROQ_API_KEY=gsk_YOUR_KEY_HERE
EOF

# Start the backend
cd backend
uvicorn main:app --reload --port 8000

# Open http://localhost:8000/docs in your browser
# POST /api/dataset/build with repo name to fetch issues
# POST /api/baseline/predict to classify test-set issues
# GET /api/eval/report to see precision/recall/F1
```

That's it. Layer 1 takes ~1 week to understand and implement. You'll have real metrics (F1 score, confusion matrix) on your repo's data before deciding if fine-tuning is worth it.

### Decide: Do You Need Layers 2–4?

**Use Layer 1 only if:**
- Repo has <1,000 labeled issues (not enough data to fine-tune)
- Baseline agreement rate (Groq zero-shot) is already ≥75%
- You're happy with API latency and Groq's availability
- Shadow-mode logging is your end goal

**Add Layers 2–4 if:**
- Repo has 1,000+ labeled issues
- Baseline agreement rate is 60–75% (room to improve)
- You want sub-100ms latency (local inference)
- You want zero API dependencies
- You're planning real auto-labeling (not just shadow mode)

### Full Stack: Layers 1–4 (8 Weeks)

**Week 1–2: Layer 1 — Baseline + Evaluation Harness**
- Fetch GitHub issues, clean labels, stratified split
- Groq zero-shot classification
- Precision/recall/F1 metrics — your baseline number

**Week 3–4: Layer 2 — Fine-Tuning**
- QLoRA fine-tune Qwen2.5-3B on your training set
- Grade with same eval harness (apples-to-apples comparison)
- See how much better your fine-tuned model is

**Week 5–6: Layer 3 — Quantization & Self-Hosting**
- Convert to GGUF, quantize to 4-bit
- Serve via Ollama locally
- LangGraph agent with ChromaDB dedup + confidence gating
- Sub-100ms predictions, zero API calls

**Week 7–8: Layer 4 — GitHub App + Shadow Mode**
- Register GitHub App, webhook verification
- Shadow-log every new issue prediction to Postgres
- Track agreement rate as maintainers label issues
- Langfuse tracing for observability
- Docker Compose for reproducible deployment

---

## Architecture

### Layer 1: Baseline

```
GitHub API
    ↓
fetch_closed_issues() → raw_issues.jsonl
    ↓
dataset_builder.py (clean, label map, stratified split)
    ↓
train.jsonl / val.jsonl / test.jsonl
    ↓
classifier.py (Groq Llama 3.3 70B zero-shot)
    ↓
eval/metrics.py (F1, precision, recall, confusion matrix)
```

**Key files:**
- `backend/services/github_fetcher.py` — pulls closed issues from GitHub
- `backend/services/classifier.py` — calls Groq for predictions
- `eval/metrics.py` — computes F1 and confusion matrix

**Time per issue:** ~800ms (network latency to Groq)
**Cost:** Free (Groq's free tier)
**Result:** Baseline F1, e.g. 0.63 macro

### Layer 2: Fine-Tuning

```
train.jsonl
    ↓
prepare_dataset.py (chat template formatting)
    ↓
train_lora.py (QLoRA, Qwen2.5-3B, 3 epochs)
    ↓
checkpoints/lora-adapter/
    ↓
merge_adapter.py → checkpoints/merged/
    ↓
eval/metrics.py (same harness, new model)
```

**Key files:**
- `training/prepare_dataset.py` — formats issues as chat examples
- `training/train_lora.py` — QLoRA training loop (SFTTrainer)
- `training/merge_adapter.py` — folds LoRA weights into base model

**Hardware:** Free T4 GPU on Colab (~90 min for 3 epochs)
**Result:** Fine-tuned model F1, e.g. 0.79 macro (outperforms baseline)

### Layer 3: Quantization & Serving

```
checkpoints/merged/
    ↓
quantize.sh (llama.cpp, Q4_K_M)
    ↓
checkpoints/issueradar-Q4_K_M.gguf (1.8GB)
    ↓
ollama create issuehawk -f serving/Modelfile
    ↓
backend/services/embeddings.py (ChromaDB semantic search)
    ↓
graph/triage_graph.py (LangGraph: dedup → classify → confidence gate)
    ↓
backend/routes/triage.py (POST /api/triage)
```

**Key files:**
- `serving/quantize.sh` — GGUF quantization
- `serving/Modelfile` — Ollama model definition
- `backend/services/embeddings.py` — ChromaDB duplicate detection
- `graph/triage_graph.py` — LangGraph agent with conditional edges
- `backend/routes/triage.py` — FastAPI endpoint

**Time per issue:** ~120ms (local GPU inference)
**Cost:** $0 (runs on your hardware)
**Result:** Same accuracy as Layer 2, but 6× faster, zero API calls

### Layer 4: GitHub App + Shadow Mode

```
GitHub Issue Opens
    ↓
POST /webhooks/github (HMAC verified)
    ↓
run_and_log_shadow_prediction() (async background task)
    ↓
triage_graph.py (full pipeline)
    ↓
Postgres: predictions table (predicted_label, confidence, duplicate_of, created_at)
    ↓
Langfuse trace (cost, latency, prompt/response)
    ↓
Later: maintainer labels issue
    ↓
Backfill: actual_label, agreed = (predicted_label == actual_label)
    ↓
Dashboard: agreement rate over time
```

**Key files:**
- `backend/routes/webhooks.py` — GitHub webhook handler + signature verification
- `backend/services/shadow_logger.py` — logs predictions to Postgres
- `backend/db/models.py` — Prediction table schema
- `docker-compose.yml` — app + Postgres + Ollama together

**Deployment:** Render, Fly.io, Railway (5 min setup)
**Shadow mode:** Predicts on real traffic, never posts back to GitHub
**Trust metric:** Agreement rate = (predictions that matched actual labels) / (total predictions)

---

## File Structure

```
issuehawk/
├── backend/
│   ├── main.py                      # FastAPI app
│   ├── config.py                    # settings + API keys
│   ├── routes/
│   │   ├── dataset.py               # POST /api/dataset/build
│   │   ├── baseline.py              # POST /api/baseline/predict
│   │   ├── eval.py                  # GET /api/eval/report
│   │   ├── triage.py                # POST /api/triage (Layer 3+)
│   │   └── webhooks.py              # POST /webhooks/github (Layer 4)
│   ├── services/
│   │   ├── github_fetcher.py        # fetches closed issues
│   │   ├── classifier.py            # Groq zero-shot
│   │   ├── serving_client.py        # Ollama local inference
│   │   ├── embeddings.py            # ChromaDB duplicate search
│   │   ├── shadow_logger.py         # Postgres logging
│   │   └── github_app_auth.py       # GitHub App JWT
│   ├── models/
│   │   └── schemas.py               # Pydantic models
│   └── db/
│       └── models.py                # SQLAlchemy Prediction table
├── data/
│   ├── raw/                         # raw API responses
│   ├── processed/                   # cleaned + labeled
│   └── splits/                      # train.jsonl / val.jsonl / test.jsonl
├── eval/
│   ├── metrics.py                   # F1, precision, recall, confusion matrix
│   └── results/                     # saved reports
├── training/                        # (added Layer 2)
│   ├── prepare_dataset.py           # chat template formatting
│   ├── train_lora.py                # QLoRA training
│   └── merge_adapter.py             # folds adapter into base
├── serving/                         # (added Layer 3)
│   ├── quantize.sh                  # GGUF conversion
│   └── Modelfile                    # Ollama definition
├── graph/                           # (added Layer 3)
│   └── triage_graph.py              # LangGraph agent
├── docker/                          # (added Layer 4)
│   ├── Dockerfile
│   └── docker-compose.yml
├── checkpoints/                     # .gitignored: LoRA, merged, GGUF models
├── .env                             # .gitignored: secrets
├── .gitignore
├── requirements.txt                 # Layer 1 deps
├── requirements-train.txt           # Layer 2+ deps
└── README.md                        # you are here
```

---

## Real Numbers

These are illustrative; your repo's numbers will differ based on issue volume, label consistency, and domain.

| Metric | Baseline (Groq 70B) | Fine-tuned (3B) | Notes |
|--------|---------------------|-----------------|-------|
| Macro F1 | 0.63 | 0.79 | Test set, held-out 20% of issues |
| Bug-class F1 | 0.81 | 0.88 | Majority class, easier |
| Duplicate-class F1 | 0.35 | 0.61 | Rare class, harder, bigger gain |
| Latency / issue | ~800ms | ~120ms | 6× faster |
| Cost / 1,000 issues | Free (Groq tier) | $0 | Runs on your hardware |
| Model size | 70B (remote) | 1.8GB quantized | Fits on a laptop |

---

## Measuring Success

### Week 1–2 (Layer 1)
**Success = you have a real F1 score.** Not "the model seems good," not "it classified 10 issues correctly." An F1 number from an evaluation harness on a held-out test set. This is your baseline.

### Week 3–4 (Layer 2)
**Success = your fine-tuned F1 beats the baseline.** Same test set, same metrics, apples-to-apples. If it doesn't, the data isn't there — go back to Layer 1 and ship shadow mode as-is.

### Week 5–6 (Layer 3)
**Success = quantized model F1 ≈ merged model F1 (within 1 point).** If quantization hurts accuracy noticeably, use Q5_K_M or Q8_0 instead.

### Week 7–8 (Layer 4)
**Success = you have real agreement-rate numbers.** After 100+ issues, your prediction agreement rate should be stable. If it's ≥80%, you're ready to *consider* real actions (comments first, labels later). If it's 60–75%, stay in shadow mode longer.

---

## How to Contribute

This project is designed for you to fork and customize. The intent is to learn and own every piece:

1. Fork the repo
2. Follow the Layer 1–4 documentation (see `docs/` folder)
3. Run on your own repo's data
4. Fine-tune if your data supports it
5. Deploy to your server or Render

---

## FAQ

**Q: Do I have to fine-tune?**
A: No. Layer 1 works forever. Fine-tune only if you have 1,000+ labeled issues and the baseline agreement rate is <75%.

**Q: Does it send data to external APIs?**
A: Layer 1 calls Groq (free tier). Layer 3+ is entirely local (Ollama). Layer 4 can optionally send traces to Langfuse for observability, but that's opt-in.

**Q: What if my repo doesn't have 1,000 labeled issues?**
A: Use Layer 1 only. Shadow-log real traffic, measure agreement rate, and collect data. Once you have enough labeled issues, Layer 2 makes sense.

**Q: Can I use this on multiple repos?**
A: Yes, but fine-tune a model for each repo. Each repo has different label patterns and issue types.

**Q: How long until I can actually auto-label?**
A: Shadow mode first (4 weeks). Measure agreement for 200+ issues (varies by volume). If ≥80%, start with comment suggestions (6 weeks). Auto-label only after 500+ high-agreement shadowed predictions.

**Q: What if the model is wrong?**
A: Shadow mode means nothing breaks. Log, measure, and iterate. The agreement rate tells you when it's trustworthy.

---

## Resources

- **Layer 1 Documentation** — `docs/IssueHawk_Layer1_Documentation.docx`
- **Layer 2 Documentation** — `docs/IssueHawk_Layer2_Documentation.docx`
- **Layer 3 Documentation** — `docs/IssueHawk_Layer3_Documentation.docx`
- **Layer 4 Documentation** — `docs/IssueHawk_Layer4_Documentation.docx`

---

## License

MIT

---

## Interview Talking Points

**"I built IssueHawk, a fine-tuned GitHub issue classifier that runs locally in shadow mode."**

Key numbers to have ready:
- Baseline F1 (Groq zero-shot)
- Fine-tuned F1 (your model)
- Real agreement rate on live traffic
- Latency / cost reduction

"The whole point is to measure trust with numbers, not demos. I shadow-logged predictions on a live repo for 3 weeks, measured agreement rate at 82%, and only then considered real actions. That discipline — evaluation first, action later — is what I want to bring to any team."

---

Built with PyTorch, PEFT/LoRA, TRL, llama.cpp, LangGraph, FastAPI, PostgreSQL, and Docker.
