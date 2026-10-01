import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Anchored to the repo root (the folder above backend/) so data and result files
# land in the same place no matter which directory the app was started from.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
EVAL_RESULTS_DIR = ROOT / "eval" / "results"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TARGET_REPO = os.getenv("TARGET_REPO", "vercel/next.js")
MAX_ISSUES = int(os.getenv("MAX_ISSUES", 4000))
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LABELS = ["bug", "feature-request", "documentation"]

# Layer 3: local serving, duplicate memory, routing thresholds
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "issuehawk")
CHROMA_DIR = ROOT / "data" / "chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", 0.88))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))