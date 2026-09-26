import json
from pathlib import Path

# Order matters: first match wins.
#
# 'duplicate' is deliberately NOT a class here. It is a cross-cutting status
# flag, not a content type -- maintainers apply "duplicate report" alongside
# the real label, so treating it as a class silently overwrote the content
# label on 49 of 56 issues (77% of them were bugs). It is captured as the
# separate is_duplicate field below and belongs to Layer 2's retrieval step,
# which scores "is this a re-report of #X" against the corpus.
LABEL_MAP = {
    "documentation": ["docs", "documentation"],
    #"question": ["usage question", "question"],
    "feature-request": ["enhancement", "feature request", "api design"],
    "bug": ["bug", "regression"],
}

DUPLICATE_VARIANTS = {"duplicate report", "duplicate"}


def is_duplicate(raw_labels: list[str]) -> bool:
    """Duplicate-ness is orthogonal to the content class -- kept for Layer 2."""
    return bool({l.strip().lower() for l in raw_labels} & DUPLICATE_VARIANTS)


def map_labels(raw_labels: list[str]) -> str | None:
    """Collapse a list of raw GitHub labels into one canonical label.
    Returns None if nothing matches (issue gets dropped)."""
    normalized = {l.strip().lower() for l in raw_labels}

    for canonical, variants in LABEL_MAP.items():
        if normalized & set(variants):
            return canonical
    return None


def build_processed(raw_path: str, out_dir: str = "../data/processed") -> tuple[str, dict]:
    """Read raw issues, attach a canonical 'label', drop unmappable ones."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    kept, dropped = [], 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            issue = json.loads(line)
            label = map_labels(issue.get("labels", []))
            if label is None:
                dropped += 1
                continue
            issue["label"] = label
            issue["is_duplicate"] = is_duplicate(issue.get("labels", []))
            kept.append(issue)

    out_path = f"{out_dir}/{Path(raw_path).stem}_processed.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for issue in kept:
            f.write(json.dumps(issue) + "\n")

    # Class distribution — check this every time, it tells you if the data is usable
    counts = {}
    for issue in kept:
        counts[issue["label"]] = counts.get(issue["label"], 0) + 1

    n_dup = sum(1 for i in kept if i["is_duplicate"])
    return out_path, {
        "kept": len(kept),
        "dropped_no_label": dropped,
        "class_counts": counts,
        "flagged_duplicate": n_dup,
    }