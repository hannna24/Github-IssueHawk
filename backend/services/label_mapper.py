import json
from pathlib import Path

# Order matters: first match wins.
# 'duplicate' is checked first because a duplicate bug is still a duplicate.
LABEL_MAP = {
    "duplicate": ["duplicate report", "duplicate"],
    "documentation": ["docs", "documentation"],
    #"question": ["usage question", "question"],
    "feature-request": ["enhancement", "feature request", "api design"],
    "bug": ["bug", "regression"],
}


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
            kept.append(issue)

    out_path = f"{out_dir}/{Path(raw_path).stem}_processed.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for issue in kept:
            f.write(json.dumps(issue) + "\n")

    # Class distribution — check this every time, it tells you if the data is usable
    counts = {}
    for issue in kept:
        counts[issue["label"]] = counts.get(issue["label"], 0) + 1

    return out_path, {"kept": len(kept), "dropped_no_label": dropped, "class_counts": counts}