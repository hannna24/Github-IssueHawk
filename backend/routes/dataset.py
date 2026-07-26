from fastapi import APIRouter
from services.github_fetcher import fetch_closed_issues
from services.label_mapper import build_processed          # NEW
from services.dataset_builder import build_splits
from models.schemas import DatasetBuildRequest

router = APIRouter()

@router.post("/api/dataset/build")
async def build_dataset(req: DatasetBuildRequest):
    raw_path, count = fetch_closed_issues(req.repo, req.limit)
    processed_path, stats = build_processed(raw_path)        # NEW
    splits = build_splits(processed_path)                    # now gets 'label'
    return {
        "raw_path": raw_path,
        "fetched": count,
        "processed": stats,
        "splits": splits,
    }