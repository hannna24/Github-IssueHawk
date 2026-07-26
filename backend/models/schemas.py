from pydantic import BaseModel
from typing import Optional

class DatasetBuildRequest(BaseModel):
    repo: str
    limit: int = 4000

class PredictionResult(BaseModel):
    label: str
    confidence: float
    reason: str

class EvalReport(BaseModel):
    macro_f1: float
    report: dict
    confusion_matrix: list