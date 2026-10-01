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


class NewIssue(BaseModel):
    title: str
    body: Optional[str] = None


class TriageResult(BaseModel):
    title: str
    body: Optional[str] = None
    duplicate_of: Optional[dict] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    route: Optional[str] = None
