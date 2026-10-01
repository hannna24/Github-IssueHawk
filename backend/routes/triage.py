# backend/routes/triage.py
from fastapi import APIRouter

from IssueHawk.graph.triage_graph import compiled_graph
from models.schemas import NewIssue, TriageResult

router = APIRouter()


@router.post('/api/triage', response_model=TriageResult)
async def triage_issue(issue: NewIssue):
    result = await compiled_graph.ainvoke({
        'title': issue.title, 'body': issue.body,
        'duplicate_of': None, 'label': None, 'confidence': None, 'route': None,
    })
    return TriageResult(**result)
