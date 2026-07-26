from fastapi import FastAPI
from routes import dataset, baseline, eval

app = FastAPI()

app.include_router(dataset.router)
app.include_router(baseline.router)
app.include_router(eval.router)

@app.get("/")
async def root():
    return {"message": "IssueHawk Layer 1 API"}