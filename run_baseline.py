import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from routes.baseline import predict_baseline
from routes.eval import get_eval_report

async def main():
    print("baseline over 3-class test split...", flush=True)
    print(json.dumps(await predict_baseline(), indent=2), flush=True)
    r = await get_eval_report()
    print(json.dumps(r["coverage"], indent=2), flush=True)
    print(f"accuracy {r['report']['accuracy']:.4f}  macroF1 {r['report']['macro avg']['f1-score']:.4f}  n={r['n']}")

asyncio.run(main())
