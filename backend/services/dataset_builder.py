import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import SPLITS_DIR

def build_splits(processed_path: str, out_dir: Path = SPLITS_DIR):
    out_dir = Path(out_dir)
    df = pd.read_json(processed_path, lines=True)
 
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp['label'], random_state=42
    )
 
    # to_json will not create missing folders, so make sure the target exists
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [('train', train), ('val', val), ('test', test)]:
        split.to_json(out_dir / f'{name}.jsonl', orient='records', lines=True)
 
    return {k: len(v) for k, v in [('train', train), ('val', val), ('test', test)]}
