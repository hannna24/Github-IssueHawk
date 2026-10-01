# backend/services/embeddings.py
"""Duplicate-detection memory: historical issues embedded once into ChromaDB,
each new issue embedded at query time and compared against them.

One-time setup (from backend/):
    python -m services.embeddings                  # index the train + val splits
    python -m services.embeddings --splits train val test   # everything (production)

The default leaves the test split OUT of the index, so evaluating on it never
finds an issue's own copy as a perfect "duplicate".
"""
import argparse
import json
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, DUPLICATE_THRESHOLD, EMBEDDING_MODEL, SPLITS_DIR


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Chroma defaults to squared-L2 distance. Similarity = 1 - distance only
    # means cosine similarity if the collection is created with the cosine metric.
    return client.get_or_create_collection('issues', metadata={'hnsw:space': 'cosine'})


def _text(title: str, body: str | None) -> str:
    return f'{title}\n{body or ""}'


def index_historical_issues(issues: list[dict], batch_size: int = 256) -> int:
    col = _collection()
    for i in range(0, len(issues), batch_size):
        chunk = issues[i:i + batch_size]
        col.upsert(   # upsert, so re-running the setup never errors on existing ids
            ids=[str(x['number']) for x in chunk],
            embeddings=_model().encode([_text(x['title'], x.get('body')) for x in chunk]).tolist(),
            metadatas=[{'label': x['label'], 'title': x['title']} for x in chunk],
        )
    return col.count()


def find_similar(title: str, body: str | None, k: int = 5) -> list[dict]:
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=_model().encode([_text(title, body)]).tolist(),
                    n_results=min(k, col.count()))
    return [
        {'issue_number': int(i), 'similarity': round(1 - d, 3),
         'label': m['label'], 'title': m['title']}
        for i, d, m in zip(res['ids'][0], res['distances'][0], res['metadatas'][0])
    ]


def find_duplicate(title: str, body: str | None, threshold: float = DUPLICATE_THRESHOLD) -> dict | None:
    top = find_similar(title, body, k=1)
    if top and top[0]['similarity'] >= threshold:
        return {'issue_number': top[0]['issue_number'], 'similarity': top[0]['similarity']}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    args = ap.parse_args()
    issues = []
    for name in args.splits:
        with open(SPLITS_DIR / f'{name}.jsonl', encoding='utf-8') as f:
            issues += [json.loads(line) for line in f]
    print(f'indexing {len(issues)} issues from {args.splits} ...')
    print('collection now holds', index_historical_issues(issues), 'issues')


if __name__ == '__main__':
    main()
