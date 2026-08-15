"""Retrieval quality eval for GET /search (v0.5 remaining item).

Loads the hand-labeled query -> relevant-article set from
tests/fixtures/retrieval_eval.json into an in-memory SQLite corpus, embeds
everything with the configured EMBEDDING_BACKEND, runs crud.find_similar for
each query, and reports Recall@5, Recall@10, and NDCG@10.

This measures the embedding backend against a fixed, deliberately
multi-topic corpus, not the production database (which currently holds one
ingestion run's worth of ~65 real articles from 4 RSS feeds plus PubMed/
bioRxiv). It answers "does this embedding model actually separate topics"
in a way eyeballing /search results doesn't.

Usage:
    cd backend
    EMBEDDING_BACKEND=hash python scripts/eval_retrieval.py
    EMBEDDING_BACKEND=sentence-transformers python scripts/eval_retrieval.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Article
from app.ml.embeddings import get_embedder
from app import crud

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "retrieval_eval.json"
K_VALUES = (5, 10)


def _load_corpus(db):
    data = json.loads(FIXTURE_PATH.read_text())
    embedder = get_embedder()
    texts = [f"{row['title']} {row.get('summary') or ''}" for row in data["corpus"]]
    vectors = embedder.embed_texts(texts)

    id_map: dict[int, int] = {}  # fixture id -> real db id
    for row, vector in zip(data["corpus"], vectors):
        article = Article(
            title=row["title"],
            url=f"https://eval.local/{row['id']}",
            source=row["source"],
            summary=row.get("summary"),
            embedding=vector,
        )
        db.add(article)
        db.flush()
        id_map[row["id"]] = article.id
    db.commit()
    return data["queries"], id_map, embedder


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_at_k(ranked_relevant: list[bool], num_relevant: int, k: int) -> float:
    relevances = [1 if hit else 0 for hit in ranked_relevant[:k]]
    ideal = [1] * min(num_relevant, k)
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(relevances) / idcg


def main():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    queries, id_map, embedder = _load_corpus(db)

    recall_hits = {k: 0 for k in K_VALUES}
    ndcg_scores: list[float] = []

    print(f"{'query':<55} recall@5  recall@10  ndcg@10")
    print("-" * 90)

    for q in queries:
        relevant_db_ids = {id_map[fid] for fid in q["relevant_ids"]}
        query_vector = embedder.embed_texts([q["query"]])[0]
        results = crud.find_similar(db, query_vector, limit=max(K_VALUES))
        ranked_ids = [article.id for article, _ in results]

        row_recalls = {}
        for k in K_VALUES:
            top_k = set(ranked_ids[:k])
            hit = len(top_k & relevant_db_ids) > 0
            recall_hits[k] += int(hit)
            row_recalls[k] = hit

        ranked_relevant = [aid in relevant_db_ids for aid in ranked_ids]
        ndcg10 = _ndcg_at_k(ranked_relevant, len(relevant_db_ids), 10)
        ndcg_scores.append(ndcg10)

        print(
            f"{q['query'][:53]:<55} "
            f"{'hit' if row_recalls[5] else 'miss':<9}"
            f"{'hit' if row_recalls[10] else 'miss':<11}"
            f"{ndcg10:.3f}"
        )

    n = len(queries)
    print("-" * 90)
    print(f"Corpus size: {len(id_map)} articles, {n} labeled queries, backend={type(embedder).__name__}")
    for k in K_VALUES:
        print(f"Recall@{k}: {recall_hits[k] / n:.3f} ({recall_hits[k]}/{n})")
    print(f"NDCG@10 (mean): {sum(ndcg_scores) / n:.3f}")


if __name__ == "__main__":
    main()
