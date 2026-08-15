"""CI-enforced floor on the v0.5 retrieval eval set.

Full per-query breakdown (Recall@5/10, NDCG@10) is
`scripts/eval_retrieval.py` — run it directly for a human-readable report.
This test just guards against a regression silently tanking retrieval
quality on the labeled set.
"""

from scripts.eval_retrieval import _load_corpus, K_VALUES
from app import crud


def test_recall_at_10_meets_floor(db_session):
    queries, id_map, embedder = _load_corpus(db_session)

    hits = 0
    for q in queries:
        relevant_db_ids = {id_map[fid] for fid in q["relevant_ids"]}
        query_vector = embedder.embed_texts([q["query"]])[0]
        results = crud.find_similar(db_session, query_vector, limit=max(K_VALUES))
        ranked_ids = {article.id for article, _ in results}
        hits += int(bool(ranked_ids & relevant_db_ids))

    recall_at_10 = hits / len(queries)
    assert recall_at_10 >= 0.9, f"Recall@10 dropped to {recall_at_10:.3f} on the labeled eval set"
