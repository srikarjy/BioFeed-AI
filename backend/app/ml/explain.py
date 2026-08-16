"""v0.9: structured "recommended because..." explanations.

The v0.6 feed already attaches a one-line `reason` string to every item
(crud._generate_reason) -- cheap, single-signal (nearest interacted
article), good enough for a list view. This module is the deeper,
multi-signal version for a "why am I seeing this" tap-through: it surfaces
the actual signals behind a recommendation as structured data (label,
detail, a 0-1 weight) instead of a single opaque sentence, which is the
"genuinely underused differentiator" BLUEPRINT.md calls for -- most feeds
don't expose this at all.

Signals composed here:
  - nearest_interaction: most similar article the user has read/bookmarked/liked
  - topic_affinity / source_affinity: from app.ml.signals, same source used
    to train the v0.7 reranker (this is the explanation for the model, not
    a separate story)
  - freshness, popularity, source_quality: item-side signals
  - shared_entities: v0.8 knowledge-graph entities co-mentioned between the
    article and the user's recently-interacted articles -- the one signal
    that didn't exist before this pass, since the KG didn't exist either

No numpy/torch/lightgbm imports -- built entirely on app.ml.signals and
app.kg.crud, so it stays on the light import path (see signals.py).
"""

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import crud
from app.kg import crud as kg_crud
from app.ml.signals import (
    compute_freshness_days,
    compute_item_popularity,
    compute_source_quality,
    compute_user_affinities,
)
from app.models import Article


def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class ExplanationSignal:
    label: str
    detail: str
    weight: float  # 0-1, roughly "how much this signal should be trusted/weighted"


@dataclass
class Explanation:
    summary: str
    signals: list[ExplanationSignal] = field(default_factory=list)


def build_explanation(db: Session, user_id: int, article: Article) -> Explanation:
    signals: list[ExplanationSignal] = []

    # Nearest-interaction signal (same computation as crud._generate_reason,
    # surfaced here with its similarity score instead of collapsed to a
    # single sentence).
    interactions = crud.get_user_positive_interactions(db, user_id, limit=100)
    if interactions and article.embedding is not None:
        interacted_ids = [i.article_id for i in interactions]
        interacted_articles = (
            db.query(Article)
            .filter(Article.id.in_(interacted_ids), Article.embedding.isnot(None))
            .all()
        )
        best_sim, best_article, best_type = 0.0, None, None
        for ia in interacted_articles:
            if ia.embedding is None:
                continue
            sim = _cosine_similarity(article.embedding, ia.embedding)
            if sim > best_sim:
                best_sim = sim
                best_article = ia
                best_type = next(
                    (i.interaction_type for i in interactions if i.article_id == ia.id), None
                )
        if best_article is not None and best_sim > 0.3:
            signals.append(
                ExplanationSignal(
                    label="similar_to_interaction",
                    detail=f'{round(best_sim, 3)} cosine similarity to "{best_article.title[:80]}" ({best_type})',
                    weight=round(min(best_sim, 1.0), 3),
                )
            )

    # Topic/source affinity -- the same signal the v0.7 reranker trains on.
    topic_affinity, source_affinity = compute_user_affinities(db, user_id)
    source_aff = source_affinity.get(article.source, 0.0)
    if source_aff > 0:
        signals.append(
            ExplanationSignal(
                label="source_affinity",
                detail=f"{round(source_aff * 100, 1)}% of your positive interactions are with {article.source}",
                weight=round(source_aff, 3),
            )
        )

    # Item-side signals.
    freshness_days = compute_freshness_days(article.published_at)
    if freshness_days < 3:
        signals.append(
            ExplanationSignal(
                label="freshness",
                detail=f"published {round(freshness_days, 1)} days ago",
                weight=round(max(0.0, 1.0 - freshness_days / 3), 3),
            )
        )

    popularity = compute_item_popularity(db, article.id)
    if popularity > 0:
        signals.append(
            ExplanationSignal(
                label="popularity",
                detail=f"{popularity} other reader interaction(s) with this article",
                weight=round(min(popularity / 20, 1.0), 3),
            )
        )

    source_quality = compute_source_quality(article.source)
    if source_quality >= 0.7:
        signals.append(
            ExplanationSignal(
                label="source_quality",
                detail=f"{article.source} is a high-quality source (score {source_quality})",
                weight=source_quality,
            )
        )

    # v0.8 knowledge-graph signal: entities this article shares with articles
    # the user has recently interacted with.
    article_entities = {e.id: e for e in kg_crud.get_article_entities(db, article.id)}
    if article_entities and interactions:
        shared_names: dict[str, int] = {}
        for interaction in interactions[:20]:
            for entity in kg_crud.get_article_entities(db, interaction.article_id):
                if entity.id in article_entities:
                    shared_names[entity.name] = shared_names.get(entity.name, 0) + 1
        if shared_names:
            top = sorted(shared_names.items(), key=lambda kv: -kv[1])[:3]
            names = ", ".join(name for name, _ in top)
            signals.append(
                ExplanationSignal(
                    label="shared_entities",
                    detail=f"mentions {names}, which also appear in articles you've engaged with",
                    weight=round(min(sum(c for _, c in top) / 5, 1.0), 3),
                )
            )

    if not signals:
        return Explanation(summary="Recommended based on recent biotech coverage", signals=[])

    top_signal = max(signals, key=lambda s: s.weight)
    return Explanation(summary=top_signal.detail, signals=sorted(signals, key=lambda s: -s.weight))
