"""Orchestrates entity extraction and co-occurrence relation-building for
articles. See extractor.py for the matching logic and models.py for why
relations are a same-article co-occurrence heuristic, not verified fact.
"""

from datetime import datetime, timezone
from itertools import combinations

from sqlalchemy.orm import Session

from app.kg import crud
from app.kg.extractor import extract_entities
from app.kg.trials import TrialFetcher, extract_nct_ids, fetch_trial_title

# Predicate chosen by the (unordered) pair of entity types that co-occurred.
# Falls back to "co_mentioned_with" for any pair not listed (disease-disease,
# gene-gene, drug-drug, or any combination not judged to have an obvious
# directed reading). This is a hand-picked labeling heuristic over the
# co-occurrence signal, not an extracted or verified relation.
PREDICATE_BY_TYPE_PAIR: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"company", "drug"}): ("develops", "developed_by"),
    frozenset({"company", "disease"}): ("researches", "researched_by"),
    frozenset({"drug", "disease"}): ("targets", "targeted_by"),
    frozenset({"drug", "gene"}): ("targets", "targeted_by"),
    frozenset({"company", "gene"}): ("researches", "researched_by"),
    frozenset({"company", "trial"}): ("sponsors", "sponsored_by"),
    frozenset({"drug", "trial"}): ("evaluated_in", "evaluates"),
    frozenset({"disease", "trial"}): ("studied_in", "studies"),
}
DEFAULT_PREDICATE = ("co_mentioned_with", "co_mentioned_with")


def _predicate_for(type_a: str, type_b: str) -> tuple[str, str]:
    return PREDICATE_BY_TYPE_PAIR.get(frozenset({type_a, type_b}), DEFAULT_PREDICATE)


def _extract_trial_entities(db: Session, article, text: str, fetcher: TrialFetcher):
    """NCT ids mentioned in text -> trial Entity rows, via a live
    ClinicalTrials.gov lookup (only for NCT ids not already known locally --
    see get_entity_by_external_id). A lookup failure (network, 404,
    malformed response) skips that NCT id rather than creating a
    placeholder entity or failing the whole article's extraction.
    """
    entities = []
    for nct_id in extract_nct_ids(text):
        entity = crud.get_entity_by_external_id(db, nct_id)
        if entity is None:
            title = fetcher(nct_id)
            if title is None:
                continue
            entity = crud.get_or_create_entity(
                db, name=title, entity_type="trial",
                external_source="ClinicalTrials.gov", external_id=nct_id, aliases=[nct_id],
            )
        crud.add_mention(db, article.id, entity.id, nct_id)
        entities.append(entity)
    return entities


def extract_for_article(db: Session, article, trial_fetcher: TrialFetcher = fetch_trial_title) -> list:
    """Extract entities from one article's title+summary, persist mentions
    and pairwise co-occurrence relations, and mark the article processed.
    Idempotent: re-running on an already-processed article is a no-op
    (mentions/relations dedup on their unique constraints).

    `trial_fetcher` is injectable so tests don't depend on network access to
    ClinicalTrials.gov -- see tests/test_kg.py.
    """
    text = f"{article.title} {article.summary or ''}"
    matches = extract_entities(text)

    db_entities = []
    for match in matches:
        entity = crud.get_or_create_entity(
            db,
            name=match.entity.name,
            entity_type=match.entity.entity_type,
            external_source=match.entity.external_source,
            external_id=match.entity.external_id,
            aliases=list(match.entity.aliases),
        )
        crud.add_mention(db, article.id, entity.id, match.mention_text)
        db_entities.append(entity)

    db_entities.extend(_extract_trial_entities(db, article, text, trial_fetcher))

    for entity_a, entity_b in combinations(db_entities, 2):
        pred_a_to_b, pred_b_to_a = _predicate_for(entity_a.entity_type, entity_b.entity_type)
        crud.add_relation(db, entity_a.id, pred_a_to_b, entity_b.id, article.id)
        if pred_a_to_b != pred_b_to_a:
            crud.add_relation(db, entity_b.id, pred_b_to_a, entity_a.id, article.id)

    article.kg_extracted_at = datetime.now(timezone.utc)
    db.commit()
    return db_entities


def extract_missing(db: Session, limit: int = 500) -> int:
    """Backfill entity extraction for articles not yet processed. Returns
    the count of articles processed (not entities found -- an article with
    zero gazetteer matches still counts).
    """
    from app import crud as article_crud

    article_ids = crud.article_ids_missing_extraction(db, limit=limit)
    for article_id in article_ids:
        article = article_crud.get_article(db, article_id)
        if article is not None:
            extract_for_article(db, article)
    return len(article_ids)
