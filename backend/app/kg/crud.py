from sqlalchemy import select
from sqlalchemy.orm import Session

from app.kg.models import Entity, EntityMention, EntityRelation


def get_or_create_entity(
    db: Session,
    name: str,
    entity_type: str,
    external_source: str | None,
    external_id: str | None,
    aliases: list[str],
) -> Entity:
    existing = None
    if external_id:
        existing = db.execute(select(Entity).where(Entity.external_id == external_id)).scalar_one_or_none()
    if existing is None:
        existing = db.execute(
            select(Entity).where(Entity.name == name, Entity.entity_type == entity_type)
        ).scalar_one_or_none()
    if existing:
        return existing

    entity = Entity(
        name=name,
        entity_type=entity_type,
        external_source=external_source,
        external_id=external_id,
        aliases=aliases,
    )
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity


def get_entity(db: Session, entity_id: int) -> Entity | None:
    return db.get(Entity, entity_id)


def list_entities(db: Session, entity_type: str | None = None, q: str | None = None, limit: int = 100) -> list[Entity]:
    stmt = select(Entity)
    if entity_type:
        stmt = stmt.where(Entity.entity_type == entity_type)
    if q:
        stmt = stmt.where(Entity.name.ilike(f"%{q}%"))
    return list(db.execute(stmt.order_by(Entity.name).limit(limit)).scalars().all())


def add_mention(db: Session, article_id: int, entity_id: int, mention_text: str) -> tuple[EntityMention, bool]:
    existing = db.execute(
        select(EntityMention).where(
            EntityMention.article_id == article_id, EntityMention.entity_id == entity_id
        )
    ).scalar_one_or_none()
    if existing:
        return existing, False

    mention = EntityMention(article_id=article_id, entity_id=entity_id, mention_text=mention_text)
    db.add(mention)
    db.commit()
    db.refresh(mention)
    return mention, True


def get_article_entities(db: Session, article_id: int) -> list[Entity]:
    stmt = (
        select(Entity)
        .join(EntityMention, EntityMention.entity_id == Entity.id)
        .where(EntityMention.article_id == article_id)
        .order_by(Entity.name)
    )
    return list(db.execute(stmt).scalars().all())


def add_relation(
    db: Session, subject_entity_id: int, predicate: str, object_entity_id: int, evidence_article_id: int
) -> tuple[EntityRelation, bool]:
    existing = db.execute(
        select(EntityRelation).where(
            EntityRelation.subject_entity_id == subject_entity_id,
            EntityRelation.predicate == predicate,
            EntityRelation.object_entity_id == object_entity_id,
            EntityRelation.evidence_article_id == evidence_article_id,
        )
    ).scalar_one_or_none()
    if existing:
        return existing, False

    relation = EntityRelation(
        subject_entity_id=subject_entity_id,
        predicate=predicate,
        object_entity_id=object_entity_id,
        evidence_article_id=evidence_article_id,
    )
    db.add(relation)
    db.commit()
    db.refresh(relation)
    return relation, True


def get_entity_relations(db: Session, entity_id: int, limit: int = 100) -> list[EntityRelation]:
    stmt = (
        select(EntityRelation)
        .where(
            (EntityRelation.subject_entity_id == entity_id) | (EntityRelation.object_entity_id == entity_id)
        )
        .order_by(EntityRelation.created_at.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def article_ids_missing_extraction(db: Session, limit: int = 500) -> list[int]:
    """Articles not yet scanned for entities (Article.kg_extracted_at is
    unset) -- the extraction backfill target, same idea as
    ml.service.embed_missing. Keyed off the timestamp rather than "has no
    EntityMention row," since an article can legitimately match zero
    gazetteer entities and still count as processed.
    """
    from app.models import Article

    stmt = select(Article.id).where(Article.kg_extracted_at.is_(None)).limit(limit)
    return list(db.execute(stmt).scalars().all())
