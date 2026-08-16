import hashlib
import re
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Article, IngestionRun, User, UserInteraction, UserEmbedding
from app.schemas import ArticleCreate, InteractionType

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize_title(title: str) -> str:
    """Collapse a title to a canonical form for cross-source matching.

    Lowercases, strips punctuation, and collapses whitespace so that the same
    paper surfaced by two sources with cosmetic differences hashes identically.
    """
    lowered = title.lower()
    stripped = _NON_ALNUM.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def content_hash_for(title: str) -> str:
    return hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()


def get_article(db: Session, article_id: int) -> Article | None:
    return db.get(Article, article_id)


def get_article_by_url(db: Session, url: str) -> Article | None:
    return db.execute(select(Article).where(Article.url == url)).scalar_one_or_none()


def get_article_by_doi(db: Session, doi: str) -> Article | None:
    return db.execute(select(Article).where(Article.doi == doi)).scalars().first()


def get_article_by_content_hash(db: Session, content_hash: str) -> Article | None:
    return (
        db.execute(select(Article).where(Article.content_hash == content_hash))
        .scalars()
        .first()
    )


def find_duplicate(db: Session, article: ArticleCreate, content_hash: str) -> Article | None:
    """Locate an existing article that represents the same content.

    Matching is tried from strongest to weakest signal: exact URL, then shared
    DOI (same paper, different landing page), then an identical normalized
    title (same paper from a source that exposes no DOI).
    """
    existing = get_article_by_url(db, article.url)
    if existing:
        return existing
    if article.doi:
        existing = get_article_by_doi(db, article.doi)
        if existing:
            return existing
    return get_article_by_content_hash(db, content_hash)


def get_articles(
    db: Session, source: str | None = None, limit: int = 50, offset: int = 0
) -> list[Article]:
    query = select(Article).order_by(Article.published_at.desc().nulls_last())
    if source:
        query = query.where(Article.source == source)
    query = query.limit(limit).offset(offset)
    return list(db.execute(query).scalars().all())


def create_article(db: Session, article: ArticleCreate) -> tuple[Article, bool]:
    """Get-or-create with cross-source dedup. Returns (article, created).

    Deduplicates on URL, DOI, and normalized-title hash (see find_duplicate).
    The URL column also carries a DB-level unique constraint as a safety net:
    two concurrent ingestion runs can both pass the existence check for the
    same URL before either commits. If that happens, the loser's commit hits
    the constraint; roll back and return the winner's row instead of leaving
    the session in an aborted-transaction state (which would break every
    subsequent query on this session).
    """
    content_hash = content_hash_for(article.title)
    existing = find_duplicate(db, article, content_hash)
    if existing:
        return existing, False

    db_article = Article(**article.model_dump(), content_hash=content_hash)
    db.add(db_article)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return get_article_by_url(db, article.url), False
    db.refresh(db_article)
    return db_article, True


def find_similar(
    db: Session,
    query_embedding: list[float],
    limit: int = 10,
    exclude_id: int | None = None,
) -> list[tuple[Article, float]]:
    """Nearest articles to a query vector, as (article, cosine similarity).

    On Postgres this is a pgvector `<=>` (cosine distance) query served by the
    HNSW index. The SQLite branch exists only for the test suite, which has no
    vector operators — it scans in Python, which is fine at test-fixture scale.
    """
    filters = [Article.embedding.is_not(None)]
    if exclude_id is not None:
        filters.append(Article.id != exclude_id)

    if db.get_bind().dialect.name == "postgresql":
        distance = Article.embedding.cosine_distance(query_embedding)
        rows = db.execute(
            select(Article, distance).where(*filters).order_by(distance).limit(limit)
        ).all()
        return [(article, 1.0 - float(dist)) for article, dist in rows]

    articles = db.execute(select(Article).where(*filters)).scalars().all()
    scored = [(a, _cosine_similarity(query_embedding, a.embedding)) for a in articles]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def record_ingestion_run(
    db: Session,
    started_at: datetime,
    added: dict[str, int],
    errors: dict[str, str],
    embedded: int = 0,
    entities_extracted: int = 0,
) -> IngestionRun:
    run = IngestionRun(
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        added_total=sum(added.values()),
        error_count=len(errors),
        detail={
            "added": added,
            "errors": errors,
            "embedded": embedded,
            "entities_extracted": entities_extracted,
        },
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_ingestion_runs(db: Session, limit: int = 20) -> list[IngestionRun]:
    return list(
        db.execute(
            select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(limit)
        )
        .scalars()
        .all()
    )


# v0.6: User and interaction CRUD

def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(select(User).where(User.email == email)).scalar_one_or_none()


def get_user_by_apple_id(db: Session, apple_user_id: str) -> User | None:
    return db.execute(select(User).where(User.apple_user_id == apple_user_id)).scalar_one_or_none()


def get_user_by_google_id(db: Session, google_user_id: str) -> User | None:
    return db.execute(select(User).where(User.google_user_id == google_user_id)).scalar_one_or_none()


def create_user(db: Session, email: str | None = None, apple_user_id: str | None = None, 
                google_user_id: str | None = None, display_name: str | None = None) -> User:
    user = User(
        email=email,
        apple_user_id=apple_user_id,
        google_user_id=google_user_id,
        display_name=display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_or_create_user(db: Session, email: str | None = None, apple_user_id: str | None = None,
                       google_user_id: str | None = None, display_name: str | None = None) -> User:
    """Get existing user or create new one. Matches by email, apple_user_id, or google_user_id."""
    if email:
        user = get_user_by_email(db, email)
        if user:
            return user
    if apple_user_id:
        user = get_user_by_apple_id(db, apple_user_id)
        if user:
            return user
    if google_user_id:
        user = get_user_by_google_id(db, google_user_id)
        if user:
            return user
    return create_user(db, email, apple_user_id, google_user_id, display_name)


def create_interaction(
    db: Session,
    user_id: int,
    article_id: int,
    interaction_type: InteractionType,
    read_time_seconds: int | None = None,
    search_query: str | None = None,
) -> UserInteraction:
    interaction = UserInteraction(
        user_id=user_id,
        article_id=article_id,
        interaction_type=interaction_type.value,
        read_time_seconds=read_time_seconds,
        search_query=search_query,
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


def get_user_interactions(
    db: Session, user_id: int, interaction_type: str | None = None, limit: int = 100
) -> list[UserInteraction]:
    query = select(UserInteraction).where(UserInteraction.user_id == user_id)
    if interaction_type:
        query = query.where(UserInteraction.interaction_type == interaction_type)
    query = query.order_by(UserInteraction.created_at.desc()).limit(limit)
    return list(db.execute(query).scalars().all())


def get_user_positive_interactions(db: Session, user_id: int, limit: int = 200) -> list[UserInteraction]:
    """Get interactions that indicate interest (read, bookmark, like). Excludes hides."""
    positive_types = ("read", "bookmark", "like")
    return list(
        db.execute(
            select(UserInteraction)
            .where(UserInteraction.user_id == user_id)
            .where(UserInteraction.interaction_type.in_(positive_types))
            .order_by(UserInteraction.created_at.desc())
            .limit(limit)
        ).scalars().all()
    )


def get_hidden_article_ids(db: Session, user_id: int) -> set[int]:
    """Get article IDs the user has explicitly hidden."""
    rows = db.execute(
        select(UserInteraction.article_id).where(
            UserInteraction.user_id == user_id,
            UserInteraction.interaction_type == "hide",
        )
    ).scalars().all()
    return set(rows)


# v0.6: User embedding computation and personalized feed

def compute_user_embedding(
    db: Session,
    user_id: int,
    min_interactions: int = 1,
) -> list[float] | None:
    """Compute user embedding from positive interactions.
    
    Uses weighted average of article embeddings:
    - bookmark/like: weight 1.0
    - read: weight proportional to read_time (capped at 5 min -> 1.0)
    
    Returns None if not enough positive interactions with embeddings.
    """
    interactions = get_user_positive_interactions(db, user_id, limit=500)
    if len(interactions) < min_interactions:
        return None

    # Get articles with embeddings
    article_ids = [i.article_id for i in interactions]
    articles = db.execute(
        select(Article).where(Article.id.in_(article_ids), Article.embedding.is_not(None))
    ).scalars().all()
    
    if not articles:
        return None

    # Build weight map
    weights = {}
    for interaction in interactions:
        if interaction.article_id not in weights:
            weights[interaction.article_id] = 0.0
        if interaction.interaction_type == "bookmark" or interaction.interaction_type == "like":
            weights[interaction.article_id] += 1.0
        elif interaction.interaction_type == "read" and interaction.read_time_seconds:
            # Cap at 5 minutes (300s) -> weight 1.0
            weight = min(interaction.read_time_seconds / 300.0, 1.0)
            weights[interaction.article_id] += weight

    # Weighted average of embeddings
    total_weight = 0.0
    weighted_sum = [0.0] * 768
    for article in articles:
        w = weights.get(article.id, 0.0)
        emb = article.embedding
        if w > 0 and emb is not None:
            # Handle both list and array-like embeddings
            try:
                emb_list = list(emb)
            except TypeError:
                continue
            for i, val in enumerate(emb_list):
                weighted_sum[i] += val * w
            total_weight += w

    if total_weight == 0.0:
        return None

    # Normalize
    embedding = [v / total_weight for v in weighted_sum]
    # L2 normalize
    norm = sum(v * v for v in embedding) ** 0.5
    if norm > 0:
        embedding = [v / norm for v in embedding]
    
    return embedding


def upsert_user_embedding(db: Session, user_id: int, embedding: list[float], interaction_count: int) -> UserEmbedding:
    """Create or update user embedding."""
    user_emb = db.get(UserEmbedding, user_id)
    if user_emb:
        user_emb.embedding = embedding
        user_emb.updated_at = datetime.now(timezone.utc)
        user_emb.interaction_count = interaction_count
    else:
        user_emb = UserEmbedding(
            user_id=user_id,
            embedding=embedding,
            interaction_count=interaction_count,
        )
        db.add(user_emb)
    db.commit()
    db.refresh(user_emb)
    return user_emb


def refresh_user_embedding(db: Session, user_id: int) -> UserEmbedding | None:
    """Recompute and store user embedding from interactions."""
    interactions = get_user_positive_interactions(db, user_id)
    embedding = compute_user_embedding(db, user_id)
    if embedding is None:
        return None
    return upsert_user_embedding(db, user_id, embedding, len(interactions))


def get_user_embedding(db: Session, user_id: int) -> UserEmbedding | None:
    return db.get(UserEmbedding, user_id)


def get_personalized_feed(
    db: Session,
    user_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[tuple[Article, float, str]], bool]:
    """Get personalized feed for user.
    
    Returns (articles with similarity and reason, cold_start_flag).
    
    If user has embedding: use it for retrieval + explainability.
    If not (cold start): fall back to recent articles + semantic search diversity.
    """
    hidden_ids = get_hidden_article_ids(db, user_id)
    
    # Try to get user embedding
    user_emb = get_user_embedding(db, user_id)
    
    if user_emb and user_emb.embedding is not None:
        # Personalized: use user embedding for retrieval
        results = find_similar(db, user_emb.embedding, limit=limit + len(hidden_ids) + 10, exclude_id=None)
        # Filter out hidden
        results = [(a, s) for a, s in results if a.id not in hidden_ids]
        results = results[:limit]
        
        # Add reasons
        feed_items = []
        for article, similarity in results:
            # Find which interaction led to this recommendation
            reason = _generate_reason(db, user_id, article)
            feed_items.append((article, similarity, reason))
        
        return feed_items, False
    
    else:
        # Cold start: recent articles from diverse sources + some semantic diversity
        return _cold_start_feed(db, user_id, hidden_ids, limit, offset), True


def _generate_reason(db: Session, user_id: int, article: Article) -> str:
    """Generate human-readable reason for recommendation."""
    # Check user's positive interactions with similar articles
    interactions = get_user_positive_interactions(db, user_id, limit=100)
    if not interactions:
        return "Recommended for you"
    
    # Find most similar article user interacted with
    user_articles = db.execute(
        select(Article).where(Article.id.in_([i.article_id for i in interactions]), Article.embedding.is_not(None))
    ).scalars().all()
    
    if not user_articles or article.embedding is None:
        return "Recommended for you"
    
    best_sim = 0.0
    best_article = None
    for ua in user_articles:
        if ua.embedding is not None:
            sim = _cosine_similarity(article.embedding, ua.embedding)
            if sim > best_sim:
                best_sim = sim
                best_article = ua
    
    if best_article and best_sim > 0.3:
        # Find interaction type
        for i in interactions:
            if i.article_id == best_article.id:
                if i.interaction_type == "bookmark":
                    return f"Similar to article you bookmarked: {best_article.title[:60]}..."
                elif i.interaction_type == "like":
                    return f"Similar to article you liked: {best_article.title[:60]}..."
                elif i.interaction_type == "read":
                    return f"Similar to article you read: {best_article.title[:60]}..."
    
    # Fallback: topic-based
    return "Recommended based on your reading interests"


def _cold_start_feed(
    db: Session,
    user_id: int,
    hidden_ids: set[int],
    limit: int,
    offset: int,
) -> list[tuple[Article, float, str]]:
    """Cold-start feed: recent articles from diverse sources."""
    # Get recent articles with embeddings, excluding hidden
    filters = [Article.embedding.is_not(None), ~Article.id.in_(hidden_ids)] if hidden_ids else [Article.embedding.is_not(None)]
    
    # Get recent articles (last 30 days) with some diversity
    from datetime import timedelta
    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    filters.append(Article.published_at >= recent_cutoff)
    
    articles = db.execute(
        select(Article).where(*filters).order_by(Article.published_at.desc().nulls_last()).limit(limit + offset)
    ).scalars().all()
    
    # Apply offset and limit
    articles = articles[offset:offset + limit]
    
    # Return with low similarity scores and generic reasons
    return [(a, 0.0, "Recent biotech news") for a in articles]
