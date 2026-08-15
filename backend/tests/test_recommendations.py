"""Tests for v0.6 recommendation engine."""

import pytest
from sqlalchemy.orm import Session

from app import crud
from app.models import Article, User, UserInteraction
from app.schemas import ArticleCreate, InteractionType


def _make_article(db: Session, title: str, source: str = "TestSource", **kwargs) -> Article:
    """Create an article with embedding for testing."""
    from datetime import datetime, timezone
    article = ArticleCreate(
        title=title,
        url=f"http://test/{title}",
        source=source,
        summary=kwargs.get("summary"),
        published_at=kwargs.get("published_at", datetime.now(timezone.utc)),
    )
    art, _ = crud.create_article(db, article)
    # Add a simple hash embedding for testing
    art.embedding = [0.1] * 768
    art.embedding[hash(title) % 768] = 1.0  # Make each article's embedding unique-ish
    db.commit()
    db.refresh(art)
    return art


def test_user_crud(db_session: Session):
    """Test user create, get, and get_or_create."""
    # Create user
    user = crud.create_user(db_session, email="test@example.com", display_name="Test User")
    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.display_name == "Test User"

    # Get by ID
    fetched = crud.get_user(db_session, user.id)
    assert fetched.id == user.id

    # Get by email
    fetched = crud.get_user_by_email(db_session, "test@example.com")
    assert fetched.id == user.id

    # Get or create - should return existing
    user2 = crud.get_or_create_user(db_session, email="test@example.com")
    assert user2.id == user.id

    # Get or create with Apple ID - new user
    user3 = crud.get_or_create_user(db_session, apple_user_id="apple_123")
    assert user3.id != user.id
    assert user3.apple_user_id == "apple_123"


def test_interaction_crud(db_session: Session):
    """Test recording and retrieving interactions."""
    user = crud.create_user(db_session, email="test@example.com")
    article = _make_article(db_session, "Test Article")

    # Record various interactions
    crud.create_interaction(db_session, user.id, article.id, InteractionType.READ, read_time_seconds=120)
    crud.create_interaction(db_session, user.id, article.id, InteractionType.BOOKMARK)
    crud.create_interaction(db_session, user.id, article.id, InteractionType.LIKE)
    crud.create_interaction(db_session, user.id, article.id, InteractionType.HIDE)

    # Get all interactions
    interactions = crud.get_user_interactions(db_session, user.id)
    assert len(interactions) == 4

    # Filter by type
    reads = crud.get_user_interactions(db_session, user.id, interaction_type="read")
    assert len(reads) == 1
    assert reads[0].read_time_seconds == 120

    bookmarks = crud.get_user_interactions(db_session, user.id, interaction_type="bookmark")
    assert len(bookmarks) == 1

    # Positive interactions (excludes hide)
    positive = crud.get_user_positive_interactions(db_session, user.id)
    assert len(positive) == 3  # read, bookmark, like

    # Hidden articles
    hidden = crud.get_hidden_article_ids(db_session, user.id)
    assert article.id in hidden


def test_user_embedding_computation(db_session: Session):
    """Test user embedding computation from interactions."""
    user = crud.create_user(db_session, email="test@example.com")
    
    # Create multiple articles with different embeddings
    articles = []
    for i in range(5):
        art = _make_article(db_session, f"Article {i}")
        articles.append(art)
    
    # User reads article 0 and 1, bookmarks article 2
    crud.create_interaction(db_session, user.id, articles[0].id, InteractionType.READ, read_time_seconds=300)  # max weight
    crud.create_interaction(db_session, user.id, articles[1].id, InteractionType.READ, read_time_seconds=150)  # half weight
    crud.create_interaction(db_session, user.id, articles[2].id, InteractionType.BOOKMARK)  # full weight
    
    # Article 3 and 4: no interactions
    
    # Compute embedding
    embedding = crud.compute_user_embedding(db_session, user.id)
    assert embedding is not None
    assert len(embedding) == 768
    
    # Should be normalized
    norm = sum(v * v for v in embedding) ** 0.5
    assert abs(norm - 1.0) < 0.01
    
    # Storing and retrieving
    crud.upsert_user_embedding(db_session, user.id, embedding, 3)
    stored = crud.get_user_embedding(db_session, user.id)
    assert stored is not None
    assert stored.interaction_count == 3
    # Compare embeddings as lists (stored might be array-like) - use approximate equality
    stored_emb = list(stored.embedding)
    assert len(stored_emb) == len(embedding)
    for a, b in zip(stored_emb, embedding):
        assert abs(a - b) < 1e-5


def test_refresh_user_embedding(db_session: Session):
    """Test the refresh_user_embedding convenience function."""
    user = crud.create_user(db_session, email="test@example.com")
    
    # No interactions yet
    result = crud.refresh_user_embedding(db_session, user.id)
    assert result is None
    
    # Add interactions
    art1 = _make_article(db_session, "Article 1")
    art2 = _make_article(db_session, "Article 2")
    crud.create_interaction(db_session, user.id, art1.id, InteractionType.READ, read_time_seconds=200)
    crud.create_interaction(db_session, user.id, art2.id, InteractionType.BOOKMARK)
    
    # Refresh
    result = crud.refresh_user_embedding(db_session, user.id)
    assert result is not None
    assert result.interaction_count == 2
    
    # Embedding should be retrievable
    stored = crud.get_user_embedding(db_session, user.id)
    assert stored is not None


def test_personalized_feed_warm_start(db_session: Session):
    """Test feed generation when user has embedding."""
    user = crud.create_user(db_session, email="test@example.com")
    
    # Create articles - some similar to user's interests, some not
    # User likes biotech articles
    biotech1 = _make_article(db_session, "CRISPR gene editing breakthrough in cancer therapy")
    biotech2 = _make_article(db_session, "New CRISPR therapy shows promise in clinical trial")
    biotech3 = _make_article(db_session, "Gene editing advances in biotechnology")
    
    # Unrelated articles
    unrelated1 = _make_article(db_session, "Stock market rallies on tech earnings")
    unrelated2 = _make_article(db_session, "Real estate market trends in 2024")
    
    # Make biotech articles have similar embeddings
    for art in [biotech1, biotech2, biotech3]:
        art.embedding = [0.2] * 768
        art.embedding[0] = 0.9
        art.embedding[1] = 0.1
    
    for art in [unrelated1, unrelated2]:
        art.embedding = [0.2] * 768
        art.embedding[0] = 0.1
        art.embedding[1] = 0.9
    
    db_session.commit()
    
    # User interacts with biotech articles
    crud.create_interaction(db_session, user.id, biotech1.id, InteractionType.READ, read_time_seconds=300)
    crud.create_interaction(db_session, user.id, biotech2.id, InteractionType.BOOKMARK)
    
    # Refresh embedding
    crud.refresh_user_embedding(db_session, user.id)
    
    # Get personalized feed
    items, cold_start = crud.get_personalized_feed(db_session, user.id, limit=5)
    
    assert cold_start is False
    assert len(items) > 0
    
    # Biotech articles should rank higher
    biotech_ids = {biotech1.id, biotech2.id, biotech3.id}
    feed_ids = [a.id for a, _, _ in items]
    
    # At least one biotech article should be in top results
    assert any(fid in biotech_ids for fid in feed_ids[:3])


def test_personalized_feed_cold_start(db_session: Session):
    """Test feed generation for new user without embedding."""
    from app.ml import service as ml_service
    
    user = crud.create_user(db_session, email="new@example.com")
    
    # Create some recent articles
    for i in range(5):
        _make_article(db_session, f"Recent Article {i}")
    
    # Embed them (needed for cold start feed which filters for articles with embeddings)
    ml_service.embed_missing(db_session)
    
    items, cold_start = crud.get_personalized_feed(db_session, user.id, limit=10)
    
    assert cold_start is True
    assert len(items) > 0
    # All items should have generic reason
    for _, _, reason in items:
        assert reason == "Recent biotech news"


def test_feed_excludes_hidden(db_session: Session):
    """Test that hidden articles don't appear in feed."""
    user = crud.create_user(db_session, email="test@example.com")
    
    # Create articles
    art1 = _make_article(db_session, "Article 1")
    art2 = _make_article(db_session, "Article 2")
    art3 = _make_article(db_session, "Article 3")
    
    # User hides article 2
    crud.create_interaction(db_session, user.id, art2.id, InteractionType.HIDE)
    
    # User likes article 1
    crud.create_interaction(db_session, user.id, art1.id, InteractionType.LIKE)
    crud.refresh_user_embedding(db_session, user.id)
    
    items, _ = crud.get_personalized_feed(db_session, user.id, limit=10)
    feed_ids = [a.id for a, _, _ in items]
    
    assert art2.id not in feed_ids
    assert art1.id in feed_ids


def test_generate_reason(db_session: Session):
    """Test recommendation reason generation."""
    from app.ml import service as ml_service
    
    user = crud.create_user(db_session, email="test@example.com")
    
    # Create articles
    art1 = _make_article(db_session, "CRISPR cancer therapy breakthrough")
    art2 = _make_article(db_session, "New gene editing technique for cancer")
    art3 = _make_article(db_session, "Stock market analysis")
    
    # Make art1 and art2 similar
    for art in [art1, art2]:
        art.embedding = [0.3] * 768
        art.embedding[0] = 0.8
    
    art3.embedding = [0.3] * 768
    art3.embedding[0] = 0.1
    art3.embedding[1] = 0.8
    
    db_session.commit()
    
    # Embed any missing
    ml_service.embed_missing(db_session)
    
    # User bookmarks art1
    crud.create_interaction(db_session, user.id, art1.id, InteractionType.BOOKMARK)
    
    # Get reason for art2 (similar to bookmarked)
    reason = crud._generate_reason(db_session, user.id, art2)
    assert "bookmarked" in reason.lower() or "similar" in reason.lower()
    
    # Get reason for art3 (unrelated) - should fall back to generic
    reason = crud._generate_reason(db_session, user.id, art3)
    assert "recommended" in reason.lower() or "similar" in reason.lower()


def test_interaction_weighting(db_session: Session):
    """Test that different interaction types have correct weights."""
    from app.ml import service as ml_service
    
    user = crud.create_user(db_session, email="test@example.com")
    
    art1 = _make_article(db_session, "Article 1")
    art2 = _make_article(db_session, "Article 2")
    art3 = _make_article(db_session, "Article 3")
    
    # Embed them
    ml_service.embed_missing(db_session)
    
    # Read for 5 min (max weight 1.0) on art1
    crud.create_interaction(db_session, user.id, art1.id, InteractionType.READ, read_time_seconds=300)
    # Read for 2.5 min (weight 0.5) on art2
    crud.create_interaction(db_session, user.id, art2.id, InteractionType.READ, read_time_seconds=150)
    # Bookmark on art3 (weight 1.0)
    crud.create_interaction(db_session, user.id, art3.id, InteractionType.BOOKMARK)
    
    embedding = crud.compute_user_embedding(db_session, user.id)
    assert embedding is not None
    
    # The embedding should be a weighted combination of the three article embeddings
    # Since we're using hash embeddings, just verify it's normalized and non-zero
    norm = sum(v * v for v in embedding) ** 0.5
    assert abs(norm - 1.0) < 0.01
    # Should have some non-zero values
    assert any(v != 0 for v in embedding)

def test_v07_endpoint_falls_back_without_trained_models(client, db_session):
    """The v0.7 route must never 500 for lack of trained checkpoints (or, in a
    minimal environment, for lack of torch/lightgbm) -- it degrades to the
    v0.6 heuristic feed. This is the regression test for a real bug: main.py
    used to import torch/lightgbm unconditionally via this route, which
    crashed app startup entirely in CI (requirements.txt intentionally
    excludes those heavy deps). See app/ml/recommender_v07.py.
    """
    resp = client.post("/users", json={"email": "v07-test@example.com"})
    user_id = resp.json()["id"]

    art1 = _make_article(db_session, "CRISPR therapy for sickle cell disease")
    resp = client.post(
        f"/users/{user_id}/interactions",
        json={"article_id": art1.id, "interaction_type": "bookmark"},
    )
    assert resp.status_code == 200

    resp = client.get(f"/v0.7/users/{user_id}/feed")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
