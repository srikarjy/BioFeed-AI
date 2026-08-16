"""Feature engineering and LightGBM reranker for v0.7.

The reranker takes candidates from two-tower retrieval and scores them
with richer features for final ranking.
"""

import numpy as np
import lightgbm as lgb
from typing import Optional
from sqlalchemy.orm import Session

from app.ml.signals import (
    compute_source_quality,
    compute_freshness_days,
    compute_item_popularity,
    compute_user_affinities,
    get_user_stats,
)
from app.models import Article, User, UserInteraction, UserEmbedding
from app import crud


# Feature names for LightGBM (must match training)
FEATURE_NAMES = [
    # User-item interaction features
    "user_item_cosine",       # Cosine similarity in retrieval space
    "user_item_dot",          # Dot product (before normalization)
    # User features
    "user_interaction_count", # Total positive interactions
    "user_avg_read_time",     # Average read time in seconds
    "user_bookmark_rate",     # Fraction of interactions that are bookmarks
    "user_like_rate",         # Fraction that are likes
    # Item features
    "item_freshness_days",    # Days since publication
    "item_popularity",        # Number of interactions across all users
    "item_source_quality",    # Source quality score (0-1)
    "item_has_doi",           # Whether article has DOI
    "item_has_embedding",     # Whether article has embedding
    # Cross features
    "user_topic_affinity",    # User's affinity for article's dominant topic
    "user_source_affinity",   # User's affinity for article's source
]


def extract_features(
    db: Session,
    user_id: int,
    article: Article,
    user_emb: Optional[np.ndarray],
    item_emb: Optional[np.ndarray],
    user_item_cosine: float,
    user_item_dot: float,
    topic_affinity: dict[str, float],
    source_affinity: dict[str, float],
    user_stats: dict,
) -> np.ndarray:
    """Extract feature vector for a user-article pair."""
    
    # User features
    user_interaction_count = user_stats.get("interaction_count", 0)
    user_avg_read_time = user_stats.get("avg_read_time", 0.0)
    user_bookmark_rate = user_stats.get("bookmark_rate", 0.0)
    user_like_rate = user_stats.get("like_rate", 0.0)
    
    # Item features
    freshness = compute_freshness_days(article.published_at)
    popularity = compute_item_popularity(db, article.id)
    source_quality = compute_source_quality(article.source)
    has_doi = 1.0 if article.doi else 0.0
    has_embedding = 1.0 if article.embedding is not None else 0.0
    
    # Cross features
    user_topic_aff = topic_affinity.get(article.source, 0.0)
    user_source_aff = source_affinity.get(article.source, 0.0)
    
    features = np.array([
        user_item_cosine,
        user_item_dot,
        float(user_interaction_count),
        user_avg_read_time,
        user_bookmark_rate,
        user_like_rate,
        freshness,
        float(popularity),
        source_quality,
        has_doi,
        has_embedding,
        user_topic_aff,
        user_source_aff,
    ], dtype=np.float32)
    
    return features


class LightGBMReranker:
    """LightGBM reranker for v0.7.
    
    Takes candidate articles from retrieval and re-ranks them using
    richer features including user/item metadata and cross features.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[lgb.Booster] = None
        self.feature_names = FEATURE_NAMES
        
        if model_path:
            self.load(model_path)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMReranker":
        """Train the reranker.
        
        Args:
            X_train: (n_samples, n_features)
            y_train: (n_samples,) - binary labels (clicked=1) or relevance scores
            X_val, y_val: Validation set for early stopping
            params: LightGBM parameters
        """
        default_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 50,
            "max_depth": -1,
            "verbosity": -1,
            "seed": 42,
        }
        if params:
            default_params.update(params)
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # lightgbm >= 4.0 dropped the early_stopping_rounds/verbose_eval train()
        # kwargs in favor of callbacks.
        callbacks = [lgb.log_evaluation(period=100)]
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = lgb.Booster(model_file=path)
    
    def get_feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            return {}
        importance = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, importance.tolist()))


def create_reranker_training_data(
    db: Session,
    min_interactions: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create training data for reranker from historical interactions.
    
    For each user with enough interactions, sample positive (clicked) and
    negative (exposed but not clicked) examples.
    
    Returns:
        X: (n_samples, n_features)
        y: (n_samples,) - 1 for positive, 0 for negative
    """
    # Get users with enough interactions
    users = db.query(User).join(UserEmbedding).filter(
        UserEmbedding.interaction_count >= min_interactions
    ).all()
    
    X_list = []
    y_list = []
    
    for user in users:
        # Get user's positive interactions (clicked/bookmarked/liked)
        pos_interactions = crud.get_user_positive_interactions(db, user.id, limit=200)
        if not pos_interactions:
            continue
        
        pos_article_ids = {i.article_id for i in pos_interactions}
        
        # Get user's embedding
        user_emb_db = db.query(UserEmbedding).filter(UserEmbedding.user_id == user.id).first()
        if not user_emb_db or user_emb_db.embedding is None:
            continue
        
        user_emb = np.array(user_emb_db.embedding, dtype=np.float32)
        
        # Get user stats and affinities
        user_stats = get_user_stats(db, user.id)
        topic_aff, source_aff = compute_user_affinities(db, user.id)
        
        # Positive samples
        for interaction in pos_interactions:
            article = db.query(Article).filter(
                Article.id == interaction.article_id,
                Article.embedding.isnot(None)
            ).first()
            
            if not article or article.embedding is None:
                continue
            
            item_emb = np.array(article.embedding, dtype=np.float32)
            
            # Compute similarities
            cos_sim = float(np.dot(user_emb, item_emb) / (
                np.linalg.norm(user_emb) * np.linalg.norm(item_emb) + 1e-8
            ))
            dot_prod = float(np.dot(user_emb, item_emb))
            
            features = extract_features(
                db, user.id, article, user_emb, item_emb,
                cos_sim, dot_prod, topic_aff, source_aff, user_stats
            )
            
            X_list.append(features)
            y_list.append(1.0)
        
        # Negative samples: articles user was exposed to but didn't interact with
        # For simplicity, sample from recent articles not in positive set
        recent_articles = db.query(Article).filter(
            Article.embedding.isnot(None),
            ~Article.id.in_(pos_article_ids),
        ).order_by(Article.published_at.desc().nulls_last()).limit(200).all()
        
        # Sample same number of negatives as positives (1:1 ratio)
        import random
        random.shuffle(recent_articles)
        neg_articles = recent_articles[:len(pos_interactions)]
        
        for article in neg_articles:
            item_emb = np.array(article.embedding, dtype=np.float32)
            
            cos_sim = float(np.dot(user_emb, item_emb) / (
                np.linalg.norm(user_emb) * np.linalg.norm(item_emb) + 1e-8
            ))
            dot_prod = float(np.dot(user_emb, item_emb))
            
            features = extract_features(
                db, user.id, article, user_emb, item_emb,
                cos_sim, dot_prod, topic_aff, source_aff, user_stats
            )
            
            X_list.append(features)
            y_list.append(0.0)
    
    if not X_list:
        return np.array([]), np.array([])
    
    return np.stack(X_list), np.array(y_list)


if __name__ == "__main__":
    # Test feature extraction
    import os
    os.chdir("/Users/srikarjy/resume_projects/Biofeed-AI/backend")
    
    from app.database import SessionLocal
    db = SessionLocal()
    
    # Test with a sample article
    article = db.query(Article).filter(Article.embedding.isnot(None)).first()
    if article:
        print(f"Article: {article.title[:60]}...")
        print(f"Source quality: {compute_source_quality(article.source)}")
        print(f"Freshness: {compute_freshness_days(article.published_at)} days")
        print(f"Popularity: {compute_item_popularity(db, article.id)}")
        print(f"Has DOI: {article.doi is not None}")
    
    # Test user stats
    user_emb = db.query(UserEmbedding).first()
    if user_emb:
        stats = get_user_stats(db, user_emb.user_id)
        print(f"User stats: {stats}")
        topic_aff, source_aff = compute_user_affinities(db, user_emb.user_id)
        print(f"Topic affinity: {topic_aff}")
        print(f"Source affinity: {source_aff}")
    
    db.close()