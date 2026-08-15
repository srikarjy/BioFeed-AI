"""Enhanced recommendation service for v0.7: Two-tower + LightGBM reranker.

Heavy deps (torch, lightgbm — requirements-ml.txt, not the light
requirements.txt CI installs) are imported lazily inside the methods that
need them, not at module level. That keeps `app.main` importable — and the
v0.7 route serving the v0.6 fallback feed — even where torch/lightgbm
aren't installed, matching the "v0.7 never 500s for lack of a model" design
(see PROJECT_STATUS.md decision #19, extended here to cover missing deps,
not just missing checkpoint files).

When both checkpoints ARE present, the process needs OMP_NUM_THREADS=1 (set
in the Dockerfile; export it yourself if running uvicorn directly outside
Docker) -- torch and lightgbm both bundle their own OpenMP runtime, and
loading both then calling into lightgbm without it segfaults on macOS. See
scripts/train_v07.py for the minimal repro.
"""

import os
from pathlib import Path
from typing import Optional

from app import crud
from app.models import Article


# Model paths. Resolved relative to this file (backend/app/ml/..), not a
# hardcoded absolute path, so it works on any machine; MODEL_DIR still
# overrides for deployments that store checkpoints elsewhere.
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "v0.7")))
TWO_TOWER_PATH = MODEL_DIR / "two_tower.pt"
RERANKER_PATH = MODEL_DIR / "reranker.txt"


class V07Recommender:
    """v0.7 recommender with two-tower retrieval + LightGBM reranking."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.two_tower = None
        self.reranker = None
        self._load_models()

    def _load_models(self):
        """Load trained models if available and their libraries are installed."""
        # Import lightgbm before torch gets imported below, even though it's
        # only used in the reranker block further down: on macOS both bundle
        # their own OpenMP runtime, and importing torch first makes lightgbm's
        # native training/predict calls segfault (SIGSEGV) the moment it
        # spins up its own OpenMP threads. See the identical note in
        # scripts/train_v07.py for a minimal repro.
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            pass

        # Load two-tower
        if TWO_TOWER_PATH.exists():
            try:
                import torch
                from app.ml.two_tower import TwoTowerModel

                checkpoint = torch.load(TWO_TOWER_PATH, map_location=self.device)
                config = checkpoint.get("config", {})
                self.two_tower = TwoTowerModel(
                    input_dim=config.get("input_dim", 768),
                    hidden_dims=config.get("hidden_dims", [512, 256]),
                    output_dim=config.get("output_dim", 128),
                )
                self.two_tower.load_state_dict(checkpoint["model_state_dict"])
                self.two_tower.to(self.device)
                self.two_tower.eval()
                print(f"Loaded two-tower model from {TWO_TOWER_PATH}")
            except ImportError:
                print("torch not installed; v0.7 two-tower retrieval unavailable, falling back to v0.6")
                self.two_tower = None
            except Exception as e:
                print(f"Failed to load two-tower model: {e}")
                self.two_tower = None
        else:
            print("Two-tower model not found, using fallback")

        # Load reranker
        if RERANKER_PATH.exists():
            try:
                from app.ml.reranker import LightGBMReranker

                self.reranker = LightGBMReranker(str(RERANKER_PATH))
                print(f"Loaded reranker from {RERANKER_PATH}")
            except ImportError:
                print("lightgbm not installed; v0.7 reranking unavailable, falling back to v0.6")
                self.reranker = None
            except Exception as e:
                print(f"Failed to load reranker: {e}")
                self.reranker = None
        else:
            print("Reranker model not found, using fallback")
    
    def get_recommendations(
        self,
        db,
        user_id: int,
        limit: int = 20,
        retrieval_k: int = 100,
        use_two_tower: bool = True,
        use_reranker: bool = True,
    ) -> list[tuple[Article, float, str]]:
        """Get enhanced recommendations for user.
        
        Pipeline:
        1. Two-tower retrieval (if available) or user embedding similarity
        2. LightGBM reranking (if available) or cosine similarity
        3. Return top-k with reasons
        
        Args:
            db: Database session
            user_id: User ID
            limit: Final number of recommendations
            retrieval_k: Number of candidates to retrieve before reranking
            use_two_tower: Whether to use two-tower for retrieval
            use_reranker: Whether to use LightGBM for reranking
            
        Returns:
            List of (article, score, reason)
        """
        # Only reached once _load_models() has already imported these
        # successfully (has_models gates the call in get_enhanced_feed), so
        # these imports are safe here.
        import numpy as np
        import torch
        from app.ml.reranker import extract_features, get_user_stats, compute_user_affinities

        # Get user embedding
        user_emb_db = crud.get_user_embedding(db, user_id)

        if user_emb_db is None or user_emb_db.embedding is None:
            # Cold start - fall back to v0.6 logic
            return crud.get_personalized_feed(db, user_id, limit, 0)

        user_emb_np = np.array(user_emb_db.embedding, dtype=np.float32)
        user_emb_torch = torch.from_numpy(user_emb_np).to(self.device)
        
        # Get hidden article IDs
        hidden_ids = crud.get_hidden_article_ids(db, user_id)
        
        # Get candidate articles with embeddings
        filters = [Article.embedding.isnot(None)]
        if hidden_ids:
            filters.append(~Article.id.in_(hidden_ids))
        
        candidates = db.query(Article).filter(*filters).all()
        
        if not candidates:
            return []
        
        # Stage 1: Retrieval
        if use_two_tower and self.two_tower is not None:
            candidate_embs = torch.tensor(
                np.stack([c.embedding for c in candidates]),
                dtype=torch.float32,
                device=self.device
            )
            
            with torch.no_grad():
                scores = self.two_tower.score_pairs(user_emb_torch, candidate_embs)
            
            scores_np = scores.cpu().numpy()
        else:
            # Fallback: cosine similarity with user embedding
            candidate_embs = np.stack([c.embedding for c in candidates])
            user_norm = np.linalg.norm(user_emb_np)
            candidate_norms = np.linalg.norm(candidate_embs, axis=1)
            scores_np = np.dot(candidate_embs, user_emb_np) / (candidate_norms * user_norm + 1e-8)
        
        # Get top retrieval_k candidates
        top_k_idx = np.argsort(scores_np)[-retrieval_k:][::-1]
        top_candidates = [candidates[i] for i in top_k_idx]
        top_scores = scores_np[top_k_idx]
        
        # Stage 2: Reranking
        if use_reranker and self.reranker is not None:
            # Extract features for each candidate
            user_stats = get_user_stats(db, user_id)
            topic_aff, source_aff = compute_user_affinities(db, user_id)
            
            feature_vectors = []
            for article, score in zip(top_candidates, top_scores):
                # Use the retrieval score as user_item_cosine
                feats = extract_features(
                    db, user_id, article, user_emb_np, 
                    np.array(article.embedding, dtype=np.float32),
                    float(score), float(np.dot(user_emb_np, article.embedding)),
                    topic_aff, source_aff, user_stats
                )
                feature_vectors.append(feats)
            
            if feature_vectors:
                X = np.stack(feature_vectors)
                rerank_scores = self.reranker.predict(X)
                
                # Combine retrieval + rerank scores (weighted)
                final_scores = 0.3 * top_scores + 0.7 * rerank_scores
            else:
                final_scores = top_scores
        else:
            final_scores = top_scores
        
        # Sort by final score
        sorted_idx = np.argsort(final_scores)[::-1]
        top_candidates = [top_candidates[i] for i in sorted_idx[:limit]]
        final_scores = final_scores[sorted_idx[:limit]]
        
        # Generate reasons
        results = []
        for article, score in zip(top_candidates, final_scores):
            reason = crud._generate_reason(db, user_id, article)
            results.append((article, float(score), reason))
        
        return results


# Global instance
_recommender: Optional[V07Recommender] = None


def get_recommender() -> V07Recommender:
    """Get or create the v0.7 recommender singleton."""
    global _recommender
    if _recommender is None:
        _recommender = V07Recommender()
    return _recommender


def get_enhanced_feed(
    db,
    user_id: int,
    limit: int = 20,
) -> tuple[list[tuple[Article, float, str]], bool]:
    """Enhanced feed endpoint for v0.7.
    
    Uses two-tower + reranker if models are available,
    falls back to v0.6 logic otherwise.
    """
    recommender = get_recommender()
    
    # Check if user has embedding
    user_emb = crud.get_user_embedding(db, user_id)
    if user_emb is None or user_emb.embedding is None:
        # Cold start
        return crud.get_personalized_feed(db, user_id, limit, 0)
    
    # Check if models are available
    has_models = recommender.two_tower is not None or recommender.reranker is not None
    
    if not has_models:
        # Fall back to v0.6
        return crud.get_personalized_feed(db, user_id, limit, 0)
    
    # Use enhanced pipeline
    items = recommender.get_recommendations(db, user_id, limit=limit)
    return items, False