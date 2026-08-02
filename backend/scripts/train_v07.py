"""Training pipeline for v0.7 models."""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.ml.two_tower import TwoTowerModel, create_training_data, train_two_tower
from app.ml.reranker import (
    create_reranker_training_data,
    LightGBMReranker,
    FEATURE_NAMES,
)


def train_two_tower_model(
    output_dir: str,
    min_interactions: int = 5,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Train two-tower retrieval model."""
    print("=" * 60)
    print("Training Two-Tower Retrieval Model")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # Create training pairs
        print("Creating training data...")
        pairs = create_training_data(db, min_interactions=min_interactions)
        print(f"Created {len(pairs)} positive pairs")
        
        if len(pairs) < 100:
            print("WARNING: Not enough training data. Need more user interactions.")
            return
        
        # Create model
        model = TwoTowerModel(
            input_dim=768,
            hidden_dims=[512, 256],
            output_dim=128,
            dropout=0.1,
        )
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        print("Training...")
        model = train_two_tower(
            model, pairs,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        
        # Save
        output_path = Path(output_dir) / "two_tower.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": 768,
                "hidden_dims": [512, 256],
                "output_dim": 128,
            }
        }, output_path)
        print(f"Saved model to {output_path}")
        
    finally:
        db.close()


def train_reranker_model(
    output_dir: str,
    min_interactions: int = 10,
    test_split: float = 0.2,
):
    """Train LightGBM reranker."""
    print("=" * 60)
    print("Training LightGBM Reranker")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        print("Creating training data...")
        X, y = create_reranker_training_data(db, min_interactions=min_interactions)
        
        if len(X) == 0:
            print("WARNING: Not enough training data for reranker.")
            return
        
        print(f"Training samples: {len(X)} (positive: {y.sum():.0f}, negative: {(1-y).sum():.0f})")
        print(f"Feature shape: {X.shape}")
        print(f"Features: {FEATURE_NAMES}")
        
        # Split
        n_train = int(len(X) * (1 - test_split))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Train
        reranker = LightGBMReranker()
        reranker.train(
            X_train, y_train,
            X_val, y_val,
            num_boost_round=500,
            early_stopping_rounds=50,
        )
        
        # Save
        output_path = Path(output_dir) / "reranker.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reranker.save(str(output_path))
        print(f"Saved reranker to {output_path}")
        
        # Feature importance
        importance = reranker.get_feature_importance("gain")
        print("\nFeature Importance (gain):")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.1f}")
        
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Train v0.7 models")
    parser.add_argument("--output-dir", default="models/v0.7", help="Output directory")
    parser.add_argument("--model", choices=["two-tower", "reranker", "both"], default="both")
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    if args.model in ["two-tower", "both"]:
        train_two_tower_model(
            args.output_dir,
            min_interactions=args.min_interactions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    
    if args.model in ["reranker", "both"]:
        train_reranker_model(
            args.output_dir,
            min_interactions=args.min_interactions,
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()