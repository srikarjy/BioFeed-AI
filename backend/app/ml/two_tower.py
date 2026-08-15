"""Two-tower retrieval model for v0.7.

Architecture:
- User Tower: User embedding → MLP → normalized user vector (d=128)
- Item Tower: Article embedding → MLP → normalized item vector (d=128)
- Score = dot(user_vec, item_vec)  (cosine similarity since normalized)

This replaces the simple user-embedding-as-query approach with a learned
retrieval model that can capture more complex user-item interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TwoTowerModel(nn.Module):
    """Two-tower retrieval model.
    
    Args:
        input_dim: Input embedding dimension (768 for PubMedBERT)
        hidden_dims: List of hidden layer sizes for each tower
        output_dim: Final embedding dimension (default 128)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list[int] = [512, 256],
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # User tower
        user_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        user_layers.append(nn.Linear(prev_dim, output_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item tower (separate params)
        item_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        item_layers.append(nn.Linear(prev_dim, output_dim))
        self.item_tower = nn.Sequential(*item_layers)
        
        # Temperature for scaling similarity (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)  # ln(14) ~ 2.659
    
    def encode_user(self, user_emb: torch.Tensor) -> torch.Tensor:
        """Encode user embedding to retrieval space."""
        user_vec = self.user_tower(user_emb)
        return F.normalize(user_vec, p=2, dim=-1)
    
    def encode_item(self, item_emb: torch.Tensor) -> torch.Tensor:
        """Encode article embedding to retrieval space."""
        item_vec = self.item_tower(item_emb)
        return F.normalize(item_vec, p=2, dim=-1)
    
    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity scores.
        
        Args:
            user_emb: (batch, 768) or (batch, 1, 768)
            item_emb: (batch, 768) or (batch, num_items, 768)
            
        Returns:
            Scores: (batch,) or (batch, num_items)
        """
        user_vec = self.encode_user(user_emb)  # (batch, 128)
        item_vec = self.encode_item(item_emb)  # (batch, 128) or (batch, num_items, 128)
        
        # Compute cosine similarity
        if item_vec.dim() == 3:
            # (batch, num_items, 128) @ (batch, 128, 1) -> (batch, num_items)
            scores = torch.bmm(item_vec, user_vec.unsqueeze(-1)).squeeze(-1)
        else:
            # (batch, 128) * (batch, 128) -> (batch,)
            scores = (user_vec * item_vec).sum(dim=-1)
        
        return scores * self.logit_scale.exp()
    
    def score_pairs(self, user_emb: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """Score one user against multiple items.
        
        Args:
            user_emb: (1, 768) or (768,)
            item_embs: (num_items, 768)
            
        Returns:
            scores: (num_items,)
        """
        # Use eval mode to avoid BatchNorm issues with batch size 1
        was_training = self.training
        self.eval()
        try:
            if user_emb.dim() == 1:
                user_emb = user_emb.unsqueeze(0)
            user_vec = self.encode_user(user_emb)  # (1, 128)
            item_vecs = self.encode_item(item_embs)  # (num_items, 128)
            
            scores = (item_vecs @ user_vec.T).squeeze(-1)  # (num_items,)
            return scores * self.logit_scale.exp()
        finally:
            if was_training:
                self.train()


class TwoTowerLoss(nn.Module):
    """In-batch negative contrastive loss for two-tower training.
    
    For each (user, positive_item) pair in batch, treats other items in batch
    as negatives. This is the standard InfoNCE loss used in retrieval.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        user_vecs: torch.Tensor,  # (batch, dim)
        item_vecs: torch.Tensor,  # (batch, dim)
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            user_vecs: Normalized user embeddings
            item_vecs: Normalized positive item embeddings (same batch size)
            
        Returns:
            Scalar loss
        """
        batch_size = user_vecs.size(0)
        
        # Similarity matrix: (batch, batch)
        logits = torch.matmul(user_vecs, item_vecs.T) / self.temperature
        
        # Labels: diagonal (each user matches its positive item)
        labels = torch.arange(batch_size, device=user_vecs.device)
        
        # Cross entropy loss (user->item and item->user)
        loss_u2i = F.cross_entropy(logits, labels)
        loss_i2u = F.cross_entropy(logits.T, labels)
        
        return (loss_u2i + loss_i2u) / 2


def create_training_data(
    db_session,
    min_interactions: int = 5,
    max_pairs_per_user: int = 100,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Create training pairs from user interactions.
    
    Returns:
        List of (user_emb, item_emb, label) where label=1 for positive,
        and we'll generate negatives during training.
    """
    from app import crud
    from app.models import Article, UserInteraction, UserEmbedding
    
    # Get users with embeddings
    user_embs = db_session.query(UserEmbedding).filter(
        UserEmbedding.interaction_count >= min_interactions
    ).all()
    
    training_pairs = []
    
    for ue in user_embs:
        user_id = ue.user_id
        user_emb = torch.tensor(ue.embedding, dtype=torch.float32)
        
        # Get positive interactions
        pos_interactions = crud.get_user_positive_interactions(
            db_session, user_id, limit=max_pairs_per_user
        )
        
        for interaction in pos_interactions:
            article = db_session.query(Article).filter(
                Article.id == interaction.article_id,
                Article.embedding.isnot(None)
            ).first()
            
            if article is not None and article.embedding is not None:
                item_emb = torch.tensor(article.embedding, dtype=torch.float32)
                training_pairs.append((user_emb, item_emb, 1.0))
    
    return training_pairs


def train_two_tower(
    model: TwoTowerModel,
    training_pairs: list[tuple[torch.Tensor, torch.Tensor, float]],
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TwoTowerModel:
    """Train two-tower model with in-batch negatives."""
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = TwoTowerLoss()
    
    # Convert to tensors
    user_embs = torch.stack([p[0] for p in training_pairs]).to(device)
    item_embs = torch.stack([p[1] for p in training_pairs]).to(device)
    
    dataset = torch.utils.data.TensorDataset(user_embs, item_embs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for user_batch, item_batch in loader:
            optimizer.zero_grad()
            
            # Forward: encode both towers
            user_vecs = model.encode_user(user_batch)
            item_vecs = model.encode_item(item_batch)
            
            # Compute loss
            loss = criterion(user_vecs, item_vecs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")
    
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test
    model = TwoTowerModel()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    user_emb = torch.randn(4, 768)
    item_emb = torch.randn(4, 768)
    scores = model(user_emb, item_emb)
    print(f"Scores shape: {scores.shape}")  # (4,)
    
    # Test score_pairs
    user_emb = torch.randn(768)
    items = torch.randn(10, 768)
    scores = model.score_pairs(user_emb, items)
    print(f"Pair scores shape: {scores.shape}")  # (10,)