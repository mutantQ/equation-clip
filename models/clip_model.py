"""
Equation-CLIP Model

Full model integrating equation encoder, text encoder, and contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

from .equation_encoder import EquationGNNEncoder, SequenceTransformerEncoder
from .text_encoder import SciTextEncoder, load_text_encoder
from .losses import CLIPLoss, compute_retrieval_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EquationCLIP(nn.Module):
    """
    Equation-CLIP: Contrastive Learning for Physics Equations.

    Architecture:
    - Equation Encoder: GNN or Sequence Transformer
    - Text Encoder: SciBERT/PhysBERT
    - Projection Heads: Map to shared embedding space
    - Contrastive Loss: CLIP-style InfoNCE
    """

    def __init__(
        self,
        # Equation encoder config
        equation_encoder_type: str = 'gnn',  # 'gnn' or 'sequence'
        equation_vocab_size: int = 1000,
        equation_hidden_dim: int = 512,
        equation_num_layers: int = 3,
        # Text encoder config
        text_encoder_type: str = 'scibert',  # 'scibert', 'physbert', 'simple'
        text_model_name: str = 'allenai/scibert_scivocab_uncased',
        text_hidden_dim: int = 768,
        freeze_text_layers: int = 6,
        # Shared config
        embedding_dim: int = 256,  # Final shared embedding dimension
        projection_hidden_dim: int = 2048,
        dropout: float = 0.1,
        # Loss config
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        """
        Initialize Equation-CLIP model.

        Args:
            equation_encoder_type: Type of equation encoder
            equation_vocab_size: Vocabulary size for equation nodes
            equation_hidden_dim: Hidden dimension for equation encoder
            equation_num_layers: Number of layers in equation encoder
            text_encoder_type: Type of text encoder
            text_model_name: Pre-trained model name for text encoder
            text_hidden_dim: Hidden dimension from text encoder
            freeze_text_layers: Number of text encoder layers to freeze
            embedding_dim: Final shared embedding dimension
            projection_hidden_dim: Hidden dimension in projection heads
            dropout: Dropout probability
            temperature: Temperature for contrastive loss
            learnable_temperature: Whether temperature is learnable
        """
        super().__init__()

        self.equation_encoder_type = equation_encoder_type
        self.text_encoder_type = text_encoder_type
        self.embedding_dim = embedding_dim

        # Initialize equation encoder
        if equation_encoder_type == 'gnn':
            self.equation_encoder = EquationGNNEncoder(
                node_vocab_size=equation_vocab_size,
                hidden_dim=equation_hidden_dim,
                num_layers=equation_num_layers,
                gnn_type='gcn',
                dropout=dropout,
                output_dim=equation_hidden_dim
            )
        elif equation_encoder_type == 'sequence':
            self.equation_encoder = SequenceTransformerEncoder(
                vocab_size=equation_vocab_size,
                hidden_dim=equation_hidden_dim,
                num_layers=equation_num_layers,
                dropout=dropout,
                output_dim=equation_hidden_dim
            )
        else:
            raise ValueError(f"Unknown equation encoder type: {equation_encoder_type}")

        # Initialize text encoder
        if text_encoder_type in ['scibert', 'physbert']:
            self.text_encoder = SciTextEncoder(
                model_name=text_model_name,
                hidden_dim=text_hidden_dim,
                output_dim=text_hidden_dim,
                freeze_layers=freeze_text_layers,
                dropout=dropout
            )
        else:
            self.text_encoder = load_text_encoder(
                model_type=text_encoder_type,
                output_dim=text_hidden_dim,
                freeze_layers=freeze_text_layers
            )

        # Projection heads to shared embedding space
        self.equation_projection = nn.Sequential(
            nn.Linear(equation_hidden_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim)
        )

        # Contrastive loss
        self.loss_fn = CLIPLoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature
        )

        self._log_model_info()

    def _log_model_info(self):
        """Log model configuration."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info("Equation-CLIP Model Initialized")
        logger.info("=" * 60)
        logger.info(f"Equation Encoder: {self.equation_encoder_type}")
        logger.info(f"Text Encoder: {self.text_encoder_type}")
        logger.info(f"Embedding Dimension: {self.embedding_dim}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info("=" * 60)

    def encode_equations(
        self,
        node_types: Optional[torch.Tensor] = None,
        node_values: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode equations to embeddings.

        Args:
            For GNN encoder:
                node_types: Node type indices
                node_values: Node value indices
                edge_index: Graph edges
                batch: Batch assignment
            For sequence encoder:
                token_ids: Token IDs
                attention_mask: Attention mask
            normalize: Whether to L2 normalize embeddings

        Returns:
            Equation embeddings (batch_size, embedding_dim)
        """
        if self.equation_encoder_type == 'gnn':
            assert node_types is not None and edge_index is not None
            hidden = self.equation_encoder(node_types, node_values, edge_index, batch)
        else:  # sequence
            assert token_ids is not None
            hidden = self.equation_encoder(token_ids, attention_mask)

        # Project to shared space
        embeddings = self.equation_projection(hidden)

        # Normalize
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def encode_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            normalize: Whether to L2 normalize embeddings

        Returns:
            Text embeddings (batch_size, embedding_dim)
        """
        # Encode text
        hidden = self.text_encoder(input_ids, attention_mask)

        # Project to shared space
        embeddings = self.text_projection(hidden)

        # Normalize
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def forward(
        self,
        equation_inputs: Dict[str, torch.Tensor],
        text_inputs: Dict[str, torch.Tensor],
        return_loss: bool = True,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            equation_inputs: Dictionary with equation encoder inputs
            text_inputs: Dictionary with text encoder inputs
            return_loss: Whether to compute and return loss
            return_embeddings: Whether to return embeddings

        Returns:
            Dictionary with loss and optionally embeddings
        """
        # Encode equations
        equation_embeds = self.encode_equations(**equation_inputs)

        # Encode texts
        text_embeds = self.encode_texts(**text_inputs)

        outputs = {}

        # Compute loss
        if return_loss:
            loss = self.loss_fn(equation_embeds, text_embeds)
            outputs['loss'] = loss

        # Return embeddings
        if return_embeddings:
            outputs['equation_embeds'] = equation_embeds
            outputs['text_embeds'] = text_embeds

        return outputs

    def get_similarity(
        self,
        equation_inputs: Dict[str, torch.Tensor],
        text_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity matrix between equations and texts.

        Args:
            equation_inputs: Equation encoder inputs
            text_inputs: Text encoder inputs

        Returns:
            Similarity matrix (batch_size, batch_size)
        """
        # Encode
        equation_embeds = self.encode_equations(**equation_inputs)
        text_embeds = self.encode_texts(**text_inputs)

        # Compute similarity
        similarity = equation_embeds @ text_embeds.T

        return similarity

    @torch.no_grad()
    def retrieve(
        self,
        query_text: str,
        equation_database: List[Dict],
        top_k: int = 10,
        device: Optional[torch.device] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-K equations given a text query.

        Args:
            query_text: Natural language query
            equation_database: List of equation dictionaries with inputs
            top_k: Number of results to return
            device: Device for computation

        Returns:
            List of (equation_index, similarity_score) tuples
        """
        if device is None:
            device = next(self.parameters()).device

        # Encode query text
        if hasattr(self.text_encoder, 'tokenizer'):
            tokenizer = self.text_encoder.get_tokenizer()
            encoded = tokenizer(
                [query_text],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            text_inputs = {
                'input_ids': encoded['input_ids'].to(device),
                'attention_mask': encoded['attention_mask'].to(device)
            }
        else:
            raise NotImplementedError("Text encoder must have tokenizer for retrieval")

        query_embed = self.encode_texts(**text_inputs)

        # Encode all equations in database
        # (In practice, pre-compute and cache these)
        equation_embeds = []
        for eq_dict in equation_database:
            eq_inputs = {k: v.to(device) for k, v in eq_dict.items()}
            eq_embed = self.encode_equations(**eq_inputs)
            equation_embeds.append(eq_embed)

        equation_embeds = torch.cat(equation_embeds, dim=0)

        # Compute similarities
        similarities = query_embed @ equation_embeds.T
        similarities = similarities.squeeze(0)

        # Get top-K
        top_k_values, top_k_indices = torch.topk(similarities, k=min(top_k, len(equation_database)))

        results = [
            (idx.item(), score.item())
            for idx, score in zip(top_k_indices, top_k_values)
        ]

        return results


def build_equation_clip(config: Dict) -> EquationCLIP:
    """
    Factory function to build Equation-CLIP model from config.

    Args:
        config: Configuration dictionary

    Returns:
        EquationCLIP model
    """
    model = EquationCLIP(
        equation_encoder_type=config.get('equation_encoder_type', 'gnn'),
        equation_vocab_size=config.get('equation_vocab_size', 1000),
        equation_hidden_dim=config.get('equation_hidden_dim', 512),
        equation_num_layers=config.get('equation_num_layers', 3),
        text_encoder_type=config.get('text_encoder_type', 'scibert'),
        text_model_name=config.get('text_model_name', 'allenai/scibert_scivocab_uncased'),
        text_hidden_dim=config.get('text_hidden_dim', 768),
        freeze_text_layers=config.get('freeze_text_layers', 6),
        embedding_dim=config.get('embedding_dim', 256),
        projection_hidden_dim=config.get('projection_hidden_dim', 2048),
        dropout=config.get('dropout', 0.1),
        temperature=config.get('temperature', 0.07),
        learnable_temperature=config.get('learnable_temperature', True)
    )

    return model


if __name__ == "__main__":
    # Test Equation-CLIP model
    logger.info("Testing Equation-CLIP Model...")

    # Configuration
    config = {
        'equation_encoder_type': 'gnn',
        'equation_vocab_size': 1000,
        'equation_hidden_dim': 512,
        'equation_num_layers': 3,
        'text_encoder_type': 'scibert',
        'text_model_name': 'allenai/scibert_scivocab_uncased',
        'text_hidden_dim': 768,
        'freeze_text_layers': 6,
        'embedding_dim': 256,
        'dropout': 0.1,
        'temperature': 0.07
    }

    # Build model
    model = build_equation_clip(config)

    # Create dummy inputs
    batch_size = 4

    # Equation inputs (GNN)
    num_nodes = 20
    equation_inputs = {
        'node_types': torch.randint(0, 100, (num_nodes,)),
        'node_values': torch.randint(0, 100, (num_nodes,)),
        'edge_index': torch.randint(0, num_nodes, (2, 30)),
        'batch': torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    }

    # Text inputs
    text_inputs = {
        'input_ids': torch.randint(0, 30000, (batch_size, 50)),
        'attention_mask': torch.ones(batch_size, 50)
    }

    # Forward pass
    outputs = model(equation_inputs, text_inputs, return_embeddings=True)

    logger.info(f"Loss: {outputs['loss'].item():.4f}")
    logger.info(f"Equation embeddings shape: {outputs['equation_embeds'].shape}")
    logger.info(f"Text embeddings shape: {outputs['text_embeds'].shape}")

    # Test retrieval metrics
    metrics = compute_retrieval_metrics(
        outputs['equation_embeds'],
        outputs['text_embeds']
    )
    logger.info(f"Retrieval metrics: {metrics}")

    logger.info("✓ Equation-CLIP model tests passed!")
