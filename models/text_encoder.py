"""
Text Encoder for Equation-CLIP

SciBERT/PhysBERT-based encoder for natural language descriptions
of equations.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SciTextEncoder(nn.Module):
    """
    Scientific text encoder using SciBERT or PhysBERT.

    Architecture:
    - Pre-trained scientific language model (SciBERT/PhysBERT)
    - Optional fine-tuning with layer freezing
    - Pooling strategy (CLS token or mean pooling)
    - MLP projection head
    """

    def __init__(
        self,
        model_name: str = 'allenai/scibert_scivocab_uncased',
        hidden_dim: int = 768,
        output_dim: int = 768,
        freeze_layers: int = 6,  # Freeze first N layers
        pooling_strategy: str = 'cls',  # 'cls' or 'mean'
        dropout: float = 0.1,
        max_length: int = 512
    ):
        """
        Initialize text encoder.

        Args:
            model_name: HuggingFace model name
                - 'allenai/scibert_scivocab_uncased' (SciBERT)
                - 'facebook/scibert_scivocab_uncased'
                - Custom PhysBERT model if available
            hidden_dim: Hidden dimension from pre-trained model
            output_dim: Output embedding dimension
            freeze_layers: Number of initial layers to freeze
            pooling_strategy: 'cls' or 'mean' pooling
            dropout: Dropout probability
            max_length: Maximum sequence length
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length

        # Load pre-trained model
        logger.info(f"Loading pre-trained model: {model_name}")
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze initial layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
            logger.info(f"Froze first {freeze_layers} layers")

        # Output projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        logger.info(f"Initialized SciTextEncoder with {model_name}")
        logger.info(f"Output dimension: {output_dim}, Pooling: {pooling_strategy}")

    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer layers."""
        # Freeze embeddings
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        if hasattr(self.transformer, 'encoder'):
            layers = self.transformer.encoder.layer
            for i in range(min(num_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_hidden_states: Whether to return intermediate hidden states

        Returns:
            Text embeddings (batch_size, output_dim)
        """
        # Pass through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states
        )

        # Get hidden states
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            pooled = last_hidden_state[:, 0, :]  # (batch, hidden)
        elif self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask  # (batch, hidden)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Apply projection head
        output = self.projection(pooled)  # (batch, output_dim)

        return output

    def encode_texts(
        self,
        texts: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode a list of text descriptions.

        Args:
            texts: List of text strings
            device: Device to use for computation

        Returns:
            Text embeddings (len(texts), output_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Encode
        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)

        return embeddings

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer


class SimpleTextEncoder(nn.Module):
    """
    Simple baseline text encoder without pre-training.

    Uses basic embeddings + transformer for comparison.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        output_dim: int = 768,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        """
        Initialize simple text encoder.

        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            output_dim: Output embedding dimension
            dropout: Dropout probability
            max_length: Maximum sequence length
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        logger.info(f"Initialized SimpleTextEncoder: {num_layers} layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Text embeddings (batch_size, output_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embed = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embed = self.pos_embedding(positions)

        # Combine
        x = token_embed + pos_embed

        # Transformer
        if attention_mask is not None:
            mask = (1.0 - attention_mask) * -10000.0
        else:
            mask = None

        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            x_sum = (x * mask_expanded).sum(dim=1)
            pooled = x_sum / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Project
        output = self.projection(pooled)

        return output


def load_text_encoder(
    model_type: str = 'scibert',
    output_dim: int = 768,
    freeze_layers: int = 6,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to load appropriate text encoder.

    Args:
        model_type: 'scibert', 'physbert', or 'simple'
        output_dim: Output embedding dimension
        freeze_layers: Number of layers to freeze
        device: Device to load model on

    Returns:
        Text encoder model
    """
    if model_type == 'scibert':
        model = SciTextEncoder(
            model_name='allenai/scibert_scivocab_uncased',
            output_dim=output_dim,
            freeze_layers=freeze_layers
        )
    elif model_type == 'physbert':
        # PhysBERT - check if available, fallback to SciBERT
        try:
            model = SciTextEncoder(
                model_name='physbert/physbert-base',  # Placeholder
                output_dim=output_dim,
                freeze_layers=freeze_layers
            )
        except:
            logger.warning("PhysBERT not available, using SciBERT")
            model = SciTextEncoder(
                model_name='allenai/scibert_scivocab_uncased',
                output_dim=output_dim,
                freeze_layers=freeze_layers
            )
    elif model_type == 'simple':
        model = SimpleTextEncoder(
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    # Test the text encoder
    logger.info("Testing Text Encoder...")

    # Test SciBERT encoder
    text_encoder = SciTextEncoder(
        model_name='allenai/scibert_scivocab_uncased',
        output_dim=768,
        freeze_layers=6
    )

    # Test encoding
    texts = [
        "This equation describes the wave equation in physics.",
        "Schrodinger equation governs quantum mechanical systems."
    ]

    embeddings = text_encoder.encode_texts(texts)
    logger.info(f"Text embeddings shape: {embeddings.shape}")

    # Test simple encoder
    simple_encoder = SimpleTextEncoder(output_dim=768)
    batch_size = 2
    seq_len = 50
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = simple_encoder(input_ids, attention_mask)
    logger.info(f"Simple encoder output shape: {output.shape}")

    logger.info("✓ Text encoder tests passed!")
