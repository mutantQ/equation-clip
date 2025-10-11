"""
Equation Encoder for Equation-CLIP

Graph Neural Network (GNN) based encoder for mathematical equations
represented as operator trees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EquationGNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for equation operator trees.

    Architecture:
    - Node embeddings (operator type, value)
    - GCN/GAT layers for message passing
    - Global pooling for graph-level representation
    - MLP projection head
    """

    def __init__(
        self,
        node_vocab_size: int = 1000,
        hidden_dim: int = 512,
        num_layers: int = 3,
        gnn_type: str = 'gcn',  # 'gcn' or 'gat'
        dropout: float = 0.1,
        output_dim: int = 768,
        use_layer_norm: bool = True
    ):
        """
        Initialize equation encoder.

        Args:
            node_vocab_size: Size of node type vocabulary
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn' or 'gat')
            dropout: Dropout probability
            output_dim: Output embedding dimension
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.node_vocab_size = node_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.output_dim = output_dim

        # Node embeddings
        self.node_type_embedding = nn.Embedding(node_vocab_size, hidden_dim)
        self.node_value_embedding = nn.Embedding(node_vocab_size, hidden_dim)

        # Initial linear layer to combine embeddings
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.gnn_layers.append(
                    GCNConv(hidden_dim, hidden_dim)
                )
            elif gnn_type == 'gat':
                self.gnn_layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        dropout=dropout
                    )
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        logger.info(f"Initialized EquationGNNEncoder: {gnn_type.upper()}, "
                   f"{num_layers} layers, hidden_dim={hidden_dim}")

    def forward(
        self,
        node_types: torch.Tensor,
        node_values: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_types: Node type indices (num_nodes,)
            node_values: Node value indices (num_nodes,)
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch assignment for nodes (num_nodes,)

        Returns:
            Graph embeddings (batch_size, output_dim)
        """
        # Embed nodes
        type_embed = self.node_type_embedding(node_types)  # (num_nodes, hidden_dim)
        value_embed = self.node_value_embedding(node_values)  # (num_nodes, hidden_dim)

        # Combine embeddings
        x = torch.cat([type_embed, value_embed], dim=-1)  # (num_nodes, hidden_dim*2)
        x = self.input_proj(x)  # (num_nodes, hidden_dim)
        x = F.relu(x)

        # Apply GNN layers with residual connections
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_prev = x
            x = gnn_layer(x, edge_index)

            # Apply layer norm
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)

            # Activation and dropout
            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection (except first layer)
            if i > 0:
                x = x + x_prev

        # Global pooling
        if batch is None:
            # Single graph - mean pooling
            graph_embed = torch.mean(x, dim=0, keepdim=True)
        else:
            # Batch of graphs - use PyG pooling
            graph_embed = global_mean_pool(x, batch)

        # Output projection
        output = self.output_proj(graph_embed)  # (batch_size, output_dim)

        return output


class SequenceTransformerEncoder(nn.Module):
    """
    Baseline sequence-based transformer encoder.

    Treats LaTeX equation as a sequence of tokens.
    Simpler alternative to GNN for comparison.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 512,
        output_dim: int = 768
    ):
        """
        Initialize sequence transformer encoder.

        Args:
            vocab_size: LaTeX token vocabulary size
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_length: Maximum sequence length
            output_dim: Output embedding dimension
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Token embedding
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
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        logger.info(f"Initialized SequenceTransformerEncoder: "
                   f"{num_layers} layers, hidden_dim={hidden_dim}")

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            token_ids: Token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Sequence embeddings (batch_size, output_dim)
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings
        token_embed = self.token_embedding(token_ids)  # (batch, seq_len, hidden)

        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_embed = self.pos_embedding(positions)  # (1, seq_len, hidden)

        # Combine embeddings
        x = token_embed + pos_embed  # (batch, seq_len, hidden)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to float and invert (0 = attend, -inf = ignore)
            mask = (1.0 - attention_mask) * -10000.0
        else:
            mask = None

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, seq_len, hidden)

        # Mean pooling over sequence
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1)
            x_sum = (x * mask_expanded).sum(dim=1)
            x_mean = x_sum / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x_mean = x.mean(dim=1)

        # Output projection
        output = self.output_proj(x_mean)  # (batch, output_dim)

        return output


def create_pyg_data_from_tree(tree_dict: Dict) -> Data:
    """
    Convert operator tree dictionary to PyTorch Geometric Data object.

    Args:
        tree_dict: Dictionary containing tree structure

    Returns:
        PyG Data object
    """
    nodes = tree_dict['nodes']

    # Create node features (type and value as separate features)
    node_types = []
    node_values = []

    # Build vocabulary mapping on the fly (in practice, use pre-built vocab)
    type_vocab = {'operator': 0, 'function': 1, 'symbol': 2, 'number': 3}
    value_vocab = {}

    for node in nodes:
        node_type = type_vocab.get(node['node_type'], 0)
        node_types.append(node_type)

        # Add value to vocabulary
        value = node['value']
        if value not in value_vocab:
            value_vocab[value] = len(value_vocab)
        node_values.append(value_vocab[value])

    # Create edge index
    edge_list = []
    for node in nodes:
        parent_id = node['node_id']
        for child_id in node['children']:
            edge_list.append([parent_id, child_id])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create PyG Data object
    data = Data(
        node_types=torch.tensor(node_types, dtype=torch.long),
        node_values=torch.tensor(node_values, dtype=torch.long),
        edge_index=edge_index,
        num_nodes=len(nodes)
    )

    return data


if __name__ == "__main__":
    # Test the encoders
    logger.info("Testing Equation Encoders...")

    # Test GNN encoder
    gnn_encoder = EquationGNNEncoder(
        node_vocab_size=1000,
        hidden_dim=512,
        num_layers=3,
        gnn_type='gcn',
        output_dim=768
    )

    # Create dummy graph
    num_nodes = 10
    node_types = torch.randint(0, 100, (num_nodes,))
    node_values = torch.randint(0, 100, (num_nodes,))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    output = gnn_encoder(node_types, node_values, edge_index)
    logger.info(f"GNN output shape: {output.shape}")

    # Test sequence encoder
    seq_encoder = SequenceTransformerEncoder(
        vocab_size=5000,
        hidden_dim=512,
        num_layers=6,
        output_dim=768
    )

    batch_size = 2
    seq_len = 50
    token_ids = torch.randint(0, 5000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = seq_encoder(token_ids, attention_mask)
    logger.info(f"Sequence encoder output shape: {output.shape}")

    logger.info("✓ Encoder tests passed!")
