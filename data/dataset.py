"""
PyTorch Dataset for Equation-CLIP Training

Handles loading and batching of (equation, text) pairs.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EquationCLIPDataset(Dataset):
    """Dataset for Equation-CLIP training."""
    
    def __init__(
        self,
        data_file: str,
        tokenizer_name: str = 'allenai/scibert_scivocab_uncased',
        max_text_length: int = 512,
        equation_vocab_file: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSON file with equation-text pairs
            tokenizer_name: HuggingFace tokenizer name
            max_text_length: Maximum text sequence length
            equation_vocab_file: Path to equation vocabulary (optional)
        """
        self.data_file = data_file
        self.max_text_length = max_text_length
        
        # Load data
        logger.info(f"Loading dataset from {data_file}")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} equation-text pairs")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Build equation vocabulary
        self.equation_vocab = self._build_equation_vocab()
        
    def _build_equation_vocab(self) -> Dict[str, int]:
        """Build vocabulary for equation nodes."""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # Node types
        for node_type in ['operator', 'function', 'symbol', 'number']:
            if node_type not in vocab:
                vocab[node_type] = len(vocab)
        
        # Collect all node values from data
        for item in self.data:
            if 'operator_tree' in item:
                for node in item['operator_tree']['nodes']:
                    value = node['value']
                    if value not in vocab:
                        vocab[value] = len(vocab)
        
        logger.info(f"Built equation vocabulary: {len(vocab)} tokens")
        return vocab
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data item."""
        item = self.data[idx]
        
        # Text encoding
        text = item['description']
        text_encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        # Equation encoding (operator tree)
        equation_data = self._encode_equation(item)
        
        return {
            'equation_id': item.get('equation_id', item['id']),
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'equation_data': equation_data
        }
    
    def _encode_equation(self, item: Dict) -> Data:
        """Encode equation as PyG Data object."""
        if 'operator_tree' not in item:
            # Fallback: create dummy tree with self-loop
            return Data(
                node_types=torch.tensor([1], dtype=torch.long),
                node_values=torch.tensor([1], dtype=torch.long),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Self-loop
                num_nodes=1
            )
        
        tree = item['operator_tree']
        nodes = tree['nodes']
        
        # Encode node types and values
        node_types = []
        node_values = []
        
        for node in nodes:
            # Node type
            node_type_str = node['node_type']
            node_type_id = self.equation_vocab.get(node_type_str, 1)  # 1 = <UNK>
            node_types.append(node_type_id)
            
            # Node value
            node_value_str = node['value']
            node_value_id = self.equation_vocab.get(node_value_str, 1)
            node_values.append(node_value_id)
        
        # Build edge index
        edge_list = []
        for node in nodes:
            parent_id = node['node_id']
            for child_id in node['children']:
                edge_list.append([parent_id, child_id])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return Data(
            node_types=torch.tensor(node_types, dtype=torch.long),
            node_values=torch.tensor(node_values, dtype=torch.long),
            edge_index=edge_index,
            num_nodes=len(nodes)
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Text inputs (already tensors, just stack)
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    
    # Equation data (need PyG batching)
    equation_data_list = [item['equation_data'] for item in batch]
    equation_batch = Batch.from_data_list(equation_data_list)
    
    return {
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'equation_node_types': equation_batch.node_types,
        'equation_node_values': equation_batch.node_values,
        'equation_edge_index': equation_batch.edge_index,
        'equation_batch': equation_batch.batch
    }


def create_dataloaders(
    train_file: str,
    val_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = EquationCLIPDataset(train_file, **dataset_kwargs)
    val_dataset = EquationCLIPDataset(val_file, **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    logger.info("Testing EquationCLIPDataset...")
    
    # This will fail without actual data, but shows the interface
    try:
        dataset = EquationCLIPDataset(
            data_file='./dataset/train.json',
            tokenizer_name='allenai/scibert_scivocab_uncased'
        )
        logger.info(f"Dataset size: {len(dataset)}")
        
        # Test single item
        item = dataset[0]
        logger.info(f"Sample keys: {item.keys()}")
        logger.info(f"Text shape: {item['text_input_ids'].shape}")
        logger.info(f"Equation nodes: {item['equation_data'].num_nodes}")
        
    except FileNotFoundError:
        logger.warning("Dataset file not found - create data first with data/build_dataset.py")
