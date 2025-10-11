"""
Dataset Builder for Equation-CLIP Project

Creates train/val/test splits and prepares (equation, description) pairs
for contrastive learning.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build Equation-CLIP dataset with train/val/test splits."""

    # Physics domain categories (from arXiv categories)
    DOMAIN_MAPPING = {
        'physics.class-ph': 'Classical Mechanics',
        'physics.optics': 'Optics',
        'quant-ph': 'Quantum Mechanics',
        'cond-mat': 'Condensed Matter',
        'hep-th': 'High Energy Physics',
        'hep-ph': 'Particle Physics',
        'gr-qc': 'General Relativity',
        'math-ph': 'Mathematical Physics',
    }

    def __init__(self, train_ratio: float = 0.875, val_ratio: float = 0.0625, test_ratio: float = 0.0625):
        """
        Initialize dataset builder.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def create_description_from_context(self, equation_context: Dict) -> str:
        """
        Create natural language description from equation context.

        Combines context_before, context_after, and any available labels/captions.
        """
        parts = []

        # Add label/caption if available
        if equation_context.get('caption'):
            parts.append(equation_context['caption'])

        if equation_context.get('label'):
            parts.append(f"Labeled as: {equation_context['label']}")

        # Add context
        if equation_context.get('context_before'):
            parts.append(equation_context['context_before'])

        if equation_context.get('context_after'):
            parts.append(equation_context['context_after'])

        description = ' '.join(parts).strip()

        # Fallback if no context
        if not description:
            description = f"Mathematical equation from physics paper {equation_context['paper_id']}"

        return description

    def filter_quality(self, equation_pairs: List[Dict]) -> List[Dict]:
        """
        Filter equation-description pairs for quality.

        Criteria:
        - Description must have minimum length (10 words)
        - Equation must not be trivial
        - Context must mention physical concepts
        """
        filtered = []

        for pair in equation_pairs:
            description = pair['description']
            equation_latex = pair['equation_latex']

            # Check description length
            word_count = len(description.split())
            if word_count < 10:
                continue

            # Check description quality (should mention physics/math terms)
            physics_keywords = [
                'equation', 'force', 'energy', 'momentum', 'field', 'wave',
                'particle', 'quantum', 'classical', 'motion', 'potential',
                'hamiltonian', 'lagrangian', 'operator', 'function', 'system'
            ]
            description_lower = description.lower()
            has_physics_term = any(keyword in description_lower for keyword in physics_keywords)

            if not has_physics_term:
                continue

            filtered.append(pair)

        logger.info(f"Filtered {len(equation_pairs)} -> {len(filtered)} pairs")
        return filtered

    def deduplicate(self, equation_pairs: List[Dict]) -> List[Dict]:
        """Remove duplicate equation-description pairs."""
        seen = set()
        unique = []

        for pair in equation_pairs:
            # Use equation ID as key
            eq_id = pair['equation_id']
            if eq_id not in seen:
                seen.add(eq_id)
                unique.append(pair)

        logger.info(f"Deduplicated {len(equation_pairs)} -> {len(unique)} pairs")
        return unique

    def assign_domains(self, equation_pairs: List[Dict]) -> List[Dict]:
        """Assign physics domain labels to equation pairs."""
        for pair in equation_pairs:
            paper_id = pair['paper_id']

            # Try to infer domain from paper ID or metadata
            # This is a placeholder - in practice, use arXiv category metadata
            domain = 'Unknown'

            # If we have category metadata, map it
            for category, domain_name in self.DOMAIN_MAPPING.items():
                if category.replace('.', '_') in paper_id:
                    domain = domain_name
                    break

            pair['domain'] = domain

        return equation_pairs

    def stratified_split(self, equation_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create stratified train/val/test splits by physics domain.

        Args:
            equation_pairs: List of equation-description pairs

        Returns:
            Tuple of (train, val, test) lists
        """
        # Group by domain
        domain_groups = defaultdict(list)
        for pair in equation_pairs:
            domain = pair.get('domain', 'Unknown')
            domain_groups[domain].append(pair)

        # Split each domain
        train_data, val_data, test_data = [], [], []

        for domain, pairs in domain_groups.items():
            # Shuffle within domain
            random.shuffle(pairs)

            n = len(pairs)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            train_data.extend(pairs[:train_end])
            val_data.extend(pairs[train_end:val_end])
            test_data.extend(pairs[val_end:])

            logger.info(f"Domain {domain}: {len(pairs)} total, "
                       f"{len(pairs[:train_end])} train, "
                       f"{len(pairs[train_end:val_end])} val, "
                       f"{len(pairs[val_end:])} test")

        # Final shuffle
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        return train_data, val_data, test_data

    def build_dataset(self, equations_file: Path, trees_file: Path,
                     output_dir: Path) -> Dict[str, int]:
        """
        Build complete dataset with splits.

        Args:
            equations_file: Path to extracted equations JSON
            trees_file: Path to parsed operator trees JSON
            output_dir: Directory to save dataset splits

        Returns:
            Dictionary with dataset statistics
        """
        # Load data
        logger.info(f"Loading equations from {equations_file}")
        with open(equations_file, 'r') as f:
            equations = json.load(f)

        logger.info(f"Loading operator trees from {trees_file}")
        with open(trees_file, 'r') as f:
            trees = json.load(f)

        # Create lookup for trees
        tree_lookup = {tree['equation_id']: tree for tree in trees}

        # Create equation-description pairs
        logger.info("Creating equation-description pairs...")
        equation_pairs = []

        for eq in equations:
            eq_id = eq['equation_id']

            # Skip if no operator tree
            if eq_id not in tree_lookup:
                continue

            # Create description
            description = self.create_description_from_context(eq)

            # Create pair
            pair = {
                'equation_id': eq_id,
                'equation_latex': eq['equation_latex'],
                'canonical_latex': tree_lookup[eq_id]['canonical_latex'],
                'description': description,
                'paper_id': eq['paper_id'],
                'operator_tree': tree_lookup[eq_id],
                'metadata': {
                    'equation_type': eq['equation_type'],
                    'label': eq.get('label'),
                    'section': eq.get('section'),
                }
            }

            equation_pairs.append(pair)

        logger.info(f"Created {len(equation_pairs)} equation-description pairs")

        # Filter for quality
        equation_pairs = self.filter_quality(equation_pairs)

        # Deduplicate
        equation_pairs = self.deduplicate(equation_pairs)

        # Assign domains
        equation_pairs = self.assign_domains(equation_pairs)

        # Create splits
        logger.info("Creating train/val/test splits...")
        train, val, test = self.stratified_split(equation_pairs)

        # Save splits
        output_dir.mkdir(parents=True, exist_ok=True)

        train_file = output_dir / 'train.json'
        val_file = output_dir / 'val.json'
        test_file = output_dir / 'test.json'

        with open(train_file, 'w') as f:
            json.dump(train, f, indent=2)
        with open(val_file, 'w') as f:
            json.dump(val, f, indent=2)
        with open(test_file, 'w') as f:
            json.dump(test, f, indent=2)

        logger.info(f"Saved train split to {train_file} ({len(train)} pairs)")
        logger.info(f"Saved val split to {val_file} ({len(val)} pairs)")
        logger.info(f"Saved test split to {test_file} ({len(test)} pairs)")

        # Compute statistics
        stats = {
            'total_pairs': len(equation_pairs),
            'train_pairs': len(train),
            'val_pairs': len(val),
            'test_pairs': len(test),
            'num_domains': len(set(pair['domain'] for pair in equation_pairs)),
            'avg_description_length': np.mean([len(pair['description'].split()) for pair in equation_pairs]),
            'avg_equation_length': np.mean([len(pair['equation_latex']) for pair in equation_pairs]),
        }

        # Save statistics
        stats_file = output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset statistics saved to {stats_file}")
        logger.info(f"Total pairs: {stats['total_pairs']}")
        logger.info(f"Train: {stats['train_pairs']}, Val: {stats['val_pairs']}, Test: {stats['test_pairs']}")

        return stats


def main():
    """Main execution function."""
    # Configuration
    EQUATIONS_FILE = Path('./data/extracted_equations.json')
    TREES_FILE = Path('./data/equation_trees.json')
    OUTPUT_DIR = Path('./data/dataset')

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    builder = DatasetBuilder(
        train_ratio=0.875,
        val_ratio=0.0625,
        test_ratio=0.0625
    )

    logger.info("Building Equation-CLIP dataset...")
    stats = builder.build_dataset(EQUATIONS_FILE, TREES_FILE, OUTPUT_DIR)

    logger.info("Dataset build complete!")
    logger.info(f"Total pairs: {stats['total_pairs']}")


if __name__ == "__main__":
    main()
