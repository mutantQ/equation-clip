#!/usr/bin/env python3
"""
Clean and preprocess the arXiv dataset.

Issues to fix:
1. Very long sequences (10k+ tokens) - need to filter or truncate intelligently
2. Descriptions are paper fragments, not proper descriptions
3. Need to create synthetic descriptions for equations
"""

import json
import re
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_latex(text):
    """Clean LaTeX text to make it more readable."""
    # Remove common LaTeX commands that don't add meaning
    text = re.sub(r'\\hspace\{[^}]+\}', '', text)
    text = re.sub(r'\\vspace\{[^}]+\}', '', text)
    text = re.sub(r'\\renewcommand\{[^}]+\}\{[^}]+\}', '', text)
    text = re.sub(r'\\arraystretch\{[^}]+\}', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def extract_equation_type(equation):
    """Heuristically determine equation type for generating description."""
    eq = equation.lower()
    
    # Common patterns
    if 'frac' in eq or '/' in eq:
        if 'sum' in eq or 'int' in eq:
            return "integral or summation expression"
        return "fractional expression"
    
    if 'matrix' in eq or 'array' in eq or 'begin{' in eq:
        return "matrix or array expression"
    
    if 'sqrt' in eq:
        return "expression involving square roots"
    
    if any(x in eq for x in ['alpha', 'beta', 'gamma', 'delta', 'sigma', 'mu', 'lambda']):
        return "expression with Greek symbols"
    
    if any(x in eq for x in ['nabla', 'partial', 'grad', 'div']):
        return "differential or gradient expression"
    
    if '=' in eq:
        return "equation"
    
    return "mathematical expression"


def generate_description(equation, max_length=150):
    """Generate a reasonable description for an equation."""
    eq_clean = clean_latex(equation)
    eq_type = extract_equation_type(eq_clean)
    
    # If equation is short enough, include it in description
    if len(eq_clean) < 50:
        desc = f"Mathematical {eq_type}: {eq_clean}"
    else:
        # For long equations, describe structure
        desc = f"Complex mathematical {eq_type}"
        
        # Add some details
        if 'matrix' in eq_clean.lower() or 'array' in eq_clean.lower():
            desc += " with matrix or array components"
        if 'sum' in eq_clean.lower() or 'int' in eq_clean.lower():
            desc += " involving summation or integration"
        if any(x in eq_clean.lower() for x in ['partial', 'nabla', 'grad']):
            desc += " with differential operators"
    
    # Truncate if too long
    if len(desc) > max_length:
        desc = desc[:max_length-3] + "..."
    
    return desc


def clean_dataset(input_file, output_file, max_eq_length=500, max_desc_length=200):
    """Clean and process the dataset."""
    logger.info(f"Loading dataset from {input_file}")
    
    with open(input_file) as f:
        data = json.load(f)
    
    logger.info(f"Original dataset size: {len(data)}")
    
    cleaned_data = []
    skipped = 0
    
    for item in tqdm(data, desc="Cleaning"):
        equation = item.get('equation', '').strip()
        
        # Skip empty equations
        if not equation:
            skipped += 1
            continue
        
        # Clean equation
        equation = clean_latex(equation)
        
        # Skip extremely long equations
        if len(equation) > max_eq_length:
            # Try to extract a meaningful portion
            equation = equation[:max_eq_length]
        
        # Generate proper description
        description = generate_description(equation, max_length=max_desc_length)
        
        cleaned_item = {
            'id': item.get('id', f"cleaned_{len(cleaned_data)}"),
            'equation': equation,
            'description': description,
            'source': 'arxiv_cleaned',
            'original_source': item.get('source', 'unknown')
        }
        
        cleaned_data.append(cleaned_item)
    
    logger.info(f"Cleaned dataset size: {len(cleaned_data)}")
    logger.info(f"Skipped: {skipped}")
    
    # Save cleaned dataset
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    logger.info(f"Saved cleaned dataset to {output_file}")
    
    # Print sample
    if cleaned_data:
        logger.info("\nSample cleaned item:")
        logger.info(json.dumps(cleaned_data[0], indent=2))
    
    return len(cleaned_data)


def main():
    data_dir = Path("data/dataset")
    
    # Clean all splits
    for split in ['train', 'val', 'test']:
        input_file = data_dir / f"{split}_arxiv.json"
        output_file = data_dir / f"{split}_arxiv_cleaned.json"
        
        if input_file.exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {split} split")
            logger.info(f"{'='*60}")
            clean_dataset(input_file, output_file, max_eq_length=500, max_desc_length=200)
        else:
            logger.warning(f"File not found: {input_file}")
    
    # Create dataset info
    stats = {}
    for split in ['train', 'val', 'test']:
        cleaned_file = data_dir / f"{split}_arxiv_cleaned.json"
        if cleaned_file.exists():
            with open(cleaned_file) as f:
                data = json.load(f)
                stats[f'num_{split}'] = len(data)
    
    stats['total'] = sum(stats.values())
    stats['source'] = 'arxiv_cleaned'
    
    info_file = data_dir / 'dataset_info_arxiv_cleaned.json'
    with open(info_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nDataset statistics:")
    logger.info(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
