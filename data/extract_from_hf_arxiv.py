"""
Extract equations from HuggingFace arxiv_papers_filtered dataset.
Fast way to get real equation-text pairs for training.
"""

import re
import json
from pathlib import Path
from typing import List, Dict
import logging
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_equations_from_text(text: str) -> List[Dict]:
    """
    Extract LaTeX equations and surrounding context from arXiv paper text.
    
    Args:
        text: Full text of arXiv paper
        
    Returns:
        List of {equation, context} dictionaries
    """
    equations = []
    
    # Pattern for display equations ($$...$$, \[...\], \begin{equation}...\end{equation})
    patterns = [
        r'\$\$(.*?)\$\$',
        r'\\\[(.*?)\\\]',
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            equation = match.group(1).strip()
            
            # Skip trivial equations
            if len(equation) < 5 or equation.isdigit():
                continue
            
            # Get surrounding context (±200 chars)
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            context = text[start:end]
            
            # Clean context
            context = re.sub(r'\s+', ' ', context).strip()
            
            equations.append({
                'equation': equation,
                'context': context,
                'position': match.start()
            })
    
    return equations


def process_dataset(
    num_papers: int = 10000,
    output_file: str = 'data/hf_arxiv_equations.jsonl'
):
    """
    Process HuggingFace arXiv dataset and extract equations.
    
    Args:
        num_papers: Number of papers to process
        output_file: Output JSON Lines file
    """
    logger.info(f"Loading HuggingFace arxiv_papers_filtered dataset...")
    
    # Load in streaming mode for memory efficiency
    dataset = load_dataset(
        "common-pile/arxiv_papers_filtered",
        split="train",
        streaming=True
    )
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_equations = 0
    papers_processed = 0
    
    with open(output_path, 'w') as f:
        for paper in tqdm(dataset, total=num_papers, desc="Processing papers"):
            if papers_processed >= num_papers:
                break
            
            text = paper['text']
            paper_id = paper.get('id', f'paper_{papers_processed}')
            
            # Extract equations
            equations = extract_equations_from_text(text)
            
            # Save each equation
            for eq_data in equations:
                entry = {
                    'id': f"{paper_id}_{eq_data['position']}",
                    'equation': eq_data['equation'],
                    'description': eq_data['context'],
                    'paper_id': paper_id,
                    'source': 'hf_arxiv',
                    'metadata': {
                        'created': str(paper.get('created', '')),
                        'position': eq_data['position']
                    }
                }
                f.write(json.dumps(entry) + '\n')
                total_equations += 1
            
            papers_processed += 1
            
            if papers_processed % 100 == 0:
                logger.info(f"Processed {papers_processed} papers, extracted {total_equations} equations")
    
    logger.info(f"✓ Extraction complete!")
    logger.info(f"Papers processed: {papers_processed}")
    logger.info(f"Equations extracted: {total_equations}")
    logger.info(f"Output: {output_path}")
    
    return total_equations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-papers', type=int, default=10000, help='Number of papers to process')
    parser.add_argument('--output', type=str, default='data/hf_arxiv_equations.jsonl')
    args = parser.parse_args()
    
    process_dataset(args.num_papers, args.output)
