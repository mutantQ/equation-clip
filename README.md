# Equation-CLIP: Contrastive Learning for Physics Equations

A research project exploring contrastive learning for joint embeddings between mathematical equations and their natural language descriptions, inspired by OpenAI's CLIP.

## Project Overview

**Goal**: Learn a joint embedding space where:
- Equations describing similar physics are close together
- Equations are close to their natural language descriptions
- Semantic search is possible: text query → relevant equations

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd equation-clip

# Install dependencies
pip install -r requirements.txt
```

### Data Collection Pipeline

**Phase 1: Download arXiv Papers**

```bash
python data/download_arxiv.py
```

This downloads metadata for physics papers from arXiv. For LaTeX source files, you'll need to use arXiv bulk access.

**Phase 2: Extract Equations**

```bash
python data/extract_equations.py
```

Extracts equations and surrounding context from LaTeX source files.

**Phase 3: Parse to Operator Trees**

```bash
python data/parse_trees.py
```

Converts LaTeX equations to operator tree representations using SymPy.

**Phase 4: Build Dataset Splits**

```bash
python data/build_dataset.py
```

Creates train/val/test splits with quality filtering and stratification by physics domain.

## Project Structure

```
equation-clip/
├── data/                   # Data collection & preprocessing
│   ├── download_arxiv.py   # Download papers from arXiv
│   ├── extract_equations.py # Extract equations from LaTeX
│   ├── parse_trees.py      # Parse to operator trees
│   └── build_dataset.py    # Create dataset splits
├── models/                 # Model architectures (TODO)
├── training/              # Training scripts (TODO)
├── evaluation/            # Evaluation metrics (TODO)
├── applications/          # Demo applications (TODO)
├── research/              # Research documentation
└── notebooks/             # Jupyter notebooks (TODO)
```

## Research Documentation

Comprehensive research analysis available in:
- `instructions` - Original research plan
- `MASTER_RESEARCH_REPORT.md` - Complete technical analysis
- `research/` - Detailed literature reviews

## Current Status

**Phase 1: Data Collection** ✅ (In Progress)
- [x] arXiv download pipeline
- [x] LaTeX equation extraction
- [x] Operator tree parsing
- [x] Dataset builder
- [ ] Quality control & annotation

**Phase 2: Model Development** 🔄 (Next)
- [ ] Equation encoder (GNN)
- [ ] Text encoder (PhysBERT/SciBERT)
- [ ] Contrastive loss implementation

**Phase 3: Training** ⏳
- [ ] Training loop
- [ ] Curriculum learning
- [ ] Hard negative mining

**Phase 4: Evaluation** ⏳
- [ ] Retrieval metrics
- [ ] Zero-shot classification
- [ ] Semantic similarity

## GPU Requirements

This project requires GPU compute for training. Currently waiting for GPU availability on Hyperbolic.

**Recommended**: 4x A100 GPUs (40GB) or equivalent

## Expected Results

- Retrieval Recall@5: >70% (vs TangentCFT ~50%)
- Zero-shot classification: >75%
- Semantic similarity: Spearman ρ > 0.7

## Citation

This is a research project in development. Paper submission planned for NeurIPS/ICLR/ICML.

## License

[To be determined]
