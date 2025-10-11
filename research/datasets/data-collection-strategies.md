# Data Collection Strategies for Equation-CLIP Dataset

**Research Date**: 2025-10-11
**Status**: Initial Research Complete
**Target Dataset Size**: 100K-500K equation-description pairs

---

## Table of Contents
1. [arXiv Data Access](#1-arxiv-data-access)
2. [LaTeX Parsing Tools](#2-latex-parsing-tools)
3. [Existing Physics & Equation Datasets](#3-existing-physics--equation-datasets)
4. [Data Extraction Pipelines](#4-data-extraction-pipelines)
5. [Data Quality & Annotation](#5-data-quality--annotation)
6. [Physics Domain Taxonomies](#6-physics-domain-taxonomies)
7. [Ethical Considerations](#7-ethical-considerations)
8. [Recommended Implementation Strategy](#8-recommended-implementation-strategy)

---

## 1. arXiv Data Access

### 1.1 arXiv API Capabilities

**Official Resources**:
- **API Documentation**: https://info.arxiv.org/help/api/index.html
- **Bulk Data Access**: https://info.arxiv.org/help/bulk_data.html
- **Python Package**: https://pypi.org/project/arxiv/

**Rate Limits**:
- **Standard API**: Maximum 1 request every 3 seconds, single connection at a time
- **Burst Rate**: 4 requests per second with 1-second sleep per burst (for bulk operations)
- **Important**: Rate limits apply to all machines under your control collectively
- **Contact Support**: If higher request rates are needed

**Best Practice**: Use dedicated site `export.arxiv.org` for programmatic harvesting - this is specifically set aside for bulk access.

### 1.2 Bulk Download Options

#### Amazon S3 (Recommended for Large-Scale)
- **Full Corpus Size**: ~9.2 TB (as of April 2025)
- **Growth Rate**: ~100 GB per month
- **Access**: https://info.arxiv.org/help/bulk_data_s3.html
- **Advantages**: Complete dataset access, no rate limits, parallel downloads
- **Cost**: AWS S3 egress fees apply

#### OAI-PMH Interface
- **Use Case**: Bulk metadata harvesting
- **Format**: Structured metadata without full text
- **Good For**: Building initial indices and category filtering

#### Source File Format
- Papers delivered as `.tar.gz` archives
- Contains LaTeX source files, figures, bibliography files
- Metadata available via arXiv API

### 1.3 Available Metadata

**Per Paper Metadata**:
- **Categories**: Primary and cross-listed (e.g., physics.comp-ph, math-ph)
- **Abstracts**: Natural language descriptions
- **Authors & Affiliations**
- **Submission/Update Dates**
- **DOI & Journal References**
- **Comments**: Often includes publication venue info

**Physics Categories Coverage**:
- All physics.* subcategories
- math-ph (Mathematical Physics)
- Approximately 1.5M+ papers in physics domains

---

## 2. LaTeX Parsing Tools

### 2.1 LaTeXML (Recommended)

**Overview**:
- Perl reimplementation of TeX parsing algorithm
- Converts LaTeX → XML/HTML/MathML
- Developed by NIST, actively maintained

**Links**:
- **Homepage**: https://dlmf.nist.gov/LaTeXML/
- **GitHub**: https://github.com/brucemiller/LaTeXML
- **Documentation**: https://math.nist.gov/~BMiller/LaTeXML/manual/

**Key Features**:
- **Semantic Preservation**: Conserves LaTeX semantic structures
- **MathML Output**: High-quality mathematical equation rendering
- **AST-like Structure**: Custom XML elements (XMath, XMApp, XMTok)
- **Two-Stage Process**:
  - `latexml`: .tex → XML representation
  - `latexmlpost`: XML → HTML5 with MathML

**Why LaTeXML for Equation-CLIP**:
- Most complete LaTeX parser available
- Produces structured representations ideal for building equation ASTs
- Handles complex TeX macros and custom commands
- Used successfully by arXiv Vanity for rendering papers

**Limitations**:
- Perl dependency (requires Perl runtime)
- Can be slow for large-scale processing
- May struggle with non-standard LaTeX packages

### 2.2 SymPy

**Overview**:
- Python library for symbolic mathematics
- Bidirectional LaTeX support (parse and generate)

**Links**:
- **Documentation**: https://docs.sympy.org/latest/modules/parsing.html
- **LaTeX Parsing**: https://docs.sympy.org/latest/modules/parsing.html
- **PyPI**: https://pypi.org/project/sympy/

**Key Features**:
- **Two Parser Backends**:
  - **ANTLR** (default): Mature, widely tested
  - **Lark** (newer): More features, better performance
- **Symbolic Operations**: Can manipulate and simplify equations
- **Canonicalization**: Convert equations to canonical forms for equivalence checking

**Code Example**:
```python
from sympy.parsing.latex import parse_latex

# Parse LaTeX to SymPy expression
expr = parse_latex(r'\frac{d}{dx} x^2 = 2x')

# Convert back to LaTeX
from sympy import latex
latex_str = latex(expr)

# Extract symbols
symbols = expr.free_symbols
```

**Supported Features**:
- Single letter symbols, Greek symbols, subscripts
- Basic operations: +, -, ×, ÷
- Implicit multiplication
- Matrices (with latex2sympy2 extension)

**Limitations**:
- Only parses mathematical expressions, not full documents
- Limited support for custom macros
- May fail on complex physics notation

### 2.3 plasTeX

**Overview**:
- Pure Python LaTeX document processing framework
- Produces XML-DOM-like object for manipulation

**Links**:
- **GitHub**: https://github.com/plastex/plastex
- **Documentation**: https://plastex.github.io/plastex/
- **PyPI**: https://pypi.org/project/plasTeX/

**Key Features**:
- **Separation**: Parsing and rendering completely separated
- **Extensible**: Easy to add custom renderers
- **Python Native**: No external dependencies
- **DOM Navigation**: Tree-like structure for document traversal

**Use Case**: Good for extracting context around equations (surrounding paragraphs, section headers)

### 2.4 tex2py / TexSoup

**Overview**:
- Lightweight Python LaTeX parser
- Navigate LaTeX as parse trees

**Links**:
- **tex2py**: https://github.com/alvinwan/tex2py
- **TexSoup**: https://github.com/alvinwan/TexSoup

**Key Features**:
- Simple API for parse tree navigation
- Custom hierarchy traversal
- Search and modification support
- Good for quick prototyping

**Code Example**:
```python
from TexSoup import TexSoup

soup = TexSoup(r'\section{Introduction} Here is $E=mc^2$.')
print(soup.find_all('section'))
print(soup.find_all('$'))  # Find inline math
```

**Use Case**: Rapid prototyping and simple extraction tasks

### 2.5 Tool Comparison Matrix

| Tool | Language | Best For | Speed | Equation AST | Full Doc |
|------|----------|----------|-------|--------------|----------|
| LaTeXML | Perl | Semantic preservation | Slow | Yes (XML) | Yes |
| SymPy | Python | Math operations | Fast | Yes (SymPy) | No |
| plasTeX | Python | Document structure | Medium | Limited | Yes |
| tex2py | Python | Quick extraction | Fast | Limited | Yes |

**Recommendation**: Use **LaTeXML for initial parsing** and **SymPy for equation canonicalization**.

---

## 3. Existing Physics & Equation Datasets

### 3.1 arXiv-Based Datasets

#### arXiv Dataset (Kaggle)
- **Size**: 1.7M+ scholarly papers
- **Coverage**: All STEM fields
- **Link**: https://www.kaggle.com/datasets/Cornell-University/arxiv
- **Format**: Metadata + abstracts
- **Use**: Category filtering, initial paper selection

#### "On the Use of ArXiv as a Dataset" (2019)
- **Paper**: https://arxiv.org/abs/1905.00075
- **Size**: 1.5M pre-print articles over 28 years
- **Citation Graph**: 6.7M edges
- **Corpus**: 11B words of full-text
- **Domains**: Physics, Mathematics, Computer Science

#### OpenWebMath
- **Paper**: https://arxiv.org/pdf/2310.06786
- **Source**: Common Crawl (HTML documents with math)
- **Size**: Large-scale (specific size in paper)
- **Format**: Text + equations (MathJax, LaTeX)
- **Coverage**: Math, Physics, CS, Statistics
- **Processing**: Specialized pipeline for LaTeX extraction

### 3.2 Equation Retrieval Datasets

#### NTCIR-12 MathIR
- **Homepage**: http://ntcir-math.nii.ac.jp/data/
- **Size**: 590,000+ mathematical formulas
- **Source**: Wikipedia
- **Queries**: 20 formula queries with relevance judgments
- **Task**: Formula retrieval and ranking
- **Baseline**: TangentCFT (state-of-the-art)

**Additional Collections**:
- **arXiv Subset**: 105,120 papers, 8.3M search units, 60M formulas
- **Use**: Benchmark for equation retrieval evaluation

#### ARQMath (CLEF 2020-2022)
- **Source**: Mathematics Stack Exchange
- **Tasks**: Answer retrieval, formula retrieval
- **Format**: Questions + answers with equations
- **Advantage**: Natural language descriptions of equations
- **Link**: CLEF conference proceedings

#### MIRB: Mathematical Information Retrieval Benchmark
- **Paper**: https://arxiv.org/html/2505.15585v1
- **Status**: Recent (2025)
- **Purpose**: Standardized benchmark for math IR
- **Multi-Source**: NTCIR, ARQMath, and custom datasets

### 3.3 Equation Image Datasets

#### im2latex-100k
- **Size**: 100,000 equation image-LaTeX pairs
- **Source**: arXiv papers
- **Link**: https://paperswithcode.com/dataset/im2latex-90k
- **Versions**:
  - im2latex-90k (cleaned)
  - im2latex-230k (extended with synthetic data)

#### im2latex-550k (Cleaned)
- **Size**: 550,000 formula-image pairs
- **Source**: ~1M LaTeX formulas from arXiv (cleaned)
- **Combined with**: im2latex-100k
- **Purpose**: Training image-to-LaTeX converters
- **Models**: Pix2Tex, TexTeller, Sumen

**Relevance to Equation-CLIP**:
- Could extract equation strings from images using pretrained models
- Provides alternative data source (equation images in papers)
- Demonstrates feasibility of large-scale equation extraction

### 3.4 Math Problem Datasets

#### MATH Dataset
- **Paper**: https://arxiv.org/abs/2103.03874
- **Size**: 12,500 competition math problems
- **Format**: Problem + step-by-step solutions
- **Coverage**: Competition-level mathematics
- **Auxiliary**: Large pretraining dataset included
- **Use**: Potentially extract equation reasoning pairs

#### Big-Math Dataset
- **Paper**: https://arxiv.org/html/2502.17387v1
- **Size**: 250,000+ problems
- **Domains**: Electromagnetic theory, thermodynamics, fluid mechanics
- **Format**: Problems with verifiable solutions
- **Physics Focus**: Directly relevant to our target domain

#### Google DeepMind Mathematics Dataset
- **GitHub**: https://github.com/google-deepmind/mathematics_dataset
- **Size**: 2M (question, answer) pairs per module
- **Coverage**: Algebra, arithmetic, numbers, polynomials
- **Format**: Text questions (≤160 chars), answers (≤30 chars)
- **Generated**: Synthetic data from templates

### 3.5 Textbook Sources

#### OpenStax
- **Homepage**: https://openstax.org/
- **Content**: Free college textbooks
- **Physics Coverage**: University Physics Vols 1-3, College Physics
- **Format**: HTML with MathML equations
- **License**: Creative Commons (CC BY)
- **Advantage**: High-quality curated content with clear explanations

**OpenStaxQA Dataset**:
- **Paper**: https://arxiv.org/html/2510.06239 (Oct 2024)
- **Size**: 43 textbooks (English, Spanish, Polish)
- **Focus**: End-of-chapter solved exercises
- **Code/Data**: Released on GitHub and Hugging Face
- **Equation-Description Pairs**: Problem-solution format ideal for our use case

#### MIT OpenCourseWare (OCW)
- **Homepage**: https://ocw.mit.edu/
- **Content**: Free course materials from MIT
- **Physics Coverage**: 8.01 Classical Mechanics, 8.04 Quantum Physics, etc.
- **Format**: Lecture notes, problem sets (PDFs and LaTeX source)
- **License**: Creative Commons (varies by course)

**Access Strategy**:
- Many OCW courses provide LaTeX source
- High-quality equation explanations in lecture notes
- Smaller scale but excellent quality

#### Wikipedia Physics Articles
- **Coverage**: Comprehensive physics encyclopedia
- **Equations**: MathML/LaTeX format
- **Descriptions**: Natural language explanations
- **License**: CC BY-SA
- **Extraction**: Use Wikipedia API or database dumps

---

## 4. Data Extraction Pipelines

### 4.1 Full Pipeline Architecture

```
┌─────────────────┐
│ arXiv S3 Bucket │
│  (Source Files) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Download .tar.gz│
│  LaTeX Sources  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract & Parse │
│   (LaTeXML)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Equation Extrac.│
│ (XMath Elements)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Context Extrac.  │
│(Surrounding Text)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Canonicalization│
│    (SymPy)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Filter  │
│  & Validation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Domain Labeling  │
│(arXiv Category) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Store in DB     │
│(Equation + Desc)│
└─────────────────┘
```

### 4.2 Existing Pipeline Implementations

#### arxiv-equations (vsoch)
- **GitHub**: https://github.com/vsoch/arxiv-equations
- **Features**:
  - Extracts equations, TeX, and metadata from .tar.gz
  - Handles multiple files per archive
  - Python-based
- **Script**: `testExtract.py`

#### arxiv_library (Whadup)
- **GitHub**: https://github.com/Whadup/arxiv_library/blob/master/equation-extractor
- **Features**:
  - Extracts tar archives of arXiv papers
  - Generates dictionaries with: citations, metadata, equations
  - Outputs: LaTeX AND MathML format
- **Advantage**: Dual format output useful for different encoders

#### arxiv-latex-extract (potamides)
- **GitHub**: https://github.com/potamides/arxiv-latex-extract
- **Focus**: Bulk extraction from arXiv.org archives
- **Source**: Uses archive.org mirror
- **Scale**: Designed for processing entire arXiv corpus

### 4.3 Equation-Description Pair Identification Strategies

#### Strategy 1: Equation Labels and References
**Method**: Parse LaTeX to find `\label{eq:...}` and corresponding `\ref{eq:...}` or `\eqref{eq:...}`

**Example**:
```latex
The energy-momentum relation is given by
\begin{equation}
E^2 = (pc)^2 + (mc^2)^2
\label{eq:energy_momentum}
\end{equation}
From Equation~\ref{eq:energy_momentum}, we see that...
```

**Extraction**: Text mentioning equation reference becomes description.

**Pros**: High precision, equations explicitly discussed
**Cons**: Not all equations are labeled/referenced

#### Strategy 2: Surrounding Sentence Context
**Method**: Extract N sentences before/after equation

**Heuristics**:
- Sentences containing: "is given by", "can be written as", "follows from"
- Preceding sentences introducing variables
- Following sentences explaining physical meaning

**Example**:
```latex
The Schrödinger equation describes the time evolution of a quantum state.
\begin{equation}
i\hbar \frac{\partial}{\partial t} \Psi = \hat{H} \Psi
\end{equation}
Here, $\Psi$ is the wave function and $\hat{H}$ is the Hamiltonian operator.
```

**Extraction**: Both preceding and following sentences form description.

**Pros**: Captures context even without labels
**Cons**: May include irrelevant sentences

#### Strategy 3: Section/Subsection Headers
**Method**: Associate equations with their section context

**Extraction**:
- Section title: "Maxwell's Equations"
- All equations in that section inherit domain label
- Combine section title + local context

**Pros**: Provides high-level semantic information
**Cons**: Less specific than sentence-level context

#### Strategy 4: Abstract Similarity
**Method**: For important equations, check if they appear in abstract

**Rationale**: Equations in abstract are usually key results with good descriptions

**Implementation**:
1. Extract abstract text
2. Find equations in abstract
3. Use abstract sentences as high-quality descriptions
4. Match similar equations in body to abstract descriptions

#### Strategy 5: Caption Mining (for Display Equations)
**Method**: Some papers include text immediately after equations that explain them

**Pattern Recognition**:
- "where $x$ is..." (variable definitions)
- "which represents..." (physical interpretation)
- "corresponding to..." (semantic meaning)

### 4.4 Context Extraction Implementation

**Python Pseudocode**:
```python
import re
from lxml import etree

def extract_equation_context(latex_content):
    """Extract equations with surrounding context."""

    # Parse LaTeX with LaTeXML
    xml_doc = latexmL_parse(latex_content)

    # Find all equation environments
    equations = xml_doc.findall('.//equation')

    pairs = []
    for eq in equations:
        # Get LaTeX source
        latex_eq = get_latex_source(eq)

        # Get preceding paragraph
        preceding = get_preceding_text(eq, num_sentences=3)

        # Get following paragraph
        following = get_following_text(eq, num_sentences=3)

        # Get section title
        section = get_section_title(eq)

        # Combine context
        description = {
            'before': preceding,
            'after': following,
            'section': section,
            'labels': get_equation_labels(eq),
            'refs': find_references_to_equation(eq, xml_doc)
        }

        # Create pair
        pairs.append({
            'equation': latex_eq,
            'description': description,
            'ast': parse_equation_ast(latex_eq)
        })

    return pairs
```

### 4.5 Filtering Low-Quality Pairs

**Filter Criteria**:

1. **Equation Complexity**:
   - Filter out trivial equations (e.g., `x = 1`)
   - Require minimum number of operators
   - Check for physics-relevant symbols

2. **Description Quality**:
   - Minimum description length (e.g., 10 words)
   - Require actual text (not just variable definitions)
   - Check for coherent sentences (language model perplexity)

3. **Relevance**:
   - Equation should be mentioned in description
   - Description shouldn't be generic ("The equation is...")
   - Check for physics terminology

4. **Parsing Success**:
   - Equation must parse successfully with SymPy
   - LaTeXML must produce valid MathML
   - No critical LaTeX errors

**Implementation**:
```python
def filter_low_quality_pairs(pairs, min_desc_length=10):
    """Filter out low-quality equation-description pairs."""
    filtered = []

    for pair in pairs:
        eq = pair['equation']
        desc = pair['description']

        # Check equation complexity
        if count_operators(eq) < 2:
            continue

        # Check description length
        if len(desc.split()) < min_desc_length:
            continue

        # Check parsing success
        try:
            parse_latex(eq)
        except:
            continue

        # Check for physics terms
        if not contains_physics_terminology(desc):
            continue

        filtered.append(pair)

    return filtered
```

---

## 5. Data Quality & Annotation

### 5.1 Inter-Annotator Agreement Metrics

**Purpose**: Measure consistency between human annotators to ensure dataset reliability.

#### Cohen's Kappa
- **Use Case**: Two annotators
- **Formula**: κ = (P_o - P_e) / (1 - P_e)
  - P_o = observed agreement
  - P_e = expected agreement by chance
- **Interpretation**:
  - κ < 0.20: Slight agreement
  - 0.21-0.40: Fair
  - 0.41-0.60: Moderate
  - 0.61-0.80: Substantial
  - 0.81-1.00: Almost perfect
- **Tools**: `sklearn.metrics.cohen_kappa_score`

#### Fleiss' Kappa
- **Use Case**: Multiple annotators (3+)
- **Extension**: Generalizes Cohen's kappa
- **Advantage**: Handles more than 2 raters
- **Tools**: `statsmodels.stats.inter_rater.fleiss_kappa`

#### Krippendorff's Alpha (Recommended)
- **Use Case**: Any number of annotators, missing data
- **Flexibility**: Handles incomplete annotations
- **Advantages**:
  - Works with various measurement levels
  - Suitable for real-world annotation workflows
  - More robust than Kappa metrics
- **Tools**: `krippendorff` package, Label Studio integration
- **Link**: https://labelstud.io/blog/how-to-use-krippendorff-s-alpha-to-measure-annotation-agreement/

**Target Agreement**: Aim for Krippendorff's α > 0.67 (minimum acceptable), ideally > 0.80

### 5.2 Annotation Workflow Best Practices

#### Phase 1: Guideline Development
**Steps**:
1. Create detailed annotation manual
2. Define clear criteria for:
   - What constitutes a valid equation-description pair
   - How to handle edge cases (e.g., equation fragments)
   - What level of description detail is needed
3. Provide examples (positive and negative)
4. Include physics domain glossary

**Example Guidelines**:
```
Valid Pair:
✓ Equation: E = mc²
✓ Description: "The energy-mass equivalence relation from special
   relativity, where E is energy, m is mass, and c is the speed of light"

Invalid Pairs:
✗ Description too vague: "This is an important equation"
✗ Description not matching: "The Schrödinger equation describes..."
   (for Maxwell's equations)
✗ Trivial equation: "x = 0"
```

#### Phase 2: Annotator Training
**Process**:
1. Review annotation guidelines with all annotators
2. Conduct training session with example annotations
3. Have annotators practice on training set (50-100 examples)
4. Review training annotations together
5. Clarify confusions and edge cases
6. Repeat until consistent understanding achieved

**Training Validation**: Calculate IAA on training set before proceeding.

#### Phase 3: Independent Annotation
**Protocol**:
1. **Double Annotation**: Each sample annotated by 2+ annotators independently
2. **No Communication**: Annotators work independently to avoid groupthink
3. **Randomization**: Shuffle sample order to avoid fatigue patterns
4. **Batch Size**: Annotate in manageable batches (100-200 examples)
5. **Time Tracking**: Monitor annotation speed to detect issues

**Tools**:
- **Label Studio**: https://labelstud.io/ (open-source)
- **Prodigy**: https://prodi.gy/ (commercial, excellent for NLP)
- **Doccano**: https://github.com/doccano/doccano (open-source)

#### Phase 4: Agreement Analysis
**Steps**:
1. Calculate IAA metrics (Cohen's κ or Krippendorff's α)
2. Identify low-agreement samples
3. Analyze disagreement patterns:
   - Systematic differences (misunderstanding guidelines)
   - Random noise (unclear samples)
   - Domain expertise gaps
4. Update guidelines if needed
5. Resolve disagreements through discussion

**Disagreement Resolution**:
- **Majority Vote**: 3+ annotators, use majority
- **Expert Adjudication**: Senior physicist resolves conflicts
- **Discussion**: Annotators discuss and reach consensus

#### Phase 5: Continuous Monitoring
**Long-term Quality Control**:
1. Periodically provide overlapping samples (10-20%)
2. Calculate IAA throughout project duration
3. Check for annotator drift (agreement decreasing over time)
4. Provide refresher training if needed
5. Monitor individual annotator quality

**Feedback Loop**:
```
Annotate Batch → Calculate IAA → Identify Issues →
Update Guidelines → Retrain → Annotate Next Batch
```

### 5.3 Active Learning for Annotation

**Goal**: Minimize annotation cost by selecting most informative samples.

#### Uncertainty Sampling
**Method**: Prioritize samples where model is most uncertain

**For Equation-CLIP**:
1. Train initial model on small labeled dataset
2. Run model on unlabeled equation-description pairs
3. Calculate uncertainty (e.g., low cosine similarity between equation and description embeddings)
4. Annotate high-uncertainty pairs first
5. Retrain and iterate

**Implementation**:
```python
def uncertainty_sampling(model, unlabeled_pairs, batch_size=100):
    """Select most uncertain pairs for annotation."""
    scores = []

    for pair in unlabeled_pairs:
        eq_emb = model.encode_equation(pair['equation'])
        desc_emb = model.encode_description(pair['description'])
        similarity = cosine_similarity(eq_emb, desc_emb)

        # Low similarity = high uncertainty
        uncertainty = 1 - abs(similarity)
        scores.append(uncertainty)

    # Select top uncertain samples
    indices = np.argsort(scores)[-batch_size:]
    return [unlabeled_pairs[i] for i in indices]
```

#### Diversity Sampling
**Method**: Select samples that are diverse/dissimilar to training data

**For Equation-CLIP**:
1. Cluster equation embeddings
2. Select samples from underrepresented clusters
3. Ensure coverage of physics subdomains
4. Prioritize rare equation types

**Implementation**:
```python
def diversity_sampling(model, unlabeled_pairs, labeled_pairs, batch_size=100):
    """Select diverse samples for annotation."""
    # Encode all equations
    unlabeled_embs = model.encode_equations([p['equation'] for p in unlabeled_pairs])
    labeled_embs = model.encode_equations([p['equation'] for p in labeled_pairs])

    # Calculate distance to nearest labeled sample
    distances = []
    for emb in unlabeled_embs:
        min_dist = np.min([euclidean(emb, l_emb) for l_emb in labeled_embs])
        distances.append(min_dist)

    # Select most distant (diverse) samples
    indices = np.argsort(distances)[-batch_size:]
    return [unlabeled_pairs[i] for i in indices]
```

#### Combined Strategy (Recommended)
**Method**: Mix uncertainty and diversity sampling

**Ratio**: 70% uncertainty, 30% diversity

**Rationale**:
- Uncertainty improves current model decision boundary
- Diversity ensures coverage of equation space
- Combined approach balances exploitation and exploration

### 5.4 Validation Dataset Creation

**Purpose**: Hold-out set for unbiased evaluation.

**Composition**:
- **Size**: 5,000-10,000 pairs
- **Coverage**: All physics subdomains proportionally
- **Quality**: Expert-reviewed annotations
- **Splits**:
  - Validation (dev set): 50%
  - Test set: 50%

**Physics Subdomain Distribution** (suggested):
- Classical Mechanics: 15%
- Electromagnetism: 15%
- Quantum Mechanics: 20%
- Thermodynamics: 10%
- Statistical Mechanics: 10%
- Relativity: 10%
- Mathematical Physics: 10%
- Other: 10%

**Expert Review**:
- All validation/test samples reviewed by physics PhDs
- Ensure correctness of equations
- Verify quality of descriptions
- Check domain labels

---

## 6. Physics Domain Taxonomies

### 6.1 arXiv Category System

**Official Taxonomy**: https://arxiv.org/category_taxonomy

#### Physics Categories (physics.*)

**Major Categories**:
- `physics.acc-ph` - Accelerator Physics
- `physics.ao-ph` - Atmospheric and Oceanic Physics
- `physics.app-ph` - Applied Physics
- `physics.atm-clus` - Atomic and Molecular Clusters
- `physics.atom-ph` - Atomic Physics
- `physics.bio-ph` - Biological Physics
- `physics.chem-ph` - Chemical Physics
- `physics.class-ph` - Classical Physics
- `physics.comp-ph` - Computational Physics
- `physics.data-an` - Data Analysis, Statistics and Probability
- `physics.ed-ph` - Physics Education
- `physics.flu-dyn` - Fluid Dynamics
- `physics.gen-ph` - General Physics
- `physics.geo-ph` - Geophysics
- `physics.hist-ph` - History and Philosophy of Physics
- `physics.ins-det` - Instrumentation and Detectors
- `physics.med-ph` - Medical Physics
- `physics.optics` - Optics
- `physics.plasm-ph` - Plasma Physics
- `physics.pop-ph` - Popular Physics
- `physics.soc-ph` - Physics and Society
- `physics.space-ph` - Space Physics

**Related Categories**:
- `math-ph` (or `math.MP`) - Mathematical Physics
- `cond-mat.*` - Condensed Matter
- `hep-*` - High Energy Physics (hep-th, hep-ph, hep-lat, hep-ex)
- `gr-qc` - General Relativity and Quantum Cosmology
- `quant-ph` - Quantum Physics
- `astro-ph.*` - Astrophysics
- `nucl-*` - Nuclear (nucl-th, nucl-ex)
- `nlin.*` - Nonlinear Sciences

#### Mathematical Physics (math-ph)
**Definition**: Articles illustrating application of mathematics to physics problems, developing mathematical methods for applications, or providing mathematically rigorous formulations of physical theories.

**Target Audience**: Both mathematically-oriented physicists and physically-oriented mathematicians.

**Historical**: Started September 1996.

### 6.2 Hierarchical Physics Taxonomy (Proposed)

**For Equation-CLIP fine-grained domain labeling**:

```
Physics
├── Classical Mechanics
│   ├── Newtonian Mechanics
│   ├── Lagrangian Mechanics
│   ├── Hamiltonian Mechanics
│   └── Rigid Body Dynamics
├── Electromagnetism
│   ├── Electrostatics
│   ├── Magnetostatics
│   ├── Electrodynamics
│   └── Maxwell's Equations
├── Quantum Mechanics
│   ├── Wave Mechanics
│   ├── Matrix Mechanics
│   ├── Quantum Field Theory
│   └── Quantum Information
├── Thermodynamics
│   ├── Classical Thermodynamics
│   ├── Statistical Mechanics
│   └── Non-Equilibrium Thermodynamics
├── Relativity
│   ├── Special Relativity
│   └── General Relativity
├── Optics
│   ├── Geometrical Optics
│   ├── Wave Optics
│   └── Quantum Optics
├── Condensed Matter
│   ├── Solid State Physics
│   ├── Soft Matter
│   └── Many-Body Physics
└── Particle Physics
    ├── Standard Model
    └── Beyond Standard Model
```

### 6.3 Domain Labeling Strategies

#### Automatic Labeling from arXiv Categories
**Method**: Use arXiv's primary category as initial label

**Implementation**:
```python
def map_arxiv_category_to_domain(arxiv_category):
    """Map arXiv category to physics domain."""
    mapping = {
        'physics.class-ph': 'Classical Mechanics',
        'quant-ph': 'Quantum Mechanics',
        'physics.optics': 'Optics',
        'gr-qc': 'General Relativity',
        'cond-mat.stat-mech': 'Statistical Mechanics',
        # ... complete mapping
    }
    return mapping.get(arxiv_category, 'Other')
```

**Advantages**: Automatic, scalable, consistent
**Limitations**: Coarse-grained, some papers span multiple domains

#### Section-Based Labeling
**Method**: Use paper section titles to infer equation domain

**Example**:
- Section: "3.2 Schrödinger Equation in Momentum Space"
- Domain: Quantum Mechanics → Wave Mechanics

**Implementation**: Train classifier on section titles → domain labels

#### Keyword-Based Labeling
**Method**: Identify physics-specific keywords in equation context

**Example Keywords**:
- Classical Mechanics: "force", "momentum", "angular velocity"
- Quantum Mechanics: "wavefunction", "operator", "eigenvalue"
- Thermodynamics: "entropy", "temperature", "free energy"

**Implementation**:
```python
domain_keywords = {
    'Quantum Mechanics': ['wavefunction', 'hamiltonian', 'eigenvalue',
                          'quantum state', 'operator', 'uncertainty'],
    'Electromagnetism': ['electric field', 'magnetic field', 'charge',
                         'current', 'maxwell', 'electromagnetic'],
    'Thermodynamics': ['entropy', 'temperature', 'heat', 'energy',
                       'free energy', 'partition function'],
    # ... more domains
}

def label_by_keywords(description, keywords):
    """Label domain based on keyword matching."""
    scores = {}
    for domain, kws in keywords.items():
        score = sum(1 for kw in kws if kw.lower() in description.lower())
        scores[domain] = score
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'Other'
```

#### Multi-Label Classification
**Reality**: Many equations appear in multiple physics domains

**Examples**:
- Fourier Transform: Classical Physics, Quantum Mechanics, Signal Processing
- Partition Function: Statistical Mechanics, Quantum Field Theory
- Hamilton's Equations: Classical Mechanics, Quantum Mechanics

**Approach**: Allow equations to have multiple domain labels

**Implementation**: Train multi-label classifier on equation + description → domain labels

---

## 7. Ethical Considerations

### 7.1 Data Licensing and Attribution

#### arXiv Licensing
**arXiv Non-Exclusive License**:
- Authors grant arXiv perpetual, non-exclusive license to distribute
- Authors retain copyright
- arXiv content can be used for research purposes
- **Key Point**: Check individual paper licenses (some specify CC-BY, CC-BY-NC, etc.)

**Best Practice**:
- Respect individual paper licenses
- Provide attribution in dataset documentation
- Include arXiv IDs for all papers
- Link back to original papers

**Dataset License Recommendation**: CC-BY 4.0 or CC-BY-SA 4.0 for derived dataset

#### OpenStax Licensing
- **License**: Creative Commons CC-BY 4.0
- **Allows**: Commercial use, modifications, distribution
- **Requires**: Attribution to original authors

#### MIT OCW Licensing
- **License**: Creative Commons (varies by course, typically CC-BY-NC-SA)
- **Check**: Individual course license before scraping
- **Non-Commercial**: Many restrict commercial use

#### Wikipedia Licensing
- **License**: Creative Commons CC-BY-SA 3.0
- **Requires**: Share-alike (derivatives under same license)
- **Attribution**: Must credit Wikipedia and original editors

### 7.2 Data Provenance and Transparency

**Recommendations from "The Data Provenance Initiative"** (arXiv:2310.16787):

1. **Document Data Lineage**:
   - Source of each equation-description pair
   - arXiv ID or DOI of original paper
   - Date of collection
   - Processing steps applied

2. **Creator Attribution**:
   - List all paper authors in dataset metadata
   - Provide links to original publications
   - Acknowledge contributions

3. **License Conditions**:
   - Clearly specify licenses for all data sources
   - Ensure derived dataset license is compatible
   - Document any restrictions

4. **Properties and Use**:
   - Document dataset statistics (size, domains, etc.)
   - Describe intended use cases
   - List known limitations

**Dataset Card Template**:
```markdown
# Equation-CLIP Dataset Card

## Dataset Description
- **Size**: 500K equation-description pairs
- **Source**: arXiv physics papers (2000-2025)
- **Collection Date**: 2025-10-11 to 2025-12-01
- **Languages**: English (descriptions)
- **License**: CC-BY 4.0

## Data Sources
- arXiv: 95% (475K pairs)
- OpenStax: 3% (15K pairs)
- Wikipedia: 2% (10K pairs)

## Attribution
All papers cited in supplementary file `sources.jsonl` with:
- arXiv ID
- Authors
- Title
- Original license

## Intended Use
- Training contrastive equation-text models
- Equation retrieval research
- Physics education applications

## Limitations
- English descriptions only
- Physics domain only (not chemistry, engineering)
- Potential OCR errors in older papers
```

### 7.3 Privacy and Bias Considerations

#### Privacy
**Minimal Risk**:
- Physics equations are factual, non-personal data
- Paper metadata is already public on arXiv
- No human subjects involved

**Precautions**:
- Remove any author contact information
- Don't include acknowledgments sections (may contain personal info)

#### Bias Considerations

**Potential Biases**:
1. **Temporal Bias**: Recent papers over-represented (more digital sources)
2. **Geographic Bias**: English-language papers, Western institutions
3. **Subdomain Bias**: Popular fields over-represented (e.g., more quantum computing than classical mechanics)
4. **Notation Bias**: Standard notation over alternative notations

**Mitigation Strategies**:
1. **Balanced Sampling**: Ensure proportional representation of physics subdomains
2. **Temporal Coverage**: Include papers from multiple decades
3. **Documentation**: Clearly document dataset composition and limitations
4. **Multiple Notations**: Include alternative forms of same equation when available

**From "Ethical Considerations for Responsible Data Curation"** (arXiv:2302.03629):
- Document purpose and intended use
- Consider diversity in data representation
- Provide transparency about curation decisions
- Enable reproducibility through detailed documentation

### 7.4 Data Curation Recommendations

**From Recent Research**:

1. **Purpose Specification** (define before collecting):
   - Primary use: Training contrastive learning models
   - Secondary uses: Equation retrieval, semantic search
   - Out-of-scope: Commercial equation plagiarism detection

2. **Consent and Copyright**:
   - arXiv authors consented to open access
   - Respect individual paper licenses
   - Provide opt-out mechanism for authors

3. **Diversity and Representation**:
   - Include papers from diverse institutions
   - Cover all major physics subdomains
   - Include historical and modern equations

4. **Quality Control**:
   - Validate equations are correctly extracted
   - Ensure descriptions accurately describe equations
   - Remove duplicates and trivial cases

5. **Documentation**:
   - Provide detailed dataset documentation
   - Include datasheets for datasets
   - Make code for data collection publicly available

6. **Ongoing Maintenance**:
   - Plan for dataset updates
   - Provide versioning
   - Accept community feedback and corrections

---

## 8. Recommended Implementation Strategy

### 8.1 Phase 1: Pilot Study (Month 1)

**Goal**: Validate extraction pipeline on small scale

**Tasks**:
1. Download 1,000 arXiv papers from `physics.comp-ph` category
2. Implement LaTeXML + SymPy extraction pipeline
3. Extract equation-description pairs
4. Manually review 100 pairs for quality
5. Calculate extraction success rate
6. Iterate on extraction heuristics

**Success Metrics**:
- 70%+ of equations successfully parsed
- 50%+ of pairs have meaningful descriptions
- Extraction pipeline processes 100 papers/hour

**Tools**:
```bash
# Download arXiv source files
pip install arxiv
python download_arxiv.py --category physics.comp-ph --limit 1000

# Extract equations
pip install latexml sympy lxml
python extract_equations.py --input papers/ --output pairs.jsonl
```

### 8.2 Phase 2: Scaling Up (Months 2-3)

**Goal**: Scale to 100K-500K pairs

**Strategy**:

#### Option A: Targeted arXiv Download
**Approach**: Use arXiv API to download specific categories
**Categories**:
- `physics.class-ph` (Classical)
- `quant-ph` (Quantum)
- `physics.optics` (Optics)
- `cond-mat.stat-mech` (Statistical Mechanics)
- `gr-qc` (Relativity)
- `math-ph` (Mathematical Physics)

**Time Range**: 2015-2025 (recent papers, better LaTeX quality)

**Estimated Volume**:
- ~10K papers per category
- ~50-100 equations per paper
- ~500K-1M total equations
- After filtering: ~100K-300K high-quality pairs

#### Option B: Bulk S3 Download
**Approach**: Download full arXiv corpus from S3
**Advantages**: Complete coverage, no rate limits
**Disadvantages**: 9.2 TB storage, AWS costs

**Filtering**: Post-download filter by category

#### Recommended: Hybrid Approach
1. Start with targeted API download (Option A)
2. If more data needed, supplement with S3 bulk download
3. Focus on recent papers (2015+) for better LaTeX quality

### 8.3 Phase 3: Quality Control (Month 3-4)

**Annotation Workflow**:

1. **Automatic Filtering** (Rule-Based):
   - Remove trivial equations
   - Filter by description length
   - Check parsing success
   - Remove duplicates
   - Result: ~200K candidates

2. **Active Learning Annotation** (Initial Round):
   - Sample 5,000 pairs using diversity sampling
   - 2 annotators per sample
   - Calculate IAA (target: α > 0.70)
   - Create annotation guidelines
   - Result: 5,000 validated pairs

3. **Model-Assisted Annotation** (Subsequent Rounds):
   - Train initial Equation-CLIP on 5,000 pairs
   - Use uncertainty sampling for next batch
   - Annotate 20,000 more pairs (4 batches of 5,000)
   - Retrain after each batch
   - Result: 25,000 high-quality labeled pairs

4. **Validation Set Creation**:
   - Expert review of 10,000 pairs
   - Ensure physics subdomain coverage
   - Split: 5,000 validation, 5,000 test
   - Result: 10,000 expert-validated pairs

### 8.4 Phase 4: Dataset Finalization (Month 4)

**Final Dataset Composition**:

| Source | Pairs | Quality | Use |
|--------|-------|---------|-----|
| Fully Annotated | 25,000 | Expert-verified | Train/Eval |
| Semi-Supervised | 150,000 | Model-filtered | Train |
| Textbooks (OpenStax) | 15,000 | High-quality | Train/Eval |
| Wikipedia | 10,000 | Curated | Train |
| **Total** | **200,000** | Mixed | Full Dataset |

**Data Splits**:
- Training: 175,000 (87.5%)
- Validation: 12,500 (6.25%)
- Test: 12,500 (6.25%)

**Domain Distribution** (Training Set):
- Classical Mechanics: 15%
- Electromagnetism: 15%
- Quantum Mechanics: 20%
- Thermodynamics: 12%
- Statistical Mechanics: 10%
- Relativity: 10%
- Optics: 8%
- Other: 10%

### 8.5 Technical Implementation

**Code Repository Structure**:
```
equation-clip-dataset/
├── scripts/
│   ├── download_arxiv.py          # Download papers
│   ├── extract_equations.py       # LaTeXML extraction
│   ├── parse_equations.py         # SymPy parsing
│   ├── extract_context.py         # Context extraction
│   ├── filter_pairs.py            # Quality filtering
│   └── create_splits.py           # Train/val/test splits
├── annotation/
│   ├── label_studio_config.json   # Annotation tool config
│   ├── guidelines.md              # Annotation manual
│   └── calculate_iaa.py           # IAA metrics
├── data/
│   ├── raw/                       # Raw arXiv papers
│   ├── processed/                 # Extracted pairs
│   ├── annotated/                 # Human annotations
│   └── final/                     # Final dataset
├── analysis/
│   ├── statistics.py              # Dataset statistics
│   ├── domain_distribution.py    # Domain analysis
│   └── quality_metrics.py        # Quality analysis
└── README.md                      # Dataset documentation
```

**Key Dependencies**:
```python
# requirements.txt
arxiv==2.1.0              # arXiv API
lxml==5.1.0               # XML parsing
sympy==1.12               # Equation parsing
numpy==1.24.3             # Numerical operations
pandas==2.0.3             # Data management
tqdm==4.65.0              # Progress bars
scikit-learn==1.3.0       # IAA metrics
krippendorff==0.6.0       # Alpha metric

# External dependencies
# - LaTeXML (install via apt/brew/conda)
# - texlive (for LaTeX compilation)
```

### 8.6 Storage and Infrastructure

**Storage Requirements**:
- **Raw Papers**: ~50 GB (50K papers × 1 MB avg)
- **Extracted Pairs**: ~5 GB (200K pairs × 25 KB avg)
- **Annotations**: ~2 GB
- **Models/Checkpoints**: ~10 GB
- **Total**: ~70 GB

**Compute Requirements**:
- **Extraction**:
  - 100 papers/hour on single CPU
  - 50K papers = 500 hours = ~21 days
  - Parallelize: 10 CPUs → ~2-3 days
- **Annotation**:
  - 5 minutes per pair
  - 25K pairs = 125K minutes = ~2,000 hours
  - 5 annotators → ~400 hours = ~10 weeks

**Infrastructure Recommendations**:
- **Compute**: AWS EC2 c5.4xlarge (16 vCPUs) or similar
- **Storage**: AWS S3 or institutional storage
- **Annotation**: Self-hosted Label Studio or Prodigy
- **Estimated AWS Cost**: ~$500-1000 for entire pipeline

### 8.7 Timeline Summary

| Month | Phase | Tasks | Deliverables |
|-------|-------|-------|--------------|
| 1 | Pilot | Pipeline development, 1K papers | Working extraction code |
| 2-3 | Scaling | 50K papers, extraction | 200K raw pairs |
| 3-4 | QC | Annotation, filtering | 25K labeled pairs |
| 4 | Finalization | Splits, docs, release | Final dataset (200K) |

**Total Duration**: 4 months
**Team Size**: 2-3 engineers + 3-5 annotators

---

## 9. Key Takeaways and Recommendations

### 9.1 Best Tools for Each Task

| Task | Recommended Tool | Alternative |
|------|------------------|-------------|
| arXiv Download | arXiv Python API | S3 bulk download |
| LaTeX Parsing | LaTeXML | plasTeX |
| Equation Parsing | SymPy | Direct AST parsing |
| Context Extraction | plasTeX + regex | TexSoup |
| Annotation | Label Studio | Prodigy |
| IAA Metrics | Krippendorff's α | Cohen's κ |
| Storage | PostgreSQL + S3 | MongoDB |

### 9.2 Critical Success Factors

1. **LaTeX Parsing Quality**: 70%+ success rate essential
   - Use LaTeXML for robust parsing
   - Handle common LaTeX errors gracefully
   - Validate with SymPy

2. **Description Quality**: Meaningful context crucial
   - Multi-sentence context windows
   - Include section headers
   - Filter generic descriptions

3. **Domain Coverage**: Balanced representation
   - Target proportional sampling across physics subdomains
   - Include both common and rare equation types
   - Mix foundational and cutting-edge equations

4. **Annotation Quality**: High IAA required
   - Comprehensive annotation guidelines
   - Thorough annotator training
   - Continuous quality monitoring

5. **Scalability**: Efficient pipeline
   - Parallelize extraction
   - Cache intermediate results
   - Handle failures gracefully

### 9.3 Potential Challenges and Mitigations

| Challenge | Risk | Mitigation |
|-----------|------|------------|
| LaTeX parsing failures | 30-40% of papers | Use multiple parsers, fallback strategies |
| Low-quality descriptions | Noisy training data | Multi-stage filtering, human annotation |
| Notation variability | Same equation, different forms | Canonicalize with SymPy |
| Domain imbalance | Bias toward popular fields | Targeted sampling by subdomain |
| Annotation cost | $50K-100K for 25K pairs | Active learning, start small |
| Storage costs | 70 GB + backups | Use efficient formats (Parquet), compress |

### 9.4 Future Enhancements

1. **Multi-Modal Pairs**: Include equation images (using im2latex data)
2. **Cross-Lingual**: Add descriptions in other languages
3. **Temporal Evolution**: Track how equations are described over time
4. **Derivation Chains**: Link related equations (A → B → C)
5. **Code Implementations**: Pair equations with code (SymPy, NumPy)

---

## 10. References and Resources

### 10.1 Key Papers

1. **Data Collection**:
   - "On the Use of ArXiv as a Dataset" (2019): https://arxiv.org/abs/1905.00075
   - "OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text" (2023): https://arxiv.org/abs/2310.06786

2. **Equation Retrieval**:
   - "Tangent-CFT: An Embedding Model for Mathematical Formulas" (2019): https://dl.acm.org/doi/10.1145/3341981.3344235
   - "NTCIR-12 MathIR Task Overview": http://ntcir-math.nii.ac.jp/

3. **Annotation & Ethics**:
   - "The Data Provenance Initiative" (2023): https://arxiv.org/abs/2310.16787
   - "Ethical Considerations for Responsible Data Curation" (2023): https://arxiv.org/abs/2302.03629

4. **Active Learning**:
   - "Contrastive Learning with Hard Negative Samples" (2020): https://arxiv.org/abs/2010.04592

### 10.2 Software Tools

| Tool | Link | Purpose |
|------|------|---------|
| LaTeXML | https://github.com/brucemiller/LaTeXML | LaTeX → XML |
| SymPy | https://www.sympy.org/ | Equation parsing |
| plasTeX | https://github.com/plastex/plastex | Document parsing |
| tex2py | https://github.com/alvinwan/tex2py | Quick LaTeX parsing |
| arXiv API | https://pypi.org/project/arxiv/ | Paper download |
| Label Studio | https://labelstud.io/ | Annotation |
| Prodigy | https://prodi.gy/ | Annotation (commercial) |

### 10.3 Datasets

| Dataset | Size | Link | Use |
|---------|------|------|-----|
| arXiv (Kaggle) | 1.7M papers | https://www.kaggle.com/datasets/Cornell-University/arxiv | Paper metadata |
| NTCIR-12 | 590K formulas | http://ntcir-math.nii.ac.jp/ | Evaluation benchmark |
| im2latex-550k | 550K pairs | Papers with Code | Image-to-LaTeX |
| OpenStaxQA | 43 textbooks | https://arxiv.org/abs/2510.06239 | Textbook problems |
| OpenWebMath | Large-scale | https://arxiv.org/abs/2310.06786 | Web math text |

### 10.4 Documentation Resources

- **arXiv Bulk Data Guide**: https://info.arxiv.org/help/bulk_data.html
- **arXiv Category Taxonomy**: https://arxiv.org/category_taxonomy
- **LaTeXML Manual**: https://math.nist.gov/~BMiller/LaTeXML/manual.pdf
- **SymPy Parsing Docs**: https://docs.sympy.org/latest/modules/parsing.html
- **Krippendorff's Alpha Guide**: https://labelstud.io/blog/how-to-use-krippendorff-s-alpha-to-measure-annotation-agreement/

---

## Appendix A: Example Code Snippets

### A.1 Download arXiv Papers by Category

```python
import arxiv
import os
from pathlib import Path

def download_papers_by_category(category, max_results=1000, output_dir='papers'):
    """Download arXiv papers from specific category."""
    Path(output_dir).mkdir(exist_ok=True)

    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for paper in search.results():
        print(f"Downloading {paper.entry_id}...")

        # Download source files
        try:
            paper.download_source(dirpath=output_dir)
        except Exception as e:
            print(f"Error downloading {paper.entry_id}: {e}")
            continue

        # Save metadata
        metadata = {
            'id': paper.entry_id,
            'title': paper.title,
            'abstract': paper.summary,
            'categories': paper.categories,
            'authors': [a.name for a in paper.authors],
            'published': str(paper.published)
        }

        with open(f"{output_dir}/{paper.get_short_id()}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

# Usage
download_papers_by_category('physics.comp-ph', max_results=1000)
```

### A.2 Extract Equations with LaTeXML

```python
import subprocess
import json
from lxml import etree

def extract_equations_latexmL(latex_file, output_file):
    """Extract equations from LaTeX file using LaTeXML."""

    # Convert LaTeX to XML
    xml_file = latex_file.replace('.tex', '.xml')
    cmd = ['latexml', '--dest=' + xml_file, latex_file]
    subprocess.run(cmd, capture_output=True)

    # Parse XML
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Find all equations
    equations = []
    namespaces = {'ltx': 'http://dlmf.nist.gov/LaTeXML'}

    for eq in root.xpath('//ltx:equation', namespaces=namespaces):
        # Extract LaTeX source
        math_elem = eq.find('.//ltx:Math', namespaces=namespaces)
        if math_elem is not None:
            latex_src = math_elem.get('tex')

            # Extract surrounding context
            parent = eq.getparent()
            context_before = get_text_before(eq, parent, n_sentences=2)
            context_after = get_text_after(eq, parent, n_sentences=2)

            equations.append({
                'latex': latex_src,
                'context_before': context_before,
                'context_after': context_after
            })

    # Save results
    with open(output_file, 'w') as f:
        json.dump(equations, f, indent=2)

    return equations
```

### A.3 Parse and Canonicalize Equations

```python
from sympy import symbols, simplify
from sympy.parsing.latex import parse_latex

def canonicalize_equation(latex_str):
    """Parse and canonicalize equation using SymPy."""
    try:
        # Parse LaTeX
        expr = parse_latex(latex_str)

        # Simplify
        canonical = simplify(expr)

        # Extract symbols
        syms = sorted([str(s) for s in canonical.free_symbols])

        return {
            'original': latex_str,
            'canonical': str(canonical),
            'symbols': syms,
            'success': True
        }
    except Exception as e:
        return {
            'original': latex_str,
            'error': str(e),
            'success': False
        }

# Example
eq = r'\frac{d^2 x}{dt^2} = -\omega^2 x'
result = canonicalize_equation(eq)
print(result)
```

### A.4 Calculate Inter-Annotator Agreement

```python
import krippendorff
import numpy as np

def calculate_iaa(annotations):
    """Calculate Krippendorff's Alpha for annotations.

    Args:
        annotations: List of [annotator_id, item_id, label]

    Returns:
        alpha: Krippendorff's alpha coefficient
    """
    # Convert to reliability matrix
    # Rows: annotators, Columns: items
    annotators = sorted(set(a[0] for a in annotations))
    items = sorted(set(a[1] for a in annotations))

    # Initialize matrix with NaN for missing values
    matrix = np.full((len(annotators), len(items)), np.nan)

    for ann_id, item_id, label in annotations:
        i = annotators.index(ann_id)
        j = items.index(item_id)
        matrix[i, j] = label

    # Calculate Krippendorff's alpha
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal')

    return alpha

# Example
annotations = [
    (1, 'eq001', 1),  # (annotator, equation_id, valid=1/invalid=0)
    (2, 'eq001', 1),
    (1, 'eq002', 0),
    (2, 'eq002', 0),
    (1, 'eq003', 1),
    (2, 'eq003', 0),  # Disagreement
]

alpha = calculate_iaa(annotations)
print(f"Krippendorff's Alpha: {alpha:.3f}")
```

---

## Document Metadata

- **Author**: Claude (Anthropic AI Research Assistant)
- **Date**: 2025-10-11
- **Project**: Equation-CLIP Research
- **Status**: Initial Research Complete
- **Next Steps**: Begin pilot study implementation (Phase 1)

---

*This research document synthesizes findings from academic papers, documentation, and best practices for data collection in scientific machine learning projects. All recommendations should be validated through pilot studies before full-scale implementation.*
