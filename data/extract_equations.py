"""
LaTeX Equation Extractor for Equation-CLIP Project

Extracts equations and surrounding context from LaTeX source files.
Uses regex patterns and heuristics to identify equation environments.
"""

import re
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EquationContext:
    """Data structure for an equation with its context."""
    equation_id: str
    equation_latex: str
    equation_type: str  # 'inline', 'display', 'numbered'
    context_before: str
    context_after: str
    full_context: str
    paper_id: str
    section: Optional[str] = None
    label: Optional[str] = None
    caption: Optional[str] = None
    page_number: Optional[int] = None


class LatexEquationExtractor:
    """Extract equations and context from LaTeX source files."""

    # Equation environment patterns
    EQUATION_PATTERNS = [
        # Display math environments
        (r'\$\$(.*?)\$\$', 'display'),
        (r'\\begin{equation}(.*?)\\end{equation}', 'numbered'),
        (r'\\begin{equation\*}(.*?)\\end{equation\*}', 'display'),
        (r'\\begin{align}(.*?)\\end{align}', 'numbered'),
        (r'\\begin{align\*}(.*?)\\end{align\*}', 'display'),
        (r'\\begin{eqnarray}(.*?)\\end{eqnarray}', 'numbered'),
        (r'\\begin{eqnarray\*}(.*?)\\end{eqnarray\*}', 'display'),
        (r'\\begin{gather}(.*?)\\end{gather}', 'numbered'),
        (r'\\begin{gather\*}(.*?)\\end{gather\*}', 'display'),
        (r'\\begin{multline}(.*?)\\end{multline}', 'numbered'),
        (r'\\begin{multline\*}(.*?)\\end{multline\*}', 'display'),
        (r'\\\[(.*?)\\\]', 'display'),
        # Inline math (handled separately due to frequency)
        # (r'\$(.*?)\$', 'inline'),
    ]

    def __init__(self, min_equation_length: int = 5, max_equation_length: int = 500,
                 context_window: int = 3):
        """
        Initialize equation extractor.

        Args:
            min_equation_length: Minimum number of characters for valid equation
            max_equation_length: Maximum number of characters for valid equation
            context_window: Number of sentences before/after equation for context
        """
        self.min_equation_length = min_equation_length
        self.max_equation_length = max_equation_length
        self.context_window = context_window

    def generate_equation_id(self, equation_latex: str, paper_id: str) -> str:
        """Generate unique ID for equation."""
        content = f"{paper_id}_{equation_latex}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def clean_latex(self, latex: str) -> str:
        """Clean LaTeX string by removing comments and extra whitespace."""
        # Remove LaTeX comments
        latex = re.sub(r'(?<!\\)%.*', '', latex)
        # Remove multiple whitespaces
        latex = re.sub(r'\s+', ' ', latex)
        # Strip leading/trailing whitespace
        latex = latex.strip()
        return latex

    def extract_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """
        Extract sentences from text.

        Simple sentence splitter (can be improved with NLTK).
        """
        # Basic sentence splitting (improved version would use NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def get_context(self, text: str, equation_start: int, equation_end: int) -> Tuple[str, str, str]:
        """
        Extract context before and after equation.

        Args:
            text: Full LaTeX text
            equation_start: Start index of equation
            equation_end: End index of equation

        Returns:
            Tuple of (context_before, context_after, full_context)
        """
        # Get text before and after equation
        text_before = text[:equation_start]
        text_after = text[equation_end:]

        # Extract last N sentences before
        sentences_before = self.extract_sentences(text_before)
        context_before = ' '.join(sentences_before[-self.context_window:])

        # Extract first N sentences after
        sentences_after = self.extract_sentences(text_after)
        context_after = ' '.join(sentences_after[:self.context_window])

        # Full context
        full_context = f"{context_before} [EQUATION] {context_after}"

        return context_before, context_after, full_context

    def is_valid_equation(self, equation_latex: str) -> bool:
        """
        Check if equation is valid (not too simple or too complex).

        Filters:
        - Length constraints
        - Not just variable assignments (x=0, y=1, etc.)
        - Contains mathematical operators
        """
        # Length check
        if len(equation_latex) < self.min_equation_length:
            return False
        if len(equation_latex) > self.max_equation_length:
            return False

        # Not just simple assignments
        simple_patterns = [
            r'^\s*[a-zA-Z]\s*=\s*\d+\s*$',  # x=0
            r'^\s*[a-zA-Z]\s*\in\s*',  # x \in R
        ]
        for pattern in simple_patterns:
            if re.match(pattern, equation_latex):
                return False

        # Must contain at least one math operator or function
        math_indicators = [
            r'\\frac', r'\\sqrt', r'\\sum', r'\\int', r'\\partial',
            r'\\nabla', r'\\times', r'\\cdot', r'\\hat', r'\\vec',
            r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\omega',
            r'\^', r'_'
        ]
        has_math = any(re.search(pattern, equation_latex) for pattern in math_indicators)

        return has_math

    def extract_equations_from_latex(self, latex_text: str, paper_id: str) -> List[EquationContext]:
        """
        Extract all equations from LaTeX text.

        Args:
            latex_text: Raw LaTeX source text
            paper_id: Identifier for the paper

        Returns:
            List of EquationContext objects
        """
        latex_text = self.clean_latex(latex_text)
        equations = []

        for pattern, eq_type in self.EQUATION_PATTERNS:
            for match in re.finditer(pattern, latex_text, re.DOTALL):
                equation_latex = match.group(1).strip()
                equation_latex = self.clean_latex(equation_latex)

                # Validate equation
                if not self.is_valid_equation(equation_latex):
                    continue

                # Extract context
                context_before, context_after, full_context = self.get_context(
                    latex_text, match.start(), match.end()
                )

                # Check for label
                label_match = re.search(r'\\label{(.*?)}', equation_latex)
                label = label_match.group(1) if label_match else None

                # Remove label from equation
                equation_latex_clean = re.sub(r'\\label{.*?}', '', equation_latex).strip()

                # Create equation context object
                eq_id = self.generate_equation_id(equation_latex_clean, paper_id)

                eq_context = EquationContext(
                    equation_id=eq_id,
                    equation_latex=equation_latex_clean,
                    equation_type=eq_type,
                    context_before=context_before,
                    context_after=context_after,
                    full_context=full_context,
                    paper_id=paper_id,
                    label=label
                )

                equations.append(eq_context)

        logger.info(f"Extracted {len(equations)} equations from paper {paper_id}")
        return equations

    def extract_from_file(self, latex_file: Path, paper_id: str = None) -> List[EquationContext]:
        """
        Extract equations from a LaTeX file.

        Args:
            latex_file: Path to LaTeX file
            paper_id: Paper identifier (uses filename if not provided)

        Returns:
            List of EquationContext objects
        """
        if paper_id is None:
            paper_id = latex_file.stem

        try:
            with open(latex_file, 'r', encoding='utf-8', errors='ignore') as f:
                latex_text = f.read()

            return self.extract_equations_from_latex(latex_text, paper_id)

        except Exception as e:
            logger.error(f"Error extracting from {latex_file}: {str(e)}")
            return []

    def extract_from_directory(self, input_dir: Path, output_file: Path,
                              file_pattern: str = "*.tex") -> int:
        """
        Extract equations from all LaTeX files in a directory.

        Args:
            input_dir: Directory containing LaTeX files
            output_file: Output JSON file for extracted equations
            file_pattern: Glob pattern for LaTeX files

        Returns:
            Number of equations extracted
        """
        all_equations = []

        latex_files = list(input_dir.rglob(file_pattern))
        logger.info(f"Found {len(latex_files)} LaTeX files in {input_dir}")

        for latex_file in latex_files:
            logger.info(f"Processing {latex_file}")
            equations = self.extract_from_file(latex_file)
            all_equations.extend(equations)

        # Save to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump([asdict(eq) for eq in all_equations], f, indent=2)

        logger.info(f"Saved {len(all_equations)} equations to {output_file}")
        return len(all_equations)


def main():
    """Main execution function."""
    # Configuration
    INPUT_DIR = Path('./arxiv_papers')
    OUTPUT_FILE = Path('./data/extracted_equations.json')

    extractor = LatexEquationExtractor(
        min_equation_length=10,
        max_equation_length=500,
        context_window=3
    )

    logger.info("Starting equation extraction...")
    num_equations = extractor.extract_from_directory(INPUT_DIR, OUTPUT_FILE)

    logger.info(f"Extraction complete. Total equations: {num_equations}")


if __name__ == "__main__":
    main()
