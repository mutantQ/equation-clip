"""
arXiv Data Downloader for Equation-CLIP Project

This script downloads physics papers from arXiv for equation extraction.
Supports bulk download via arXiv API with category filtering.
"""

import arxiv
import os
import time
import logging
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArxivDownloader:
    """Download physics papers from arXiv with LaTeX source files."""

    # Physics categories to target
    PHYSICS_CATEGORIES = [
        'physics.class-ph',  # Classical Physics
        'physics.optics',    # Optics
        'quant-ph',          # Quantum Physics
        'cond-mat',          # Condensed Matter
        'hep-th',            # High Energy Physics - Theory
        'hep-ph',            # High Energy Physics - Phenomenology
        'gr-qc',             # General Relativity and Quantum Cosmology
        'math-ph',           # Mathematical Physics
    ]

    def __init__(self, output_dir: str = './arxiv_papers', max_papers: int = 1000):
        """
        Initialize the arXiv downloader.

        Args:
            output_dir: Directory to save downloaded papers
            max_papers: Maximum number of papers to download
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_papers = max_papers
        self.metadata_file = self.output_dir / 'metadata.jsonl'

    def build_query(self, categories: List[str] = None,
                   date_from: str = "2015-01-01",
                   date_to: str = "2025-12-31") -> str:
        """
        Build arXiv API query string.

        Args:
            categories: List of arXiv categories to search
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)

        Returns:
            Query string for arXiv API
        """
        if categories is None:
            categories = self.PHYSICS_CATEGORIES

        # Build category query (OR between categories)
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])

        # Add date range
        query = f"({category_query}) AND submittedDate:[{date_from.replace('-', '')} TO {date_to.replace('-', '')}]"

        return query

    def download_papers(self,
                       categories: List[str] = None,
                       papers_per_category: int = 100,
                       skip_existing: bool = True) -> List[Dict]:
        """
        Download papers from arXiv.

        Args:
            categories: List of arXiv categories
            papers_per_category: Number of papers to download per category
            skip_existing: Skip papers that already exist

        Returns:
            List of metadata dictionaries for downloaded papers
        """
        if categories is None:
            categories = self.PHYSICS_CATEGORIES

        all_metadata = []

        for category in categories:
            logger.info(f"Downloading papers from category: {category}")

            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=papers_per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            category_dir = self.output_dir / category.replace('.', '_')
            category_dir.mkdir(parents=True, exist_ok=True)

            downloaded_count = 0
            for paper in search.results():
                try:
                    paper_id = paper.get_short_id()
                    paper_dir = category_dir / paper_id

                    # Skip if already downloaded
                    if skip_existing and paper_dir.exists():
                        logger.debug(f"Skipping existing paper: {paper_id}")
                        continue

                    paper_dir.mkdir(parents=True, exist_ok=True)

                    # Download paper source (LaTeX)
                    source_file = paper_dir / f"{paper_id}.tar.gz"

                    # Note: arxiv-py library doesn't directly support source download
                    # We'll save metadata and provide instructions for bulk download

                    metadata = {
                        'paper_id': paper_id,
                        'title': paper.title,
                        'authors': [author.name for author in paper.authors],
                        'abstract': paper.summary,
                        'categories': paper.categories,
                        'primary_category': paper.primary_category,
                        'published': paper.published.isoformat(),
                        'updated': paper.updated.isoformat(),
                        'pdf_url': paper.pdf_url,
                        'entry_id': paper.entry_id,
                        'download_dir': str(paper_dir),
                        'downloaded_at': datetime.now().isoformat()
                    }

                    # Save metadata
                    with open(paper_dir / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)

                    # Append to global metadata file
                    with open(self.metadata_file, 'a') as f:
                        f.write(json.dumps(metadata) + '\n')

                    all_metadata.append(metadata)
                    downloaded_count += 1

                    if downloaded_count >= papers_per_category:
                        break

                    # Rate limiting
                    time.sleep(3)  # 3 seconds between requests

                except Exception as e:
                    logger.error(f"Error downloading paper {paper.get_short_id()}: {str(e)}")
                    continue

            logger.info(f"Downloaded {downloaded_count} papers from {category}")

        logger.info(f"Total papers downloaded: {len(all_metadata)}")
        return all_metadata

    def download_source_bulk(self, metadata_file: str = None):
        """
        Instructions for bulk download of LaTeX source files.

        The arxiv Python library doesn't support source download directly.
        Use arXiv bulk download or AWS S3 mirror instead.

        See: https://info.arxiv.org/help/bulk_data.html
        """
        logger.info("""
        To download LaTeX source files:

        1. Via arXiv Bulk Access:
           - Apply for bulk access: https://info.arxiv.org/help/bulk_data.html
           - Use AWS S3 mirror: s3://arxiv/src/

        2. Via Individual Paper Download:
           - Use wget or curl with paper ID:
             wget https://arxiv.org/e-print/{paper_id}

        3. Via arxiv-vanity or similar tools:
           - https://www.arxiv-vanity.com/

        Metadata has been saved to: {self.metadata_file}
        Use paper IDs from metadata to download sources.
        """)


def main():
    """Main execution function."""
    # Configuration
    OUTPUT_DIR = './arxiv_papers'
    MAX_PAPERS_PER_CATEGORY = 100  # Start with 100 per category for pilot

    downloader = ArxivDownloader(
        output_dir=OUTPUT_DIR,
        max_papers=1000
    )

    # Download papers (metadata only)
    logger.info("Starting arXiv paper download...")
    metadata = downloader.download_papers(
        papers_per_category=MAX_PAPERS_PER_CATEGORY
    )

    logger.info(f"Download complete. Metadata saved to {downloader.metadata_file}")
    logger.info("Next step: Download LaTeX source files using bulk access or individual downloads")

    # Print instructions for source download
    downloader.download_source_bulk()


if __name__ == "__main__":
    main()
