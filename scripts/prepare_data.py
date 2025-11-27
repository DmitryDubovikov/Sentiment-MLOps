#!/usr/bin/env python3
"""Script to download and prepare the IMDb dataset for training."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import download_imdb_subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Download and prepare the IMDb dataset."""
    parser = argparse.ArgumentParser(description="Prepare IMDb sentiment dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/imdb_sample.csv",
        help="Output path for the dataset (default: data/imdb_sample.csv)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=2000,
        help="Number of samples to download (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if file exists",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        logger.info(f"Dataset already exists at {output_path}. Use --force to re-download.")
        return

    if output_path.exists() and args.force:
        logger.info(f"Removing existing dataset at {output_path}")
        output_path.unlink()

    logger.info(f"Downloading {args.samples} samples to {output_path}")

    result_path = download_imdb_subset(
        output_path=output_path,
        n_samples=args.samples,
        random_state=args.seed,
    )

    logger.info(f"Dataset prepared successfully: {result_path}")

    # Print basic stats
    import pandas as pd

    df = pd.read_csv(result_path)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()
