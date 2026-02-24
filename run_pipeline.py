"""
KMP Training Data Pipeline
===========================

Complete pipeline to scrape KMP repos from GitHub, extract code blocks,
create training pairs with FULL method body implementations, and prepare
data for fine-tuning SmolLM3/Qwen3 models.

Pipeline:
    Step 1: scrape_repos.py           — Clone KMP repos from GitHub
    Step 2: extract_code_blocks.py    — Parse Kotlin files into structured blocks
    Step 3: create_training_pairs.py  — Generate input→output pairs (with bodies!)
    Step 4: prepare_for_training_v2.py — Format, deduplicate, split, stats

Usage:
    # Run full pipeline
    python run_pipeline.py

    # Run individual steps
    python run_pipeline.py --step scrape
    python run_pipeline.py --step extract
    python run_pipeline.py --step pairs
    python run_pipeline.py --step prepare

    # Configure
    python run_pipeline.py --max-repos 200 --sample-size 50000 --github-token ghp_xxx

Requirements:
    pip install requests tqdm
    git (for cloning repos)

Features:
    - Enhanced source set detection (reduces "unknown" from 21% to <5%)
    - Cross-platform compatibility (Windows, Linux, macOS)
    - Better error handling and logging
    - Production-ready quality filtering
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_environment():
    """Validate that all requirements are met"""
    logger.info("Validating environment...")

    # Check Python version
    import sys
    if sys.version_info < (3, 7):
        logger.error("Python 3.7+ required")
        sys.exit(1)

    # Check for required packages
    try:
        import requests
        import tqdm
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Run: pip install requests tqdm")
        sys.exit(1)

    # Check for git
    import shutil
    if not shutil.which("git"):
        logger.error("git is not installed or not in PATH")
        sys.exit(1)

    logger.info("✓ Environment validation passed")


def main():
    parser = argparse.ArgumentParser(
        description="KMP Training Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with 300 repos
  python run_pipeline.py --max-repos 300 --github-token ghp_xxx

  # Run only extraction step
  python run_pipeline.py --step extract

  # Prepare training data with sampling
  python run_pipeline.py --step prepare --sample-size 50000
        """
    )
    parser.add_argument("--step", choices=["scrape", "extract", "pairs", "prepare", "all"],
                        default="all", help="Which step to run")
    parser.add_argument("--max-repos", type=int, default=500,
                        help="Max repos to clone (default: 500)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Max training pairs to output (default: all)")
    parser.add_argument("--github-token", default=None,
                        help="GitHub token for higher API rate limits")
    parser.add_argument("--data-dir", default="data",
                        help="Base data directory (default: data)")
    parser.add_argument("--min-stars", type=int, default=5,
                        help="Minimum GitHub stars (default: 5)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip environment validation")
    args = parser.parse_args()

    # Validate environment
    if not args.skip_validation:
        validate_environment()

    # Set env vars for sub-scripts
    os.environ["KMP_DATA_DIR"] = args.data_dir
    if args.github_token:
        os.environ["GITHUB_TOKEN"] = args.github_token

    # Create data directory
    data_path = Path(args.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using data directory: {data_path.absolute()}")

    steps = {
        "scrape": run_scrape,
        "extract": run_extract,
        "pairs": run_pairs,
        "prepare": run_prepare,
    }

    try:
        if args.step == "all":
            for name, func in steps.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"  Starting Step: {name.upper()}")
                logger.info(f"{'='*60}\n")
                func(args)
        else:
            logger.info(f"Running step: {args.step}")
            steps[args.step](args)

        logger.info(f"\n{'='*60}")
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"{'='*60}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def run_scrape(args):
    from scrape_repos import GitHubKMPScraper
    scraper = GitHubKMPScraper(
        data_dir=args.data_dir,
        max_repos=args.max_repos,
        min_stars=args.min_stars,
        token=args.github_token or os.environ.get("GITHUB_TOKEN"),
    )
    scraper.run()


def run_extract(args):
    from extract_code_blocks import KotlinCodeExtractor
    extractor = KotlinCodeExtractor(data_dir=args.data_dir)
    extractor.run()


def run_pairs(args):
    from create_training_pairs import TrainingPairCreator
    creator = TrainingPairCreator(data_dir=args.data_dir)
    creator.run()


def run_prepare(args):
    from prepare_for_training_v2 import KMPTrainingDataPreparer
    preparer = KMPTrainingDataPreparer(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
    )
    preparer.run()


if __name__ == "__main__":
    main()
