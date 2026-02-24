"""
Quick validation script to test pipeline integration.

This script performs basic checks to ensure all components work together.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if not Path(filepath).exists():
        print(f"❌ Missing: {description} ({filepath})")
        return False
    print(f"✓ Found: {description}")
    return True


def check_imports():
    """Check if all scripts can be imported"""
    print("\n📦 Checking script imports...")

    scripts = [
        ("scrape_repos", "GitHubKMPScraper"),
        ("extract_code_blocks", "KotlinCodeExtractor"),
        ("create_training_pairs", "TrainingPairCreator"),
        ("prepare_for_training_v2", "KMPTrainingDataPreparer"),
    ]

    all_good = True
    for module_name, class_name in scripts:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"✓ {module_name}.{class_name}")
            else:
                print(f"❌ {module_name} missing {class_name}")
                all_good = False
        except ImportError as e:
            print(f"❌ Cannot import {module_name}: {e}")
            all_good = False

    return all_good


def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")

    deps = ["requests", "tqdm"]
    all_good = True

    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"❌ Missing: {dep}")
            all_good = False

    return all_good


def check_git():
    """Check if git is available"""
    print("\n🔧 Checking git...")
    import shutil
    if shutil.which("git"):
        print("✓ git found")
        return True
    else:
        print("❌ git not found in PATH")
        return False


def main():
    print("=" * 60)
    print("KMP Training Pipeline - Validation Check")
    print("=" * 60)

    all_checks = []

    # Check main scripts exist
    print("\n📄 Checking script files...")
    all_checks.append(check_file_exists("run_pipeline.py", "Main pipeline runner"))
    all_checks.append(check_file_exists("scrape_repos.py", "Repo scraper"))
    all_checks.append(check_file_exists("extract_code_blocks.py", "Code extractor"))
    all_checks.append(check_file_exists("create_training_pairs.py", "Pair creator"))
    all_checks.append(check_file_exists("prepare_for_training_v2.py", "Data preparer"))

    # Check imports
    all_checks.append(check_imports())

    # Check dependencies
    all_checks.append(check_dependencies())

    # Check git
    all_checks.append(check_git())

    # Summary
    print("\n" + "=" * 60)
    if all(all_checks):
        print("✅ All validation checks passed!")
        print("=" * 60)
        print("\nYou're ready to run the pipeline:")
        print("  python run_pipeline.py --max-repos 10 --github-token YOUR_TOKEN")
        return 0
    else:
        print("❌ Some validation checks failed")
        print("=" * 60)
        print("\nPlease fix the issues above before running the pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
