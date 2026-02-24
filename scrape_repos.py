"""
Step 1: Scrape KMP repositories from GitHub.

Searches for Kotlin Multiplatform repos, filters by quality,
and clones them locally for code extraction.

Features:
- Multiple search queries to maximize coverage
- Rate-limit aware (respects GitHub API limits)
- Deduplicates repos across queries
- Filters by stars, size, and KMP indicators
- Shallow clone (--depth 1) to save disk space
- Resume support (skips already-cloned repos)
- Saves repo metadata for analysis

Usage:
    python scrape_repos.py
    python scrape_repos.py --max-repos 200 --min-stars 10
"""

import os
import json
import time
import subprocess
import hashlib
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Install: pip install requests tqdm")
    exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RepoInfo:
    """Metadata about a scraped repository."""
    name: str
    full_name: str
    url: str
    clone_url: str
    stars: int
    forks: int
    language: str
    description: str
    topics: list
    size_kb: int
    updated_at: str
    license: str
    has_kmp_indicator: bool
    search_query: str
    cloned: bool = False
    clone_path: str = ""
    kt_file_count: int = 0


class GitHubKMPScraper:
    """Scrape KMP repositories from GitHub."""

    # Multiple search queries to maximize KMP repo coverage
    SEARCH_QUERIES = [
        # Direct KMP queries
        "kotlin multiplatform mobile",
        "kotlin multiplatform",
        "KMM shared module",
        "compose multiplatform",
        "kotlin multiplatform ios android",

        # Framework-specific
        "ktor client multiplatform",
        "sqldelight multiplatform",
        "koin multiplatform",
        "decompose multiplatform",
        "voyager multiplatform",
        "moko multiplatform",

        # Architecture patterns
        "expect actual kotlin",
        "commonMain androidMain iosMain",
        "kotlin multiplatform clean architecture",
        "KMP MVVM",

        # Build config patterns
        "kotlin(\"multiplatform\")",
        "KotlinMultiplatform",
        "iosArm64 iosX64 iosSimulatorArm64",

        # Specific libraries/samples
        "kotlin multiplatform sample",
        "kotlin multiplatform template",
        "kotlin multiplatform starter",
        "CMP compose multiplatform app",
    ]

    # Files/dirs that indicate a real KMP project
    KMP_INDICATORS = [
        "shared/src/commonMain",
        "composeApp/src/commonMain",
        "commonMain/kotlin",
        "androidMain/kotlin",
        "iosMain/kotlin",
        "build.gradle.kts",  # We'll check content for multiplatform
    ]

    def __init__(self, data_dir="data", max_repos=500, min_stars=5, token=None):
        self.data_dir = Path(data_dir)
        self.repos_dir = self.data_dir / "repos"
        self.meta_dir = self.data_dir / "metadata"
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.max_repos = max_repos
        self.min_stars = min_stars
        self.token = token or os.environ.get("GITHUB_TOKEN")

        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            print(f"✅ Using GitHub token (higher rate limits)")
        else:
            print(f"⚠️  No GitHub token — limited to 10 search requests/min")
            print(f"   Set GITHUB_TOKEN env var or pass --github-token")

        self.session.headers["Accept"] = "application/vnd.github.v3+json"

        # Track seen repos to avoid duplicates
        self.seen_repos = set()
        self.all_repos = []

    def run(self):
        """Execute the full scraping pipeline."""
        print(f"\n🔍 Scraping KMP repos from GitHub")
        print(f"   Max repos: {self.max_repos}")
        print(f"   Min stars: {self.min_stars}")
        print(f"   Output:    {self.repos_dir}\n")

        # Phase 1: Search
        self._search_all_queries()

        # Phase 2: Filter & rank
        filtered = self._filter_repos()

        # Phase 3: Clone
        self._clone_repos(filtered)

        # Phase 4: Validate (check for actual KMP structure)
        validated = self._validate_repos(filtered)

        # Save metadata
        self._save_metadata(validated)

        # Summary
        cloned_count = sum(1 for r in validated if r.cloned)
        print(f"\n{'='*60}")
        print(f"✅ Scraping complete!")
        print(f"   Searched:   {len(self.all_repos)} repos found")
        print(f"   Filtered:   {len(filtered)} repos (≥{self.min_stars} stars)")
        print(f"   Cloned:     {cloned_count} repos")
        print(f"   Validated:  {sum(1 for r in validated if r.has_kmp_indicator)} with KMP structure")
        print(f"   Location:   {self.repos_dir}")
        print(f"{'='*60}")

    def _search_all_queries(self):
        """Run all search queries and collect unique repos."""
        for i, query in enumerate(self.SEARCH_QUERIES):
            if len(self.all_repos) >= self.max_repos * 2:  # Collect 2x for filtering
                break

            print(f"\n🔍 [{i+1}/{len(self.SEARCH_QUERIES)}] Searching: {query}")
            repos = self._search_github(query)
            new_count = 0

            for repo in repos:
                if repo.full_name not in self.seen_repos:
                    self.seen_repos.add(repo.full_name)
                    self.all_repos.append(repo)
                    new_count += 1

            print(f"   Found {len(repos)} results, {new_count} new (total: {len(self.all_repos)})")

        print(f"\n📊 Total unique repos found: {len(self.all_repos)}")

    def _search_github(self, query: str, max_pages=5) -> list:
        """Search GitHub API with pagination."""
        repos = []
        per_page = 100  # Max allowed

        for page in range(1, max_pages + 1):
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"{query} language:kotlin",
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }

            response = self._api_request(url, params)
            if not response:
                break

            items = response.get("items", [])
            if not items:
                break

            for item in items:
                repo = RepoInfo(
                    name=item["name"],
                    full_name=item["full_name"],
                    url=item["html_url"],
                    clone_url=item["clone_url"],
                    stars=item["stargazers_count"],
                    forks=item["forks_count"],
                    language=item.get("language", ""),
                    description=item.get("description", "") or "",
                    topics=item.get("topics", []),
                    size_kb=item.get("size", 0),
                    updated_at=item.get("updated_at", ""),
                    license=(item.get("license") or {}).get("spdx_id", ""),
                    has_kmp_indicator=False,
                    search_query=query,
                )
                repos.append(repo)

            # Check if we got all results
            if len(items) < per_page:
                break

        return repos

    def _api_request(self, url: str, params: dict = None) -> dict:
        """Make a rate-limit-aware API request."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                # Check rate limit
                remaining = int(response.headers.get("X-RateLimit-Remaining", 999))
                if remaining < 5:
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    wait = max(reset_time - time.time(), 0) + 5
                    print(f"   ⏳ Rate limit near ({remaining} left). Waiting {wait:.0f}s...")
                    time.sleep(wait)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    # Rate limited
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    wait = max(reset_time - time.time(), 0) + 10
                    print(f"   ⚠️ Rate limited. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                elif response.status_code == 422:
                    # Validation error (e.g., too many results)
                    return {}
                else:
                    print(f"   ⚠️ HTTP {response.status_code}: {response.text[:200]}")
                    return {}

            except requests.exceptions.RequestException as e:
                print(f"   ❌ Request error: {e}")
                time.sleep(5 * (attempt + 1))

        return {}

    def _filter_repos(self) -> list:
        """Filter and rank repos by quality."""
        filtered = []
        for repo in self.all_repos:
            # Skip repos below star threshold
            if repo.stars < self.min_stars:
                continue

            # Skip very small repos (likely empty/template)
            if repo.size_kb < 10:
                continue

            # Skip very large repos (likely monorepos with non-KMP content)
            if repo.size_kb > 500_000:  # >500MB
                continue

            # Boost score for KMP-related topics
            kmp_topics = {"kotlin-multiplatform", "kmp", "kmm", "compose-multiplatform",
                          "kotlin-multiplatform-mobile", "multiplatform"}
            topic_bonus = len(set(repo.topics) & kmp_topics) * 10

            # Boost for recent updates
            recency_bonus = 0
            try:
                updated = datetime.fromisoformat(repo.updated_at.replace("Z", "+00:00"))
                days_ago = (datetime.now(updated.tzinfo) - updated).days
                if days_ago < 180:
                    recency_bonus = 20
                elif days_ago < 365:
                    recency_bonus = 10
            except Exception:
                pass

            # Boost for KMP-related description
            desc_bonus = 0
            desc_lower = repo.description.lower()
            kmp_keywords = ["multiplatform", "kmp", "kmm", "compose", "shared",
                            "ios", "android", "cross-platform", "ktor", "sqldelight"]
            desc_bonus = sum(5 for kw in kmp_keywords if kw in desc_lower)

            repo._score = repo.stars + topic_bonus + recency_bonus + desc_bonus
            filtered.append(repo)

        # Sort by score and limit
        filtered.sort(key=lambda r: r._score, reverse=True)
        filtered = filtered[:self.max_repos]

        print(f"\n📋 Filtered: {len(filtered)} repos (from {len(self.all_repos)} total)")
        if filtered:
            print(f"   Top:    {filtered[0].full_name} ({filtered[0].stars}⭐)")
            print(f"   Bottom: {filtered[-1].full_name} ({filtered[-1].stars}⭐)")

        return filtered

    def _clone_repos(self, repos: list):
        """Clone repos with shallow clone (--depth 1)."""
        print(f"\n📥 Cloning {len(repos)} repositories...")

        for i, repo in enumerate(tqdm(repos, desc="Cloning")):
            clone_path = self.repos_dir / repo.name

            # Skip if already cloned
            if clone_path.exists() and any(clone_path.iterdir()):
                repo.cloned = True
                repo.clone_path = str(clone_path)
                continue

            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "--quiet",
                     repo.clone_url, str(clone_path)],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    repo.cloned = True
                    repo.clone_path = str(clone_path)
                else:
                    # Clean up failed clone
                    if clone_path.exists():
                        shutil.rmtree(clone_path, ignore_errors=True)
            except subprocess.TimeoutExpired:
                logger.warning(f"Clone timeout for {repo.name}")
                if clone_path.exists():
                    shutil.rmtree(clone_path, ignore_errors=True)
            except Exception as e:
                logger.debug(f"Clone error for {repo.name}: {e}")
                if clone_path.exists():
                    shutil.rmtree(clone_path, ignore_errors=True)

            # Small delay to be nice to GitHub
            if (i + 1) % 50 == 0:
                time.sleep(2)

    def _validate_repos(self, repos: list) -> list:
        """Validate that cloned repos actually have KMP structure."""
        print(f"\n🔎 Validating KMP structure...")

        for repo in repos:
            if not repo.cloned:
                continue

            clone_path = Path(repo.clone_path)

            # Check for KMP indicators
            for indicator in self.KMP_INDICATORS:
                if (clone_path / indicator).exists():
                    repo.has_kmp_indicator = True
                    break

            # Also check gradle files for multiplatform plugin
            if not repo.has_kmp_indicator:
                for gradle_file in clone_path.rglob("*.gradle.kts"):
                    try:
                        content = gradle_file.read_text(errors="ignore")
                        if any(kw in content for kw in [
                            "multiplatform", "KotlinMultiplatform",
                            'kotlin("multiplatform")', "commonMain"
                        ]):
                            repo.has_kmp_indicator = True
                            break
                    except Exception:
                        pass

            # Count Kotlin files
            try:
                repo.kt_file_count = len(list(clone_path.rglob("*.kt")))
            except Exception:
                repo.kt_file_count = 0

        valid = sum(1 for r in repos if r.has_kmp_indicator)
        print(f"   {valid}/{sum(1 for r in repos if r.cloned)} repos have KMP structure")

        return repos

    def _save_metadata(self, repos: list):
        """Save repo metadata for analysis."""
        meta_file = self.meta_dir / "repos.jsonl"
        with open(meta_file, "w") as f:
            for repo in repos:
                d = asdict(repo)
                d.pop("_score", None)
                f.write(json.dumps(d) + "\n")

        # Summary stats
        stats = {
            "total_found": len(self.all_repos),
            "total_filtered": len(repos),
            "total_cloned": sum(1 for r in repos if r.cloned),
            "total_validated": sum(1 for r in repos if r.has_kmp_indicator),
            "total_kt_files": sum(r.kt_file_count for r in repos),
            "scrape_date": datetime.now().isoformat(),
            "queries_used": len(self.SEARCH_QUERIES),
        }
        stats_file = self.meta_dir / "scrape_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n📄 Metadata saved: {meta_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-repos", type=int, default=500)
    parser.add_argument("--min-stars", type=int, default=5)
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    scraper = GitHubKMPScraper(
        data_dir=args.data_dir,
        max_repos=args.max_repos,
        min_stars=args.min_stars,
        token=args.github_token,
    )
    scraper.run()
