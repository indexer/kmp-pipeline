# KMP Training Data Pipeline

Complete pipeline to build training data for fine-tuning LLMs on Kotlin Multiplatform code generation.

## The Problem

Previous training pairs only contained **interfaces and signatures without method bodies**. The model learned to output declarations without implementations:

```kotlin
// Model would generate THIS (bad):
interface UserRepository {
    suspend fun getUser(id: String): User
}

// Instead of THIS (good):
class UserRepositoryImpl(
    private val api: UserApi,
    private val db: UserDatabase,
) : UserRepository {
    override suspend fun getUser(id: String): User = withContext(Dispatchers.Default) {
        try {
            val user = api.fetchUser(id)
            db.userQueries.upsert(user.toEntity())
            user.toDomain()
        } catch (e: Exception) {
            db.userQueries.selectById(id).executeAsOne().toDomain()
        }
    }
}
```

## The Fix

This pipeline generates 7 types of training pairs that **all require full method body implementations**:

| Pair Type | Input | Target |
|-----------|-------|--------|
| `expect_actual` | expect declaration | actual with full body |
| `interface_implementation` | interface | class with all method bodies |
| `description_to_code` | natural description + signature | complete implementation |
| `skeleton_to_complete` | class with TODO placeholders | same class with bodies filled |
| `composable` | @Composable signature | full UI component |
| `full_file` | file description | complete Kotlin file |
| `gradle` | project description | full build.gradle.kts |

A quality filter **rejects** any pair where the target doesn't contain real implementation logic.

## Quick Start

```bash
# Install dependencies
pip install requests tqdm

# Validate pipeline (recommended first run)
python3 validate_pipeline.py

# Set GitHub token (optional but recommended for higher rate limits)
export GITHUB_TOKEN=ghp_your_token_here

# Run full pipeline
python3 run_pipeline.py

# Or run individual steps
python3 run_pipeline.py --step scrape --max-repos 300
python3 run_pipeline.py --step extract
python3 run_pipeline.py --step pairs
python3 run_pipeline.py --step prepare --sample-size 50000
```

## Recent Improvements (v2.0)

- **Enhanced Source Set Detection**: Reduced "unknown" classifications from ~21% to <5%
- **Cross-Platform Compatibility**: Works on Windows, Linux, and macOS
- **Better Error Handling**: Comprehensive logging and graceful failure recovery
- **Production-Ready Quality Filtering**: Multi-dimensional quality scoring
- **Improved Integration**: All scripts work seamlessly with `run_pipeline.py`

## Pipeline Steps

### Step 1: `scrape_repos.py` — Clone KMP Repos
- Searches GitHub with 20+ KMP-specific queries
- Filters by stars, size, recency
- Validates KMP project structure
- Shallow clone (--depth 1) to save space
- Resume support (skips already-cloned repos)

### Step 2: `extract_code_blocks.py` — Parse Kotlin Files
- Extracts functions, classes, interfaces, objects
- Detects expect/actual declarations
- Tags blocks with `has_body` flag (real implementation?)
- Detects architecture patterns (ViewModel, Repository, UseCase)
- Extracts @Composable functions
- Includes full-file blocks for substantial files

### Step 3: `create_training_pairs.py` — Generate Pairs
- 7 pair generators (see table above)
- Quality filter requires real method bodies
- Deduplicates by content hash
- Splits into train/val/test (90/5/5)

### Step 4: `prepare_for_training_v2.py` — Final Preparation (Production)
- Advanced deduplication (exact + structural fingerprinting)
- Quality-weighted balancing with multi-dimensional scoring
- KMP-specific validation (syntax, braces, real implementations)
- Stratified splits for consistent type distribution
- Optional sampling for smaller GPU runs
- Outputs `input_text`/`target_text` JSONL format
- Comprehensive quality metrics and statistics

## Output Structure

```
data/
├── repos/                    # Cloned GitHub repos
├── metadata/
│   ├── repos.jsonl          # Repo metadata
│   └── scrape_stats.json
├── raw_blocks/
│   ├── all_blocks.jsonl     # Extracted code blocks
│   └── extraction_stats.json
├── training_pairs/
│   ├── train.jsonl          # Raw pairs (90%)
│   ├── val.jsonl            # Validation (5%)
│   └── test.jsonl           # Test (5%)
└── final_training/
    ├── train.jsonl          # ← Upload to Google Drive
    ├── val.jsonl            # ← Upload to Google Drive
    ├── test.jsonl
    └── all_train.jsonl      # Combined for notebooks
```

## Using with SmolLM3 Notebook

1. Upload `data/final_training/` to Google Drive
2. In notebook Cell 10, update `DATA_PATHS`:

```python
DATA_PATHS = [
    f"{SAVE_DIR}/final_training/train.jsonl",
    f"{SAVE_DIR}/final_training/val.jsonl",
]
```

The notebook's `format_sample()` already reads `input_text`/`target_text` keys — no other changes needed.

## Using with Qwen3-Coder Notebooks

Same process — the notebooks use the same data format.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--max-repos` | 500 | Max repos to clone |
| `--min-stars` | 5 | Minimum GitHub stars |
| `--sample-size` | None | Cap total training pairs |
| `--github-token` | env | GitHub API token |
| `--data-dir` | data | Base data directory |

## Tips

- **More repos = better.** 500+ repos gives good coverage.
- **GitHub token required** for >60 requests/hour. Create at github.com/settings/tokens.
- **Disk space:** ~5-10GB for 500 repos (shallow clones).
- **Time:** Full pipeline takes ~30-60 min depending on network.
- **Quality > Quantity:** 30K high-quality pairs with bodies > 1M pairs without bodies.
