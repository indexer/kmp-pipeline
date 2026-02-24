"""
Diagnostic script to analyze "unknown" source_set assignments.

This helps identify:
1. Which file paths are being marked as "unknown"
2. What patterns they have that could be added to SOURCE_SET_PATTERNS
3. Statistics on unknown paths
4. Suggestions for fixing the patterns

Usage:
    python diagnose_unknown_sources.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import re


def analyze_unknown_sources(data_dir="data"):
    """Analyze blocks with 'unknown' source_set"""
    
    data_dir = Path(data_dir)
    blocks_file = data_dir / "raw_blocks" / "all_blocks.jsonl"
    
    if not blocks_file.exists():
        print(f"❌ {blocks_file} not found")
        return
    
    print("=" * 70)
    print("Unknown Source Set Diagnostic")
    print("=" * 70)
    
    # Load all blocks
    print("\n📂 Loading blocks...")
    all_blocks = []
    unknown_blocks = []
    
    with open(blocks_file) as f:
        for line in f:
            block = json.loads(line)
            all_blocks.append(block)
            if block.get("source_set") == "unknown":
                unknown_blocks.append(block)
    
    total = len(all_blocks)
    unknown_count = len(unknown_blocks)
    
    print(f"  Total blocks: {total:,}")
    print(f"  Unknown source_set: {unknown_count:,} ({100*unknown_count/total:.1f}%)")
    
    if not unknown_blocks:
        print("\n✅ No unknown source sets found!")
        return
    
    # Analyze file paths
    print("\n" + "=" * 70)
    print("File Path Analysis")
    print("=" * 70)
    
    path_patterns = defaultdict(int)
    repos = Counter()
    file_types = Counter()
    
    for block in unknown_blocks:
        path = block.get("file_path", "")
        repo = block.get("repo", "unknown")
        
        repos[repo] += 1
        
        # Extract path pattern
        path_normalized = path.replace("\\", "/").lower()
        
        # Check for common patterns
        if "/src/" in path_normalized:
            parts = path_normalized.split("/src/")
            if len(parts) > 1:
                src_path = parts[1].split("/")[0] if "/" in parts[1] else parts[1]
                path_patterns[f"src/{src_path}"] += 1
        
        # Check file type
        if path.endswith(".kt"):
            file_types[".kt"] += 1
        elif path.endswith(".kts"):
            file_types[".kts"] += 1
    
    print("\n📊 Top Path Patterns in Unknown Files:")
    for pattern, count in sorted(path_patterns.items(), key=lambda x: -x[1])[:20]:
        pct = 100 * count / unknown_count
        print(f"  {pattern:50s} {count:>6,}  ({pct:5.1f}%)")
    
    print("\n📊 Top Repos with Unknown Paths:")
    for repo, count in repos.most_common(10):
        pct = 100 * count / unknown_count
        print(f"  {repo:50s} {count:>6,}  ({pct:5.1f}%)")
    
    # Sample unknown paths
    print("\n" + "=" * 70)
    print("Sample Unknown File Paths (first 30)")
    print("=" * 70)
    
    seen_patterns = set()
    samples_shown = 0
    
    for block in unknown_blocks:
        if samples_shown >= 30:
            break
        
        path = block.get("file_path", "")
        
        # Extract a pattern to avoid showing too many similar paths
        pattern = extract_pattern(path)
        if pattern in seen_patterns:
            continue
        
        seen_patterns.add(pattern)
        samples_shown += 1
        
        print(f"  {samples_shown:2d}. {path}")
        print(f"      Repo: {block.get('repo', 'unknown')}")
        print(f"      Type: {block.get('type', 'unknown')}")
        print()
    
    # Suggest patterns
    print("=" * 70)
    print("Suggested SOURCE_SET_PATTERNS Additions")
    print("=" * 70)
    
    suggestions = analyze_and_suggest_patterns(unknown_blocks)
    
    if suggestions:
        print("\nAdd these to SOURCE_SET_PATTERNS in extract_code_blocks.py:\n")
        print("SOURCE_SET_PATTERNS = {")
        for source_set, patterns in suggestions.items():
            print(f'    "{source_set}": [')
            for pattern in patterns:
                print(f'        "{pattern}",')
            print(f'    ],')
        print("    # ... existing patterns ...")
        print("}\n")
    else:
        print("\n⚠️  Could not identify clear patterns.")
        print("Manual review of file paths is recommended.")
    
    # Categorization suggestions
    print("\n" + "=" * 70)
    print("Categorization Hints")
    print("=" * 70)
    
    print("\nBased on common KMP project structures:")
    print("  • 'src/main' → likely commonMain or androidMain")
    print("  • 'src/test' → likely commonTest or androidTest")
    print("  • Root-level .kt files → might be build scripts or standalone files")
    print("  • 'app/src' → typically androidMain")
    print("  • 'shared/src' → check subdirectories for commonMain, androidMain, etc.")


def extract_pattern(path: str) -> str:
    """Extract a general pattern from a path for grouping"""
    path = path.replace("\\", "/")
    
    # Remove repo name (first part)
    parts = path.split("/")
    if len(parts) > 2:
        # Keep structure after repo name
        return "/".join(parts[1:4])  # repo/module/src/...
    return path


def analyze_and_suggest_patterns(unknown_blocks):
    """Analyze unknown blocks and suggest new SOURCE_SET_PATTERNS"""
    suggestions = defaultdict(set)
    
    for block in unknown_blocks:
        path = block.get("file_path", "").replace("\\", "/").lower()
        
        # Heuristic detection
        if "src/main" in path and "test" not in path:
            if "android" in path:
                suggestions["androidMain"].add(extract_source_marker(path, "android"))
            elif "ios" in path:
                suggestions["iosMain"].add(extract_source_marker(path, "ios"))
            elif "desktop" in path or "jvm" in path:
                suggestions["desktopMain"].add(extract_source_marker(path, "desktop"))
            elif "js" in path or "wasm" in path:
                suggestions["jsMain"].add(extract_source_marker(path, "js"))
            else:
                suggestions["commonMain"].add(extract_source_marker(path, "main"))
        
        elif "src/test" in path:
            if "android" in path:
                suggestions["androidTest"].add(extract_source_marker(path, "android"))
            elif "ios" in path:
                suggestions["iosTest"].add(extract_source_marker(path, "ios"))
            else:
                suggestions["commonTest"].add(extract_source_marker(path, "test"))
        
        # Check for module-based paths
        elif "/kotlin/" in path or "/java/" in path:
            if "test" in path:
                suggestions["commonTest"].add(extract_source_marker(path, "test"))
            else:
                suggestions["commonMain"].add(extract_source_marker(path, "main"))
    
    # Convert sets to lists and filter
    filtered_suggestions = {}
    for source_set, patterns in suggestions.items():
        # Only include patterns that appear frequently
        pattern_list = sorted(patterns)
        if len(pattern_list) <= 5:  # Reasonable number of patterns
            filtered_suggestions[source_set] = pattern_list
    
    return filtered_suggestions


def extract_source_marker(path: str, hint: str) -> str:
    """Extract a source set marker from path"""
    # Try to find a distinctive part of the path
    if "/src/" in path:
        parts = path.split("/src/")
        if len(parts) > 1:
            after_src = parts[1].split("/")[0]
            return f"src/{after_src}"
    
    # Fallback: use hint
    return hint


def suggest_manual_fixes():
    """Print manual fix suggestions"""
    print("\n" + "=" * 70)
    print("Manual Fix Options")
    print("=" * 70)
    
    print("""
If automated detection doesn't work, you can:

1. **Examine actual file paths** and add patterns manually to SOURCE_SET_PATTERNS
   
2. **Use repo structure conventions**:
   - Check if repos follow standard Gradle conventions
   - Look for build.gradle.kts to understand source set structure
   
3. **Default assignment strategy**:
   - Assign all "unknown" to "commonMain" if they're not platform-specific
   - Filter out unknown blocks in prepare_for_training_v2.py
   
4. **Add a fallback detector** in extract_code_blocks.py:
   ```python
   def _detect_source_set_enhanced(self, path: str) -> str:
       # Try existing patterns first
       result = self._detect_source_set(path)
       if result != "unknown":
           return result
       
       # Fallback heuristics
       path_lower = path.lower()
       if "test" in path_lower:
           return "commonTest"
       if "android" in path_lower:
           return "androidMain"
       if "ios" in path_lower:
           return "iosMain"
       
       # Default to commonMain for .kt files
       if path.endswith(".kt"):
           return "commonMain"
       
       return "unknown"
   ```

5. **Filter in training data prep**:
   Add this to prepare_for_training_v2.py quality validation:
   ```python
   # In validate_kotlin_pair():
   if pair.get("source_set") == "unknown":
       return False, "unknown_source_set"
   ```
""")


if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose unknown source sets")
    parser.add_argument("--data-dir", default=os.environ.get("KMP_DATA_DIR", "data"))
    args = parser.parse_args()
    
    analyze_unknown_sources(args.data_dir)
    suggest_manual_fixes()
    
    print("\n✅ Diagnostic complete!")
