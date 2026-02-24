"""
Post-Extraction Fix: Re-classify unknown source_sets in raw blocks.

This fixes unknown source_sets in the extracted blocks BEFORE creating
training pairs, which is more effective than fixing training pairs later.

Usage:
    python fix_raw_blocks.py
"""

import json
from pathlib import Path
from collections import Counter


def detect_source_set_from_block(block: dict) -> str:
    """
    Detect source set from block data.
    
    Uses multiple signals:
    1. File path analysis (better heuristics)
    2. Code content (imports, platform APIs)
    3. Block type (expect/actual patterns)
    4. Architecture patterns
    """
    path = block.get("file_path", "").replace("\\", "/").lower()
    code = block.get("code", "").lower()
    block_type = block.get("type", "")
    imports = " ".join(block.get("imports", [])).lower()
    
    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 1: File Path Analysis (Enhanced)
    # ═══════════════════════════════════════════════════════════════════
    
    # Test paths (highest priority)
    if any(marker in path for marker in ["/test/", "/androidtest/", "/iostest/", "/commontest/"]):
        if "android" in path or "app/" in path:
            return "androidTest"
        elif "ios" in path:
            return "iosTest"
        else:
            return "commonTest"
    
    # Standard Gradle source paths
    if "/src/main/" in path:
        # Check module context
        if any(marker in path for marker in ["android/", "app/", "/app/"]):
            return "androidMain"
        elif "ios" in path:
            return "iosMain"
        elif any(marker in path for marker in ["desktop/", "jvm/"]):
            return "desktopMain"
        elif any(marker in path for marker in ["js/", "wasm/"]):
            return "jsMain"
        elif "native/" in path:
            return "nativeMain"
        # Default for src/main without platform indicators
        return "commonMain"
    
    # Platform-specific directories
    if any(marker in path for marker in ["android/", "/app/src/"]):
        return "androidMain"
    
    if "ios/" in path:
        return "iosMain"
    
    if any(marker in path for marker in ["desktop/", "jvm/"]):
        return "desktopMain"
    
    if any(marker in path for marker in ["js/", "wasm/"]):
        return "jsMain"
    
    # Common/shared modules (very common pattern)
    if any(marker in path for marker in ["shared/", "common/", "core/", "domain/", "data/"]):
        if "test" not in path:
            return "commonMain"
        else:
            return "commonTest"
    
    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 2: Code Content Analysis
    # ═══════════════════════════════════════════════════════════════════
    
    # Android platform indicators
    android_indicators = [
        "import android.", "import androidx.", 
        "android.content", "android.os", "android.app",
        "activity", "fragment", "bundle", "context",
        "androidx.compose", "androidx.lifecycle"
    ]
    if any(ind in imports or ind in code for ind in android_indicators):
        return "androidMain"
    
    # iOS platform indicators
    ios_indicators = [
        "import platform.uikit", "import platform.foundation",
        "platform.darwin", "uiview", "uiviewcontroller",
        "nsstring", "nsobject", "platform.posix"
    ]
    if any(ind in imports or ind in code for ind in ios_indicators):
        return "iosMain"
    
    # JVM/Desktop indicators
    jvm_indicators = [
        "import java.", "import javax.", 
        "import kotlin.jvm",
        "swing", "javafx", "jframe"
    ]
    if any(ind in imports or ind in code for ind in jvm_indicators):
        return "desktopMain"
    
    # JS indicators
    js_indicators = [
        "import js.", "external interface", "external class",
        "js(\"", "window.", "document.", "import kotlinx.browser"
    ]
    if any(ind in imports or ind in code for ind in js_indicators):
        return "jsMain"
    
    # Test indicators (if not caught by path)
    test_indicators = [
        "@test", "import kotlin.test", 
        "import io.kotest", "import org.junit",
        "assertequals", "asserttrue", "junit"
    ]
    if any(ind in imports or ind in code for ind in test_indicators):
        return "commonTest"
    
    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 3: Block Type Patterns
    # ═══════════════════════════════════════════════════════════════════
    
    # Expect declarations are always in commonMain
    if block.get("is_expect"):
        return "commonMain"
    
    # Actual implementations - check what platform
    if block.get("is_actual"):
        # Check code for platform-specific APIs
        if any(ind in code for ind in ["android.", "androidx."]):
            return "androidMain"
        elif any(ind in code for ind in ["platform.uikit", "platform.foundation"]):
            return "iosMain"
        elif any(ind in code for ind in ["java.", "javax."]):
            return "desktopMain"
        # Default actual to commonMain (some actuals are in common with typealiases)
        return "commonMain"
    
    # Composables - usually common or android
    if block_type == "composable" or "@composable" in code:
        if any(ind in code for ind in ["android.", "androidx."]):
            return "androidMain"
        return "commonMain"
    
    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 4: Architecture Patterns
    # ═══════════════════════════════════════════════════════════════════
    
    arch = block.get("arch_pattern", "")
    
    # ViewModels are typically in common or android
    if arch == "viewmodel":
        if "android" in code:
            return "androidMain"
        return "commonMain"
    
    # Repositories and UseCases are typically in commonMain
    if arch in ["repository", "usecase", "service"]:
        return "commonMain"
    
    # DI modules are usually in commonMain
    if arch == "di_module":
        return "commonMain"
    
    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 5: Default Strategy
    # ═══════════════════════════════════════════════════════════════════
    
    # Most KMP code is in commonMain
    # This is the safest default for .kt files
    return "commonMain"


def fix_raw_blocks(data_dir="data"):
    """Fix unknown source_sets in raw blocks before training pair creation"""
    
    data_dir = Path(data_dir)
    input_file = data_dir / "raw_blocks" / "all_blocks.jsonl"
    output_file = data_dir / "raw_blocks" / "all_blocks_fixed.jsonl"
    backup_file = data_dir / "raw_blocks" / "all_blocks.backup.jsonl"
    
    if not input_file.exists():
        print(f"❌ {input_file} not found")
        return
    
    print("=" * 70)
    print("Fix Unknown Source Sets in Raw Blocks")
    print("=" * 70)
    
    # Load blocks
    print(f"\n📂 Loading blocks from {input_file}...")
    blocks = []
    with open(input_file) as f:
        for line in f:
            blocks.append(json.loads(line))
    
    total = len(blocks)
    unknown_before = sum(1 for b in blocks if b.get("source_set") == "unknown")
    
    print(f"  Total blocks: {total:,}")
    print(f"  Unknown before: {unknown_before:,} ({100*unknown_before/total:.1f}%)")
    
    # Fix unknowns
    print(f"\n🔄 Re-classifying unknowns...")
    fixed_count = 0
    new_assignments = Counter()
    
    for block in blocks:
        if block.get("source_set") == "unknown":
            new_source_set = detect_source_set_from_block(block)
            block["source_set"] = new_source_set
            fixed_count += 1
            new_assignments[new_source_set] += 1
    
    unknown_after = sum(1 for b in blocks if b.get("source_set") == "unknown")
    
    print(f"  Fixed: {fixed_count:,}")
    print(f"  Unknown after: {unknown_after:,} ({100*unknown_after/total:.1f}%)")
    
    if new_assignments:
        print(f"\n  Re-assigned to:")
        for source_set, count in new_assignments.most_common():
            pct = 100 * count / fixed_count
            print(f"    {source_set:20s} {count:>7,}  ({pct:5.1f}%)")
    
    # Backup original
    print(f"\n💾 Creating backup...")
    import shutil
    shutil.copy2(input_file, backup_file)
    print(f"  Original saved to: {backup_file}")
    
    # Save fixed blocks
    print(f"\n💾 Saving fixed blocks...")
    with open(output_file, "w", encoding="utf-8") as f:
        for block in blocks:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")
    print(f"  Fixed blocks saved to: {output_file}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("Final Source Set Distribution")
    print("=" * 70)
    
    source_set_counts = Counter(b.get("source_set") for b in blocks)
    total = len(blocks)
    
    for source_set, count in source_set_counts.most_common():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {source_set:20s} {count:>7,}  ({pct:5.1f}%)  {bar}")
    
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    
    print(f"""
1. Replace original with fixed blocks:
   mv {input_file} {input_file}.old
   mv {output_file} {input_file}

2. Re-run training pair creation:
   python create_training_pairs.py

3. Prepare final training data:
   python prepare_for_training_v2.py

Your training pairs will now have proper source_set labels!
""")


def analyze_before_after(data_dir="data"):
    """Compare before and after distributions"""
    
    data_dir = Path(data_dir)
    original_file = data_dir / "raw_blocks" / "all_blocks.backup.jsonl"
    fixed_file = data_dir / "raw_blocks" / "all_blocks_fixed.jsonl"
    
    if not original_file.exists() or not fixed_file.exists():
        return
    
    print("\n" + "=" * 70)
    print("Before vs After Comparison")
    print("=" * 70)
    
    # Load both
    original = []
    fixed = []
    
    with open(original_file) as f:
        for line in f:
            original.append(json.loads(line))
    
    with open(fixed_file) as f:
        for line in f:
            fixed.append(json.loads(line))
    
    orig_counts = Counter(b.get("source_set") for b in original)
    fixed_counts = Counter(b.get("source_set") for b in fixed)
    
    print(f"\n{'Source Set':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 60)
    
    all_source_sets = set(orig_counts.keys()) | set(fixed_counts.keys())
    
    for source_set in sorted(all_source_sets):
        before = orig_counts.get(source_set, 0)
        after = fixed_counts.get(source_set, 0)
        change = after - before
        change_str = f"+{change:,}" if change > 0 else f"{change:,}"
        
        print(f"{source_set:<20} {before:>10,} {after:>10,} {change_str:>10}")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Fix unknown source sets in raw blocks")
    parser.add_argument("--data-dir", default=os.environ.get("KMP_DATA_DIR", "data"))
    parser.add_argument("--analyze", action="store_true", help="Analyze before/after if backup exists")
    args = parser.parse_args()
    
    fix_raw_blocks(args.data_dir)
    
    if args.analyze:
        analyze_before_after(args.data_dir)
    
    print("\n✅ Complete!")
