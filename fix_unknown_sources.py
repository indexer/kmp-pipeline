"""
Quick Fix: Re-classify unknown source_sets in existing training data.

This reads your existing training pairs and attempts to fix "unknown" 
source_sets using heuristic detection, without re-extracting everything.

Usage:
    python fix_unknown_sources.py
    python fix_unknown_sources.py --data-dir data
"""

import json
from pathlib import Path
from collections import Counter


def detect_source_set_from_context(pair: dict) -> str:
    """
    Detect source set from pair context.
    
    Uses:
    - input_text hints (e.g., "Source set: commonMain")
    - Code content (imports, platform-specific APIs)
    - Metadata if available
    """
    input_text = pair.get("input_text", "").lower()
    target_text = pair.get("target_text", "").lower()
    
    # Check if input_text has source set hint
    if "source set:" in input_text:
        for source_set in ["commonmain", "androidmain", "iosmain", "desktopmain", 
                           "jsmain", "nativemain", "commontest", "androidtest", "iostest"]:
            if source_set in input_text:
                # Capitalize properly
                return source_set.replace("main", "Main").replace("test", "Test")
    
    # Platform-specific indicators in code
    android_indicators = [
        "android.", "androidx.", "import android", 
        "activity", "fragment", "bundle",
        "context:", "context)", "context,"
    ]
    if any(ind in target_text for ind in android_indicators):
        return "androidMain"
    
    ios_indicators = [
        "platform.uikit", "platform.foundation", "import platform",
        "uiview", "uiviewcontroller", "nsstring"
    ]
    if any(ind in target_text for ind in ios_indicators):
        return "iosMain"
    
    # Desktop/JVM indicators
    desktop_indicators = [
        "import java.", "import javax.", "swing", "javafx",
        "jframe", "jpanel"
    ]
    if any(ind in target_text for ind in desktop_indicators):
        return "desktopMain"
    
    # JS indicators
    js_indicators = [
        "import js.", "external ", "js(\"", "window.", "document."
    ]
    if any(ind in target_text for ind in js_indicators):
        return "jsMain"
    
    # Test indicators
    test_indicators = [
        "@test", "import kotlin.test", "import io.kotest",
        "import org.junit", "fun test", "class test",
        "assertequals", "asserttrue", "assertfalse"
    ]
    if any(ind in target_text for ind in test_indicators):
        # Check which platform
        if any(ind in target_text for ind in android_indicators):
            return "androidTest"
        elif any(ind in target_text for ind in ios_indicators):
            return "iosTest"
        else:
            return "commonTest"
    
    # Expect/actual patterns strongly suggest commonMain
    if pair.get("pair_type") == "expect_actual":
        return "commonMain"
    
    # Composables are usually in common or android
    if "@composable" in target_text:
        if any(ind in target_text for ind in android_indicators):
            return "androidMain"
        return "commonMain"
    
    # Default: most KMP code is commonMain
    return "commonMain"


def fix_unknown_sources(data_dir="data"):
    """Re-classify unknown source sets in training pairs"""
    
    data_dir = Path(data_dir)
    pairs_dir = data_dir / "training_pairs"
    output_dir = data_dir / "training_pairs_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Fix Unknown Source Sets")
    print("=" * 70)
    
    for split in ["train", "val", "test"]:
        input_file = pairs_dir / f"{split}.jsonl"
        output_file = output_dir / f"{split}.jsonl"
        
        if not input_file.exists():
            print(f"\n⚠️  {split}.jsonl not found, skipping...")
            continue
        
        print(f"\n📂 Processing {split}.jsonl...")
        
        # Load pairs
        pairs = []
        with open(input_file) as f:
            for line in f:
                pairs.append(json.loads(line))
        
        # Count unknowns before
        unknown_before = sum(1 for p in pairs if p.get("source_set") == "unknown")
        
        # Fix unknowns
        fixed_count = 0
        new_assignments = Counter()
        
        for pair in pairs:
            if pair.get("source_set") == "unknown":
                new_source_set = detect_source_set_from_context(pair)
                if new_source_set != "unknown":
                    pair["source_set"] = new_source_set
                    fixed_count += 1
                    new_assignments[new_source_set] += 1
        
        # Save fixed pairs
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
        unknown_after = sum(1 for p in pairs if p.get("source_set") == "unknown")
        
        print(f"  Total pairs: {len(pairs):,}")
        print(f"  Unknown before: {unknown_before:,} ({100*unknown_before/len(pairs):.1f}%)")
        print(f"  Unknown after:  {unknown_after:,} ({100*unknown_after/len(pairs):.1f}%)")
        print(f"  Fixed: {fixed_count:,}")
        
        if new_assignments:
            print(f"\n  Re-assigned to:")
            for source_set, count in new_assignments.most_common():
                print(f"    {source_set:20s} {count:>6,}")
        
        print(f"  ✅ Saved to {output_file}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"""
Fixed training pairs saved to: {output_dir}/

To use the fixed data:
1. Backup original: mv {pairs_dir} {pairs_dir}.backup
2. Use fixed data:  mv {output_dir} {pairs_dir}
3. Re-run prepare_for_training_v2.py

Or keep both and compare results.
""")


def analyze_remaining_unknowns(data_dir="data"):
    """Analyze what's still unknown after the fix"""
    
    data_dir = Path(data_dir)
    fixed_dir = data_dir / "training_pairs_fixed"
    
    print("\n" + "=" * 70)
    print("Analyzing Remaining Unknowns")
    print("=" * 70)
    
    all_unknowns = []
    
    for split in ["train", "val", "test"]:
        file_path = fixed_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue
        
        with open(file_path) as f:
            for line in f:
                pair = json.loads(line)
                if pair.get("source_set") == "unknown":
                    all_unknowns.append(pair)
    
    if not all_unknowns:
        print("\n✅ No unknowns remaining!")
        return
    
    print(f"\n  Remaining unknowns: {len(all_unknowns):,}")
    
    # Analyze pair types
    pair_types = Counter(p.get("pair_type") for p in all_unknowns)
    print(f"\n  By pair type:")
    for ptype, count in pair_types.most_common():
        print(f"    {ptype:30s} {count:>6,}")
    
    # Show samples
    print(f"\n  Sample unknowns (first 5):\n")
    for i, pair in enumerate(all_unknowns[:5], 1):
        print(f"  {i}. Type: {pair.get('pair_type')}")
        print(f"     Input: {pair.get('input_text', '')[:80]}...")
        print(f"     Target: {pair.get('target_text', '')[:80]}...")
        print()


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Fix unknown source sets")
    parser.add_argument("--data-dir", default=os.environ.get("KMP_DATA_DIR", "data"))
    parser.add_argument("--analyze", action="store_true", help="Analyze remaining unknowns after fix")
    args = parser.parse_args()
    
    fix_unknown_sources(args.data_dir)
    
    if args.analyze:
        analyze_remaining_unknowns(args.data_dir)
    
    print("\n✅ Complete!")
