"""
Production-ready KMP Training Data Preparation Pipeline
========================================================

Takes training pairs and prepares them for fine-tuning with:
- Advanced deduplication (exact + structural)
- Quality-weighted balancing
- KMP-specific validation
- Stratified splits
- Comprehensive quality metrics

Usage:
    python prepare_for_training_v2.py
    python prepare_for_training_v2.py --sample-size 50000 --min-quality 1.5
    python prepare_for_training_v2.py --max-target-tokens 4000
"""

import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

random.seed(42)


@dataclass
class QualityMetrics:
    """Quality assessment results for a training pair"""
    score: float
    flags: List[str]
    is_valid: bool
    rejection_reason: Optional[str] = None


class KMPTrainingDataPreparer:
    """Production pipeline for KMP training data preparation"""
    
    def __init__(
        self,
        data_dir: str = "data",
        sample_size: Optional[int] = None,
        min_quality_score: float = 1.0,
        max_target_tokens: int = 6000,
        min_input_words: int = 5,
        max_input_tokens: int = 1000,
    ):
        self.data_dir = Path(data_dir)
        self.pairs_dir = self.data_dir / "training_pairs"
        self.output_dir = self.data_dir / "final_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.sample_size = sample_size
        self.min_quality_score = min_quality_score
        self.max_target_tokens = max_target_tokens
        self.min_input_words = min_input_words
        self.max_input_tokens = max_input_tokens
        
        # Statistics tracking
        self.stats = defaultdict(int)
        self.rejection_reasons = Counter()

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Kotlin code.
        
        Kotlin code typically: ~1 token per 3.5-4 characters
        This is a rough estimate but works well for filtering.
        """
        return len(text) // 4

    def get_code_fingerprint(self, code: str, prefix_len: int = 500) -> str:
        """
        Create structural fingerprint for better deduplication.
        
        Combines:
        - First N characters (captures package/imports)
        - Structural elements (first 15 meaningful lines)
        
        This catches near-duplicates that differ only in minor details.
        """
        prefix = code[:prefix_len].strip().lower()
        
        # Extract structural elements (skip comments)
        lines = [
            l.strip() 
            for l in code.split('\n') 
            if l.strip() and not l.strip().startswith('//')
        ]
        structure = '|'.join(lines[:15])  # First 15 meaningful lines
        
        combined = f"{prefix}||{structure}"
        return hashlib.md5(combined.encode()).hexdigest()

    def validate_kotlin_pair(self, pair: dict) -> Tuple[bool, str]:
        """
        Validate KMP-specific code quality.
        
        Checks:
        - Length constraints (input and target)
        - Kotlin syntax indicators
        - Code completeness (balanced braces/parens)
        - Input quality (not just extracted comments)
        """
        inp = pair["input_text"]
        tgt = pair["target_text"]
        
        # Token-based length validation
        input_tokens = self.estimate_tokens(inp)
        target_tokens = self.estimate_tokens(tgt)
        
        if len(inp.split()) < self.min_input_words:
            return False, "input_too_short"
        
        if input_tokens > self.max_input_tokens:
            return False, "input_too_long"
            
        if len(tgt.strip()) < 50:
            return False, "target_too_short"
            
        if target_tokens > self.max_target_tokens:
            return False, "target_too_long"
        
        # Kotlin syntax validation
        kotlin_keywords = [
            "fun ", "class ", "object ", "interface ", 
            "val ", "var ", "data class"
        ]
        if not any(kw in tgt for kw in kotlin_keywords):
            return False, "not_kotlin_code"
        
        # Code completeness checks
        if tgt.count("{") != tgt.count("}"):
            return False, "unbalanced_braces"
        if tgt.count("(") != tgt.count(")"):
            return False, "unbalanced_parens"
        
        # Input quality checks
        inp_stripped = inp.strip()
        if inp_stripped.startswith("//") or inp_stripped.startswith("/*"):
            if inp_stripped in tgt:
                return False, "input_is_code_comment"
        
        # Avoid overly generic inputs
        generic_inputs = [
            "write code", "create code", "generate code",
            "write function", "create function"
        ]
        if inp.lower().strip() in generic_inputs:
            return False, "input_too_generic"
        
        # Check input isn't identical to target
        if inp.strip() == tgt.strip():
            return False, "input_equals_target"
        
        return True, "valid"

    def calculate_quality_score(self, pair: dict) -> QualityMetrics:
        """
        Calculate multi-dimensional quality score for KMP pairs.
        
        Scoring factors:
        - Input quality (clear, specific descriptions)
        - Target complexity (moderate is best)
        - KMP-specific patterns (expect/actual, platform-specific)
        - Documentation quality
        - Boilerplate ratio
        - Pair type rarity/value
        """
        score = 0.0
        flags = []
        
        inp = pair["input_text"]
        tgt = pair["target_text"]
        
        # === Input Quality (0-1.0 points) ===
        input_words = len(inp.split())
        if 10 <= input_words <= 100:
            score += 1.0
        elif 5 <= input_words < 10:
            score += 0.5
            flags.append("input_somewhat_short")
        else:
            flags.append("input_length_suboptimal")
        
        # === Target Complexity (0-1.3 points) ===
        # Count structural elements
        functions = tgt.count("fun ")
        classes = tgt.count("class ") + tgt.count("object ") + tgt.count("interface ")
        
        # Moderate complexity is ideal for learning
        if 1 <= functions <= 6:
            score += 0.8
        elif functions > 10:
            flags.append("very_complex")
            score += 0.3  # Still valuable but harder to learn
        
        if 1 <= classes <= 3:
            score += 0.5
        
        # === KMP-Specific Patterns (0-2.0 points) ===
        # These are highly valuable for KMP training
        if "expect " in tgt or "actual " in tgt:
            score += 1.5
            flags.append("kmp_expect_actual")
        
        platform_keywords = ["commonMain", "androidMain", "iosMain", "jvmMain", "jsMain"]
        if any(platform in tgt for platform in platform_keywords):
            score += 0.5
            flags.append("kmp_platform_specific")
        
        # === Boilerplate Penalty (0 to -0.8 points) ===
        total_lines = len([l for l in tgt.split('\n') if l.strip()])
        import_lines = tgt.count("import ")
        package_lines = tgt.count("package ")
        
        if total_lines > 0:
            boilerplate_ratio = (import_lines + package_lines) / total_lines
            if boilerplate_ratio > 0.4:
                score -= 0.8
                flags.append("excessive_boilerplate")
            elif boilerplate_ratio > 0.3:
                score -= 0.3
                flags.append("high_boilerplate")
        
        # === Pair Type Bonus (0-0.7 points) ===
        # Reward rare but valuable pair types
        rare_valuable_types = [
            "interface_implementation",
            "refactoring",
            "bug_fix",
            "platform_specific_implementation",
            "expect_actual_pair",
            "multiplatform_refactor"
        ]
        if pair.get("pair_type") in rare_valuable_types:
            score += 0.7
            flags.append("valuable_pair_type")
        
        # === Documentation Bonus (0-0.3 points) ===
        if "/**" in tgt or "///" in tgt:
            score += 0.3
            flags.append("well_documented")
        
        # === Additional Quality Indicators ===
        # Has tests or test-related code
        if "test" in tgt.lower() or "@Test" in tgt:
            score += 0.2
            flags.append("includes_tests")
        
        # Uses modern Kotlin features
        modern_features = ["sealed ", "suspend ", "flow", "coroutine", "@Composable"]
        if any(feat in tgt for feat in modern_features):
            score += 0.3
            flags.append("modern_kotlin")
        
        # Validate the pair
        is_valid, reason = self.validate_kotlin_pair(pair)
        
        return QualityMetrics(
            score=score,
            flags=flags,
            is_valid=is_valid,
            rejection_reason=None if is_valid else reason
        )

    def deduplicate_advanced(self, pairs: List[dict]) -> List[dict]:
        """
        Advanced deduplication using exact + structural matching.
        
        Two-stage process:
        1. Exact MD5 hash deduplication (catches identical code)
        2. Structural fingerprint (catches near-duplicates)
        """
        print("\n🔄 Advanced deduplication...")
        
        # Stage 1: Exact deduplication
        seen_exact = set()
        exact_deduped = []
        for p in tqdm(pairs, desc="Exact dedup"):
            h = hashlib.md5(p["target_text"].encode()).hexdigest()
            if h not in seen_exact:
                seen_exact.add(h)
                exact_deduped.append(p)
        print(f"  Exact dedup: {len(pairs):,} → {len(exact_deduped):,}")
        
        # Stage 2: Structural fingerprint deduplication
        seen_structural = set()
        structural_deduped = []
        for p in tqdm(exact_deduped, desc="Structural dedup"):
            fingerprint = self.get_code_fingerprint(p["target_text"])
            if fingerprint not in seen_structural:
                seen_structural.add(fingerprint)
                structural_deduped.append(p)
        print(f"  Structural dedup: {len(exact_deduped):,} → {len(structural_deduped):,}")
        
        return structural_deduped

    def quality_weighted_balance(self, pairs: List[dict]) -> List[dict]:
        """
        Balance pair types while preserving high-quality examples.
        
        Strategy:
        - Sort each type by quality score
        - Take top-quality examples up to a cap
        - Cap is based on median type count (prevents dominance)
        - Allows some oversampling for rare valuable types
        """
        print("\n⚖️ Quality-weighted balancing...")
        
        # Group by type
        by_type = defaultdict(list)
        for p in pairs:
            by_type[p["pair_type"]].append(p)
        
        # Sort each type by quality score (best first)
        for ptype in by_type:
            by_type[ptype].sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        # Calculate balanced cap
        type_counts = Counter(p["pair_type"] for p in pairs)
        counts = sorted(type_counts.values())
        median_count = counts[len(counts) // 2] if counts else 1000
        
        # Allow 2x median or minimum 2000 examples per type
        max_per_type = max(median_count * 2, 2000)
        
        # Select top-quality examples per type
        balanced = []
        for ptype, type_pairs in sorted(by_type.items()):
            selected = type_pairs[:max_per_type]
            balanced.extend(selected)
            
            avg_quality = sum(p.get("quality_score", 0) for p in selected) / len(selected)
            print(f"  {ptype:40s} {len(type_pairs):>7,} → {len(selected):>7,}  (avg quality: {avg_quality:.2f})")
        
        random.shuffle(balanced)
        print(f"\n  Total after balancing: {len(balanced):,}")
        return balanced

    def stratified_split(
        self, 
        pairs: List[dict], 
        ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05)
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Create stratified splits ensuring consistent type distribution.
        
        Each pair type is split according to the same ratios,
        ensuring train/val/test have similar distributions.
        """
        print("\n📑 Creating stratified splits...")
        
        by_type = defaultdict(list)
        for p in pairs:
            by_type[p["pair_type"]].append(p)
        
        train, val, test = [], [], []
        
        for ptype, type_pairs in by_type.items():
            random.shuffle(type_pairs)
            n = len(type_pairs)
            
            train_end = int(n * ratios[0])
            val_end = int(n * (ratios[0] + ratios[1]))
            
            train.extend(type_pairs[:train_end])
            val.extend(type_pairs[train_end:val_end])
            test.extend(type_pairs[val_end:])
            
            print(f"  {ptype:40s} T:{len(type_pairs[:train_end]):>5,} V:{len(type_pairs[train_end:val_end]):>4,} T:{len(type_pairs[val_end:]):>4,}")
        
        # Shuffle each split
        for split in [train, val, test]:
            random.shuffle(split)
        
        print(f"\n  Final splits: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
        return train, val, test

    def run(self):
        """Execute the full preparation pipeline"""
        print("=" * 70)
        print("KMP Training Data Preparation Pipeline - Production Ready")
        print("=" * 70)
        
        # === STEP 1: Load Data ===
        print("\n📂 STEP 1: Loading training pairs...")
        all_pairs = []
        for split in ["train", "val", "test"]:
            path = self.pairs_dir / f"{split}.jsonl"
            if path.exists():
                with open(path) as f:
                    pairs = [json.loads(l) for l in f]
                all_pairs.extend(pairs)
                print(f"  {split:5s}: {len(pairs):>8,} pairs")
        
        if not all_pairs:
            print("\n❌ No pairs found. Run create_training_pairs.py first.")
            return
        
        print(f"\n  Total loaded: {len(all_pairs):,} pairs")
        
        # === STEP 2: Deduplication ===
        pairs = self.deduplicate_advanced(all_pairs)
        
        # === STEP 3: Quality Scoring & Filtering ===
        print("\n📊 STEP 3: Calculating quality scores and filtering...")
        valid_pairs = []
        
        for p in tqdm(pairs, desc="Quality assessment"):
            quality = self.calculate_quality_score(p)
            
            if not quality.is_valid:
                self.rejection_reasons[quality.rejection_reason] += 1
                continue
            
            if quality.score < self.min_quality_score:
                self.rejection_reasons["quality_too_low"] += 1
                continue
            
            p["quality_score"] = quality.score
            p["quality_flags"] = quality.flags
            valid_pairs.append(p)
        
        print(f"\n  Quality filtered: {len(pairs):,} → {len(valid_pairs):,}")
        
        if self.rejection_reasons:
            print(f"\n  Rejection reasons (top 10):")
            for reason, count in self.rejection_reasons.most_common(10):
                pct = 100 * count / len(pairs)
                print(f"    {reason:30s} {count:>7,}  ({pct:5.1f}%)")
        
        if not valid_pairs:
            print("\n❌ No valid pairs after filtering. Adjust quality thresholds.")
            return
        
        # === STEP 4: Balance Types ===
        balanced = self.quality_weighted_balance(valid_pairs)
        
        # === STEP 5: Sampling (if requested) ===
        if self.sample_size and len(balanced) > self.sample_size:
            print(f"\n🎲 STEP 5: Sampling {self.sample_size:,} from {len(balanced):,}...")
            # Quality-weighted sampling
            weights = [p["quality_score"] for p in balanced]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            sampled_indices = random.choices(
                range(len(balanced)), 
                weights=weights, 
                k=self.sample_size
            )
            balanced = [balanced[i] for i in sampled_indices]
            print(f"  Sampled: {len(balanced):,} pairs")
        
        # === STEP 6: Stratified Split ===
        train, val, test = self.stratified_split(balanced)
        
        splits = {
            "train": train,
            "val": val,
            "test": test,
        }
        
        # === STEP 7: Save Output ===
        print("\n💾 STEP 7: Saving final datasets...")
        for split_name, split_pairs in splits.items():
            out_file = self.output_dir / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for p in split_pairs:
                    row = {
                        "input_text": p["input_text"],
                        "target_text": p["target_text"],
                        "pair_type": p.get("pair_type", ""),
                        "source_set": p.get("source_set", ""),
                        "quality_score": round(p.get("quality_score", 0), 2),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {split_name:5s}: {len(split_pairs):>8,} → {out_file.name}")
        
        # Combined training file (train + val for actual training)
        combined_file = self.output_dir / "all_train.jsonl"
        with open(combined_file, "w", encoding="utf-8") as f:
            for p in train + val:
                row = {
                    "input_text": p["input_text"],
                    "target_text": p["target_text"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  combined: {len(train + val):>8,} → {combined_file.name}")
        
        # === STEP 8: Statistics ===
        self.print_comprehensive_stats(train, val, test)
        
        print(f"\n{'=' * 70}")
        print(f"✅ Production-ready data preparation complete!")
        print(f"{'=' * 70}")
        print(f"\nNext steps:")
        print(f"  1. Upload {self.output_dir}/ to Google Drive")
        print(f"  2. In Colab, mount Drive and set DATA_PATHS")
        print(f"  3. Use train.jsonl for training, val.jsonl for validation")
        print(f"  4. Evaluate on test.jsonl for final metrics")

    def print_comprehensive_stats(self, train: List[dict], val: List[dict], test: List[dict]):
        """Print detailed statistics about the final dataset"""
        all_pairs = train + val + test
        
        print(f"\n{'=' * 70}")
        print(f"📊 COMPREHENSIVE STATISTICS")
        print(f"{'=' * 70}")
        
        # === Dataset Sizes ===
        print(f"\n  Dataset Distribution:")
        print(f"    Train:  {len(train):>8,}  ({100*len(train)/len(all_pairs):>5.1f}%)")
        print(f"    Val:    {len(val):>8,}  ({100*len(val)/len(all_pairs):>5.1f}%)")
        print(f"    Test:   {len(test):>8,}  ({100*len(test)/len(all_pairs):>5.1f}%)")
        print(f"    {'─' * 40}")
        print(f"    Total:  {len(all_pairs):>8,}")
        
        # === Type Distribution ===
        type_counts = Counter(p.get("pair_type", "unknown") for p in all_pairs)
        print(f"\n  Pair Type Distribution:")
        for ptype, count in type_counts.most_common():
            pct = 100 * count / len(all_pairs)
            bar = "█" * int(pct / 2)
            print(f"    {ptype:40s} {count:>7,}  ({pct:5.1f}%)  {bar}")
        
        # === Quality Distribution ===
        quality_scores = [p.get("quality_score", 0) for p in all_pairs]
        sorted_scores = sorted(quality_scores)
        
        print(f"\n  Quality Score Distribution:")
        print(f"    Mean:       {sum(quality_scores)/len(quality_scores):>6.2f}")
        print(f"    Median:     {sorted_scores[len(sorted_scores)//2]:>6.2f}")
        print(f"    Min:        {min(quality_scores):>6.2f}")
        print(f"    Max:        {max(quality_scores):>6.2f}")
        print(f"    25th %ile:  {sorted_scores[len(sorted_scores)//4]:>6.2f}")
        print(f"    75th %ile:  {sorted_scores[3*len(sorted_scores)//4]:>6.2f}")
        
        # === Token Statistics ===
        input_tokens = [self.estimate_tokens(p["input_text"]) for p in all_pairs]
        target_tokens = [self.estimate_tokens(p["target_text"]) for p in all_pairs]
        
        print(f"\n  Token Count Statistics:")
        print(f"    Input Tokens:")
        print(f"      Mean:   {sum(input_tokens)/len(input_tokens):>7.0f}")
        print(f"      Median: {sorted(input_tokens)[len(input_tokens)//2]:>7,}")
        print(f"      Max:    {max(input_tokens):>7,}")
        
        print(f"    Target Tokens:")
        print(f"      Mean:   {sum(target_tokens)/len(target_tokens):>7.0f}")
        print(f"      Median: {sorted(target_tokens)[len(target_tokens)//2]:>7,}")
        print(f"      Max:    {max(target_tokens):>7,}")
        
        # === KMP-Specific Stats ===
        kmp_flags = ["kmp_expect_actual", "kmp_platform_specific"]
        kmp_count = sum(1 for p in all_pairs if any(
            flag in p.get("quality_flags", []) for flag in kmp_flags
        ))
        
        modern_count = sum(1 for p in all_pairs if "modern_kotlin" in p.get("quality_flags", []))
        documented_count = sum(1 for p in all_pairs if "well_documented" in p.get("quality_flags", []))
        
        print(f"\n  KMP-Specific Features:")
        print(f"    KMP patterns (expect/actual):    {kmp_count:>7,}  ({100*kmp_count/len(all_pairs):>5.1f}%)")
        print(f"    Modern Kotlin features:          {modern_count:>7,}  ({100*modern_count/len(all_pairs):>5.1f}%)")
        print(f"    Well-documented:                 {documented_count:>7,}  ({100*documented_count/len(all_pairs):>5.1f}%)")
        
        # === Source Distribution ===
        source_counts = Counter(p.get("source_set", "unknown") for p in all_pairs)
        if len(source_counts) > 1:
            print(f"\n  Source Set Distribution:")
            for source, count in source_counts.most_common():
                pct = 100 * count / len(all_pairs)
                print(f"    {source:25s} {count:>7,}  ({pct:5.1f}%)")
        
        # === Sample High-Quality Examples ===
        high_quality = sorted(
            all_pairs, 
            key=lambda x: x.get("quality_score", 0), 
            reverse=True
        )[:20]
        
        if high_quality:
            print(f"\n  High-Quality Sample:")
            sample = random.choice(high_quality[:5])
            
            print(f"    Type:         {sample.get('pair_type', 'unknown')}")
            print(f"    Quality:      {sample.get('quality_score', 0):.2f}")
            print(f"    Flags:        {', '.join(sample.get('quality_flags', [])[:5])}")
            print(f"    Input length: {len(sample['input_text'])} chars, ~{self.estimate_tokens(sample['input_text'])} tokens")
            print(f"    Target length: {len(sample['target_text'])} chars, ~{self.estimate_tokens(sample['target_text'])} tokens")
            
            print(f"\n    INPUT:")
            input_preview = sample['input_text'][:280]
            print(f"    {input_preview}{'...' if len(sample['input_text']) > 280 else ''}")
            
            print(f"\n    TARGET (preview):")
            target_lines = sample['target_text'].split('\n')[:12]
            for line in target_lines:
                print(f"    {line[:100]}")
            total_target_lines = len(sample['target_text'].split('\n'))
            if total_target_lines > 12:
                remaining_lines = total_target_lines - 12
                print(f"    ... ({remaining_lines} more lines)")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Prepare production-ready KMP training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None,
        help="Limit dataset to N samples (quality-weighted sampling)"
    )
    parser.add_argument(
        "--data-dir", 
        default=os.environ.get("KMP_DATA_DIR", "data"),
        help="Data directory containing training_pairs/"
    )
    parser.add_argument(
        "--min-quality", 
        type=float, 
        default=1.0,
        help="Minimum quality score threshold (higher = stricter)"
    )
    parser.add_argument(
        "--max-target-tokens", 
        type=int, 
        default=6000,
        help="Maximum target tokens (for model context window)"
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1000,
        help="Maximum input tokens"
    )
    parser.add_argument(
        "--min-input-words",
        type=int,
        default=5,
        help="Minimum words in input description"
    )
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Data dir:           {args.data_dir}")
    print(f"  Min quality score:  {args.min_quality}")
    print(f"  Max target tokens:  {args.max_target_tokens:,}")
    print(f"  Max input tokens:   {args.max_input_tokens:,}")
    print(f"  Min input words:    {args.min_input_words}")
    if args.sample_size:
        print(f"  Sample size:        {args.sample_size:,}")
    print()
    
    preparer = KMPTrainingDataPreparer(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        min_quality_score=args.min_quality,
        max_target_tokens=args.max_target_tokens,
        max_input_tokens=args.max_input_tokens,
        min_input_words=args.min_input_words,
    )
    
    preparer.run()
