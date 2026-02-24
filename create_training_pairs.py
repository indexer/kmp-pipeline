"""
Step 3: Create training pairs from extracted code blocks.

CRITICAL: This generator ensures targets contain FULL method body
implementations — not just signatures/interfaces.

Pair types:
1. expect → actual (with body)
2. interface → implementation class (with all method bodies)
3. description → complete code (functions, classes with bodies)
4. skeleton → complete (fill in TODO bodies)
5. composable description → full UI component
6. full file generation
7. gradle config

Usage:
    python create_training_pairs.py
"""

import json
import re
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# ── Configuration ──────────────────────────────────────────────────────────

random.seed(42)


@dataclass
class TrainingPair:
    id: str
    pair_type: str
    input_text: str
    target_text: str
    context: str
    source_set: str
    repo: str
    metadata: dict


# ── Helpers ────────────────────────────────────────────────────────────────

BODY_INDICATORS = [
    "return ", " = ", ".launch", ".collect", ".map(", ".filter(",
    "try {", "catch ", "when (", "when {", "if (", "for (", "while (",
    "Log.", "println", "emit(", "send(", ".update {", ".value =",
    "install(", "single(", "factory(", "get()", "inject(",
    "withContext", "runCatching", "viewModelScope", "coroutineScope",
    "remember {", "mutableStateOf", "LaunchedEffect",
    "HttpClient(", "Json.decodeFrom", "async {", "flow {",
]


def has_method_bodies(code: str) -> bool:
    """Check if code has real implementation logic."""
    count = sum(1 for ind in BODY_INDICATORS if ind in code)
    return count >= 2


def extract_signature(code: str) -> str:
    """Extract just the declaration signature (no body)."""
    lines = code.strip().split("\n")
    sig_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@") or stripped.startswith("//"):
            sig_lines.append(line)
            continue

        if any(kw in stripped for kw in
               ["class ", "interface ", "object ", "fun ", "val ", "var "]):
            sig_lines.append(line)
            if "{" in stripped:
                break
            continue

        if sig_lines and stripped == "{":
            break

    return "\n".join(sig_lines)


def build_imports_str(imports: list, max_imports: int = 15) -> str:
    """Build an import block string."""
    if not imports:
        return ""
    return "\n".join(f"import {imp}" for imp in imports[:max_imports])


def create_description(block: dict) -> str:
    """Create a natural language description of a code block."""
    name = block["name"]
    btype = block["type"]
    source_set = block.get("source_set", "commonMain")
    arch = block.get("arch_pattern", "")

    parts = [f"// Source set: {source_set}"]

    if btype == "composable":
        parts.append(f"// Implement @Composable function '{name}' with full UI layout")
    elif btype == "class":
        if arch == "viewmodel":
            parts.append(f"// Implement ViewModel '{name}' with state management, coroutines, and event handling")
        elif arch == "repository":
            parts.append(f"// Implement Repository '{name}' with data fetching, caching, and error handling")
        elif arch == "usecase":
            parts.append(f"// Implement UseCase '{name}' with business logic")
        elif arch == "mapper":
            parts.append(f"// Implement mapper '{name}' with conversion logic")
        else:
            parts.append(f"// Implement class '{name}' with complete method bodies")
    elif btype == "function":
        parts.append(f"// Implement function '{name}' with full logic")
    elif btype == "object":
        parts.append(f"// Implement object '{name}'")
    elif btype == "interface":
        parts.append(f"// Define interface '{name}' with method signatures")
    else:
        parts.append(f"// Implement '{name}'")

    # Add signature hint
    sig = extract_signature(block["code"])
    if sig and len(sig) < 500:
        parts.append(f"\n{sig}")

    return "\n".join(parts)


def create_skeleton(code: str) -> str:
    """Replace method bodies with TODO comments."""
    lines = code.split("\n")
    result = []
    brace_depth = 0
    in_method_body = False
    method_start_depth = 0

    for line in lines:
        stripped = line.strip()
        cleaned = re.sub(r'"[^"]*"', '""', stripped)
        cleaned = re.sub(r'//.*$', '', cleaned)
        opens = cleaned.count("{")
        closes = cleaned.count("}")

        # Detect function/property with body
        if re.match(r'\s*(override\s+)?(suspend\s+)?(private\s+)?(fun|val|var)\s+', stripped) and "{" in cleaned:
            indent = len(line) - len(line.lstrip())
            result.append(line.split("{")[0].rstrip() + " {")
            result.append(" " * (indent + 4) + "// TODO: implement")
            result.append(" " * indent + "}")
            in_method_body = True
            method_start_depth = brace_depth
            brace_depth += opens - closes
            continue

        if in_method_body:
            brace_depth += opens - closes
            if brace_depth <= method_start_depth:
                in_method_body = False
            continue

        brace_depth += opens - closes
        result.append(line)

    skeleton = "\n".join(result)
    if "// TODO: implement" in skeleton:
        return skeleton
    return ""


# ── Pair Generators ────────────────────────────────────────────────────────

class ExpectActualPairGenerator:
    """Match expect declarations with actual implementations."""

    def generate(self, blocks: list) -> list:
        pairs = []
        expects = defaultdict(list)
        actuals = defaultdict(list)

        for b in blocks:
            if b["is_expect"]:
                expects[(b["package"], b["name"])].append(b)
            elif b["is_actual"]:
                actuals[(b["package"], b["name"])].append(b)

        for key, exp_list in expects.items():
            if key not in actuals:
                continue
            for exp in exp_list:
                for act in actuals[key]:
                    # Skip trivial one-liner actuals unless they use typealias
                    if act["lines"] < 3 and "typealias" not in act["code"]:
                        if not has_method_bodies(act["code"]):
                            continue

                    imports = build_imports_str(act.get("imports", []))
                    target = f"{imports}\n\n{act['code']}".strip() if imports else act["code"]

                    pairs.append(TrainingPair(
                        id=hashlib.md5(f"ea_{exp['id']}_{act['id']}".encode()).hexdigest()[:12],
                        pair_type="expect_actual",
                        input_text=(
                            f"// Source set: {act['source_set']}\n"
                            f"// Implement the actual for this expect declaration:\n"
                            f"{exp['code']}"
                        ),
                        target_text=target,
                        context=imports,
                        source_set=act["source_set"],
                        repo=exp["repo"],
                        metadata={"name": key[1], "package": key[0]},
                    ))

        print(f"    expect → actual: {len(pairs)}")
        return pairs


class InterfaceImplementationPairGenerator:
    """Match interfaces with their implementation classes."""

    def generate(self, blocks: list) -> list:
        pairs = []
        interfaces = [b for b in blocks if b["type"] == "interface"]
        classes = [b for b in blocks if b["type"] == "class" and b.get("has_body", False)]

        # Build implementation index
        impl_index = defaultdict(list)
        for cls in classes:
            name = cls["name"]
            for suffix in ["Impl", "Implementation", "Default", "Real"]:
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    impl_index[base].append(cls)
            # Also match by "implements" in code
            for iface in interfaces:
                if f": {iface['name']}" in cls["code"] or f", {iface['name']}" in cls["code"]:
                    impl_index[iface["name"]].append(cls)

        for iface in interfaces:
            candidates = impl_index.get(iface["name"], [])
            for impl in candidates:
                if not has_method_bodies(impl["code"]):
                    continue
                if impl["lines"] < 5:
                    continue

                imports = build_imports_str(impl.get("imports", []))
                target = f"{imports}\n\n{impl['code']}".strip() if imports else impl["code"]

                pairs.append(TrainingPair(
                    id=hashlib.md5(f"ii_{iface['id']}_{impl['id']}".encode()).hexdigest()[:12],
                    pair_type="interface_implementation",
                    input_text=(
                        f"// Source set: {impl['source_set']}\n"
                        f"// Implement this interface with full method bodies:\n"
                        f"{iface['code']}"
                    ),
                    target_text=target,
                    context=imports,
                    source_set=impl["source_set"],
                    repo=impl["repo"],
                    metadata={"interface": iface["name"], "implementation": impl["name"]},
                ))

        print(f"    interface → implementation: {len(pairs)}")
        return pairs


class DescriptionToCodePairGenerator:
    """Generate description → full code pairs from blocks with real bodies."""

    def generate(self, blocks: list) -> list:
        pairs = []

        good_blocks = [
            b for b in blocks
            if b["type"] in ("function", "class", "object", "composable")
            and b.get("has_body", False)
            and 5 <= b["lines"] <= 200
        ]

        for block in good_blocks:
            description = create_description(block)
            if not description or len(description) < 20:
                continue

            imports = build_imports_str(block.get("imports", []))
            target = f"{imports}\n\n{block['code']}".strip() if imports else block["code"]

            pairs.append(TrainingPair(
                id=hashlib.md5(f"dc_{block['id']}".encode()).hexdigest()[:12],
                pair_type="description_to_code",
                input_text=description,
                target_text=target,
                context=imports,
                source_set=block["source_set"],
                repo=block["repo"],
                metadata={"name": block["name"], "type": block["type"],
                          "arch": block.get("arch_pattern", "")},
            ))

        print(f"    description → code: {len(pairs)}")
        return pairs


class SkeletonToCompletePairGenerator:
    """Given a skeleton with TODO, teach model to fill in implementations."""

    def generate(self, blocks: list) -> list:
        pairs = []

        candidates = [
            b for b in blocks
            if b["type"] in ("class", "object")
            and b.get("has_body", False)
            and b["lines"] >= 15
        ]

        for block in candidates:
            skeleton = create_skeleton(block["code"])
            if not skeleton:
                continue

            imports = build_imports_str(block.get("imports", []))
            target = f"{imports}\n\n{block['code']}".strip() if imports else block["code"]

            pairs.append(TrainingPair(
                id=hashlib.md5(f"sk_{block['id']}".encode()).hexdigest()[:12],
                pair_type="skeleton_to_complete",
                input_text=(
                    f"// Complete this code — replace all TODO comments with real implementations:\n"
                    f"{skeleton}"
                ),
                target_text=target,
                context=imports,
                source_set=block["source_set"],
                repo=block["repo"],
                metadata={"name": block["name"]},
            ))

        print(f"    skeleton → complete: {len(pairs)}")
        return pairs


class ComposablePairGenerator:
    """Generate pairs specifically for Compose UI code."""

    def generate(self, blocks: list) -> list:
        pairs = []

        composables = [
            b for b in blocks
            if (b["type"] == "composable" or
                ("@Composable" in b["code"] and b.get("has_body", False)))
            and 5 <= b["lines"] <= 200
        ]

        for block in composables:
            name = block["name"]
            sig = extract_signature(block["code"])

            pairs.append(TrainingPair(
                id=hashlib.md5(f"cmp_{block['id']}".encode()).hexdigest()[:12],
                pair_type="composable",
                input_text=(
                    f"// Source set: {block['source_set']}\n"
                    f"// Implement this @Composable function with full UI layout:\n"
                    f"{sig}"
                ),
                target_text=block["code"],
                context="",
                source_set=block["source_set"],
                repo=block["repo"],
                metadata={"name": name},
            ))

        print(f"    composable UI: {len(pairs)}")
        return pairs


class FullFilePairGenerator:
    """Generate pairs from complete files with real implementations."""

    def generate(self, blocks: list) -> list:
        pairs = []

        files = [
            b for b in blocks
            if b["type"] == "full_file"
            and b.get("has_body", False)
            and 20 <= b["lines"] <= 300
        ]

        for block in files:
            file_name = block.get("file_path", "").split("/")[-1]
            arch = block.get("arch_pattern", "")
            arch_hint = f"\n// Architecture: {arch}" if arch else ""

            pairs.append(TrainingPair(
                id=hashlib.md5(f"ff_{block['id']}".encode()).hexdigest()[:12],
                pair_type="full_file",
                input_text=(
                    f"// Generate complete KMP file: {file_name}\n"
                    f"// Package: {block.get('package', '')}\n"
                    f"// Source set: {block['source_set']}"
                    f"{arch_hint}\n"
                    f"// Include all imports, classes, and full method implementations."
                ),
                target_text=block["code"],
                context="",
                source_set=block["source_set"],
                repo=block["repo"],
                metadata={"file": file_name, "name": block["name"], "arch": arch},
            ))

        print(f"    full file: {len(pairs)}")
        return pairs


class GradlePairGenerator:
    """Generate Gradle build file pairs."""

    def generate(self, blocks: list) -> list:
        pairs = []

        gradle_blocks = [b for b in blocks if b["type"] == "gradle" and b["lines"] >= 10]

        for block in gradle_blocks:
            pairs.append(TrainingPair(
                id=hashlib.md5(f"gr_{block['id']}".encode()).hexdigest()[:12],
                pair_type="gradle",
                input_text=(
                    f"// Generate KMP build.gradle.kts for project: {block['repo']}\n"
                    f"// Include multiplatform targets, dependencies, and source set configuration."
                ),
                target_text=block["code"],
                context="",
                source_set="build_config",
                repo=block["repo"],
                metadata={"name": block["name"]},
            ))

        print(f"    gradle: {len(pairs)}")
        return pairs


# ── Quality Filter ─────────────────────────────────────────────────────────

def quality_filter(pair: TrainingPair) -> bool:
    """Filter out low-quality pairs."""
    target = pair.target_text
    inp = pair.input_text

    # Length checks
    target_tokens = len(target.split())
    input_tokens = len(inp.split())
    if target_tokens < 10 or target_tokens > 4000:
        return False
    if input_tokens < 5:
        return False

    # Skip auto-generated
    auto_markers = ["AUTO-GENERATED", "DO NOT EDIT", "Generated by", "AUTO_GENERATED"]
    if any(m.lower() in target.lower() for m in auto_markers):
        return False

    # Skip mostly blank
    non_blank = [l for l in target.split("\n") if l.strip()]
    if len(non_blank) < 3:
        return False

    # KEY: For code pairs (not gradle/expect_actual), REQUIRE method bodies
    if pair.pair_type in ("description_to_code", "interface_implementation",
                          "skeleton_to_complete", "composable", "full_file"):
        if not has_method_bodies(target):
            return False

    # Skip if target is just imports
    code_lines = [l for l in target.split("\n")
                  if l.strip() and not l.strip().startswith(("import ", "package ", "//"))]
    if len(code_lines) < 3:
        return False

    return True


# ── Main ───────────────────────────────────────────────────────────────────

class TrainingPairCreator:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.blocks_file = self.data_dir / "raw_blocks" / "all_blocks.jsonl"
        self.output_dir = self.data_dir / "training_pairs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        if not self.blocks_file.exists():
            print(f"❌ {self.blocks_file} not found. Run extract_code_blocks.py first.")
            return

        # Load blocks
        print("📂 Loading extracted blocks...")
        blocks = []
        with open(self.blocks_file) as f:
            for line in f:
                blocks.append(json.loads(line))
        print(f"  Loaded {len(blocks)} blocks")

        with_bodies = sum(1 for b in blocks if b.get("has_body", False))
        print(f"  With real method bodies: {with_bodies} ({100*with_bodies/max(len(blocks),1):.1f}%)")

        # Generate pairs
        print("\n🔄 Generating training pairs...")
        all_pairs = []

        generators = [
            ("Expect/Actual", ExpectActualPairGenerator()),
            ("Interface→Implementation", InterfaceImplementationPairGenerator()),
            ("Description→Code", DescriptionToCodePairGenerator()),
            ("Skeleton→Complete", SkeletonToCompletePairGenerator()),
            ("Composable UI", ComposablePairGenerator()),
            ("Full File", FullFilePairGenerator()),
            ("Gradle", GradlePairGenerator()),
        ]

        for name, gen in generators:
            print(f"\n  [{name}]")
            pairs = gen.generate(blocks)
            all_pairs.extend(pairs)

        print(f"\n  Total before filtering: {len(all_pairs)}")

        # Quality filter
        filtered = [p for p in all_pairs if quality_filter(p)]
        print(f"  Total after filtering:  {len(filtered)}")

        # Deduplicate by target hash
        seen_targets = set()
        deduped = []
        for p in filtered:
            h = hashlib.md5(p.target_text.encode()).hexdigest()
            if h not in seen_targets:
                seen_targets.add(h)
                deduped.append(p)
        print(f"  After deduplication:    {len(deduped)}")

        # Stats
        type_counts = defaultdict(int)
        for p in deduped:
            type_counts[p.pair_type] += 1
        print(f"\n  Breakdown:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t:30s} {c:>6d}")

        # Split: 90/5/5
        random.shuffle(deduped)
        pair_dicts = [asdict(p) for p in deduped]
        n = len(pair_dicts)
        train_end = int(n * 0.90)
        val_end = int(n * 0.95)

        splits = {
            "train": pair_dicts[:train_end],
            "val": pair_dicts[train_end:val_end],
            "test": pair_dicts[val_end:],
        }

        # Save
        for split_name, split_pairs in splits.items():
            out_file = self.output_dir / f"{split_name}.jsonl"
            with open(out_file, "w") as f:
                for pair in split_pairs:
                    f.write(json.dumps(pair) + "\n")
            print(f"\n  {split_name}: {len(split_pairs)} pairs → {out_file}")

        # Token stats
        target_tokens = [len(p["target_text"].split()) for p in pair_dicts]
        if target_tokens:
            print(f"\n  Target token stats:")
            print(f"    Mean:   {sum(target_tokens)/len(target_tokens):.0f}")
            print(f"    Median: {sorted(target_tokens)[len(target_tokens)//2]}")
            print(f"    Max:    {max(target_tokens)}")
            print(f"    Min:    {min(target_tokens)}")

        # Sample
        if pair_dicts:
            # Pick a sample with a real body
            good_samples = [p for p in pair_dicts
                            if p["pair_type"] in ("interface_implementation", "description_to_code")]
            sample = random.choice(good_samples or pair_dicts)
            print(f"\n📝 Sample pair ({sample['pair_type']}):")
            print(f"  INPUT:\n{sample['input_text'][:400]}")
            print(f"\n  TARGET (first 500 chars):\n{sample['target_text'][:500]}")

        print(f"\n✅ Done! Pairs saved to {self.output_dir}/")


if __name__ == "__main__":
    import os
    creator = TrainingPairCreator(
        data_dir=os.environ.get("KMP_DATA_DIR", "data")
    )
    creator.run()
