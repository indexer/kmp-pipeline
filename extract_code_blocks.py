"""
Step 2: Extract structured code blocks from cloned KMP repositories.

ENHANCED version that:
- Tags blocks that have REAL method body implementations
- Extracts Composable functions and UI components
- Better brace matching for nested classes
- Detects architecture patterns (ViewModel, Repository, UseCase)
- Extracts full expect/actual pairs across source sets

Usage:
    python extract_code_blocks.py
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

SOURCE_SET_PATTERNS = {
    "commonMain": [
        "commonMain", "shared/src/commonMain", "composeApp/src/commonMain",
        "src/main/kotlin", "src/commonMain", "common/src/main",
        "core/src/main", "domain/src/main", "data/src/main",
    ],
    "commonTest": [
        "commonTest", "shared/src/commonTest",
        "src/test/kotlin", "src/commonTest", "common/src/test", "core/src/test",
    ],
    "androidMain": [
        "androidMain", "shared/src/androidMain", "composeApp/src/androidMain",
        "android/src/main", "app/src/main", "src/androidMain",
    ],
    "androidTest": ["androidTest", "android/src/test", "app/src/test", "src/androidTest"],
    "iosMain": [
        "iosMain", "shared/src/iosMain", "composeApp/src/iosMain",
        "ios/src/main", "src/iosMain",
    ],
    "iosTest": ["iosTest", "ios/src/test", "src/iosTest"],
    "desktopMain": [
        "desktopMain", "jvmMain", "composeApp/src/desktopMain",
        "desktop/src/main", "src/jvmMain", "src/desktopMain",
    ],
    "jsMain": [
        "jsMain", "wasmJsMain", "wasmMain",
        "js/src/main", "src/jsMain", "src/wasmJsMain",
    ],
    "nativeMain": ["nativeMain", "native/src/main", "src/nativeMain"],
}


@dataclass
class CodeBlock:
    id: str
    type: str                    # function, class, object, interface, expect, actual, gradle, composable, full_file
    name: str
    code: str
    source_set: str
    file_path: str
    repo: str
    package: str
    imports: list
    annotations: list
    is_expect: bool
    is_actual: bool
    has_body: bool               # NEW: does this block have real implementation?
    arch_pattern: str            # NEW: viewmodel, repository, usecase, service, etc.
    token_count: int
    lines: int


class KotlinCodeExtractor:
    """Extract code blocks from Kotlin files in cloned repos."""

    DECLARATION_PATTERNS = [
        # Expect/Actual (highest priority for KMP)
        (r'(expect\s+(?:class|abstract\s+class|interface|object|fun|val|var|enum\s+class|sealed\s+class|annotation\s+class)\s+\w+)', "expect"),
        (r'(actual\s+(?:class|abstract\s+class|interface|object|fun|val|var|enum\s+class|sealed\s+class|annotation\s+class)\s+\w+)', "actual"),

        # Composable functions
        (r'(@Composable\s+(?:(?:internal|private|public)\s+)?fun\s+\w+)', "composable"),

        # Standard declarations
        (r'((?:data\s+|sealed\s+|abstract\s+|open\s+|internal\s+|private\s+|public\s+|enum\s+|value\s+)*class\s+\w+)', "class"),
        (r'((?:internal\s+|private\s+|public\s+)?interface\s+\w+)', "interface"),
        (r'((?:internal\s+|private\s+|public\s+|companion\s+)?object\s+\w+)', "object"),
        (r'((?:suspend\s+|internal\s+|private\s+|public\s+|inline\s+|operator\s+|override\s+|tailrec\s+)*fun\s+(?:\w+\.)?[\w<>]+\s*\()', "function"),
    ]

    # Patterns that indicate real implementation logic
    BODY_INDICATORS = [
        "return ", " = ", ".launch", ".collect", ".map(", ".filter(",
        "try {", "catch ", "when (", "when {", "if (", "for (", "while (",
        "Log.", "println", "emit(", "send(", ".update {", ".value =",
        "install(", "single(", "factory(", "get()", "inject(",
        "withContext", "runCatching", "viewModelScope", "coroutineScope",
        "remember {", "mutableStateOf", "LaunchedEffect", "derivedStateOf",
        "HttpClient(", "Json.decodeFrom", "transaction {",
        "async {", "await()", "delay(", "flow {", "channelFlow {",
    ]

    ARCH_PATTERNS = {
        "viewmodel": ["ViewModel", "StateFlow", "MutableStateFlow", "viewModelScope"],
        "repository": ["Repository", "suspend fun get", "suspend fun fetch", "Flow<"],
        "usecase": ["UseCase", "invoke(", "operator fun invoke"],
        "service": ["Service", "Api", "Client"],
        "di_module": ["module {", "single(", "factory(", "koinModule"],
        "composable_screen": ["@Composable", "Scaffold", "Column", "LazyColumn"],
        "mapper": ["Mapper", "toDomain", "toEntity", "toDto"],
        "state": ["UiState", "sealed class", "data class.*State"],
    }

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.repos_dir = self.data_dir / "repos"
        self.output_dir = self.data_dir / "raw_blocks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Extract from all repos."""
        if not self.repos_dir.exists():
            print(f"❌ {self.repos_dir} not found. Run scrape_repos.py first.")
            return

        all_blocks = []
        seen_hashes = set()
        stats = {
            "repos_processed": 0,
            "files_processed": 0,
            "blocks_by_type": defaultdict(int),
            "blocks_by_source_set": defaultdict(int),
            "blocks_with_body": 0,
            "blocks_by_arch": defaultdict(int),
        }

        repo_dirs = [d for d in self.repos_dir.iterdir() if d.is_dir()]
        print(f"📂 Processing {len(repo_dirs)} repositories...\n")

        for repo_dir in tqdm(repo_dirs, desc="Extracting"):
            repo_name = repo_dir.name
            stats["repos_processed"] += 1

            # Kotlin files
            for kt_file in repo_dir.rglob("*.kt"):
                blocks = self._extract_from_file(kt_file, repo_name)
                stats["files_processed"] += 1

                for block in blocks:
                    if block.id not in seen_hashes:
                        seen_hashes.add(block.id)
                        all_blocks.append(asdict(block))
                        stats["blocks_by_type"][block.type] += 1
                        stats["blocks_by_source_set"][block.source_set] += 1
                        if block.has_body:
                            stats["blocks_with_body"] += 1
                        if block.arch_pattern:
                            stats["blocks_by_arch"][block.arch_pattern] += 1

            # Gradle files
            for kts_file in repo_dir.rglob("*.gradle.kts"):
                blocks = self._extract_gradle(kts_file, repo_name)
                for block in blocks:
                    if block.id not in seen_hashes:
                        seen_hashes.add(block.id)
                        all_blocks.append(asdict(block))
                        stats["blocks_by_type"]["gradle"] += 1

        # Save
        output_file = self.output_dir / "all_blocks.jsonl"
        with open(output_file, "w") as f:
            for block in all_blocks:
                f.write(json.dumps(block) + "\n")

        # Save stats
        stats_dict = dict(stats)
        stats_dict["total_blocks"] = len(all_blocks)
        stats_dict["blocks_by_type"] = dict(stats["blocks_by_type"])
        stats_dict["blocks_by_source_set"] = dict(stats["blocks_by_source_set"])
        stats_dict["blocks_by_arch"] = dict(stats["blocks_by_arch"])

        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats_dict, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Extraction Complete!")
        print(f"{'='*60}")
        print(f"  Repos:  {stats['repos_processed']}")
        print(f"  Files:  {stats['files_processed']}")
        print(f"  Blocks: {len(all_blocks)}")
        print(f"  With method bodies: {stats['blocks_with_body']} ({100*stats['blocks_with_body']/max(len(all_blocks),1):.1f}%)")
        print(f"\n  By type:")
        for t, c in sorted(stats["blocks_by_type"].items(), key=lambda x: -x[1]):
            print(f"    {t:20s} {c:>6d}")
        print(f"\n  By source set:")
        for s, c in sorted(stats["blocks_by_source_set"].items(), key=lambda x: -x[1]):
            print(f"    {s:20s} {c:>6d}")
        if stats["blocks_by_arch"]:
            print(f"\n  By architecture:")
            for a, c in sorted(stats["blocks_by_arch"].items(), key=lambda x: -x[1]):
                print(f"    {a:20s} {c:>6d}")
        print(f"\n  Output: {output_file}")

    def _extract_from_file(self, file_path: Path, repo_name: str) -> list:
        """Extract all code blocks from a Kotlin file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return []

        if len(content) < 20 or len(content) > 500_000:
            return []

        source_set = self._detect_source_set(str(file_path))
        package = self._extract_package(content)
        imports = self._extract_imports(content)

        blocks = []

        # Extract individual declarations
        declarations = self._find_declarations(content)
        for decl_type, name, code, annotations in declarations:
            is_expect = code.strip().startswith("expect ")
            is_actual = code.strip().startswith("actual ")
            if is_expect:
                decl_type = "expect"
            elif is_actual:
                decl_type = "actual"

            block = CodeBlock(
                id=hashlib.md5(code.encode()).hexdigest()[:12],
                type=decl_type,
                name=name,
                code=code.strip(),
                source_set=source_set,
                file_path=str(file_path),
                repo=repo_name,
                package=package,
                imports=imports,
                annotations=annotations,
                is_expect=is_expect,
                is_actual=is_actual,
                has_body=self._has_method_body(code),
                arch_pattern=self._detect_arch_pattern(code, name),
                token_count=len(code.split()),
                lines=code.count("\n") + 1,
            )
            blocks.append(block)

        # Full file block for substantial files
        if 30 < len(content.split("\n")) < 400:
            file_block = CodeBlock(
                id=hashlib.md5(content.encode()).hexdigest()[:12],
                type="full_file",
                name=file_path.stem,
                code=content.strip(),
                source_set=source_set,
                file_path=str(file_path),
                repo=repo_name,
                package=package,
                imports=imports,
                annotations=[],
                is_expect="expect " in content,
                is_actual="actual " in content,
                has_body=self._has_method_body(content),
                arch_pattern=self._detect_arch_pattern(content, file_path.stem),
                token_count=len(content.split()),
                lines=content.count("\n") + 1,
            )
            blocks.append(file_block)

        return blocks

    def _has_method_body(self, code: str) -> bool:
        """Check if code has real implementation logic, not just declarations."""
        indicator_count = 0
        for indicator in self.BODY_INDICATORS:
            if indicator in code:
                indicator_count += 1
        return indicator_count >= 2

    def _detect_arch_pattern(self, code: str, name: str) -> str:
        """Detect the architecture pattern of a code block."""
        combined = code + " " + name
        best_match = ""
        best_score = 0

        for pattern, keywords in self.ARCH_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_match = pattern

        return best_match if best_score >= 1 else ""

    def _detect_source_set(self, path: str) -> str:
        """
        Enhanced source set detection with fallback heuristics.

        Strategy:
        1. Try pattern matching (existing approach)
        2. Apply heuristic rules for common structures
        3. Use file location hints
        4. Default to commonMain for .kt files (most are shared code)
        """
        path_normalized = path.replace("\\", "/").lower()

        # Stage 1: Pattern Matching
        for source_set, markers in SOURCE_SET_PATTERNS.items():
            for marker in markers:
                if marker.lower() in path_normalized:
                    return source_set

        # Stage 2: Heuristic Detection
        # Check for test paths first (higher specificity)
        if any(test_marker in path_normalized for test_marker in
               ["/test/", "test/kotlin", "androidtest", "iostest", "commontest"]):
            if "android" in path_normalized:
                return "androidTest"
            elif "ios" in path_normalized:
                return "iosTest"
            else:
                return "commonTest"

        # Platform-specific main code
        if any(android_marker in path_normalized for android_marker in
               ["android/", "app/src/main", "/androidmain"]):
            return "androidMain"

        if "ios" in path_normalized and "/src/" in path_normalized:
            return "iosMain"

        if any(desktop_marker in path_normalized for desktop_marker in
               ["desktop/", "jvm/", "/jvmmain"]):
            return "desktopMain"

        if any(js_marker in path_normalized for js_marker in
               ["js/", "wasm/", "/jsmain"]):
            return "jsMain"

        if "native/" in path_normalized:
            return "nativeMain"

        # Stage 3: Source Directory Structure Hints
        # Standard Gradle kotlin/java structure
        if "/src/main/kotlin/" in path_normalized or "/src/main/java/" in path_normalized:
            if any(x in path_normalized for x in ["android", "app/"]):
                return "androidMain"
            return "commonMain"

        # Shared/common modules
        if any(x in path_normalized for x in ["shared/", "common/", "core/", "domain/", "data/"]):
            if "test" not in path_normalized:
                return "commonMain"

        # Stage 4: File Type Defaults
        # Most .kt files in KMP projects are in commonMain
        if path.endswith(".kt"):
            return "commonMain"

        # Gradle build files
        if path.endswith(("build.gradle.kts", "build.gradle", "settings.gradle.kts")):
            return "build_config"

        return "unknown"

    def _extract_package(self, content: str) -> str:
        match = re.search(r'^package\s+([\w.]+)', content, re.MULTILINE)
        return match.group(1) if match else ""

    def _extract_imports(self, content: str) -> list:
        return re.findall(r'^import\s+([\w.*]+)', content, re.MULTILINE)

    def _find_declarations(self, content: str) -> list:
        """Find top-level declarations with their full bodies."""
        declarations = []
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                i += 1
                continue

            # Check for @Composable on previous lines
            composable_start = None
            if line.startswith("@Composable") or (i > 0 and lines[i-1].strip().startswith("@Composable")):
                composable_start = i - 1 if i > 0 and lines[i-1].strip().startswith("@Composable") else i

            decl_info = self._match_declaration(line)
            if decl_info:
                decl_type, name = decl_info
                annotations = self._collect_annotations(lines, i)
                start = i - len(annotations)
                end = self._find_block_end(lines, i)
                full_code = "\n".join(lines[start:end + 1])

                if (end - start >= 2) or ("expect " in line) or ("actual " in line):
                    declarations.append((decl_type, name, full_code, annotations))

                i = end + 1
            else:
                i += 1

        return declarations

    def _match_declaration(self, line: str) -> tuple:
        if line.startswith(("import ", "package ")):
            return None
        for pattern, decl_type in self.DECLARATION_PATTERNS:
            match = re.search(pattern, line)
            if match:
                name_match = re.search(
                    r'(?:class|interface|object|fun|val|var)\s+(?:\w+\.)?(\w+)',
                    match.group(1)
                )
                name = name_match.group(1) if name_match else "unknown"
                return (decl_type, name)
        return None

    def _collect_annotations(self, lines: list, decl_line: int) -> list:
        annotations = []
        i = decl_line - 1
        while i >= 0:
            line = lines[i].strip()
            if line.startswith("@") or line.startswith("//"):
                annotations.insert(0, line)
                i -= 1
            elif line == "":
                i -= 1
            else:
                break
        return annotations

    def _find_block_end(self, lines: list, start: int) -> int:
        brace_count = 0
        found_open = False

        for i in range(start, min(start + 500, len(lines))):
            line = lines[i]
            # Remove string literals and comments
            cleaned = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
            cleaned = re.sub(r"'(?:[^'\\]|\\.)*'", "''", cleaned)
            cleaned = re.sub(r'//.*$', '', cleaned)

            brace_count += cleaned.count("{") - cleaned.count("}")

            if "{" in cleaned:
                found_open = True

            if found_open and brace_count <= 0:
                return i

            # Single-line (no braces) — like expect fun signatures
            if not found_open and i > start:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not next_line or self._match_declaration(next_line):
                        return i

        return min(start + 300, len(lines) - 1)

    def _extract_gradle(self, file_path: Path, repo_name: str) -> list:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to read gradle file {file_path}: {e}")
            return []

        kmp_indicators = ["multiplatform", "KotlinMultiplatform", 'kotlin("multiplatform")',
                          "commonMain", "androidMain", "iosMain", "cocoapods"]
        if not any(ind in content for ind in kmp_indicators):
            return []

        return [CodeBlock(
            id=hashlib.md5(content.encode()).hexdigest()[:12],
            type="gradle",
            name=file_path.stem,
            code=content.strip(),
            source_set="build_config",
            file_path=str(file_path),
            repo=repo_name,
            package="",
            imports=[],
            annotations=[],
            is_expect=False,
            is_actual=False,
            has_body=True,
            arch_pattern="",
            token_count=len(content.split()),
            lines=content.count("\n") + 1,
        )]


if __name__ == "__main__":
    extractor = KotlinCodeExtractor(
        data_dir=os.environ.get("KMP_DATA_DIR", "data")
    )
    extractor.run()
