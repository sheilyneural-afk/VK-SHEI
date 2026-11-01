#!/usr/bin/env python3
"""AST-backed source inventory for KV-SHEI.

Declaration discovery never uses source-text regular expressions:
- Rust: rust-analyzer syntax symbols; rustdoc JSON enriches effective API data.
- OpenCL/C++: Clang AST through libclang.
- Python: the standard-library AST.
- JavaScript/Shell: tree-sitter through tools/inventory-syntax.

Rust callers are derived from rust-analyzer LSIF definition/reference edges.
Clang supplies resolved C++/OpenCL references; Python and tree-sitter resolve
only unique same-file names. Unknown is kept distinct from an empty verified
caller list, and every record names its resolution scope.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = (
    ".git/",
    ".mcp_trash/",
    "target/",
    "deploy/local-production/runtime/",
    "tools/inventory-syntax/target/",
)
CODE_SUFFIXES = {".rs", ".cl", ".cpp", ".hpp", ".py", ".sh", ".html"}
BUILD_INPUT_NAMES = {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "config.toml"}
SYNTAX_TOOL_MANIFEST = ROOT / "tools/inventory-syntax/Cargo.toml"
CLANG_LIBRARY = Path("/usr/lib/llvm-18/lib/libclang.so.1")


def run(arguments: list[str], *, input_text: str | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n{completed.stderr}"
        )
    return completed.stdout


def git(*arguments: str) -> str:
    return run(["git", "-C", str(ROOT), *arguments]).strip()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def excluded(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def source_files() -> list[Path]:
    tracked = set(git("ls-files").splitlines())
    untracked = set(git("ls-files", "--others", "--exclude-standard").splitlines())
    result = []
    for rel in sorted(tracked | untracked):
        path = ROOT / rel
        if not excluded(rel) and path.is_file() and path.suffix in CODE_SUFFIXES:
            result.append(path)
    return result


def build_input_files(code: list[Path]) -> list[Path]:
    selected = set(code)
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = relative(path)
        if excluded(rel):
            continue
        if path.name in BUILD_INPUT_NAMES or rel == ".cargo/config.toml":
            selected.add(path)
        if rel.startswith("scripts/inventory-") and path.suffix == ".json":
            selected.add(path)
        if path.suffix in {".cl", ".html", ".json"} and (
            rel.startswith("assets/") or rel.startswith("deploy/local-production/")
        ) and rel != "deploy/local-production/manifest.json":
            # manifest.json is the installed release receipt and stores the
            # build_input_digest itself. Including it would make that digest
            # self-referential and impossible to reproduce.
            selected.add(path)
    return sorted(selected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_digest(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for rel, value in sorted(hashes.items()):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def offset_to_line_column(source: bytes, offset: int) -> tuple[int, int]:
    line = source.count(b"\n", 0, offset) + 1
    last = source.rfind(b"\n", 0, offset)
    return line, offset - last


def documentation_before(lines: list[str], line: int, marker: str) -> str:
    docs: list[str] = []
    index = line - 2
    while index >= 0:
        value = lines[index].strip()
        if value.startswith(marker):
            docs.append(value[len(marker) :].strip())
        elif value == "" or value.startswith("#["):
            index -= 1
            continue
        else:
            break
        index -= 1
    return " ".join(reversed(docs))


def area_for(rel: str) -> str:
    if rel.startswith("src/bin/"):
        return "entrypoint_or_cli"
    if rel.startswith("src/gpu_llm/") or rel.startswith("assets/") and rel.endswith(".cl"):
        return "gpu_inference"
    if rel.startswith("src/torch_runtime/"):
        return "torch_inference"
    if rel.startswith("src/sheily_port/"):
        return "test_only_sheily_port_inspiration"
    if rel.startswith("crates/mslim-adapter-api/"):
        return "adapter_contract"
    if rel.startswith("tests/"):
        return "integration_test"
    if rel.startswith("scripts/") or rel.startswith("deploy/") or rel.startswith("tools/"):
        return "operations_and_validation"
    if rel.startswith("assets/"):
        return "user_interface"
    if rel.startswith("native/"):
        return "native_acceleration"
    return "core_library"


def base_record(path: Path, name: str, line: int, column: int, end_line: int, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "file": relative(path),
        "line": line,
        "column": column,
        "end_line": end_line,
        "area": area_for(relative(path)),
    }
    record.update(extra)
    return record


@dataclass
class RustSymbol:
    index: int
    parent: int | None
    label: str
    name_start: int
    name_end: int
    node_start: int
    node_end: int
    kind: str
    detail: str | None


def parse_rust_symbols(path: Path) -> list[dict[str, Any]]:
    source_bytes = path.read_bytes()
    source = source_bytes.decode("utf-8")
    lines = source.splitlines()
    output = run(["rust-analyzer", "symbols"], input_text=source)
    symbols: list[RustSymbol] = []
    prefix = "StructureNode { parent: "
    for index, line in enumerate(output.splitlines()):
        if not line.startswith(prefix):
            continue
        if "kind: SymbolKind(" not in line:
            continue
        if 'label: "' not in line:
            raise RuntimeError(f"unexpected rust-analyzer symbols output for {path}: {line}")
        parent_text = line[len(prefix) :].split(", label: ", 1)[0]
        parent = None if parent_text == "None" else int(parent_text.removeprefix("Some(").removesuffix(")"))
        label_part = line.split('label: "', 1)[1]
        label = label_part.split('", navigation_range:', 1)[0]
        nav = label_part.split("navigation_range: ", 1)[1].split(", node_range: ", 1)[0]
        node = label_part.split("node_range: ", 1)[1].split(", kind: ", 1)[0]
        kind = label_part.split("kind: SymbolKind(", 1)[1].split(")", 1)[0]
        detail_part = label_part.split(", detail: ", 1)[1].rsplit(", deprecated:", 1)[0]
        detail = None if detail_part == "None" else detail_part.removeprefix('Some("').removesuffix('")')
        name_start, name_end = (int(value) for value in nav.split(".."))
        node_start, node_end = (int(value) for value in node.split(".."))
        symbols.append(RustSymbol(index, parent, label, name_start, name_end, node_start, node_end, kind, detail))

    by_index = {symbol.index: symbol for symbol in symbols}
    records: list[dict[str, Any]] = []
    for symbol in symbols:
        if symbol.kind not in {"Function", "Method"}:
            continue
        ancestors: list[RustSymbol] = []
        parent = symbol.parent
        while parent is not None and parent in by_index:
            ancestor = by_index[parent]
            ancestors.append(ancestor)
            parent = ancestor.parent
        ancestor_kinds = {ancestor.kind for ancestor in ancestors}
        if "Trait" in ancestor_kinds:
            declaration_kind = "rust_public_or_private_trait_item"
        elif "Impl" in ancestor_kinds:
            declaration_kind = "rust_impl_method"
        elif any(ancestor.kind == "Module" for ancestor in ancestors):
            declaration_kind = "rust_module_function"
        else:
            declaration_kind = "rust_free_function"
        prefix_text = source_bytes[symbol.node_start : symbol.name_start].decode("utf-8")
        declaration_line, column = offset_to_line_column(source_bytes, symbol.name_start)
        end_line, _ = offset_to_line_column(source_bytes, symbol.node_end)
        declared_visibility = "private"
        tokens = prefix_text.replace("(", " ").replace(")", " ").replace("::", " ").split()
        if "pub(crate)" in prefix_text:
            declared_visibility = "crate"
        elif "pub(super)" in prefix_text or "pub(in " in prefix_text:
            declared_visibility = "restricted"
        elif "pub" in tokens:
            declared_visibility = "public"
        ancestor_labels = [ancestor.label for ancestor in reversed(ancestors) if ancestor.kind in {"Module", "Trait", "Struct", "Enum", "Impl"}]
        test = (
            "#[test]" in prefix_text
            or any(ancestor.label == "tests" for ancestor in ancestors)
            or relative(path).startswith("tests/")
        )
        body_start = source_bytes.find(b"{", symbol.name_end, symbol.node_end)
        body = (
            source_bytes[body_start:symbol.node_end].decode("utf-8") if body_start >= 0 else ""
        )
        records.append(
            base_record(
                path,
                symbol.label,
                declaration_line,
                column,
                end_line,
                language="rust",
                kind=declaration_kind,
                signature=symbol.detail
                or source_bytes[symbol.node_start:body_start].decode("utf-8").strip(),
                documentation=documentation_before(lines, declaration_line, "///"),
                declared_visibility=declared_visibility,
                enclosing_path=ancestor_labels,
                test=test,
                parser="rust-analyzer-syntax",
                byte_range=[symbol.node_start, symbol.node_end],
                name_byte_range=[symbol.name_start, symbol.name_end],
                normalized_body_sha256=normalized_body_digest(body),
                verified_callers=[],
                caller_resolution="pending_rust_analyzer_lsif",
                effective_visibility_by_feature={},
            )
        )
    return records


def normalized_body_digest(body: str) -> str | None:
    if not body:
        return None
    # Tokenization is delegated to rust-analyzer for discovery. For duplicate
    # regression only comments and ASCII whitespace are discarded; identifiers
    # and literals remain significant.
    compact: list[str] = []
    index = 0
    while index < len(body):
        if body.startswith("//", index):
            end = body.find("\n", index)
            index = len(body) if end < 0 else end + 1
            continue
        if body.startswith("/*", index):
            end = body.find("*/", index + 2)
            index = len(body) if end < 0 else end + 2
            continue
        char = body[index]
        if not char.isspace():
            compact.append(char)
        index += 1
    value = "".join(compact)
    if len(value) < 80:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def normalized_token_digest(value: str) -> str | None:
    if len(value) < 80:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def configure_clang() -> Any:
    from clang import cindex

    if CLANG_LIBRARY.exists() and not cindex.Config.loaded:
        cindex.Config.set_library_file(str(CLANG_LIBRARY))
    return cindex


def clang_entries(path: Path, language: str) -> list[dict[str, Any]]:
    cindex = configure_clang()
    index = cindex.Index.create()
    args: list[str]
    unsaved = None
    if language == "opencl":
        args = ["-x", "cl", "-cl-std=CL1.2"]
        if path.name == "tiled_attention.cl":
            # The production builder supplies these four identity parameters.
            # Any positive, internally consistent instance is sufficient to
            # let Clang build the declaration AST.
            args += [
                "-DAT_D=64",
                "-DAT_BM=1",
                "-DAT_BN=16",
                "-DAT_WG=64",
                "-include",
                str(ROOT / "assets/gpu_llm.cl"),
            ]
    else:
        source = path.read_text(encoding="utf-8")
        source = source.replace(
            "#include <hip/hip_runtime.h>", "// inventory declarations supplied by -include"
        ).replace(
            "#include <hipblaslt/hipblaslt.h>", "// inventory declarations supplied by -include"
        )
        stub = ROOT / "tools/inventory-syntax/hip_ast_stubs.hpp"
        args = ["-std=c++17", "-include", str(stub)]
        unsaved = [(str(path), source)]
    translation = index.parse(str(path), args=args, unsaved_files=unsaved)
    errors = [str(item) for item in translation.diagnostics if item.severity >= cindex.Diagnostic.Error]
    if errors:
        raise RuntimeError(f"Clang AST failed for {path}: {'; '.join(errors)}")
    target = path.resolve()
    accepted = {cindex.CursorKind.FUNCTION_DECL}
    if language == "cpp":
        accepted |= {cindex.CursorKind.CXX_METHOD, cindex.CursorKind.DESTRUCTOR, cindex.CursorKind.FUNCTION_TEMPLATE}
    result: list[dict[str, Any]] = []
    definitions: dict[str, dict[str, Any]] = {}
    edges: list[tuple[dict[str, Any], str]] = []

    def visit(cursor: Any, caller: dict[str, Any] | None = None) -> None:
        location_file = cursor.location.file
        if (
            cursor.kind in accepted
            and cursor.is_definition()
            and location_file is not None
            and Path(location_file.name).resolve() == target
        ):
            tokens = list(cursor.get_tokens())
            token_text = " ".join(token.spelling for token in tokens)
            is_kernel = language == "opencl" and "__kernel" in {token.spelling for token in tokens}
            body_tokens = token_text.split("{", 1)[1] if "{" in token_text else ""
            record = base_record(
                    path,
                    cursor.spelling,
                    cursor.location.line,
                    cursor.location.column,
                    cursor.extent.end.line,
                    language=language,
                    kind=("opencl_kernel" if is_kernel else f"{language}_function"),
                    signature=token_text.split("{", 1)[0].strip(),
                    documentation=cursor.raw_comment or "",
                    declared_visibility="language_default",
                    effective_visibility_by_feature={},
                    test=False,
                    parser="clang-ast",
                    verified_callers=[],
                    caller_resolution="clang_ast_semantic_references",
                    normalized_body_sha256=normalized_token_digest(body_tokens),
                    clang_usr=cursor.get_usr(),
                )
            result.append(record)
            if cursor.get_usr():
                definitions[cursor.get_usr()] = record
            caller = record
        if cursor.kind == cindex.CursorKind.CALL_EXPR and caller is not None:
            referenced = cursor.referenced
            if referenced is not None and referenced.get_usr():
                edges.append((caller, referenced.get_usr()))
        for child in cursor.get_children():
            visit(child, caller)

    visit(translation.cursor)
    for caller, target in edges:
        callee = definitions.get(target)
        if callee is not None and callee is not caller:
            callee["verified_callers"].append(
                {"file": caller["file"], "line": caller["line"], "name": caller["name"]}
            )
    return result


def python_entries(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), type_comments=True)
    result: list[dict[str, Any]] = []
    parents: list[str] = []
    nodes: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, dict[str, Any]]] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            parents.append(node.name)
            self.generic_visit(node)
            parents.pop()

        def add(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            segment = ast.get_source_segment(source, node) or ""
            signature = segment.split(":", 1)[0].strip()
            body_dump = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
            record = base_record(
                    path,
                    node.name,
                    node.lineno,
                    node.col_offset + 1,
                    node.end_lineno or node.lineno,
                    language="python",
                    kind="python_async_function" if isinstance(node, ast.AsyncFunctionDef) else "python_function",
                    signature=signature,
                    documentation=ast.get_docstring(node, clean=True) or "",
                    declared_visibility="language_default",
                    effective_visibility_by_feature={},
                    enclosing_path=list(parents),
                    test=node.name.startswith("test_") or "/tests/" in f"/{relative(path)}",
                    parser="python-ast",
                    verified_callers=[],
                    caller_resolution="python_ast_same_file_unique_name",
                    normalized_body_sha256=normalized_token_digest(body_dump),
                )
            result.append(record)
            nodes.append((node, record))

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.add(node)
            parents.append(node.name)
            self.generic_visit(node)
            parents.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    by_name: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in result:
        by_name[record["name"]].append(record)
    def direct_calls(node: ast.AST) -> Iterable[ast.Call]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                yield child
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            yield from direct_calls(child)

    for node, caller in nodes:
        for child in direct_calls(ast.Module(body=node.body, type_ignores=[])):
            target = None
            if isinstance(child.func, ast.Name):
                target = child.func.id
            elif isinstance(child.func, ast.Attribute):
                target = child.func.attr
            matches = by_name.get(target or "", [])
            if len(matches) == 1 and matches[0] is not caller:
                matches[0]["verified_callers"].append(
                    {"file": caller["file"], "line": caller["line"], "name": caller["name"]}
                )
    return result


class ScriptCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.active = False
        self.start_line = 0
        self.parts: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "script" and not any(key.lower() == "src" for key, _ in attrs):
            self.active = True
            self.start_line = self.getpos()[0]

    def handle_data(self, data: str) -> None:
        if self.active:
            self.parts.append((self.start_line, data))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self.active = False


def syntax_tool(language: str, path: Path) -> list[dict[str, Any]]:
    output = run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(SYNTAX_TOOL_MANIFEST),
            "--",
            language,
            str(path),
        ]
    )
    return json.loads(output)


def javascript_entries(path: Path) -> list[dict[str, Any]]:
    collector = ScriptCollector()
    collector.feed(path.read_text(encoding="utf-8"))
    result: list[dict[str, Any]] = []
    for start_line, source in collector.parts:
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as temporary:
            temporary.write(source)
            temporary.flush()
            rows = syntax_tool("javascript", Path(temporary.name))
        for row in rows:
            result.append(
                base_record(
                    path,
                    row["name"],
                    start_line + row["line"] - 1,
                    row["column"],
                    start_line + row["end_line"] - 1,
                    language="javascript",
                    kind=row["kind"],
                    signature=row["signature"],
                    documentation="",
                    declared_visibility="language_default",
                    effective_visibility_by_feature={},
                    test=False,
                    parser="tree-sitter-javascript",
                    verified_callers=[],
                    caller_resolution="tree_sitter_same_file_unique_name",
                    normalized_body_sha256=normalized_token_digest(row["normalized_body"]),
                    call_targets=row["call_targets"],
                )
            )
    return result


def shell_entries(path: Path) -> list[dict[str, Any]]:
    result = []
    for row in syntax_tool("bash", path):
        result.append(
            base_record(
                path,
                row["name"],
                row["line"],
                row["column"],
                row["end_line"],
                language="shell",
                kind=row["kind"],
                signature=row["signature"],
                documentation=documentation_before(path.read_text().splitlines(), row["line"], "#"),
                declared_visibility="language_default",
                effective_visibility_by_feature={},
                test=False,
                parser="tree-sitter-bash",
                verified_callers=[],
                caller_resolution="tree_sitter_same_file_unique_name",
                normalized_body_sha256=normalized_token_digest(row["normalized_body"]),
                call_targets=row["call_targets"],
            )
        )
    return result


def resolve_script_callers(records: list[dict[str, Any]]) -> None:
    by_file_and_name: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["language"] in {"javascript", "shell"}:
            by_file_and_name[(record["file"], record["name"])].append(record)
    for caller in records:
        for target in caller.pop("call_targets", []):
            matches = by_file_and_name.get((caller["file"], target), [])
            if len(matches) == 1 and matches[0] is not caller:
                matches[0]["verified_callers"].append(
                    {"file": caller["file"], "line": caller["line"], "name": caller["name"]}
                )


def nested_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "id" and isinstance(child, int):
                yield str(child)
            yield from nested_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_ids(child)


def rustdoc_semantics(profile: str, cargo_arguments: list[str]) -> dict[tuple[str, int, str], str]:
    environment = os.environ.copy()
    environment["RUSTC_BOOTSTRAP"] = "1"
    if profile == "torch-backend":
        environment["LIBTORCH_USE_PYTORCH"] = "1"
        environment["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
    run(
        [
            "cargo",
            "rustdoc",
            "--locked",
            "--lib",
            *cargo_arguments,
            "--",
            "-Z",
            "unstable-options",
            "--output-format",
            "json",
            "--document-private-items",
        ],
        env=environment,
    )
    path = ROOT / "deploy/local-production/runtime/build-target/doc/mslim.json"
    data = json.loads(path.read_text())
    index: dict[str, Any] = data["index"]
    paths: dict[str, Any] = data["paths"]
    parent: dict[str, tuple[str, Any]] = {}
    for item_id, item in index.items():
        inner = item.get("inner", {})
        if "impl" in inner:
            for child in inner["impl"].get("items", []):
                parent[str(child)] = ("impl", inner["impl"])
        if "trait" in inner:
            for child in inner["trait"].get("items", []):
                parent[str(child)] = ("trait", inner["trait"])
        if "module" in inner:
            for child in inner["module"].get("items", []):
                parent.setdefault(str(child), ("module", inner["module"]))
    result: dict[tuple[str, int, str], str] = {}
    for item_id, item in index.items():
        if "function" not in item.get("inner", {}) or not item.get("span") or not item.get("name"):
            continue
        filename = item["span"]["filename"]
        try:
            rel = Path(filename).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = Path(filename).as_posix()
        visibility = item.get("visibility")
        classification = "private_or_crate_local"
        if item_id in paths and visibility == "public":
            classification = "public_free_function"
        elif item_id in parent:
            parent_kind, parent_data = parent[item_id]
            if parent_kind == "trait":
                trait_ids = [candidate for candidate, value in index.items() if value.get("inner", {}).get("trait") is parent_data]
                classification = (
                    "public_trait_item" if any(candidate in paths for candidate in trait_ids) else "private_trait_item"
                )
            elif parent_kind == "impl":
                trait = parent_data.get("trait")
                referenced = set(nested_ids(parent_data.get("for")))
                target_public = any(candidate in paths for candidate in referenced)
                trait_public = trait is not None and any(candidate in paths for candidate in nested_ids(trait))
                if trait_public:
                    classification = "public_trait_implementation_method"
                elif target_public and visibility == "public":
                    classification = "public_inherent_method"
                elif target_public:
                    classification = "private_method_on_public_type"
                else:
                    classification = "method_on_non_public_type"
        result[(rel, item["span"]["begin"][0], item["name"])] = classification
    return result


def enrich_rust_visibility(records: list[dict[str, Any]], enabled: bool) -> dict[str, str]:
    profiles = {
        "no-default-features": ["--no-default-features"],
        "default": [],
        "accelerator-backend": ["--features", "accelerator-backend"],
        "torch-backend": ["--features", "torch-backend"],
    }
    status: dict[str, str] = {}
    if not enabled:
        for record in records:
            if record["language"] == "rust":
                record["effective_visibility_status"] = "not_computed"
        return {profile: "not_run" for profile in profiles}
    for profile, arguments in profiles.items():
        try:
            semantics = rustdoc_semantics(profile, arguments)
        except Exception as error:
            status[profile] = f"unavailable: {error}"
            continue
        status[profile] = "computed"
        for record in records:
            if record["language"] != "rust":
                continue
            key = (record["file"], record["line"], record["name"])
            if key in semantics:
                record["effective_visibility_by_feature"][profile] = semantics[key]
    for record in records:
        if record["language"] == "rust":
            record["effective_visibility_status"] = (
                "compiler_resolved_for_listed_feature_sets"
                if record["effective_visibility_by_feature"]
                else "not_compiled_in_analyzed_feature_sets_or_binary_test_only"
            )
    return status


def rust_lsif_edges(
    rust: list[dict[str, Any]], project: Path, environment: dict[str, str]
) -> tuple[set[tuple[str, int, int]], defaultdict[tuple[str, int, int], set[tuple[str, int, str]]]]:
    with tempfile.NamedTemporaryFile("w+b", suffix=".lsif") as stream:
        completed = subprocess.run(
            ["rust-analyzer", "lsif", str(project)],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode(errors="replace"))
        stream.seek(0)
        documents: dict[int, str] = {}
        ranges: dict[int, tuple[int, int, int, int]] = {}
        range_document: dict[int, int] = {}
        next_edge: dict[int, int] = {}
        reference_result: dict[int, int] = {}
        items: defaultdict[int, list[tuple[str | None, int, list[int]]]] = defaultdict(list)
        pending_contains: list[tuple[int, list[int]]] = []
        for raw in stream:
            value = json.loads(raw)
            if value.get("type") == "vertex" and value.get("label") == "document":
                parsed = urlparse(value["uri"])
                documents[value["id"]] = unquote(parsed.path)
            elif value.get("type") == "vertex" and value.get("label") == "range":
                start, end = value["start"], value["end"]
                ranges[value["id"]] = (
                    start["line"], start["character"], end["line"], end["character"]
                )
            elif value.get("label") == "contains":
                pending_contains.append((value["outV"], value["inVs"]))
            elif value.get("label") == "next":
                next_edge[value["outV"]] = value["inV"]
            elif value.get("label") == "textDocument/references":
                reference_result[value["outV"]] = value["inV"]
            elif value.get("label") == "item":
                items[value["outV"]].append(
                    (value.get("property"), value["document"], value["inVs"])
                )
        for document, contained in pending_contains:
            for range_id in contained:
                range_document[range_id] = document

    records_by_file: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rust:
        records_by_file[record["file"]].append(record)
    ranges_by_location: defaultdict[tuple[str, int, int], list[int]] = defaultdict(list)
    for range_id, (line, column, _, _) in ranges.items():
        document = range_document.get(range_id)
        if document is None or document not in documents:
            continue
        try:
            rel = Path(documents[document]).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            continue
        ranges_by_location[(rel, line + 1, column + 1)].append(range_id)

    indexed: set[tuple[str, int, int]] = set()
    callers_by_key: defaultdict[
        tuple[str, int, int], set[tuple[str, int, str]]
    ] = defaultdict(set)
    for record in rust:
        key = (record["file"], record["line"], record["column"])
        declaration_ranges = ranges_by_location.get(key, [])
        result_sets = {
            next_edge[range_id]
            for range_id in declaration_ranges
            if range_id in next_edge
        }
        for result_set in result_sets:
            references = reference_result.get(result_set)
            if references is None:
                continue
            indexed.add(key)
            for property_name, document, range_ids in items.get(references, []):
                if property_name == "definitions" or document not in documents:
                    continue
                try:
                    ref_file = (
                        Path(documents[document]).resolve().relative_to(ROOT).as_posix()
                    )
                except ValueError:
                    continue
                for range_id in range_ids:
                    if range_id not in ranges:
                        continue
                    ref_line = ranges[range_id][0] + 1
                    candidates = [
                        candidate
                        for candidate in records_by_file.get(ref_file, [])
                        if candidate["line"] <= ref_line <= candidate["end_line"]
                    ]
                    if candidates:
                        caller = min(
                            candidates,
                            key=lambda candidate: candidate["end_line"]
                            - candidate["line"],
                        )
                        caller_key = (caller["file"], caller["line"], caller["name"])
                        if caller_key != (record["file"], record["line"], record["name"]):
                            callers_by_key[key].add(caller_key)
    return indexed, callers_by_key


def rust_feature_overlay(parent: Path, name: str, features: list[str]) -> Path:
    overlay = parent / name
    overlay.mkdir()
    for item in [
        ".cargo",
        "assets",
        "build.rs",
        "Cargo.lock",
        "crates",
        "native",
        "rust-toolchain.toml",
        "src",
        "tests",
    ]:
        source = ROOT / item
        if source.exists():
            (overlay / item).symlink_to(source, target_is_directory=source.is_dir())
    manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    original = 'default = ["native-cpu-backend"]'
    if original not in manifest:
        raise RuntimeError("Cargo default feature declaration changed; update inventory overlay")
    replacement = "default = [" + ", ".join(json.dumps(item) for item in features) + "]"
    (overlay / "Cargo.toml").write_text(
        manifest.replace(original, replacement, 1), encoding="utf-8"
    )
    return overlay


def enrich_rust_callers(records: list[dict[str, Any]], enabled: bool) -> str:
    rust = [record for record in records if record["language"] == "rust"]
    if not enabled:
        for record in rust:
            record["caller_resolution"] = "not_computed"
        return "not_run"
    if shutil.which("rust-analyzer") is None:
        for record in rust:
            record["caller_resolution"] = "rust_analyzer_unavailable"
        return "rust_analyzer_unavailable"

    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    all_features = sorted(
        feature for feature in manifest.get("features", {}) if feature != "default"
    )
    environment = os.environ.copy()
    environment["LIBTORCH_USE_PYTORCH"] = "1"
    environment["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
    union_indexed: set[tuple[str, int, int]] = set()
    union_callers: defaultdict[
        tuple[str, int, int], set[tuple[str, int, str]]
    ] = defaultdict(set)
    with tempfile.TemporaryDirectory(prefix="mslim-rust-graphs-") as directory:
        parent = Path(directory)
        profiles = {
            "no-default-features": rust_feature_overlay(parent, "no-default", []),
            "all-features": rust_feature_overlay(parent, "all-features", all_features),
        }
        for project in profiles.values():
            indexed, callers = rust_lsif_edges(rust, project, environment)
            union_indexed.update(indexed)
            for key, values in callers.items():
                union_callers[key].update(values)
        syntax_project = ROOT / "tools/inventory-syntax"
        indexed, callers = rust_lsif_edges(rust, syntax_project, environment)
        union_indexed.update(indexed)
        for key, values in callers.items():
            union_callers[key].update(values)

    for record in rust:
        key = (record["file"], record["line"], record["column"])
        record["verified_callers"] = [
            {"file": file, "line": line, "name": caller}
            for file, line, caller in sorted(union_callers[key])
        ]
        record["caller_resolution"] = (
            "rust_analyzer_lsif_no_default_and_all_features_union"
            if key in union_indexed
            else "not_indexed_in_rust_feature_graph_union"
        )
    return "computed_no_default_and_all_features_union"


def lexical_occurrences(records: list[dict[str, Any]], files: list[Path]) -> None:
    corpus = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in files)

    def identifier_count(name: str) -> int:
        count = 0
        start = 0
        while True:
            index = corpus.find(name, start)
            if index < 0:
                return count
            end = index + len(name)
            before = corpus[index - 1] if index else ""
            after = corpus[end] if end < len(corpus) else ""
            if not (before.isalnum() or before == "_") and not (
                after.isalnum() or after == "_"
            ):
                count += 1
            start = end

    counts = {name: identifier_count(name) for name in {record["name"] for record in records}}
    for record in records:
        record["lexical_occurrences"] = counts[record["name"]]


def duplicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        digest = record.get("normalized_body_sha256")
        if digest:
            grouped[digest].append(record)
    result = []
    for digest, members in sorted(grouped.items()):
        locations = {(member["file"], member["line"]) for member in members}
        if len(locations) > 1:
            result.append(
                {
                    "normalized_body_sha256": digest,
                    "members": [
                        {"file": member["file"], "line": member["line"], "name": member["name"], "test": member["test"]}
                        for member in members
                    ],
                }
            )
    return result


def name_collision_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["name"]].append(record)
    return [
        {
            "name": name,
            "members": [
                {
                    "file": member["file"],
                    "line": member["line"],
                    "language": member["language"],
                    "kind": member["kind"],
                    "enclosing_path": member.get("enclosing_path", []),
                    "test": member["test"],
                }
                for member in members
            ],
        }
        for name, members in sorted(grouped.items())
        if len({(member["file"], member["line"]) for member in members}) > 1
    ]


def render_markdown(snapshot: dict[str, Any]) -> str:
    records = snapshot["functions"]
    by_language = Counter(record["language"] for record in records)
    by_area = Counter(record["area"] for record in records)
    lines = [
        "# KV-SHEI: inventario sintáctico y semántico de funciones",
        "",
        "```yaml",
        "project: KV-SHEI",
        "product: M-SLIM",
        "technical_namespace: mslim",
        "```",
        "",
        f"- Instantánea UTC: `{snapshot['generated_at_utc']}`",
        f"- Commit base: `{snapshot['commit_base']}`",
        f"- Archivos de código censados: **{snapshot['source_file_count']}**",
        f"- Declaraciones con nombre: **{snapshot['function_count']}**",
        f"- `inventory_code_digest`: `{snapshot['inventory_code_digest']}`",
        f"- `build_input_digest`: `{snapshot['build_input_digest']}`",
        "- Descubrimiento: AST real; no se usan expresiones regulares para descubrir declaraciones.",
        "- `lexical_occurrences` es frecuencia textual; `verified_callers` procede del analizador indicado en `caller_resolution` y nunca de esa frecuencia.",
        "",
        "## Cobertura de analizadores",
        "",
        "| Lenguaje | Analizador | Declaraciones |",
        "|---|---|---:|",
    ]
    parser_by_language: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        parser_by_language[record["language"]].add(record["parser"])
    for language, count in sorted(by_language.items()):
        lines.append(f"| {language} | {', '.join(sorted(parser_by_language[language]))} | {count} |")
    lines += ["", "## Áreas", "", "| Área | Declaraciones |", "|---|---:|"]
    for area, count in sorted(by_area.items()):
        lines.append(f"| `{area}` | {count} |")
    lines += [
        "",
        "## Resolución Rust",
        "",
        f"- Grafo de callers: `{snapshot['rust_call_graph_status']}`.",
        f"- Visibilidad por features: `{json.dumps(snapshot['rust_visibility_profiles'], ensure_ascii=False)}`.",
        "- Las funciones de binarios y tests no se presentan como API pública aunque tengan `pub` sintáctico.",
        "- Un array vacío de callers solo significa 'ningún caller demostrado en el grafo analizado'; el campo `caller_resolution` indica si la función fue indexada.",
        "",
        "| Perfil | Clasificación efectiva | Cantidad |",
        "|---|---|---:|",
    ]
    rust_records = [record for record in records if record["language"] == "rust"]
    for profile in snapshot["rust_visibility_profiles"]:
        counts = Counter(
            record["effective_visibility_by_feature"].get(profile, "absent_from_profile")
            for record in rust_records
        )
        for classification, count in sorted(counts.items()):
            lines.append(f"| `{profile}` | `{classification}` | {count} |")
    lines += [
        "",
        "## Clones normalizados",
        "",
        f"Grupos detectados: **{len(snapshot['normalized_duplicate_groups'])}**.",
        f"Colisiones de nombre (no equivalen por sí solas a duplicados): **{len(snapshot['declaration_name_collision_groups'])}**; el detalle íntegro está en JSON.",
        "",
    ]
    for group in snapshot["normalized_duplicate_groups"]:
        lines.append(f"- `{group['normalized_body_sha256']}`")
        for member in group["members"]:
            lines.append(f"  - `{member['file']}:{member['line']}` · `{member['name']}` · test={str(member['test']).lower()}")
    lines += ["", "## Índice completo", ""]
    by_file: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_file[record["file"]].append(record)
    for file, members in sorted(by_file.items()):
        lines += [f"### `{file}`", ""]
        for record in members:
            callers = record.get("verified_callers", [])
            lines.append(
                f"- **L{record['line']} · `{record['name']}`** · {record['kind']} · parser `{record['parser']}`"
            )
            lines.append(f"  - Firma: `{record['signature'].replace('`', '\\`')}`")
            lines.append(
                f"  - Frecuencia léxica: **{record['lexical_occurrences']}**; callers verificados: **{len(callers)}**; estado: `{record['caller_resolution']}`."
            )
            if record["language"] == "rust":
                lines.append(
                    f"  - Visibilidad declarada: `{record['declared_visibility']}`; efectiva: `{json.dumps(record['effective_visibility_by_feature'], ensure_ascii=False)}`."
                )
            if record.get("documentation"):
                lines.append(f"  - Documentación: {record['documentation']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--print-build-input-digest", action="store_true")
    parser.add_argument("--semantic-rust", action="store_true")
    parser.add_argument("--call-graph", action="store_true")
    options = parser.parse_args()

    files = source_files()
    build_files = build_input_files(files)
    code_hashes_before = {relative(path): sha256(path) for path in files}
    build_hashes_before = {relative(path): sha256(path) for path in build_files}
    if options.print_build_input_digest:
        build_hashes_after = {relative(path): sha256(path) for path in build_files}
        if build_hashes_before != build_hashes_after:
            raise SystemExit("build input changed while hashing; retry required")
        print(aggregate_digest(build_hashes_before))
        return
    if options.json is None or options.markdown is None:
        parser.error("--json and --markdown are required unless --print-build-input-digest is used")
    tracked = set(git("ls-files").splitlines())
    records: list[dict[str, Any]] = []
    for path in files:
        if path.suffix == ".rs":
            entries = parse_rust_symbols(path)
        elif path.suffix == ".cl":
            entries = clang_entries(path, "opencl")
        elif path.suffix == ".cpp":
            entries = clang_entries(path, "cpp")
        elif path.suffix == ".py":
            entries = python_entries(path)
        elif path.suffix == ".sh":
            entries = shell_entries(path)
        elif path.suffix == ".html":
            entries = javascript_entries(path)
        else:
            entries = []
        for record in entries:
            record["tracked"] = record["file"] in tracked
            record["file_sha256"] = code_hashes_before[record["file"]]
        records.extend(entries)
    records.sort(key=lambda record: (record["file"], record["line"], record["column"], record["name"]))
    resolve_script_callers(records)
    for record in records:
        unique_callers = {
            (caller["file"], caller["line"], caller["name"])
            for caller in record["verified_callers"]
        }
        record["verified_callers"] = [
            {"file": file, "line": line, "name": name}
            for file, line, name in sorted(unique_callers)
        ]
    lexical_occurrences(records, files)
    visibility_status = enrich_rust_visibility(records, options.semantic_rust)
    call_graph_status = enrich_rust_callers(records, options.call_graph)
    duplicates = duplicate_groups(records)
    name_collisions = name_collision_groups(records)

    code_hashes_after = {relative(path): sha256(path) for path in files}
    build_hashes_after = {relative(path): sha256(path) for path in build_files}
    if code_hashes_before != code_hashes_after or build_hashes_before != build_hashes_after:
        raise SystemExit("source or build input changed during inventory; retry required")

    snapshot = {
        "identity": {"project": "KV-SHEI", "product": "M-SLIM", "technical_namespace": "mslim"},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "commit_base": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short", "--branch").splitlines(),
        "inventory_code_digest": aggregate_digest(code_hashes_before),
        "build_input_digest": aggregate_digest(build_hashes_before),
        "digest_algorithm": "SHA-256(sorted(relative UTF-8 path + NUL + lowercase file SHA-256 + LF))",
        "code_file_sha256": code_hashes_before,
        "build_input_file_sha256": build_hashes_before,
        "source_file_count": len(files),
        "function_count": len(records),
        "rust_visibility_profiles": visibility_status,
        "rust_call_graph_status": call_graph_status,
        "normalized_duplicate_groups": duplicates,
        "declaration_name_collision_groups": name_collisions,
        "scope": "Project-owned code, tracked and non-ignored untracked; dependencies, build outputs, runtime, models, Git internals and trash excluded.",
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "functions": records,
    }
    options.json.parent.mkdir(parents=True, exist_ok=True)
    options.markdown.parent.mkdir(parents=True, exist_ok=True)
    options.json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    options.markdown.write_text(render_markdown(snapshot) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "files": len(files),
                "functions": len(records),
                "languages": Counter(record["language"] for record in records),
                "duplicate_groups": len(duplicates),
            },
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
