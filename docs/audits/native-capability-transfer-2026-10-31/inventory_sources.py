#!/usr/bin/env python3
"""Read-only source inventory for a native M-SLIM capability transfer assessment.

This is an audit utility, not a runtime dependency or an integration bridge.
Source projects are only read. Outputs stay in this audit directory.
Python symbols use AST; Rust symbols use a lexical approximation, not a compiler.
Static test counts never imply execution or certification.
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3]
PROJECTS = ("SHEILY-COGNITIVA", "INICIO", "KV-SHEI")
EXTENSIONS = {".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".sh", ".sql", ".cl", ".cpp", ".h", ".c"}
EXCLUDED_PARTS = {".git", "target", "node_modules", ".venv", "__pycache__", ".runtime", "vendor-patches", "vendor", ".hypothesis", ".pytest_cache"}
RS_SYMBOL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:async|unsafe|const)\s+)*(fn|struct|enum|trait|type)\s+([A-Za-z_]\w*)", re.M)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def selected_files(root: Path) -> list[str]:
    names = sorted(set(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")) - {""})
    return [name for name in names if not any(part in EXCLUDED_PARTS for part in Path(name).parts)
            and not name.startswith(("docs/audits/native-capability-transfer-", "deploy/local-production/runtime/"))
            and (root / name).is_file() and not (root / name).is_symlink()]


def classify(path: str) -> str:
    parts = Path(path).parts
    name = Path(path).name
    if "live-plane" in parts or "patches" in parts:
        return "external_or_vendored_component"
    if path.startswith("evidence/") or "fixtures" in parts or "research" in parts:
        return "historical_fixture_or_research"
    if any(part in {"tests", "test", "fuzz", "benches", "benchmarks"} for part in parts) or name.startswith("test_"):
        return "test_or_benchmark"
    if parts[0] in {"docs", "tools", "tooling", "scripts", "ops", "deploy"} or "/tools/" in path:
        return "development_or_operations"
    return "implementation_candidate"


def source_record(root: Path, name: str) -> dict:
    raw = (root / name).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    record = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
              "lines": len(text.splitlines()), "kind": classify(name), "language": Path(name).suffix,
              "symbols": [], "imports": [], "static_test_count": 0}
    if name.endswith(".py"):
        try:
            tree = ast.parse(text, filename=name)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    record["symbols"].append({"name": node.name, "line": node.lineno, "end_line": node.end_lineno,
                                              "kind": type(node).__name__})
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                        record["static_test_count"] += 1
                elif isinstance(node, ast.Import):
                    record["imports"].extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    record["imports"].append("." * node.level + (node.module or ""))
            record["symbol_method"] = "python_ast"
        except SyntaxError as exc:
            record["parse_error"] = {"line": exc.lineno, "message": exc.msg}
    elif name.endswith(".rs"):
        record["symbol_method"] = "rust_lexical_not_semantic"
        for match in RS_SYMBOL.finditer(text):
            record["symbols"].append({"name": match.group(2), "kind": match.group(1),
                                      "line": text.count("\n", 0, match.start(1)) + 1})
        record["imports"] = re.findall(r"^\s*(?:pub\s+)?use\s+([^;]+);", text, re.M)
        record["static_test_count"] = len(re.findall(r"#\[(?:tokio::|async_std::)?test(?:\([^]]*\))?\]", text))
    record["imports"] = sorted(set(record["imports"]))
    record["symbols"].sort(key=lambda symbol: (symbol["line"], symbol["name"]))
    return record


def build_inventory() -> dict:
    output = {"schema": "mslim.native_transfer.source_inventory/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Git-visible regular source files; no runtime data, credentials, dependency caches or symlink traversal",
              "limitations": ["Structural inventory is not a semantic review of every line.",
                              "Rust symbol extraction is lexical and may include comments or miss macros.",
                              "Test counts are static, not test results.",
                              "Historical sources and vendored application code are classified separately."], "projects": {}}
    for project in PROJECTS:
        root = BASE / project
        names = selected_files(root)
        source_names = [name for name in names if Path(name).suffix in EXTENSIONS]
        sources = [source_record(root, name) for name in source_names]
        manifests = []
        for name in names:
            if Path(name).name not in {"Cargo.toml", "pyproject.toml"}:
                continue
            try:
                raw = (root / name).read_bytes()
                manifest = tomllib.loads(raw.decode())
                package = manifest.get("package", manifest.get("project", {}))
                deps = manifest.get("dependencies", {})
                manifests.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
                                  "name": package.get("name"), "license": package.get("license"),
                                  "dependencies": deps, "workspace": manifest.get("workspace", {})})
            except (ValueError, UnicodeError) as exc:
                manifests.append({"path": name, "error": str(exc)})
        fingerprint = hashlib.sha256("".join(f"{source['path']}\0{source['sha256']}\n" for source in sources).encode()).hexdigest()
        status = git(root, "status", "--short")
        output["projects"][project] = {"root": str(root), "head": git(root, "rev-parse", "HEAD").strip(),
            "branch": git(root, "branch", "--show-current").strip(), "working_tree_dirty_at_snapshot": bool(status.strip()),
            "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(), "git_visible_files_in_scope": len(names),
            "source_fingerprint": fingerprint, "source_count": len(sources), "source_lines": sum(s["lines"] for s in sources),
            "kinds": dict(collections.Counter(s["kind"] for s in sources)),
            "languages": dict(collections.Counter(s["language"] for s in sources)),
            "static_tests": sum(s["static_test_count"] for s in sources),
            "manifests": manifests, "sources": sources}
    return output


def check_unchanged(path: Path) -> dict:
    inventory = json.loads(path.read_text())
    results = {}
    for project, snapshot in inventory["projects"].items():
        root = BASE / project
        changes = []
        for source in snapshot["sources"]:
            target = root / source["path"]
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != source["sha256"]:
                changes.append(source["path"])
        before = {s["path"] for s in snapshot["sources"]}
        after = {p for p in selected_files(root) if Path(p).suffix in EXTENSIONS}
        results[project] = {"changed_or_missing": changes, "added": sorted(after - before),
                            "unchanged": not changes and before == after}
    return results


def main() -> None:
    inventory_path = HERE / "source_inventory.v2.json"
    if "--check-unchanged" in sys.argv:
        result = check_unchanged(inventory_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if all(item["unchanged"] for item in result.values()) else 2)
    if inventory_path.exists():
        raise SystemExit("Snapshot already exists; use a new audit directory instead of overwriting evidence.")
    inventory = build_inventory()
    with inventory_path.open("x") as stream:
        json.dump(inventory, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (HERE / "source_inventory.v2.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["project", "path", "kind", "language", "lines", "bytes", "sha256", "symbols", "static_tests"])
        for project, data in inventory["projects"].items():
            for item in data["sources"]:
                writer.writerow([project, item["path"], item["kind"], item["language"], item["lines"], item["bytes"],
                                 item["sha256"], len(item["symbols"]), item["static_test_count"]])
    for project, data in inventory["projects"].items():
        print(project, json.dumps({key: data[key] for key in ("head", "source_count", "source_lines", "kinds", "languages", "static_tests", "source_fingerprint")}, ensure_ascii=False))
    print("SAVED", inventory_path)


if __name__ == "__main__":
    main()
