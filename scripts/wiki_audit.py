#!/usr/bin/env python3
"""Generate and audit the KV-SHEI GitHub wiki without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "wiki" / "wiki-manifest.json"
CLAIMS_PATH = ROOT / "wiki" / "claims.json"
SCHEMA = "mslim.wiki_audit_receipt/v1"
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
FENCE = re.compile(r"^```(?P<kind>[^\s`]*)\s*$")
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)
INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
TEXT_SUFFIXES = {
    ".cl",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


class AuditError(RuntimeError):
    pass


def run_git(*arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and process.returncode:
        raise AuditError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {path.relative_to(ROOT)}: {error}") from error


def manifest() -> dict:
    data = load_json(MANIFEST_PATH)
    if data.get("schema") != "mslim.wiki_manifest/v1":
        raise AuditError("unsupported wiki manifest schema")
    return data


def source_revision(path: str) -> str:
    revision = run_git("log", "-1", "--format=%H", "--", path)
    if not revision:
        raise AuditError(f"source is not tracked: {path}")
    return revision


def working_tree_dirty() -> bool:
    return bool(run_git("status", "--porcelain", check=True))


def source_dirty(path: str) -> bool:
    process = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", path],
        check=False,
    )
    if process.returncode not in (0, 1):
        raise AuditError(f"cannot determine source cleanliness: {path}")
    return process.returncode == 1


def page_map(configuration: dict) -> dict[str, str]:
    return {entry["source"]: Path(entry["target"]).stem for entry in configuration["derived_pages"]}


def rewrite_target(target: str, source: str, mapping: dict[str, str], revision: str) -> str:
    if target.startswith(("http://", "https://", "#", "mailto:")):
        return target
    clean_target, anchor = (target.split("#", 1) + [""])[:2]
    source_dir = Path(source).parent
    resolved = os.path.normpath((source_dir / clean_target).as_posix())
    while resolved.startswith("../"):
        resolved = resolved[3:]
    if resolved in mapping:
        result = mapping[resolved]
        return f"{result}#{anchor}" if anchor else result
    local = ROOT / resolved
    if local.exists():
        base = f"https://github.com/sheilyneural-afk/KV-SHEI/blob/{revision}/{resolved}"
        return f"{base}#{anchor}" if anchor else base
    return target


def derive_page(entry: dict, configuration: dict) -> str:
    source = entry["source"]
    source_path = ROOT / source
    if not source_path.is_file():
        raise AuditError(f"missing canonical wiki source: {source}")
    revision = source_revision(source)
    short_revision = revision[:12]
    dirty = source_dirty(source)
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise AuditError(f"canonical page must start with one H1: {source}")
    provenance = (
        f"> Página generada desde [`{source}`]"
        f"(https://github.com/sheilyneural-afk/KV-SHEI/blob/{revision}/{source}) "
        f"en el commit `{short_revision}`. Si difieren, manda el archivo del repositorio."
    )
    if dirty:
        provenance += " **Propuesta local no publicable: la fuente tiene cambios sin commit.**"
    remainder = lines[1:]
    while remainder and not remainder[0].strip():
        remainder.pop(0)
    body = "\n".join([lines[0], "", provenance, "", *remainder]).rstrip() + "\n"
    mapping = page_map(configuration)

    def replace_link(match: re.Match[str]) -> str:
        label, target = match.groups()
        return f"[{label}]({rewrite_target(target, source, mapping, revision)})"

    return MARKDOWN_LINK.sub(replace_link, body)


def copy_editorial_pages(existing: Path, output: Path, configuration: dict) -> None:
    replacements = configuration.get("editorial_replacements", {})

    def copy_with_migrations(source: Path, target: Path) -> None:
        text = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        target.write_text(text, encoding="utf-8", newline="\n")

    for name in configuration["required_editorial_pages"]:
        source = existing / name
        if not source.is_file():
            raise AuditError(f"published wiki is missing required editorial page: {name}")
        copy_with_migrations(source, output / name)
    for source in sorted(existing.glob("*.md")):
        target = output / source.name
        if not target.exists() and source.name not in {entry["target"] for entry in configuration["derived_pages"]}:
            copy_with_migrations(source, target)


def generate(existing: Path, output: Path) -> None:
    configuration = manifest()
    if not existing.is_dir():
        raise AuditError(f"existing wiki directory does not exist: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.md"):
        stale.unlink()
    copy_editorial_pages(existing, output, configuration)
    for entry in configuration["derived_pages"]:
        (output / entry["target"]).write_text(
            derive_page(entry, configuration), encoding="utf-8", newline="\n"
        )


def markdown_files(wiki: Path) -> list[Path]:
    files = sorted(wiki.glob("*.md"))
    if not files:
        raise AuditError(f"no Markdown pages found in {wiki}")
    return files


def fenced_blocks(text: str, filename: str) -> tuple[int, list[str]]:
    open_kind: str | None = None
    content: list[str] = []
    mermaid: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if match:
            if open_kind is None:
                open_kind = match.group("kind")
                content = []
            else:
                if open_kind == "mermaid":
                    mermaid.append("\n".join(content))
                open_kind = None
            continue
        if open_kind is not None:
            content.append(line)
    if open_kind is not None:
        raise AuditError(f"unclosed code fence in {filename}")
    return len(mermaid), mermaid


def validate_mermaid(block: str, location: str) -> None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines or not re.match(
        r"^(?:flowchart|graph|sequenceDiagram|stateDiagram(?:-v2)?|classDiagram|erDiagram|gantt|journey|pie)\b",
        lines[0],
    ):
        raise AuditError(f"unsupported or missing Mermaid declaration in {location}")
    if "[/" in block or "/]" in block:
        raise AuditError(f"ambiguous Mermaid node delimiter in {location}; quote the label")
    square = 0
    curly = 0
    for character in block:
        square += character == "["
        square -= character == "]"
        curly += character == "{"
        curly -= character == "}"
        if square < 0 or curly < 0:
            raise AuditError(f"unbalanced Mermaid delimiters in {location}")
    if square or curly:
        raise AuditError(f"unbalanced Mermaid delimiters in {location}")


def internal_target_exists(wiki: Path, target: str) -> bool:
    target = target.split("#", 1)[0]
    if not target:
        return True
    candidate = wiki / target
    if candidate.suffix:
        return candidate.exists()
    return candidate.with_suffix(".md").exists()


def audit_claims() -> int:
    data = load_json(CLAIMS_PATH)
    if data.get("schema") != "mslim.wiki_claims/v1":
        raise AuditError("unsupported wiki claims schema")
    seen: set[str] = set()
    for claim in data.get("claims", []):
        claim_id = claim.get("id", "")
        if not claim_id or claim_id in seen:
            raise AuditError(f"missing or duplicate claim id: {claim_id!r}")
        seen.add(claim_id)
        source = ROOT / claim["source"]
        if not source.is_file():
            raise AuditError(f"claim {claim_id} references missing source {claim['source']}")
        if claim["pattern"] not in source.read_text(encoding="utf-8"):
            raise AuditError(f"claim {claim_id} no longer matches {claim['source']}")
    return len(seen)


def canonical_text_corpus() -> str:
    parts: list[str] = []
    excluded_parts = {
        ".git",
        "target",
        "runtime",
        "wiki",
        "__pycache__",
        ".agent-cache",
        ".mcp_trash",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"Cargo.toml", "Cargo.lock"}:
            continue
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(parts)


def is_technical_identifier(value: str) -> bool:
    if not value or any(character.isspace() for character in value):
        return False
    if value.startswith(("MSLIM_", "mslim.", "mslim-", "--", "/v1/", "/ready", "/health")):
        return True
    if "/" in value and not value.startswith(("http://", "https://")):
        return True
    if Path(value).suffix in TEXT_SUFFIXES:
        return True
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9_]{3,}", value))


def identifier_resolves(value: str, corpus: str) -> bool:
    if value in corpus:
        return True
    if "<" in value and ">" in value:
        return True
    if any(character in value for character in "*?["):
        return any(ROOT.glob(value))
    candidate = ROOT / value
    if candidate.exists():
        return True
    if "/" not in value:
        return any(path.is_file() for path in ROOT.rglob(value))
    return False


def audit_identifiers(files: list[Path], configuration: dict) -> tuple[int, int, list[str]]:
    candidates: set[str] = set()
    for path in files:
        for value in INLINE_CODE.findall(path.read_text(encoding="utf-8")):
            value = value.strip()
            if is_technical_identifier(value):
                candidates.add(value)
    corpus = canonical_text_corpus()
    historical = set(configuration.get("historical_identifiers", []))
    unresolved = sorted(
        value for value in candidates if value not in historical and not identifier_resolves(value, corpus)
    )
    observed_historical = len(candidates & historical)
    return len(candidates), observed_historical, unresolved


def tree_digest(files: list[Path], wiki: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(wiki).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit(wiki: Path) -> dict:
    configuration = manifest()
    files = markdown_files(wiki)
    names = {path.name for path in files}
    expected = {entry["target"] for entry in configuration["derived_pages"]}
    expected.update(configuration["required_editorial_pages"])
    missing = sorted(expected - names)
    if missing:
        raise AuditError(f"wiki is missing required pages: {', '.join(missing)}")
    links = 0
    mermaid_blocks = 0
    secret_findings = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        count, blocks = fenced_blocks(text, path.name)
        mermaid_blocks += count
        for index, block in enumerate(blocks, start=1):
            validate_mermaid(block, f"{path.name} block {index}")
        for pattern in SECRET_PATTERNS:
            findings = pattern.findall(text)
            secret_findings += len(findings)
        for match in MARKDOWN_LINK.finditer(text):
            links += 1
            target = match.group(2)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not internal_target_exists(wiki, target):
                raise AuditError(f"broken internal link in {path.name}: {target}")
    if secret_findings:
        raise AuditError(f"possible secrets found in wiki: {secret_findings}")
    claims = audit_claims()
    identifier_count, historical_identifier_count, unresolved_identifiers = audit_identifiers(
        files, configuration
    )
    if unresolved_identifiers:
        sample = ", ".join(unresolved_identifiers[:12])
        suffix = "" if len(unresolved_identifiers) <= 12 else ", ..."
        raise AuditError(
            f"unresolved technical identifiers ({len(unresolved_identifiers)}): {sample}{suffix}"
        )
    head = run_git("rev-parse", "HEAD")
    dirty = working_tree_dirty()
    return {
        "schema": SCHEMA,
        "source_revision": head,
        "source_tree_clean": not dirty,
        "reusable_evidence": not dirty,
        "wiki_content_sha256": tree_digest(files, wiki),
        "page_count": len(files),
        "derived_page_count": len(configuration["derived_pages"]),
        "claim_count": claims,
        "technical_identifier_count": identifier_count,
        "historical_identifier_count": historical_identifier_count,
        "unresolved_technical_identifiers": 0,
        "markdown_link_count": links,
        "mermaid_block_count": mermaid_blocks,
        "broken_internal_links": 0,
        "secret_findings": 0,
        "status": "passed" if not dirty else "passed-non-reusable-dirty-tree",
    }


def write_receipt(receipt: dict, path: Path | None) -> None:
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(rendered)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8", newline="\n")


def diff_wikis(existing: Path, generated: Path) -> int:
    names = {path.name for path in existing.glob("*.md")}
    names.update(path.name for path in generated.glob("*.md"))
    changed = False
    null = Path(os.devnull)
    for name in sorted(names):
        left = existing / name
        right = generated / name
        process = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--",
                str(left if left.exists() else null),
                str(right if right.exists() else null),
            ],
            check=False,
            text=True,
        )
        if process.returncode == 1:
            changed = True
        elif process.returncode != 0:
            raise AuditError(f"unable to compare wiki page: {name}")
    return 1 if changed else 0


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        wiki = Path(directory)
        for entry in manifest()["derived_pages"]:
            (wiki / entry["target"]).write_text("# Test\n", encoding="utf-8")
        for name in manifest()["required_editorial_pages"]:
            (wiki / name).write_text("# Test\n", encoding="utf-8")
        receipt = audit(wiki)
        assert receipt["page_count"] >= 20
        assert receipt["claim_count"] == 10
        try:
            validate_mermaid("flowchart LR\n E --> R[/ready]", "test")
        except AuditError:
            pass
        else:
            raise AssertionError("ambiguous Mermaid node was accepted")
        (wiki / "Home.md").write_text("[missing](Not-a-page)\n", encoding="utf-8")
        try:
            audit(wiki)
        except AuditError:
            pass
        else:
            raise AssertionError("broken internal link was accepted")
    print("wiki audit self-test: passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--existing-wiki", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--wiki", type=Path, required=True)
    check_parser.add_argument("--receipt", type=Path)
    diff_parser = subparsers.add_parser("diff")
    diff_parser.add_argument("--existing-wiki", type=Path, required=True)
    diff_parser.add_argument("--wiki", type=Path, required=True)
    subparsers.add_parser("self-test")
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "generate":
            generate(arguments.existing_wiki.resolve(), arguments.output.resolve())
        elif arguments.command == "check":
            write_receipt(audit(arguments.wiki.resolve()), arguments.receipt)
        elif arguments.command == "diff":
            return diff_wikis(arguments.existing_wiki.resolve(), arguments.wiki.resolve())
        else:
            self_test()
        return 0
    except AuditError as error:
        print(f"wiki audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
