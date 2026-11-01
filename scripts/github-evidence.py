#!/usr/bin/env python3
"""Create and verify bounded GitHub Actions evidence for M-SLIM.

The manifest deliberately records what a workflow observed and what it did not
prove.  It is not a replacement for the runtime validation matrix, calibration
receipt, installed-release verification, persistence readback, or restart E2E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Iterable


SCHEMA = "mslim.github_evidence/v1"
CHUNK_SIZE = 1024 * 1024


def run(command: list[str], root: Path, *, required: bool = True) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=required,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        if required:
            raise
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def describe_artifacts(paths: Iterable[Path], display_root: Path) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    seen: set[Path] = set()
    for candidate in paths:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"evidence artifact must be a regular non-symlink file: {candidate}")
        artifacts.append(
            {
                "path": display_path(path, display_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    artifacts.sort(key=lambda item: str(item["path"]))
    return artifacts


def source_identity(root: Path) -> dict[str, object]:
    commit = run(["git", "rev-parse", "HEAD"], root)
    tree = run(["git", "rev-parse", "HEAD^{tree}"], root)
    branch = run(["git", "branch", "--show-current"], root, required=False) or None
    dirty_output = run(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    github_sha = os.environ.get("GITHUB_SHA") or None
    return {
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "tracked_files_modified": bool(dirty_output),
        "github_sha": github_sha,
        "github_sha_matches_checkout": github_sha is None or github_sha == commit,
    }


def tool_versions(root: Path) -> dict[str, str | None]:
    return {
        "rustc": run(["rustc", "--version"], root, required=False),
        "cargo": run(["cargo", "--version"], root, required=False),
        "python": run(["python3", "--version"], root, required=False),
    }


def github_context() -> dict[str, str | None]:
    names = (
        "GITHUB_REPOSITORY",
        "GITHUB_WORKFLOW",
        "GITHUB_WORKFLOW_REF",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_NUMBER",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "GITHUB_EVENT_NAME",
        "GITHUB_REF",
        "GITHUB_REF_TYPE",
        "GITHUB_REF_NAME",
        "GITHUB_ACTOR",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_NAME",
    )
    return {name.lower(): os.environ.get(name) or None for name in names}


def boundaries(kind: str) -> dict[str, object]:
    return {
        "evidence_kind": kind,
        "proves": [
            "the listed files existed with the recorded hashes in this workflow",
            "the checkout and GitHub source identities recorded in this manifest",
            "the explicitly recorded workflow status",
        ],
        "does_not_by_itself_prove": [
            "configuration-matching validation matrix",
            "valid sustained calibration receipt",
            "installed artifact identity",
            "resident process identity",
            "authenticated product request and external effect",
            "durable persistence readback",
            "provider restart recovery",
            "production readiness",
        ],
    }


def atomic_json_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(rendered)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def build(arguments: argparse.Namespace) -> int:
    root = arguments.root.resolve()
    artifact_base = (arguments.artifact_base or root).resolve()
    artifacts = describe_artifacts(arguments.artifact, artifact_base)
    identity = source_identity(root)
    if identity["tracked_files_modified"] and not arguments.allow_dirty:
        raise SystemExit("refusing reusable GitHub evidence from modified tracked source")
    if not identity["github_sha_matches_checkout"]:
        raise SystemExit("GITHUB_SHA does not match the checked-out source revision")

    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "status": arguments.status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": identity,
        "github": github_context(),
        "toolchain": tool_versions(root),
        "artifacts": artifacts,
        "claims": sorted(set(arguments.claim)),
        "boundaries": boundaries(arguments.kind),
    }
    atomic_json_write(arguments.output.resolve(), manifest)
    return 0


def verify(arguments: argparse.Namespace) -> int:
    manifest_path = arguments.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise SystemExit(f"unsupported evidence schema: {manifest.get('schema')!r}")
    if manifest.get("status") not in {"passed", "failed", "cancelled", "skipped"}:
        raise SystemExit("invalid evidence status")
    if not isinstance(manifest.get("boundaries", {}).get("does_not_by_itself_prove"), list):
        raise SystemExit("missing evidence boundaries")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SystemExit("missing artifact inventory")
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise SystemExit("invalid artifact entry")
        path_value = artifact.get("path")
        digest = artifact.get("sha256")
        size = artifact.get("size_bytes")
        if not isinstance(path_value, str) or not path_value or path_value in seen:
            raise SystemExit("invalid or duplicate artifact path")
        seen.add(path_value)
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("artifact path escapes its evidence root")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit("invalid artifact SHA-256")
        try:
            int(digest, 16)
        except ValueError as error:
            raise SystemExit("invalid artifact SHA-256") from error
        if not isinstance(size, int) or size < 0:
            raise SystemExit("invalid artifact size")
        if arguments.artifact_root:
            local = arguments.artifact_root.resolve() / relative
            if not local.is_file() or local.is_symlink():
                raise SystemExit(f"missing regular artifact: {relative}")
            if local.stat().st_size != size or sha256(local) != digest:
                raise SystemExit(f"artifact digest mismatch: {relative}")
    if arguments.require_source:
        current = run(["git", "rev-parse", "HEAD"], arguments.root.resolve())
        if manifest.get("source", {}).get("commit") != current:
            raise SystemExit("manifest source commit does not match the current checkout")
    return 0


def self_test(_: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="mslim-github-evidence-") as directory:
        root = Path(directory)
        first = root / "first.bin"
        second = root / "second.json"
        first.write_bytes(b"mslim-evidence\n")
        second.write_text('{"status":"passed"}\n', encoding="utf-8")
        described = describe_artifacts([second, first, first], root)
        assert [item["path"] for item in described] == ["first.bin", "second.json"]
        assert described[0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()

        output = root / "receipt.json"
        atomic_json_write(
            output,
            {
                "schema": SCHEMA,
                "status": "passed",
                "boundaries": boundaries("self-test"),
            },
        )
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["schema"] == SCHEMA
        assert "production readiness" in loaded["boundaries"]["does_not_by_itself_prove"]
    print("OK: GitHub evidence self-test passed")
    return 0


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)

    create = subcommands.add_parser("build", help="write an evidence manifest")
    create.add_argument("--root", type=Path, default=Path.cwd())
    create.add_argument(
        "--artifact-base",
        type=Path,
        help="base used to store portable relative artifact paths",
    )
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--kind", required=True)
    create.add_argument(
        "--status", choices=("passed", "failed", "cancelled", "skipped"), required=True
    )
    create.add_argument("--artifact", type=Path, action="append", default=[])
    create.add_argument("--claim", action="append", default=[])
    create.add_argument("--allow-dirty", action="store_true")
    create.set_defaults(handler=build)

    validate = subcommands.add_parser("verify", help="validate a manifest boundary")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--require-source", action="store_true")
    validate.add_argument(
        "--artifact-root", type=Path, help="recompute every recorded artifact hash here"
    )
    validate.set_defaults(handler=verify)

    tests = subcommands.add_parser("self-test", help="run dependency-free regression tests")
    tests.set_defaults(handler=self_test)
    return command


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.handler(parsed))
