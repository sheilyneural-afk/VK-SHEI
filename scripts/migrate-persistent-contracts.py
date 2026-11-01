#!/usr/bin/env python3
"""Migrate identity-bearing M-SLIM JSON contracts without rewriting history.

Every legacy source byte sequence is archived by SHA-256 before an atomic
replacement is activated.  Calibration transaction JSONL chains are
deliberately excluded: the Rust verifier preserves their v1 hash domain and
new transactions must use a fresh v2 root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


IDENTITY = {
    "project": "KV-SHEI",
    "product": "M-SLIM",
    "technical_namespace": "mslim",
}

SCHEMAS = {
    "mslim.model_bundle/v1": "mslim.model_bundle/v2",
    "mslim.validation_receipt/v1": "mslim.validation_receipt/v2",
    "mslim.validation-matrix/v1": "mslim.validation-matrix/v2",
    "mslim.calibration_receipt/v3": "mslim.calibration_receipt/v4",
}

CANONICAL = frozenset(SCHEMAS.values())
MIGRATION_SCHEMA = "mslim.contract_migration/v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def candidates(runtime: Path) -> list[Path]:
    paths: list[Path] = []
    models = runtime / "models"
    if models.is_dir():
        paths.extend(sorted(models.glob("*/bundle-manifest.json")))
    validation = runtime / "validation"
    if validation.is_dir():
        paths.extend(sorted(validation.glob("*.json")))
    calibration = runtime / "calibration"
    if calibration.is_dir():
        paths.extend(sorted(calibration.glob("*.json")))
    return paths


def validate_shape(path: Path, document: dict[str, object], runtime: Path) -> None:
    schema = document.get("schema")
    if not isinstance(schema, str):
        raise ValueError(f"{path}: missing string schema")
    bundle = document.get("bundle_sha256", document.get("bundle"))
    if schema.startswith("mslim.model_bundle/"):
        bundle = document.get("bundle_sha256")
        directory = path.parent.name
        if bundle != directory or len(directory) != 64:
            raise ValueError(f"{path}: bundle identity does not match directory")
        files = document.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{path}: empty bundle inventory")
        installed = sorted(
            item.relative_to(path.parent).as_posix()
            for item in path.parent.rglob("*")
            if item.is_file() and item != path
        )
        if sorted(files) != installed:
            raise ValueError(f"{path}: bundle file inventory drift")
    elif bundle is not None:
        if not isinstance(bundle, str) or len(bundle) != 64:
            raise ValueError(f"{path}: invalid bundle digest")
        if not (runtime / "models" / bundle).is_dir():
            raise ValueError(f"{path}: referenced bundle is not installed")


def canonical_bytes(document: dict[str, object]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def migrate_one(path: Path, runtime: Path, apply: bool) -> dict[str, str] | None:
    original = path.read_bytes()
    try:
        document = json.loads(original)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path}: top level must be an object")
    validate_shape(path, document, runtime)
    schema = document.get("schema")
    if schema in CANONICAL:
        if any(document.get(key) != value for key, value in IDENTITY.items()):
            raise ValueError(f"{path}: canonical contract has invalid project identity")
        return None
    if schema not in SCHEMAS:
        return None

    source_digest = sha256(original)
    archive_relative = Path("receipts/contract-migration/archive") / f"{source_digest}.json"
    archive = runtime / archive_relative
    if archive.exists() and archive.read_bytes() != original:
        raise ValueError(f"{archive}: content-addressed archive collision")

    migrated = dict(document)
    migrated.update(IDENTITY)
    migrated["schema"] = SCHEMAS[str(schema)]
    migrated["migration"] = {
        "schema": MIGRATION_SCHEMA,
        "source_schema": schema,
        "source_sha256": source_digest,
        "archive_relative_path": archive_relative.as_posix(),
    }
    replacement = canonical_bytes(migrated)
    result = {
        "path": path.relative_to(runtime).as_posix(),
        "source_schema": str(schema),
        "target_schema": SCHEMAS[str(schema)],
        "source_sha256": source_digest,
        "target_sha256": sha256(replacement),
        "archive_relative_path": archive_relative.as_posix(),
    }
    if apply:
        if not archive.exists():
            atomic_write(archive, original)
        atomic_write(path, replacement)
    return result


def run(runtime: Path, apply: bool) -> list[dict[str, str]]:
    results = []
    for path in candidates(runtime):
        result = migrate_one(path, runtime, apply)
        if result is not None:
            results.append(result)
    if apply and results:
        receipt = {
            **IDENTITY,
            "schema": MIGRATION_SCHEMA,
            "created_unix_seconds": int(time.time()),
            "migrations": results,
            "historical_transaction_policy": (
                "v1 calibration transaction chains are preserved byte-for-byte, "
                "verified in their original hash domain, and read-only"
            ),
        }
        receipt_bytes = canonical_bytes(receipt)
        receipt_path = runtime / "receipts/contract-migration" / f"{sha256(receipt_bytes)}.receipt.json"
        atomic_write(receipt_path, receipt_bytes)
    return results


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = Path(directory)
        digest = "a" * 64
        model = runtime / "models" / digest
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        manifest = model / "bundle-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "mslim.model_bundle/v1",
                    "bundle_sha256": digest,
                    "files": ["config.json"],
                }
            ),
            encoding="utf-8",
        )
        original = manifest.read_bytes()
        preview = run(runtime, apply=False)
        assert len(preview) == 1 and manifest.read_bytes() == original
        applied = run(runtime, apply=True)
        assert len(applied) == 1
        migrated = json.loads(manifest.read_text(encoding="utf-8"))
        assert migrated["schema"] == "mslim.model_bundle/v2"
        assert migrated["project"] == "KV-SHEI"
        archive = runtime / applied[0]["archive_relative_path"]
        assert archive.read_bytes() == original
        assert run(runtime, apply=False) == []


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report pending migrations without writing")
    mode.add_argument("--apply", action="store_true", help="archive and atomically migrate legacy contracts")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "deploy/local-production/runtime",
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        print("contract-migration-self-test: ok")
        return
    pending = run(arguments.runtime.resolve(), apply=arguments.apply)
    print(json.dumps({"mode": "apply" if arguments.apply else "check", "pending_or_applied": pending}, indent=2))
    if arguments.check and pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
