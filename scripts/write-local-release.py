#!/usr/bin/env python3
"""Atomically bind a local release manifest and environment to built inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise SystemExit(f"{name} is not a SHA-256 digest")


def atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def update_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0] if "=" in line and not line.lstrip().startswith("#") else None
        if key in values:
            output.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in values.items():
        if key not in seen:
            output.append(f"{key}={value}")
    atomic_write(path, ("\n".join(output) + "\n").encode(), 0o600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build-input-digest", required=True)
    parser.add_argument("--model-digest", required=True)
    arguments = parser.parse_args()
    require_sha256("build_input_digest", arguments.build_input_digest)
    require_sha256("model_digest", arguments.model_digest)
    root = arguments.root.resolve()
    runtime = root / "deploy/local-production/runtime"
    gateway = runtime / "bin/mslim-gateway"
    worker = runtime / "bin/mslim-runtime-worker"
    model = runtime / "models" / arguments.model_digest
    for binary in (gateway, worker):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise SystemExit(f"missing executable: {binary}")
    if not model.is_dir():
        raise SystemExit(f"missing model: {model}")
    gateway_digest = file_sha256(gateway)
    worker_digest = file_sha256(worker)
    manifest = {
        "project": "KV-SHEI",
        "product": "M-SLIM",
        "technical_namespace": "mslim",
        "schema": "mslim.local_release/v3",
        "build_input_digest": arguments.build_input_digest,
        "binary": {"path": "runtime/bin/mslim-gateway", "sha256": gateway_digest},
        "worker_binary": {
            "path": "runtime/bin/mslim-runtime-worker",
            "sha256": worker_digest,
        },
        "model": {
            "relative_path": f"runtime/models/{arguments.model_digest}",
            "bundle_sha256": arguments.model_digest,
        },
    }
    atomic_write(
        root / "deploy/local-production/manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        0o600,
    )
    update_env(
        root / "deploy/local-production/local-production.env",
        {
            "MSLIM_MODEL_DIGEST": arguments.model_digest,
            "MSLIM_BINARY_DIGEST": gateway_digest,
            "MSLIM_WORKER_BINARY_DIGEST": worker_digest,
        },
    )


if __name__ == "__main__":
    main()
