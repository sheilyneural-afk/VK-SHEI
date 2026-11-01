#!/usr/bin/env python3
"""Compute the content-addressed M-SLIM bundle digest used by importers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile


def bundle_digest(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"not a model directory: {root}")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"model bundle contains a symbolic link: {path}")
        for name in names:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"model bundle contains a symbolic link: {path}")
            if not path.is_file():
                raise ValueError(f"model bundle contains an unsupported file type: {path}")
            if name != "bundle-manifest.json":
                files.append(path.relative_to(root))
    files.sort()
    if not files:
        raise ValueError("model bundle is empty")
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with (root / relative).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "nested").mkdir()
        (root / "config.json").write_bytes(b"config")
        (root / "nested/weights.bin").write_bytes(b"weights")
        (root / "bundle-manifest.json").write_bytes(b"ignored")
        first = bundle_digest(root)
        (root / "bundle-manifest.json").write_bytes(b"still ignored")
        assert bundle_digest(root) == first
        (root / "nested/weights.bin").write_bytes(b"changed")
        assert bundle_digest(root) != first
        if hasattr(os, "symlink"):
            os.symlink(root / "config.json", root / "linked-config.json")
            try:
                bundle_digest(root)
            except ValueError as error:
                assert "symbolic link" in str(error)
            else:
                raise AssertionError("symbolic link was accepted")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("model-bundle-digest-self-test: ok")
    elif len(sys.argv) == 2:
        print(bundle_digest(Path(sys.argv[1]).resolve()))
    else:
        raise SystemExit("usage: model-bundle-digest.py MODEL_DIRECTORY | --self-test")
