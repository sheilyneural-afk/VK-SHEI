"""Shared process boundary for the retrieval evaluation scripts."""

from __future__ import annotations

import json
import subprocess


def run(binary: str, *arguments: str):
    completed = subprocess.run(
        [binary, *arguments], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        command = " ".join(arguments)
        raise SystemExit(f"{binary} {command} failed: {completed.stderr[:500]}")
    return (
        json.loads(completed.stdout)
        if completed.stdout.lstrip().startswith("{")
        else completed.stdout
    )
