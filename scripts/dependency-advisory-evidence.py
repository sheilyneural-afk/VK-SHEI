#!/usr/bin/env python3
"""Build and verify a source-bound dependency advisory inventory.

This script deliberately separates two claims:

* the advisory inventory and its dependency versions are reproducible; and
* no advisory is new or has drifted from the reviewed baseline.

A passing delta gate does not mean that the known findings are resolved or
that an installed runtime is safe.  Known open findings remain explicit in the
receipt until the vulnerable dependency disappears from the observed inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from evidence_io import atomic_json_write, file_sha256


BASELINE_SCHEMA = "mslim.dependency_advisory_baseline/v1"
EVIDENCE_SCHEMA = "mslim.dependency_advisory_evidence/v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(command: list[str], root: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def source_identity(root: Path) -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"], root)
    tree = command_output(["git", "rev-parse", "HEAD^{tree}"], root)
    dirty = command_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], root
    )
    github_sha = os.environ.get("GITHUB_SHA") or None
    return {
        "commit": commit,
        "tree": tree,
        "tracked_files_modified": bool(dirty),
        "github_sha": github_sha,
        "github_sha_matches_checkout": github_sha is None or github_sha == commit,
    }


def clean_version_requirement(requirement: str | None) -> str | None:
    if requirement is None:
        return None
    value = requirement.strip()
    if value.startswith("="):
        return value[1:].strip()
    return value


def cargo_findings(document: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in document.get("vulnerabilities", {}).get("list", []):
        advisory = item["advisory"]
        package = item["package"]
        findings.append(
            {
                "source": "cargo-audit",
                "id": advisory["id"],
                "kind": "vulnerability",
                "package": package["name"],
                "version": package["version"],
                "title": advisory["title"].strip(),
                "aliases": sorted(advisory.get("aliases") or []),
                "patched": sorted(item.get("versions", {}).get("patched") or []),
            }
        )
    for kind, items in sorted((document.get("warnings") or {}).items()):
        for item in items:
            advisory = item["advisory"]
            package = item["package"]
            findings.append(
                {
                    "source": "cargo-audit",
                    "id": advisory["id"],
                    "kind": kind,
                    "package": package["name"],
                    "version": package["version"],
                    "title": advisory["title"].strip(),
                    "aliases": sorted(advisory.get("aliases") or []),
                    "patched": sorted(item.get("versions", {}).get("patched") or []),
                }
            )
    return findings


def dependabot_findings(document: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in document:
        advisory = item["security_advisory"]
        dependency = item["dependency"]
        vulnerability = item["security_vulnerability"]
        aliases = [value for value in [advisory.get("cve_id")] if value]
        patched = vulnerability.get("first_patched_version")
        findings.append(
            {
                "source": "dependabot",
                "id": advisory["ghsa_id"],
                "kind": "vulnerability",
                "package": dependency["package"]["name"],
                "version": clean_version_requirement(
                    dependency.get("version_requirement")
                ),
                "title": advisory["summary"].strip(),
                "aliases": sorted(aliases),
                "severity": advisory.get("severity"),
                "patched": [patched["identifier"]] if patched else [],
                "url": item.get("html_url"),
            }
        )
    return findings


def finding_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item["source"]), str(item["id"])


def validate_baseline(document: dict[str, Any]) -> None:
    if document.get("schema") != BASELINE_SCHEMA:
        raise SystemExit(f"unsupported baseline schema: {document.get('schema')!r}")
    entries = document.get("known_advisories")
    if not isinstance(entries, list):
        raise SystemExit("baseline does not contain known_advisories")
    seen: set[tuple[str, str]] = set()
    required = {
        "source",
        "id",
        "kind",
        "package",
        "version",
        "dependency_path",
        "feature_surface",
        "reachability",
        "disposition",
        "tracking_issue",
    }
    for entry in entries:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise SystemExit("baseline contains an incomplete advisory entry")
        key = finding_key(entry)
        if key in seen:
            raise SystemExit(f"duplicate baseline advisory: {key}")
        seen.add(key)


def compare(
    detected: list[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    expected = {finding_key(item): item for item in baseline["known_advisories"]}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in detected:
        key = finding_key(item)
        if key in observed:
            raise SystemExit(f"duplicate detected advisory: {key}")
        observed[key] = item

    known: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    unregistered: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    identity_fields = ("kind", "package", "version")

    for key, finding in sorted(observed.items()):
        reviewed = expected.get(key)
        if reviewed is None:
            unregistered.append(finding)
            continue
        differences = {
            field: {"expected": reviewed.get(field), "observed": finding.get(field)}
            for field in identity_fields
            if reviewed.get(field) != finding.get(field)
        }
        enriched = {**finding, "review": reviewed}
        if differences:
            enriched["differences"] = differences
            changed.append(enriched)
        else:
            known.append(enriched)

    for key, reviewed in sorted(expected.items()):
        if key not in observed:
            resolved.append(reviewed)

    return {
        "known_open": known,
        "changed": changed,
        "unregistered": unregistered,
        "not_detected": resolved,
    }


def markdown_summary(evidence: dict[str, Any]) -> str:
    result = evidence["result"]
    lines = [
        "## Dependency advisory evidence",
        "",
        f"**Delta gate:** `{result['delta_gate']}` — "
        f"**risk state:** `{result['risk_state']}`",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Known findings still detected | {result['known_open']} |",
        f"| New unregistered findings | {result['unregistered']} |",
        f"| Findings with package/version drift | {result['changed']} |",
        f"| Baseline findings no longer detected | {result['not_detected']} |",
        "",
        "A passing delta gate means only that no unreviewed advisory or identity "
        "drift was observed against the exact baseline. It does **not** mean "
        "the known findings are fixed, reachable, or unreachable.",
        "",
        "| Source | Advisory | Package | Version | Surface | Reachability review |",
        "|---|---|---|---|---|---|",
    ]
    for finding in evidence["findings"]["known_open"]:
        review = finding["review"]
        reachability = str(review["reachability"]).replace("|", "\\|")
        lines.append(
            "| {source} | `{id}` | `{package}` | `{version}` | {surface} | {reachability} |".format(
                source=finding["source"],
                id=finding["id"],
                package=finding["package"],
                version=finding["version"],
                surface=review["feature_surface"],
                reachability=reachability,
            )
        )
    if evidence["findings"]["unregistered"]:
        lines.extend(["", "### Unregistered findings", ""])
        for finding in evidence["findings"]["unregistered"]:
            lines.append(
                f"- `{finding['source']}:{finding['id']}` — "
                f"`{finding['package']} {finding['version']}`"
            )
    if evidence["findings"]["changed"]:
        lines.extend(["", "### Identity drift", ""])
        for finding in evidence["findings"]["changed"]:
            lines.append(
                f"- `{finding['source']}:{finding['id']}`: "
                f"`{json.dumps(finding['differences'], sort_keys=True)}`"
            )
    return "\n".join(lines) + "\n"


def build(arguments: argparse.Namespace) -> int:
    root = arguments.root.resolve()
    baseline_path = arguments.baseline.resolve()
    cargo_path = arguments.cargo_audit_json.resolve()
    dependabot_path = arguments.dependabot_json.resolve()
    lockfile_path = arguments.lockfile.resolve()

    baseline = load_json(baseline_path)
    validate_baseline(baseline)
    cargo_document = load_json(cargo_path)
    dependabot_document = load_json(dependabot_path)
    if not isinstance(cargo_document, dict):
        raise SystemExit("cargo audit input is not a JSON object")
    if not isinstance(dependabot_document, list):
        raise SystemExit("Dependabot input is not a JSON array")

    detected = cargo_findings(cargo_document) + dependabot_findings(dependabot_document)
    comparison = compare(detected, baseline)
    lockfile_sha = file_sha256(lockfile_path)
    expected_lockfile_sha = baseline["captured_from"]["cargo_lock_sha256"]
    lockfile_matches = lockfile_sha == expected_lockfile_sha
    delta_passed = (
        not comparison["unregistered"]
        and not comparison["changed"]
        and lockfile_matches
    )
    known_count = len(comparison["known_open"])
    result = {
        "delta_gate": "passed" if delta_passed else "failed",
        "risk_state": "known_findings_open" if known_count else "no_known_findings",
        "known_open": known_count,
        "unregistered": len(comparison["unregistered"]),
        "changed": len(comparison["changed"]),
        "not_detected": len(comparison["not_detected"]),
        "lockfile_matches_baseline": lockfile_matches,
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_identity(root),
        "inputs": {
            "baseline": {
                "path": baseline_path.relative_to(root).as_posix(),
                "sha256": file_sha256(baseline_path),
            },
            "cargo_audit": {
                "path": cargo_path.name,
                "sha256": file_sha256(cargo_path),
                "database": cargo_document.get("database"),
            },
            "dependabot": {
                "path": dependabot_path.name,
                "sha256": file_sha256(dependabot_path),
            },
            "cargo_lock": {
                "path": lockfile_path.relative_to(root).as_posix(),
                "sha256": lockfile_sha,
                "baseline_sha256": expected_lockfile_sha,
            },
        },
        "result": result,
        "findings": comparison,
        "boundaries": {
            "proves": [
                "the raw Cargo audit and open Dependabot inputs were normalized",
                "the detected advisory identities were compared to the reviewed baseline",
                "the exact Cargo.lock digest was compared to the baseline digest",
            ],
            "does_not_by_itself_prove": [
                "runtime reachability or exploitability",
                "absence of advisories missing from both source databases",
                "installed or resident artifact identity",
                "physical backend safety",
                "production readiness",
            ],
        },
    }
    if evidence["source"]["tracked_files_modified"] and not arguments.allow_dirty:
        raise SystemExit("refusing reusable evidence from modified tracked source")
    if not evidence["source"]["github_sha_matches_checkout"]:
        raise SystemExit("GITHUB_SHA does not match the checked-out source revision")
    atomic_json_write(arguments.output.resolve(), evidence)
    if arguments.summary:
        arguments.summary.resolve().write_text(
            markdown_summary(evidence), encoding="utf-8"
        )
    return 0


def verify(arguments: argparse.Namespace) -> int:
    evidence = load_json(arguments.evidence.resolve())
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise SystemExit(f"unsupported evidence schema: {evidence.get('schema')!r}")
    result = evidence.get("result")
    if not isinstance(result, dict):
        raise SystemExit("evidence does not contain a result")
    if result.get("delta_gate") != "passed":
        raise SystemExit(
            "dependency advisory delta failed: "
            f"unregistered={result.get('unregistered')}, "
            f"changed={result.get('changed')}, "
            f"lockfile_matches_baseline={result.get('lockfile_matches_baseline')}"
        )
    print(
        "OK: advisory delta passed; "
        f"known_open={result.get('known_open')} "
        f"not_detected={result.get('not_detected')}"
    )
    return 0


def self_test(_: argparse.Namespace) -> int:
    baseline = {
        "schema": BASELINE_SCHEMA,
        "captured_from": {"cargo_lock_sha256": "unused"},
        "known_advisories": [
            {
                "source": "cargo-audit",
                "id": "RUSTSEC-TEST-0001",
                "kind": "vulnerability",
                "package": "example",
                "version": "1.0.0",
                "dependency_path": "root -> example",
                "feature_surface": "test",
                "reachability": "not evaluated",
                "disposition": "open",
                "tracking_issue": "https://example.invalid/1",
            }
        ],
    }
    validate_baseline(baseline)
    finding = {
        "source": "cargo-audit",
        "id": "RUSTSEC-TEST-0001",
        "kind": "vulnerability",
        "package": "example",
        "version": "1.0.0",
    }
    result = compare([finding], baseline)
    assert len(result["known_open"]) == 1
    assert not result["unregistered"]
    drifted = {**finding, "version": "2.0.0"}
    assert len(compare([drifted], baseline)["changed"]) == 1
    new = {**finding, "id": "RUSTSEC-TEST-0002"}
    assert len(compare([new], baseline)["unregistered"]) == 1
    assert len(compare([], baseline)["not_detected"]) == 1
    print("OK: dependency advisory evidence self-test passed")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--root", type=Path, default=Path.cwd())
    build_parser.add_argument("--baseline", type=Path, required=True)
    build_parser.add_argument("--cargo-audit-json", type=Path, required=True)
    build_parser.add_argument("--dependabot-json", type=Path, required=True)
    build_parser.add_argument("--lockfile", type=Path, default=Path("Cargo.lock"))
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--summary", type=Path)
    build_parser.add_argument("--allow-dirty", action="store_true")
    build_parser.set_defaults(handler=build)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("evidence", type=Path)
    verify_parser.set_defaults(handler=verify)

    test_parser = subparsers.add_parser("self-test")
    test_parser.set_defaults(handler=self_test)
    return result


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
