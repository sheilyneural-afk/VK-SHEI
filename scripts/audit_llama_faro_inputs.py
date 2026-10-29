#!/usr/bin/env python3
"""Read-only, reproducible Faro audit using original Llama/MiniCPM tokenizers.

This script NEVER runs the 111K/112K corpus through the model; no claim
of full-volume LLM parity, retrieval fidelity, or V omission follows.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import struct
import sys
import urllib.error
import urllib.request
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = {
    "faro_short_7k": pathlib.Path("/tmp/faro-full.txt"),
    "faro_volume_full": ROOT / "assets/el-faro-volumen-completo.txt",
}
TOKENIZERS = {
    "Llama-3.1-8B-Instruct": pathlib.Path(
        "/home/yo/models/Llama-3.1-8B-Instruct/tokenizer.json"
    ),
    "MiniCPM5-1B": pathlib.Path("/home/yo/models/MiniCPM5-1B/tokenizer.json"),
}
QUESTION = "¿Cuál es el sello del primer cuaderno de bitácora?"


def digest_token_ids(ids: list[int]) -> str:
    payload = bytearray()
    for token in ids:
        payload.extend(struct.pack("<I", token))
    return hashlib.sha256(payload).hexdigest()


def llama_render(messages: list[tuple[str, str]]) -> str:
    prompt = "<|begin_of_text|>"
    for role, message in messages:
        if role not in {"system", "user", "assistant"}:
            raise ValueError("invalid Llama role")
        prompt += (
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{message}<|eot_id|>"
        )
    return prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def live_llama_ids() -> dict:
    """Check identifiers only; never trigger expensive Llama inference."""
    base = "http://127.0.0.1:8099"
    with urllib.request.urlopen(base + "/v1/models", timeout=6) as response:
        models = json.load(response)
    ids = [entry["id"] for entry in models["data"]]
    wrong = {
        "model": "minicpm-dense-bf16",
        "messages": [{"role": "user", "content": "Hola"}],
        "max_tokens": 1,
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(wrong).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=6) as response:
            wrong_status = response.status
    except urllib.error.HTTPError as exc:
        wrong_status = exc.code
    native_document = {
        "model": "llama-dense-bf16",
        "messages": [{"role": "user", "content": "Pregunta"}],
        "document": "Texto de referencia",
        "max_tokens": 1,
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(native_document).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=6) as response:
            native_doc_status = response.status
    except urllib.error.HTTPError as exc:
        native_doc_status = exc.code
    assert "llama-dense-bf16" in ids, ids
    assert "minicpm-dense-bf16" not in ids, ids
    assert wrong_status == 404, wrong_status
    assert native_doc_status == 422, native_doc_status
    return {
        "server": base,
        "advertised_model_ids": ids,
        "wrong_minicpm_native_id_http_status": wrong_status,
        "native_document_parameter_http_status": native_doc_status,
        "inference_executed": False,
    }


def main() -> None:
    report: dict = {
        "corpus_scope": "raw text bytes and tokenizer IDs, not model full-volume inference",
        "models": {k: str(v) for k, v in TOKENIZERS.items()},
        "input_files": {},
    }
    source = (ROOT / "src/torch_runtime.rs").read_text(encoding="utf-8")
    match = re.search(r'const DEFAULT_LLAMA_CHAT_SYSTEM: &str = "([^"]+)";', source)
    if not match:
        raise RuntimeError("Cannot locate actual Llama default system template")
    system = match.group(1)
    tokenizers = {name: Tokenizer.from_file(str(path)) for name, path in TOKENIZERS.items()}
    for name, path in FILES.items():
        if not path.is_file():
            report["input_files"][name] = {"available": False, "path": str(path)}
            continue
        original = path.read_bytes()
        raw = original.decode("utf-8")
        entry: dict = {
            "available": True,
            "path": str(path),
            "bytes": len(original),
            "characters": len(raw),
            "lines": raw.count("\n") + 1,
            "sha256_raw": hashlib.sha256(original).hexdigest(),
            "tokenizers": {},
        }
        for model, tokenizer in tokenizers.items():
            encoded = tokenizer.encode(raw, add_special_tokens=False).ids
            entry["tokenizers"][model] = {
                "raw_text_tokens": len(encoded),
                "raw_text_token_ids_sha256": digest_token_ids(encoded),
            }
        llama = tokenizers["Llama-3.1-8B-Instruct"]
        cases = {
            "native_virgin_user_document": [("user", raw)],
            "terrsvusion_system_plus_user_document": [
                ("system", system),
                ("user", raw),
            ],
            "terrsvusion_document_parameter_and_question": [
                ("system", system),
                (
                    "user",
                    "Texto de referencia (única fuente para las siguientes preguntas):"
                    + "\n\n"
                    + raw.strip(),
                ),
                (
                    "assistant",
                    "He leído el texto. Responderé solo con datos que aparezcan en él.",
                ),
                (
                    "user",
                    "Responde con un dato concreto del texto de referencia. "
                    "No repitas la pregunta. Pregunta: "
                    + QUESTION,
                ),
            ],
        }
        entry["llama_chat_templates"] = {}
        for label, messages in cases.items():
            ids = llama.encode(llama_render(messages), add_special_tokens=False).ids
            entry["llama_chat_templates"][label] = {
                "token_count": len(ids),
                "token_ids_sha256": digest_token_ids(ids),
                "explicit_messages": [role for role, _ in messages],
            }
        unique_hashes = {
            value["token_ids_sha256"]
            for value in entry["llama_chat_templates"].values()
        }
        assert len(unique_hashes) == len(cases), "chat input cases conflated"
        report["input_files"][name] = entry
        print(
            f"{name}: {len(original)} bytes, "
            f"Llama raw={entry['tokenizers']['Llama-3.1-8B-Instruct']['raw_text_tokens']}, "
            f"MiniCPM raw={entry['tokenizers']['MiniCPM5-1B']['raw_text_tokens']}",
            flush=True,
        )
    short = report["input_files"].get("faro_short_7k", {})
    full = report["input_files"].get("faro_volume_full", {})
    if short.get("available") and full.get("available"):
        assert short["sha256_raw"] != full["sha256_raw"]
        assert full["bytes"] > 10 * short["bytes"]
        report["faro_7k_is_not_full_volume"] = True
    if "--skip-http" not in sys.argv:
        report["active_llama_model_ids"] = live_llama_ids()
    destination = ROOT / "docs/AUDITORIA_LLAMA_FARO_IDS_Y_PLANTILLAS_20261029.json"
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("AUDIT_SAVED", destination)
    print("FULL_VOLUME_INFERENCE_COMPLETED false")


if __name__ == "__main__":
    main()

