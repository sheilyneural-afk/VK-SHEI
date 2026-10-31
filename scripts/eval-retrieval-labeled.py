#!/usr/bin/env python3
"""Hand-labelled retrieval benchmark for mslim-document.

Usage: eval-retrieval-labeled.py BIN QUESTIONS.json [--page-characters C]
Gold pages = pages whose original text contains the question's `marker`.
Questions whose marker is missing or straddles a page boundary are skipped
and reported. Compares the variants below on identical questions and prints a
JSON report with hit rate, MRR, paired differences and the misses.
"""
import argparse, json, os, subprocess, tempfile

VARIANTS = {
    "routing": ["--rank", "routing"],
    "fused": ["--rank", "fused"],
    "fused+prf4": ["--rank", "fused", "--expansion-terms", "4"],
    "fused+prf8": ["--rank", "fused", "--expansion-terms", "8"],
}


def run(bin_, *args):
    r = subprocess.run([bin_, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{bin_} {' '.join(args)} failed: {r.stderr[:500]}")
    return json.loads(r.stdout) if r.stdout.lstrip().startswith("{") else r.stdout


def pages_of(res):
    ev = res["evidence"] if "evidence" in res else res["phase"]["evidence"]
    return [e["page_number"] for e in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("questions")
    ap.add_argument("--page-characters", type=int, default=1000)
    a = ap.parse_args()
    spec = json.load(open(a.questions))
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.questions)))
    rows, skipped = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(spec["sets"]):
            doc = os.path.join(base, s["doc"])
            graph = os.path.join(tmp, f"g{i}.json")
            run(a.bin, "build-text", "--document-id", f"d{i}", "--version", "v1",
                "--text", doc, "--output", graph,
                "--page-characters", str(a.page_characters))
            pages = {int(k): v["original_text"]
                     for k, v in json.load(open(graph))["pages"].items()}
            for q in s["questions"]:
                gold = sorted(p for p, t in pages.items() if q["marker"] in t)
                if not gold:
                    skipped.append(q["q"])
                    continue
                for k in (1, 2, 3):
                    for name, flags in VARIANTS.items():
                        res = run(a.bin, "query", "--graph", graph, "--text", q["q"],
                                  *flags, "--max-pages", str(k))
                        got = pages_of(res)
                        rank = next((j + 1 for j, p in enumerate(got) if p in gold), None)
                        rows.append({"q": q["q"], "k": k, "variant": name,
                                     "gold": gold, "got": got, "rank": rank})
    report = {"page_characters": a.page_characters,
              "skipped_marker_not_in_a_page": skipped, "by_k": {}, "paired_vs_routing": {}, "misses_at_3": {}}
    for k in (1, 2, 3):
        report["by_k"][str(k)] = {}
        report["paired_vs_routing"][str(k)] = {}
        base_rows = {r["q"]: r for r in rows if r["k"] == k and r["variant"] == "routing"}
        for name in VARIANTS:
            sel = [r for r in rows if r["k"] == k and r["variant"] == name]
            hit = [r for r in sel if r["rank"] is not None]
            report["by_k"][str(k)][name] = {
                "n": len(sel), "hit_rate": round(len(hit) / len(sel), 4),
                "mrr": round(sum(1 / r["rank"] for r in hit) / len(sel), 4)}
            if name != "routing":
                report["paired_vs_routing"][str(k)][name] = {
                    "gained": sum(1 for r in sel if r["rank"] and not base_rows[r["q"]]["rank"]),
                    "lost": sum(1 for r in sel if not r["rank"] and base_rows[r["q"]]["rank"]),
                }
    for name in VARIANTS:
        report["misses_at_3"][name] = [r["q"] for r in rows
                                       if r["k"] == 3 and r["variant"] == name and r["rank"] is None]
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
