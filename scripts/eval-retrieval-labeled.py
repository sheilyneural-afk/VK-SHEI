#!/usr/bin/env python3
"""Hand-labelled retrieval benchmark for mslim-document (routing vs fused).

Usage: eval-retrieval-labeled.py BIN QUESTIONS.json [--page-characters C]
Gold pages = pages whose original text contains the question's `marker`.
Questions whose marker is missing or straddles a page boundary are skipped
and reported. Prints a JSON report with per-question outcomes.
"""
import argparse, json, os, subprocess, tempfile


def run(bin_, *args):
    r = subprocess.run([bin_, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{bin_} {' '.join(args)} failed: {r.stderr[:500]}")
    return json.loads(r.stdout) if r.stdout.lstrip().startswith("{") else r.stdout


def pages_of(res, mode):
    ev = res["evidence"] if mode == "routing" else res["phase"]["evidence"]
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
                    for mode in ("routing", "fused"):
                        res = run(a.bin, "query", "--graph", graph, "--text", q["q"],
                                  "--rank", mode, "--max-pages", str(k))
                        got = pages_of(res, mode)
                        rank = next((j + 1 for j, p in enumerate(got) if p in gold), None)
                        rows.append({"doc": s["doc"], "q": q["q"], "k": k, "mode": mode,
                                     "gold": gold, "got": got, "rank": rank})
    report = {"page_characters": a.page_characters, "skipped_marker_not_in_a_page": skipped,
              "by_k": {}, "paired": {}, "misses": []}
    for k in (1, 2, 3):
        rep = {}
        for mode in ("routing", "fused"):
            sel = [r for r in rows if r["k"] == k and r["mode"] == mode]
            hit = [r for r in sel if r["rank"] is not None]
            rep[mode] = {"n": len(sel), "hit_rate": round(len(hit) / len(sel), 4),
                         "mrr": round(sum(1 / r["rank"] for r in hit) / len(sel), 4)}
        report["by_k"][str(k)] = rep
        pairs = [(r, f) for r, f in zip(
            [x for x in rows if x["k"] == k and x["mode"] == "routing"],
            [x for x in rows if x["k"] == k and x["mode"] == "fused"])]
        report["paired"][str(k)] = {
            "fused_only": sum(1 for r, f in pairs if f["rank"] and not r["rank"]),
            "routing_only": sum(1 for r, f in pairs if r["rank"] and not f["rank"]),
            "both": sum(1 for r, f in pairs if r["rank"] and f["rank"]),
            "neither": sum(1 for r, f in pairs if not r["rank"] and not f["rank"]),
        }
    for r in rows:
        if r["k"] == 3 and r["rank"] is None:
            report["misses"].append({"mode": r["mode"], "q": r["q"], "gold": r["gold"], "got": r["got"]})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
