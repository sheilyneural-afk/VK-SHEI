#!/usr/bin/env python3
"""Reproducible retrieval benchmark: mslim-document `routing` vs `fused`.

Ground truth is constructed from the documents themselves, never from the
retrieval code under test:
  * "single":    4 content words sampled from ONE sentence of a page.
  * "scattered": 2 words from one sentence + 2 words from another sentence
                 of the same page (terms spread across the page).
A page is "gold" when its original text contains ALL query words, so a
query whose words also occur together elsewhere is not penalised.

Usage: eval-retrieval.py BIN DOC [DOC ...] [--queries N] [--seed S]
                          [--page-characters C] [--max-pages K]
Prints a JSON report. Synthetic queries measure lexical recall only; they do
not replace a human-labelled relevance set.
"""
import argparse, json, random, re, sys, tempfile, os, statistics

from retrieval_eval_common import run

WORD = re.compile(r"[^\W\d_]{5,}", re.UNICODE)
STOP = set("""
which their there these those about would could should being where while after before
entre sobre desde hasta cuando donde porque también ningún ninguna dentro mismo misma
puede pueden tiene tienen están estos estas otros otras cada todo todos todas
""".split())


def words(text):
    return [w.lower() for w in WORD.findall(text) if w.lower() not in STOP]


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(words(s)) >= 6]


def make_queries(pages, n, rng):
    out = []
    page_numbers = [p for p in pages if len(sentences(pages[p]["original_text"])) >= 2]
    if not page_numbers:
        return out
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        pn = rng.choice(page_numbers)
        sents = sentences(pages[pn]["original_text"])
        kind = "single" if len(out) % 2 == 0 else "scattered"
        if kind == "single":
            ws = list(dict.fromkeys(words(rng.choice(sents))))
            if len(ws) < 4:
                continue
            idx = sorted(rng.sample(range(len(ws)), 4))
            q = [ws[i] for i in idx]
        else:
            a, b = rng.sample(sents, 2)
            wa, wb = list(dict.fromkeys(words(a))), list(dict.fromkeys(words(b)))
            if len(wa) < 2 or len(wb) < 2:
                continue
            q = rng.sample(wa, 2) + rng.sample(wb, 2)
            if len(set(q)) < 4:
                continue
        gold = {
            p for p, pg in pages.items()
            if all(w in set(words(pg["original_text"])) for w in q)
        }
        out.append({"kind": kind, "text": " ".join(q), "gold": sorted(gold)})
    return out


def pages_of(result, mode):
    ev = result["evidence"] if mode == "routing" else result["phase"]["evidence"]
    return [e["page_number"] for e in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--page-characters", type=int, default=2000)
    ap.add_argument("--max-pages", type=int, default=3)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for d, doc in enumerate(a.docs):
            graph = os.path.join(tmp, f"g{d}.json")
            run(a.bin, "build-text", "--document-id", f"doc{d}", "--version", "v1",
                "--text", doc, "--output", graph,
                "--page-characters", str(a.page_characters))
            pages = {int(k): v for k, v in json.load(open(graph))["pages"].items()}
            for q in make_queries(pages, a.queries, rng):
                for mode in ("routing", "fused"):
                    res = run(a.bin, "query", "--graph", graph, "--text", q["text"],
                              "--rank", mode, "--max-pages", str(a.max_pages))
                    got = pages_of(res, mode)
                    rank = next((i + 1 for i, p in enumerate(got) if p in q["gold"]), None)
                    rows.append({"doc": os.path.basename(doc), "kind": q["kind"],
                                 "mode": mode, "returned": len(got),
                                 "pages_total": len(pages), "rank": rank})
    report = {"seed": a.seed, "max_pages": a.max_pages, "queries_per_doc": a.queries,
              "page_characters": a.page_characters, "results": {}}
    for mode in ("routing", "fused"):
        for kind in ("single", "scattered", "all"):
            sel = [r for r in rows if r["mode"] == mode and (kind == "all" or r["kind"] == kind)]
            if not sel:
                continue
            hit = [r for r in sel if r["rank"] is not None]
            report["results"][f"{mode}/{kind}"] = {
                "n": len(sel),
                "hit_rate": round(len(hit) / len(sel), 4),
                "mrr": round(sum(1 / r["rank"] for r in hit) / len(sel), 4),
                "empty_rate": round(sum(1 for r in sel if r["returned"] == 0) / len(sel), 4),
                "mean_pages_returned": round(statistics.mean(r["returned"] for r in sel), 2),
            }
    # paired comparison on identical queries
    by = {}
    for i in range(0, len(rows), 2):
        r, f = rows[i], rows[i + 1]
        by[i] = (r["rank"] is not None, f["rank"] is not None)
    report["paired"] = {
        "fused_only_hits": sum(1 for r, f in by.values() if f and not r),
        "routing_only_hits": sum(1 for r, f in by.values() if r and not f),
        "both": sum(1 for r, f in by.values() if r and f),
        "neither": sum(1 for r, f in by.values() if not r and not f),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
