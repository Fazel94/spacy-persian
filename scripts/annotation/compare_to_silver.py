#!/usr/bin/env python
"""Compare token-aligned LLM annotations against the PerDT silver layer.

    python scripts/annotation/compare_to_silver.py \
        --llm annotation/data/llm/ner-v1-default/select-100.jsonl \
        --ref annotation/data/select-100.jsonl \
        --out annotation/data/llm/ner-v1-default/select-100.agreement.json

Exact span match (start, end, label); silver is the reference, so "precision" reads as
"share of LLM spans the silver layer also has". Silver is not gold — the disagreement list
is the human review queue, not an error list. `expected_corrections` counts silver spans
that end on an opening parenthesis, the tokenization artefact GUIDELINES.md rule 10 tells
the annotator to correct, so those disagreements are not read as LLM mistakes.

With --self <other llm jsonl>, also reports span agreement between two LLM runs over their
overlapping ids (a run-to-run consistency sample).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

LABELS = ("PER", "LOC", "ORG", "DAT")


def load(path):
    return {json.loads(line)["id"]: json.loads(line) for line in Path(path).open(encoding="utf8")}


def span_set(spans):
    return {(s["start"], s["end"], s["label"]) for s in spans}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "p": round(p, 4), "r": round(r, 4), "f": round(f, 4)}


def score(pairs):
    """pairs: [(ref_spans, hyp_spans)] -> {label: prf, micro: prf}."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for ref, hyp in pairs:
        for s in hyp & ref:
            tp[s[2]] += 1
        for s in hyp - ref:
            fp[s[2]] += 1
        for s in ref - hyp:
            fn[s[2]] += 1
    out = {lab: prf(tp[lab], fp[lab], fn[lab]) for lab in LABELS}
    out["micro"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return out


def render(tokens, spans):
    return [f"[{' '.join(tokens[s['start']:s['end']])}]({s['label']})"
            for s in sorted(spans, key=lambda s: s["start"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--self", dest="self_path", default=None)
    args = ap.parse_args()

    llm, ref = load(args.llm), load(args.ref)
    missing_ids = sorted(set(ref) - set(llm), key=lambda i: int(i.split(":")[1]))

    pairs, disagreements, expected = [], [], 0
    for sid, r in sorted(ref.items(), key=lambda kv: int(kv[0].split(":")[1])):
        h = llm.get(sid)
        if h is None:
            continue
        tokens = r["tokens"]
        rs, hs = span_set(r["silver"]), span_set(h["entities"])
        pairs.append((rs, hs))
        artefacts = [s for s in r["silver"] if tokens[s["end"] - 1] in ("(", ")")]
        expected += len(artefacts)
        if rs != hs:
            disagreements.append({
                "id": sid,
                "text": r["text"],
                "silver": render(tokens, r["silver"]),
                "llm": render(tokens, h["entities"]),
                "silver_paren_artefact": bool(artefacts),
            })

    result = {
        "llm": args.llm,
        "ref": args.ref,
        "sentences": len(pairs),
        "missing_ids": missing_ids,
        "scores": score(pairs),
        "expected_corrections": {
            "rule": "10 (silver span ending on a parenthesis token)",
            "spans": expected,
            "sentences": sum(1 for d in disagreements if d["silver_paren_artefact"]),
        },
        "disagreements": disagreements,
    }

    if args.self_path:
        other = load(args.self_path)
        shared = sorted(set(other) & set(llm), key=lambda i: int(i.split(":")[1]))
        sp = [(span_set(other[i]["entities"]), span_set(llm[i]["entities"])) for i in shared]
        result["self_agreement"] = {
            "against": args.self_path,
            "sentences": len(shared),
            "identical_sentences": sum(1 for a, b in sp if a == b),
            "scores": score(sp),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")

    m = result["scores"]["micro"]
    print(f"sentences={result['sentences']} micro p={m['p']} r={m['r']} f={m['f']} "
          f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    for lab in LABELS:
        s = result["scores"][lab]
        print(f"  {lab}: p={s['p']} r={s['r']} f={s['f']} (tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    print(f"disagreeing sentences={len(disagreements)} "
          f"expected_corrections={result['expected_corrections']['spans']} spans -> {args.out}")
    if "self_agreement" in result:
        sa = result["self_agreement"]
        print(f"self_agreement: {sa['identical_sentences']}/{sa['sentences']} sentences identical, "
              f"micro f={sa['scores']['micro']['f']}")


if __name__ == "__main__":
    main()
