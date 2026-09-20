#!/usr/bin/env python
"""Score the silver layer and the LLM annotations against human gold.

    python scripts/annotation/score_gold.py --gold annotation/human/gold-200.iob

Reads the completed worksheet (the `O` column edited by hand), joins it to
gold-200.key.jsonl, and reports precision/recall/F for BOTH annotators against the human,
per label and overall.

Two things this does that a naive script would get wrong:

1. **Reweighting.** The sample is stratified with deliberately unequal rates — 30 of 916
   agree-empty sentences, 120 of 229 disagreeing ones. Raw sample counts therefore describe
   the sample, not the split. Each sentence carries weight `stratum_size / stratum_sampled`,
   and the reweighted figures are the ones that generalise to the 1,455-sentence test split.
   Both are printed; they differ a lot, and the raw numbers are the misleading ones.
2. **`--judge`.** Given the adjudication dump that produced the ~0.94 precision estimate,
   it also scores the LLM judge's verdicts against the human, which is the only way to learn
   whether the judge's bias toward its own family was real and how large it was.

With `--second <file>` it reports inter-annotator agreement between two human worksheets
instead, which is the number that says whether the guideline is teachable at all.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

LABELS = ("PER", "LOC", "ORG", "DAT")


def read_worksheet(path, column=2):
    """-> {id: [(start, end, label)]}, driven by the `# <id>` header before each block.

    `column` is 1-based: 2 for the two-column worksheets, 4 for the verdict column of
    gold-200.review.tsv (token, silver, llm, verdict).
    """
    out, sid, tags = {}, None, []
    for line in Path(path).open(encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("# "):
            sid, tags = line[2:].strip(), []
            continue
        if not line.strip():
            if sid is not None and tags:
                out[sid] = spans_from_iob(tags)
                sid, tags = None, []
            continue
        parts = line.split("\t")
        if len(parts) < column:
            raise SystemExit(f"{path}: line {line!r} has {len(parts)} columns, need {column}")
        tags.append(parts[column - 1].strip())
    if sid is not None and tags:
        out[sid] = spans_from_iob(tags)
    return out


def spans_from_iob(tags):
    spans, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            label, j = tags[i][2:], i + 1
            while j < len(tags) and tags[j] == f"I-{label}":
                j += 1
            spans.append((i, j, label))
            i = j
        else:
            if tags[i] not in ("O", "") and not tags[i].startswith("I-"):
                raise SystemExit(f"unknown tag {tags[i]!r}")
            i += 1
    return spans


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def report(title, counts):
    print(f"\n== {title}")
    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    p, r, f = prf(tp, fp, fn)
    print(f"   micro p={p:.3f} r={r:.3f} f={f:.3f}  (tp={tp:.1f} fp={fp:.1f} fn={fn:.1f})")
    for lab in LABELS:
        t, fp_, fn_ = counts[lab]
        if t + fp_ + fn_ == 0:
            continue
        p, r, f = prf(t, fp_, fn_)
        print(f"   {lab}: p={p:.3f} r={r:.3f} f={f:.3f}")


def score(gold, systems, weights):
    """-> {system: (raw counts, weighted counts)} keyed by label."""
    out = {}
    for name, sysspans in systems.items():
        raw = defaultdict(lambda: [0.0, 0.0, 0.0])
        wtd = defaultdict(lambda: [0.0, 0.0, 0.0])
        for sid, g in gold.items():
            h, w = set(sysspans.get(sid, [])), weights[sid]
            g = set(g)
            for s in h & g:
                raw[s[2]][0] += 1; wtd[s[2]][0] += w
            for s in h - g:
                raw[s[2]][1] += 1; wtd[s[2]][1] += w
            for s in g - h:
                raw[s[2]][2] += 1; wtd[s[2]][2] += w
        out[name] = (raw, wtd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="annotation/human/gold-200.iob")
    ap.add_argument("--key", default="annotation/human/gold-200.key.jsonl")
    ap.add_argument("--second", help="a second annotator's worksheet, for IAA")
    ap.add_argument("--judge", help="adjudication dump with per-span verdicts")
    ap.add_argument("--column", type=int, default=2,
                    help="1-based tag column; use 4 for a corrected gold-200.review.tsv")
    args = ap.parse_args()

    key = {json.loads(l)["id"]: json.loads(l) for l in Path(args.key).open(encoding="utf8")}
    gold = read_worksheet(args.gold, args.column)
    missing = [sid for sid in key if sid not in gold]
    if missing:
        raise SystemExit(f"{len(missing)} sentences are unannotated, e.g. {missing[:3]}")

    if args.second:
        other = read_worksheet(args.second, args.column)
        shared = sorted(set(gold) & set(other))
        counts = defaultdict(lambda: [0.0, 0.0, 0.0])
        same = 0
        for sid in shared:
            a, b = set(gold[sid]), set(other[sid])
            same += a == b
            for s in a & b:
                counts[s[2]][0] += 1
            for s in b - a:
                counts[s[2]][1] += 1
            for s in a - b:
                counts[s[2]][2] += 1
        report(f"inter-annotator agreement over {len(shared)} shared sentences", counts)
        print(f"   {same}/{len(shared)} sentences annotated identically")
        return

    weights = {sid: k["stratum_size"] / k["stratum_sampled"] for sid, k in key.items()}
    systems = {
        "silver (PerDT)": {sid: [(s["start"], s["end"], s["label"]) for s in k["silver"]]
                           for sid, k in key.items()},
        "LLM (v2.2)": {sid: [(e["start"], e["end"], e["label"]) for e in k["llm"]]
                       for sid, k in key.items()},
    }
    n_gold = sum(len(v) for v in gold.values())
    print(f"human gold: {len(gold)} sentences, {n_gold} spans")
    for name, (raw, wtd) in score(gold, systems, weights).items():
        report(f"{name} against human gold — raw sample counts", raw)
        report(f"{name} against human gold — reweighted to the test split", wtd)

    if args.judge:
        verdicts = json.load(Path(args.judge).open(encoding="utf8"))
        agree = total = 0
        wrong = []
        for v in verdicts:
            sid, kind = v["id"], v["kind"]
            if sid not in gold or kind not in ("fp", "fn") or v["verdict"] not in ("correct", "incorrect"):
                continue
            tokens = key[sid]["tokens"]
            human = {(" ".join(tokens[s[0]:s[1]]), s[2]) for s in gold[sid]}
            # fp: a span only the LLM has. fn: a span only the silver layer has. Either way
            # "correct" claims the span belongs, and the human worksheet settles it.
            in_gold = (v["span"], v["label"]) in human
            judged_correct = v["verdict"] == "correct"
            total += 1
            if in_gold == judged_correct:
                agree += 1
            else:
                wrong.append(f"{sid} [{v['span']}]({v['label']}) {kind}: judge said "
                             f"{v['verdict']}, human {'has' if in_gold else 'lacks'} it")
        if not total:
            print("\n== judge review: no adjudicated spans fall inside the gold sample")
            return
        print(f"\n== judge review: {agree}/{total} verdicts match the human ({agree/total:.1%})")
        print("   Every precision estimate quoted for this dataset rests on these verdicts;")
        print("   this percentage is how much that estimate can be trusted.")
        for line in wrong[:20]:
            print(f"   MISJUDGED {line}")


if __name__ == "__main__":
    main()
