#!/usr/bin/env python
"""Draw a stratified, blind 200-sentence sample for human gold annotation.

    python scripts/annotation/sample_gold.py

Nothing in this repo has been measured against a human. Every quality claim — the LLM
annotations at ~0.94 adjudicated precision, the silver layer at ~0.84, the 16% silver error
rate — rests on an LLM judge from the annotator's own model family, and recall is unmeasured
for both sides because entities *both* miss are invisible to a pairwise comparison. This
sample is what closes that gap.

Stratification, over the test split (1,455 sentences), because that is where every published
score is computed:

    agree-empty   silver and the LLM both find nothing   -> the only stratum that can expose
                                                            entities both annotators miss
    agree-spans   both find exactly the same spans       -> measures the agreed mass, which
                                                            a disagreement-only sample would
                                                            wrongly assume correct
    disagree      any difference at all                  -> the contested spans, oversampled

Strata are sampled at different rates on purpose, so estimates MUST be reweighted back to
the split; score_gold.py does this and prints both the raw and reweighted figures.

Writes, under annotation/human/:
    gold-200.iob        blind worksheet, every tag pre-filled O, this is what a human edits
    gold-200.jsonl      the same sentences as JSON, ids and tokens only
    gold-200.key.jsonl  silver spans, LLM spans and stratum — DO NOT OPEN before annotating
"""

import argparse
import json
import random
from pathlib import Path

STRATA = (("agree-empty", 30), ("agree-spans", 50), ("disagree", 120))


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in Path(path).open(encoding="utf8")}


def spans_of(entities):
    return {(e["start"], e["end"], e["label"]) for e in entities}


def to_iob(tokens, entities):
    tags = ["O"] * len(tokens)
    for e in entities:
        tags[e["start"]] = f"B-{e['label']}"
        for i in range(e["start"] + 1, e["end"]):
            tags[i] = f"I-{e['label']}"
    return tags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="annotation/data/test-all.jsonl")
    ap.add_argument("--llm", default="annotation/data/llm/ner-v2.2-default/test-all.jsonl")
    ap.add_argument("--out-dir", default="annotation/human")
    ap.add_argument("--name", default="gold-200")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ref, llm = load(args.ref), load(args.llm)
    pools = {name: [] for name, _ in STRATA}
    for sid in sorted(ref, key=lambda s: int(s.split(":")[1])):
        silver = spans_of(ref[sid]["silver"])
        hyp = spans_of(llm[sid]["entities"])
        if silver == hyp:
            pools["agree-empty" if not silver else "agree-spans"].append(sid)
        else:
            pools["disagree"].append(sid)

    rng = random.Random(args.seed)
    picked = []
    for name, n in STRATA:
        pool = pools[name]
        if len(pool) < n:
            raise SystemExit(f"stratum {name} has {len(pool)} sentences, need {n}")
        picked += [(sid, name, len(pool)) for sid in rng.sample(pool, n)]
    rng.shuffle(picked)  # so the annotator cannot read strata off the ordering

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks, blind, key = [], [], []
    silver_blocks, llm_blocks, review_blocks = [], [], []
    n_marked = 0
    for sid, stratum, pool_size in picked:
        tokens = ref[sid]["tokens"]
        s_tags = to_iob(tokens, ref[sid]["silver"])
        l_tags = to_iob(tokens, llm[sid]["entities"])
        head = f"# {sid}"
        blocks.append(head + "\n" + "\n".join(f"{t}\tO" for t in tokens))
        silver_blocks.append(head + "\n" + "\n".join(f"{t}\t{g}" for t, g in zip(tokens, s_tags)))
        llm_blocks.append(head + "\n" + "\n".join(f"{t}\t{g}" for t, g in zip(tokens, l_tags)))
        # Review sheet: both annotators side by side, a verdict column seeded with the LLM
        # tag, and a marker on every token where they differ so the eye goes straight there.
        rows = []
        for t, s, l in zip(tokens, s_tags, l_tags):
            mark = "" if s == l else "\t*"
            n_marked += s != l
            rows.append(f"{t}\t{s}\t{l}\t{l}{mark}")
        review_blocks.append(head + "\n" + "\n".join(rows))
        blind.append({"id": sid, "tokens": tokens, "text": ref[sid]["text"], "entities": []})
        key.append({
            "id": sid,
            "stratum": stratum,
            "stratum_size": pool_size,
            "stratum_sampled": dict(STRATA)[stratum],
            "tokens": tokens,
            "silver": ref[sid]["silver"],
            "llm": llm[sid]["entities"],
        })
    for fname, rows in ((f"{args.name}.jsonl", blind), (f"{args.name}.key.jsonl", key)):
        with (out / fname).open("w", encoding="utf8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for fname, bs in ((f"{args.name}.iob", blocks),
                      (f"{args.name}.silver.iob", silver_blocks),
                      (f"{args.name}.llm.iob", llm_blocks),
                      (f"{args.name}.review.tsv", review_blocks)):
        (out / fname).write_text("\n\n".join(bs) + "\n\n", encoding="utf8")

    tokens = sum(len(ref[sid]["tokens"]) for sid, _, _ in picked)
    print(f"{len(picked)} sentences, {tokens} tokens")
    for name, n in STRATA:
        print(f"  {name}: {n} sampled of {len(pools[name])} in the split "
              f"(weight {len(pools[name]) / n:.2f})")
    print(f"  {out}/{args.name}.iob         blind worksheet, all tags O")
    print(f"  {out}/{args.name}.silver.iob  pre-filled with the silver layer, for review")
    print(f"  {out}/{args.name}.llm.iob     pre-filled with the LLM annotation, for review")
    print(f"  {out}/{args.name}.review.tsv  token, silver, llm, verdict; {n_marked} tokens "
          f"marked * where the two differ")


if __name__ == "__main__":
    main()
