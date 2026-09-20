#!/usr/bin/env python
"""Deterministic sentence selection for the LLM NER annotation kit.

Reads the PerDT test split in two-column IOB2 (`corpus/perdt-ner-iob/test.txt`, the
`--merge-subtokens` tokenization) and writes two id-sorted JSONL files:

    annotation/data/pool-500.jsonl    400 entity-rich + 100 entity-free sentences
    annotation/data/select-100.jsonl  80 rich + 20 empty, a subset of the pool

Only PER/LOC/ORG/DAT are kept as silver spans; MON/TIM/PCT are dropped (too thin to
annotate: 205/135/121 train spans). A sentence is "empty" only if it has no silver span
of any label at all, so the 8 MON/TIM/PCT-only sentences land in neither bucket.

Selection is seeded with random.Random(0) and re-runs byte-identically.
"""

import argparse
import json
import random
from pathlib import Path

KEEP = ("PER", "LOC", "ORG", "DAT")
MIN_TOKENS, MAX_TOKENS = 6, 45
# select-100 per-label sentence minimums, applied greedily in this order
MINIMUMS = (("DAT", 20), ("ORG", 25), ("PER", 25), ("LOC", 25))


def iob_sents(path):
    """[(tokens, tags)] — blank line separates sentences, same reader as transfer_perdt_ner."""
    sents, cur = [], []
    for line in path.open(encoding="utf8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        p = line.split("\t")
        if len(p) < 2:
            p = line.split()
        if len(p) < 2:
            continue
        cur.append((p[0], p[-1]))
    if cur:
        sents.append(cur)
    return [([t for t, _ in s], [g for _, g in s]) for s in sents]


def spans_from_iob(tags):
    """[(start, end, label)] over token indices; end is exclusive."""
    spans, i = [], 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            label, j = tag[2:], i + 1
            while j < len(tags) and tags[j] == f"I-{label}":
                j += 1
            spans.append((i, j, label))
            i = j
        else:
            i += 1
    return spans


def record(split, idx, tokens, spans):
    return {
        "id": f"{split}:{idx}",
        "tokens": tokens,
        "text": " ".join(tokens),
        "silver": [
            {"start": s, "end": e, "label": lab, "text": " ".join(tokens[s:e])}
            for s, e, lab in spans
            if lab in KEEP
        ],
    }


def labels_of(rec):
    return {e["label"] for e in rec["silver"]}


def dump(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        for r in sorted(recs, key=lambda r: int(r["id"].split(":")[1])):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(name, recs, n_rich, n_empty):
    print(f"{name}: {len(recs)} sentences ({n_rich} rich, {n_empty} empty)")
    for lab in KEEP:
        sents = sum(1 for r in recs if lab in labels_of(r))
        spans = sum(1 for r in recs for e in r["silver"] if e["label"] == lab)
        print(f"  {lab}: {sents} sentences, {spans} spans")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus/perdt-ner-iob/test.txt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="annotation/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("kit", "split"), default="kit",
                    help="kit: test-all + pool-500 + select-100; "
                         "split: the whole split as shards, for bulk relabelling")
    ap.add_argument("--shard-size", type=int, default=2000,
                    help="sentences per shard in split mode; 0 writes one file")
    args = ap.parse_args()

    sents = iob_sents(Path(args.corpus))
    rich, empty, every = [], [], []
    for idx, (tokens, tags) in enumerate(sents):
        spans = spans_from_iob(tags)
        rec = record(args.split, idx, tokens, spans)
        every.append(rec)
        if not spans:
            empty.append(rec)
        elif rec["silver"] and MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
            rich.append(rec)
    print(f"corpus: {len(sents)} sentences, {len(rich)} rich (after length filter), "
          f"{len(empty)} empty")

    if args.mode == "split":
        out = Path(args.out_dir)
        if args.shard_size:
            shards = [every[i:i + args.shard_size] for i in range(0, len(every), args.shard_size)]
            for n, shard in enumerate(shards):
                dump(out / f"shard-{n:03d}.jsonl", shard)
            where = f"{len(shards)} shards of <= {args.shard_size} sentences in {out}"
        else:
            dump(out / f"{args.split}-all.jsonl", every)
            where = f"one file, {out}/{args.split}-all.jsonl"
        n_any = sum(1 for r in every if r["silver"])
        report(args.split, every, n_any, len(every) - n_any)
        print(where)
        return

    rng = random.Random(args.seed)

    # --- pool: 400 rich, seeded with every DAT and every ORG sentence, then a shuffled fill
    chosen = {r["id"] for r in rich if "DAT" in labels_of(r)}
    chosen |= {r["id"] for r in rich if "ORG" in labels_of(r)}
    pool_rich = [r for r in rich if r["id"] in chosen]
    if len(pool_rich) > 400:
        raise SystemExit(f"DAT+ORG seed is {len(pool_rich)} sentences, over the 400 budget")
    remainder = [r for r in rich if r["id"] not in chosen]
    rng.shuffle(remainder)
    pool_rich += remainder[: 400 - len(pool_rich)]

    pool_empty = list(empty)
    rng.shuffle(pool_empty)
    pool_empty = pool_empty[:100]
    pool = pool_rich + pool_empty

    # --- select-100: 80 rich meeting per-label minimums, 20 empty
    order = list(pool_rich)
    rng.shuffle(order)
    picked, picked_ids = [], set()

    def take(rec):
        picked.append(rec)
        picked_ids.add(rec["id"])

    for label, minimum in MINIMUMS:
        have = sum(1 for r in picked if label in labels_of(r))
        for r in order:
            if have >= minimum:
                break
            if r["id"] in picked_ids or label not in labels_of(r):
                continue
            take(r)
            have += 1
        if have < minimum:
            raise SystemExit(f"cannot reach {label} >= {minimum}: only {have} available")
    for r in order:
        if len(picked) >= 80:
            break
        if r["id"] not in picked_ids:
            take(r)

    sel_empty = list(pool_empty)
    rng.shuffle(sel_empty)
    sel_empty = sel_empty[:20]
    select = picked + sel_empty

    out = Path(args.out_dir)
    dump(out / "test-all.jsonl", every)
    dump(out / "pool-500.jsonl", pool)
    dump(out / "select-100.jsonl", select)
    n_any = sum(1 for r in every if r["silver"])
    report("test-all", every, n_any, len(every) - n_any)
    report("pool-500", pool, len(pool_rich), len(pool_empty))
    report("select-100", select, len(picked), len(sel_empty))


if __name__ == "__main__":
    main()
