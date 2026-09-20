#!/usr/bin/env python
"""Assemble the LLM annotations into a drop-in replacement for corpus/perdt-ner-iob.

    python scripts/annotation/export_iob.py

Writes two-column IOB2 (`token\\tTAG`, blank line between sentences) in the exact sentence
order and tokenization of the source corpus, so the result can be swapped into project.yml's
`spacy convert` step by changing one path:

    corpus/perdt-ner-iob-llm/{train,dev,test}.txt

Sentence order and token strings are asserted against corpus/perdt-ner-iob, so a partial or
misaligned annotation set fails loudly instead of silently training on shifted labels. The
output lives under corpus/ because it is a build artifact regenerable from the committed
JSONL in annotation/data/llm/.
"""

import argparse
import json
from pathlib import Path

SOURCES = {
    "train": [f"train/shard-{i:03d}" for i in range(14)],
    "dev": ["dev-all"],
    "test": ["test-all"],
}


def iob_sents(path):
    sents, cur = [], []
    for line in path.open(encoding="utf8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        cur.append(parts[0])
    if cur:
        sents.append(cur)
    return sents


def tag(tokens, entities):
    tags = ["O"] * len(tokens)
    for e in entities:
        tags[e["start"]] = f"B-{e['label']}"
        for i in range(e["start"] + 1, e["end"]):
            tags[i] = f"I-{e['label']}"
    return tags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-dir", default="annotation/data/llm/ner-v2.2-default")
    ap.add_argument("--reference", default="corpus/perdt-ner-iob")
    ap.add_argument("--out-dir", default="corpus/perdt-ner-iob-llm")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, parts in SOURCES.items():
        rows = []
        for part in parts:
            path = Path(args.llm_dir) / f"{part}.jsonl"
            rows += [json.loads(line) for line in path.open(encoding="utf8")]

        ref = iob_sents(Path(args.reference) / f"{split}.txt")
        if len(rows) != len(ref):
            raise SystemExit(f"{split}: {len(rows)} annotated sentences against {len(ref)} in corpus")
        for i, (row, tokens) in enumerate(zip(rows, ref)):
            if row["tokens"] != tokens:
                raise SystemExit(f"{split}: token mismatch at sentence {i} ({row['id']})")

        blocks, spans, unannotated = [], 0, 0
        for row in rows:
            spans += len(row["entities"])
            unannotated += 1 if row.get("error") == "missing" else 0
            tags = tag(row["tokens"], row["entities"])
            blocks.append("\n".join(f"{t}\t{g}" for t, g in zip(row["tokens"], tags)))
        # trailing blank line, matching corpus/perdt-ner-iob byte for byte in column 1
        (out_dir / f"{split}.txt").write_text("\n\n".join(blocks) + "\n\n", encoding="utf8")
        print(f"{split}.txt: {len(rows)} sentences, {spans} entities, "
              f"{unannotated} unannotated -> {out_dir / f'{split}.txt'}")


if __name__ == "__main__":
    main()
