#!/usr/bin/env python
"""Validate raw LLM NER responses and align entity strings back to corpus tokens.

    python scripts/annotation/validate_responses.py \
        --input annotation/data/select-100.jsonl \
        --responses annotation/work/responses/select-100.jsonl \
        --out-dir annotation/data/llm/ner-v1-default --name select-100

An entity is accepted only if its text reproduces a consecutive token subsequence of the
sentence. Search runs forward from the previously accepted entity's end (entities come in
sentence order, which disambiguates repeat mentions); if that fails, any non-overlapping
occurrence is accepted, and an entity that can only land on already-taken tokens is dropped
as `overlap`. Everything dropped is written to <name>.invalid.jsonl with its reason.
"""

import argparse
import json
from pathlib import Path

LABELS = ("PER", "LOC", "ORG", "DAT")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf8")]


def occurrences(tokens, parts):
    """Start indices where `parts` appears as a consecutive token subsequence."""
    n, m = len(tokens), len(parts)
    if m == 0 or m > n:
        return []
    return [i for i in range(n - m + 1) if tokens[i : i + m] == parts]


def align(tokens, entities):
    """-> (accepted spans, dropped [(text, label, reason)])."""
    accepted, dropped, cursor = [], [], 0
    taken = set()
    for ent in entities:
        text = (ent.get("text") or "").strip()
        label = ent.get("label")
        if label not in LABELS:
            dropped.append((text, label, "bad-label"))
            continue
        parts = text.split()
        starts = occurrences(tokens, parts)
        if not starts:
            dropped.append((text, label, "no-align"))
            continue
        free = [s for s in starts if not (taken & set(range(s, s + len(parts))))]
        if not free:
            dropped.append((text, label, "overlap"))
            continue
        forward = [s for s in free if s >= cursor]
        start = forward[0] if forward else free[0]
        end = start + len(parts)
        accepted.append({"start": start, "end": end, "label": label, "text": " ".join(parts)})
        taken |= set(range(start, end))
        cursor = end
    accepted.sort(key=lambda s: s["start"])
    return accepted, dropped


def to_iob(tokens, spans):
    tags = ["O"] * len(tokens)
    for s in spans:
        tags[s["start"]] = f"B-{s['label']}"
        for i in range(s["start"] + 1, s["end"]):
            tags[i] = f"I-{s['label']}"
    return tags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    sents = load_jsonl(args.input)
    responses = load_jsonl(args.responses)

    items, dupes = {}, 0
    model = prompt_hash = None
    for resp in responses:
        model = resp.get("model", model)
        prompt_hash = resp.get("prompt_hash", prompt_hash)
        for item in resp.get("items", []):
            if item["id"] in items:
                dupes += 1
                continue
            items[item["id"]] = item

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"no-align": 0, "bad-label": 0, "overlap": 0}
    n_ent = missing = 0
    rows, iob_blocks, invalid = [], [], []

    for sent in sents:
        item = items.get(sent["id"])
        if item is None:
            missing += 1
            spans = []
        else:
            spans, dropped = align(sent["tokens"], item.get("entities", []))
            for text, label, reason in dropped:
                counts[reason] += 1
                invalid.append({"id": sent["id"], "text": text, "label": label, "reason": reason})
        n_ent += len(spans)
        row = {
            "id": sent["id"],
            "tokens": sent["tokens"],
            "entities": spans,
            "model": model,
            "prompt_hash": prompt_hash,
        }
        if item is None:
            row["error"] = "missing"
        rows.append(row)
        tags = to_iob(sent["tokens"], spans)
        iob_blocks.append("\n".join(f"{t}\t{g}" for t, g in zip(sent["tokens"], tags)))

    with (out_dir / f"{args.name}.jsonl").open("w", encoding="utf8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"{args.name}.iob").write_text("\n\n".join(iob_blocks) + "\n", encoding="utf8")
    with (out_dir / f"{args.name}.invalid.jsonl").open("w", encoding="utf8") as fh:
        for row in invalid:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dropped_total = sum(counts.values())
    print(f"sentences={len(rows)} entities={n_ent} dropped={dropped_total} "
          f"(no-align={counts['no-align']} bad-label={counts['bad-label']} "
          f"overlap={counts['overlap']}) missing={missing} duplicate_items={dupes}")


if __name__ == "__main__":
    main()
