#!/usr/bin/env python
"""Build batched annotation requests for the LLM NER kit.

Renders the system prompt by substituting `{{GUIDELINES}}` in the prompt template with the
full text of GUIDELINES.md, so the guideline is the single source of truth: editing it
changes `prompt_hash` and therefore every `cache_key`, invalidating stale responses.

    python scripts/annotation/build_requests.py \
        --input annotation/data/select-100.jsonl \
        --prompt annotation/prompts/ner-v1.md \
        --batch 10 --out annotation/work/requests/select-100.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

INSTRUCTION = "Annotate every sentence. Return one item per id."

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string", "enum": ["PER", "LOC", "ORG", "DAT"]},
                            },
                            "required": ["text", "label"],
                        },
                    },
                },
                "required": ["id", "entities"],
            },
        }
    },
    "required": ["items"],
}


def sha256(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def render_prompt(prompt_path, guidelines_path):
    template = prompt_path.read_text(encoding="utf8")
    if "{{GUIDELINES}}" not in template:
        raise SystemExit(f"{prompt_path} has no {{{{GUIDELINES}}}} placeholder")
    return template.replace("{{GUIDELINES}}", guidelines_path.read_text(encoding="utf8").strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--prompt", default="annotation/prompts/ner-v2.1.md")
    ap.add_argument("--guidelines", default="annotation/GUIDELINES.md")
    # 20 measured on pool-500 against batch 10, same prompt: 25/25 batches returned every
    # id with no retry, micro F vs silver 0.7778 against 0.7789, and the two runs agree at
    # F 0.976 — tighter than run-to-run noise at a fixed batch size (0.949-0.951).
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--model", default="default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    system = render_prompt(Path(args.prompt), Path(args.guidelines))
    prompt_hash = sha256(system)[:12]

    sents = [json.loads(line) for line in Path(args.input).open(encoding="utf8")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name[: -len(".jsonl")] if out.name.endswith(".jsonl") else out.name

    n = 0
    with out.open("w", encoding="utf8") as fh:
        for i in range(0, len(sents), args.batch):
            chunk = sents[i : i + args.batch]
            pairs = [{"id": s["id"], "text": s["text"]} for s in chunk]
            lines = [f"{p['id']}: {p['text']}" for p in pairs]
            body = INSTRUCTION + "\n\n" + "\n".join(lines)
            req = {
                "batch": f"{name}/{i // args.batch:03d}",
                "model": args.model,
                "ids": [p["id"] for p in pairs],
                "sentences": pairs,
                "system": system,
                "body": body,
                "schema": SCHEMA,
                "prompt_hash": prompt_hash,
                "cache_key": sha256(prompt_hash + args.model + "\n".join(lines)),
            }
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")
            n += 1

    print(f"requests={n} sentences={len(sents)} batch={args.batch} model={args.model} "
          f"prompt_hash={prompt_hash} -> {out}")


if __name__ == "__main__":
    main()
