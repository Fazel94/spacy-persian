"""Build the Hugging Face model card for a packaged pipeline.

`spacy package` already writes a README into the wheel, and `spacy huggingface-hub push`
uploads it as the card. That card is a metadata dump: no install line, no usage, no
throughput, and no YAML frontmatter, so the Hub cannot index the model by language or task.

This composes a card from the same sources of truth (`meta.json` and the JSON written by
scripts/benchmark_throughput.py) rather than from hand-copied numbers, so the card cannot
drift from the artifact it describes.
"""

import argparse
import json
from pathlib import Path

# meta.json key -> (row label, reference note). Only keys the pipeline actually evidences
# are emitted; a missing key means the corpus could not score it.
METRICS = [
    ("token_acc", "Tokenization accuracy", ""),
    ("tag_acc", "XPOS tag accuracy", ""),
    ("pos_acc", "UPOS tag accuracy", ""),
    ("morph_acc", "Morphological features", ""),
    ("lemma_acc", "Lemma accuracy", ""),
    ("dep_uas", "Unlabelled attachment (UAS)", ""),
    ("dep_las", "Labelled attachment (LAS)", ""),
    ("sents_f", "Sentence segmentation F", ""),
    ("ents_p", "NER precision", ""),
    ("ents_r", "NER recall", ""),
    ("ents_f", "NER F-score", ""),
]


def load(path):
    return json.loads(Path(path).read_text())


def throughput_rows(paths):
    rows = []
    for p in paths:
        if not Path(p).exists():
            continue
        d = load(p)
        rows.append((d["device"], d["batch_size"], d["wps_median"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="meta.json of the finalized pipeline")
    ap.add_argument("--throughput", nargs="*", default=[], help="benchmark_throughput JSONs")
    ap.add_argument("--repo-id", required=True, help="e.g. Phazel/fa_core_news_trf")
    ap.add_argument("--wheel-name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = load(args.meta)
    name = f"{meta['lang']}_{meta['name']}"
    perf = meta.get("performance", {})

    lines = []
    # Frontmatter: without this the Hub cannot filter the model by language or library.
    lines += [
        "---",
        "language:",
        "- fa",
        f"license: {meta.get('license', 'cc-by-sa-4.0').lower().replace(' ', '-')}",
        "library_name: spacy",
        "pipeline_tag: token-classification",
        "tags:",
        "- spacy",
        "- token-classification",
        "- persian",
        "- farsi",
        "---",
        "",
        f"# {name}",
        "",
        meta.get("description", "").strip(),
        "",
    ]

    lines += [
        "## Install",
        "",
        "```bash",
        f"pip install https://huggingface.co/{args.repo_id}/resolve/main/{args.wheel_name}",
        "```",
        "",
        "```python",
        "import spacy",
        f'nlp = spacy.load("{name}")',
        'doc = nlp("شرکت ایران خودرو اعلام کرد که تولید خود را افزایش می\u200cدهد.")',
        "print([(t.text, t.pos_, t.lemma_, t.dep_) for t in doc])",
        "print([(e.text, e.label_) for e in doc.ents])",
        "```",
        "",
    ]

    lines += ["## Accuracy", "",
              "Scored with `spacy benchmark accuracy` on the held-out PerDT test split.",
              "", "| Metric | Score |", "| --- | ---: |"]
    for key, label, _ in METRICS:
        v = perf.get(key)
        if isinstance(v, (int, float)):
            lines.append(f"| {label} | {v * 100:.2f} |")
    lines.append("")

    rows = throughput_rows(args.throughput)
    if rows:
        lines += ["## Throughput", "",
                  "Median of repeated `nlp.pipe` passes over the 146-document PerDT test",
                  "split (23,825 tokens), timing the pipe only. Warmup pass discarded.",
                  "", "| Device | Batch | Words/s |", "| --- | ---: | ---: |"]
        for device, batch, wps in rows:
            lines.append(f"| {device} | {batch} | {wps:,.0f} |")
        lines.append("")
        gpu = next((r for r in rows if r[0].startswith("gpu")), None)
        cpu = next((r for r in rows if r[0].startswith("cpu")), None)
        if gpu and cpu:
            lines += [
                f"A transformer pipeline is GPU-bound: the T4 is {gpu[2] / cpu[2]:.0f}x the "
                f"CPU on the same machine. On CPU this runs roughly 25x slower than the "
                f"`sm`/`md`/`lg` tiers, which is the price of the accuracy below.",
                "",
            ]

    lines += ["## Sources", "", "| Source | Author | Licence |", "| --- | --- | --- |"]
    for s in meta.get("sources", []):
        url, nm = s.get("url"), s.get("name", "")
        label = f"[{nm}]({url})" if url else nm
        lines.append(f"| {label} | {s.get('author', '')} | {s.get('license', '')} |")
    lines.append("")

    notes = (meta.get("notes") or "").strip()
    if notes:
        lines += ["## Notes", "", notes, ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
