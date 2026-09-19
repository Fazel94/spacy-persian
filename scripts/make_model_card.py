"""Build the Hugging Face model cards for every published package.

`spacy package` already writes a README into the wheel, and `spacy huggingface-hub push`
uploads it as the card. That card is a metadata dump: no install line, no usage, no
throughput, and no YAML frontmatter, so the Hub cannot index the model by language or task.

Cards are composed from the same sources of truth as the artifacts they describe
(`meta.json`, the wheel on disk, and the JSON written by scripts/benchmark_throughput.py)
rather than from hand-copied numbers, so a card cannot drift from its artifact.

Two kinds of package are handled:

* pipelines (`meta["pipeline"]` non-empty) get usage, accuracy and throughput;
* vectors-only packages (`pipeline: []`) get the vector table, its training corpus and
  hyperparameters, and a `--paths.vectors` snippet instead.

Every card carries the same cross-link table, built from the catalogue below, so a reader
landing on one package can find the other tiers.

    python scripts/make_model_card.py --all --out-dir cards
    python scripts/make_model_card.py --package fa_core_news_trf --out cards/trf.md
"""

import argparse
import json
import zipfile
from pathlib import Path

REPO_OWNER = "Phazel"
PROJECT_URL = "https://github.com/Fazel94/spacy-persian"

# meta.json key -> row label. Only keys the pipeline actually evidences are emitted; a
# missing key means the corpus could not score it.
METRICS = [
    ("token_acc", "Tokenization accuracy"),
    ("tag_acc", "XPOS tag accuracy"),
    ("pos_acc", "UPOS tag accuracy"),
    ("morph_acc", "Morphological features"),
    ("lemma_acc", "Lemma accuracy"),
    ("dep_uas", "Unlabelled attachment (UAS)"),
    ("dep_las", "Labelled attachment (LAS)"),
    ("sents_f", "Sentence segmentation F"),
    ("ents_p", "NER precision"),
    ("ents_r", "NER recall"),
    ("ents_f", "NER F-score"),
]

# One entry per published package. `hf` differs from `pkg` only where the Hub repo was
# created under a different name (fa-floret-wiki-vectors), so install URLs must be built
# from `hf` and never assumed equal to the package name.
CATALOG = [
    {"pkg": "fa_dep_news_sm", "tier": "sm", "dir": "packages/fa_dep_news_sm-3.8.0"},
    {"pkg": "fa_core_news_sm", "tier": "sm", "dir": "packages/fa_core_news_sm-3.8.0",
     "throughput": ["metrics/throughput-sm-cpu.json", "metrics/throughput-sm-gpu.json"]},
    {"pkg": "fa_ent_news_sm", "tier": "sm", "dir": "packages/fa_ent_news_sm-3.8.0",
     "throughput": ["metrics/throughput-fa_ent_news_sm-cpu.json",
                    "metrics/throughput-fa_ent_news_sm-gpu.json"]},
    {"pkg": "fa_dep_news_md", "tier": "md", "dir": "packages/fa_dep_news_md-3.8.0",
     "throughput": ["metrics/throughput-fa_dep_news_md-cpu.json",
                    "metrics/throughput-fa_dep_news_md-gpu.json"]},
    {"pkg": "fa_core_news_md", "tier": "md", "dir": "packages/fa_core_news_md-3.8.0",
     "throughput": ["metrics/throughput-md-cpu.json", "metrics/throughput-md-gpu.json"]},
    {"pkg": "fa_ent_news_md", "tier": "md", "dir": "packages/fa_ent_news_md-3.8.0"},
    {"pkg": "fa_dep_news_lg", "tier": "lg", "dir": "packages/fa_dep_news_lg-3.8.0"},
    {"pkg": "fa_core_news_lg", "tier": "lg", "dir": "packages/fa_core_news_lg-3.8.0",
     "throughput": ["metrics/throughput-lg-cpu.json", "metrics/throughput-lg-gpu.json"]},
    {"pkg": "fa_ent_news_lg", "tier": "lg", "dir": "packages/fa_ent_news_lg-3.8.0"},
    {"pkg": "fa_core_news_trf", "tier": "trf", "dir": "packages/fa_core_news_trf-1.0.0",
     "throughput": ["metrics/throughput-trf-cpu.json", "metrics/throughput-trf-940mx.json",
                    "metrics/throughput-trf-colab-cpu.json",
                    "metrics/throughput-trf-t4-gpu.json"]},
    {"pkg": "fa_floret_400k", "tier": "md", "dir": "packages_hf/fa_floret_400k-0.1.0"},
    {"pkg": "fa_floret_full_wiki", "tier": "-",
     "dir": "packages_hf/fa_floret_full_wiki-0.1.0"},
    {"pkg": "fa_floret_wiki_200k", "tier": "lg", "hf": "fa-floret-wiki-vectors",
     "dir": "packages_hf/fa_floret_wiki_200k-0.1.0"},
]

TIER_BLURB = {
    "sm": "hash embeddings, no vectors",
    "md": "50k x 300d floret vectors",
    "lg": "200k x 300d floret vectors",
    "trf": "fine-tuned transformer, GPU recommended",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wheel_path(entry):
    """The versioned wheel. `spacy huggingface-hub push` also leaves a
    `<name>-any-py3-none-any.whl` copy on the Hub that pip rejects as an invalid version,
    so install lines must point at this filename instead."""
    dist = Path(entry["dir"]) / "dist"
    wheels = [w for w in sorted(dist.glob("*.whl")) if "-any-py3" not in w.name]
    if not wheels:
        raise SystemExit(f"no versioned wheel under {dist}")
    return wheels[-1]


def read_meta(entry):
    """Prefer the meta beside the package; fall back to the copy inside the wheel, which is
    the only one present for packages built elsewhere (trf, on a rented GPU)."""
    beside = Path(entry["dir"]) / "meta.json"
    if beside.exists():
        return load(beside)
    with zipfile.ZipFile(wheel_path(entry)) as z:
        names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith("meta.json")]
        if not names:
            raise SystemExit(f"no meta.json in {wheel_path(entry)}")
        return json.loads(z.read(names[0]))


def mb(path):
    """Decimal MB, matching the sizes quoted in README.md and docs/MODELS.md."""
    return Path(path).stat().st_size / 1_000_000


def repo_id(entry):
    return f"{REPO_OWNER}/{entry.get('hf', entry['pkg'])}"


def frontmatter(meta, vectors_only):
    """Without this block the Hub cannot filter the model by language, library or task."""
    notes = meta.get("notes") or ""
    lines = ["---", "language:", "- fa"]
    if "REDISTRIBUTION WARNING" in notes:
        # The pipeline's own code and annotations are CC BY-SA 4.0, but the wheel embeds
        # encoder weights whose terms are unknown. Tagging the whole artifact cc-by-sa-4.0
        # would assert a licence nobody granted.
        lines += ["license: other", "license_name: mixed-see-notes"]
    else:
        lines.append(f"license: {meta.get('license', 'cc-by-sa-4.0').lower().replace(' ', '-')}")
    lines += ["library_name: spacy"]
    if vectors_only:
        lines += ["pipeline_tag: feature-extraction", "tags:", "- spacy",
                  "- feature-extraction", "- floret", "- word-embeddings"]
    else:
        lines += ["pipeline_tag: token-classification", "tags:", "- spacy",
                  "- token-classification"]
        for component, tag in (("parser", "dependency-parsing"), ("ner", "named-entity-recognition")):
            if component in meta.get("pipeline", []):
                lines.append(f"- {tag}")
    lines += ["- persian", "- farsi", "---", ""]
    return lines


def usage(meta, name):
    """The snippet must only print attributes this pipeline actually sets: `doc.ents` on a
    dep pipeline is always empty, and `token.pos_` on an ent-only pipeline is always ''."""
    pipes = meta.get("pipeline", [])
    lines = ["```python", "import spacy", "", f'nlp = spacy.load("{name}")',
             'doc = nlp("شرکت ایران خودرو اعلام کرد که تولید خود را افزایش می\u200cدهد.")', ""]
    # UPOS comes from the morphologizer, XPOS from the tagger: this project trains both,
    # and printing `t.pos_` for a tagger-only pipeline would print empty strings.
    attrs = [("morphologizer", "t.pos_"), ("tagger", "t.tag_"),
             ("trainable_lemmatizer", "t.lemma_"), ("parser", "t.dep_")]
    shown = [expr for component, expr in attrs if component in pipes]
    if shown:
        lines.append(f"print([(t.text, {', '.join(shown)}) for t in doc])")
    if "ner" in pipes:
        lines.append("print([(e.text, e.label_) for e in doc.ents])")
    lines += ["```", ""]
    return lines


def accuracy(meta):
    perf = meta.get("performance", {})
    lines = ["## Accuracy", "",
             "Scored with `spacy benchmark accuracy` on the held-out PerDT test split.", "",
             "| Metric | Score |", "| --- | ---: |"]
    for key, label in METRICS:
        value = perf.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"| {label} | {value * 100:.2f} |")
    lines.append("")
    per_type = perf.get("ents_per_type")
    if per_type:
        lines += ["Per entity label:", "", "| Label | P | R | F |", "| --- | ---: | ---: | ---: |"]
        for label in sorted(per_type):
            s = per_type[label]
            lines.append(f"| `{label}` | {s['p'] * 100:.2f} | {s['r'] * 100:.2f} | "
                         f"{s['f'] * 100:.2f} |")
        lines.append("")
    return lines


def throughput(entry):
    rows = []
    for path in entry.get("throughput", []):
        if Path(path).exists():
            d = load(path)
            rows.append((d["device"], d["batch_size"], d["wps_median"]))
    if not rows:
        return []
    lines = ["## Throughput", "",
             "Median of repeated `nlp.pipe` passes over the 146-document PerDT test split",
             "(23,825 tokens), timing the pipe only. Warmup pass discarded.", "",
             "| Device | Batch | Words/s |", "| --- | ---: | ---: |"]
    for device, batch, wps in rows:
        lines.append(f"| {device} | {batch} | {wps:,.0f} |")
    lines.append("")
    return lines


def vector_table(meta):
    v = meta.get("vectors", {})
    training = meta.get("vectors_training", {})
    lines = ["## The table", "", "| Property | Value |", "| --- | --- |",
             f"| Rows | {v.get('vectors', 0):,} |",
             f"| Dimensions | {v.get('width', 0)} |",
             f"| Mode | {v.get('name', 'floret')} |"]
    for key, label in [("minn", "`minn` / `maxn`"), ("hash_count", "`hash_count`"),
                       ("corpus", "Corpus"), ("tokens", "Tokens"), ("epochs", "Epochs"),
                       ("tool", "Trained with")]:
        if key == "minn" and "minn" in training:
            lines.append(f"| {label} | {training['minn']} / {training.get('maxn')} |")
        elif key in training:
            lines.append(f"| {label} | {training[key]} |")
    lines.append("")
    cli = training.get("cli")
    if cli:
        lines += ["Trained with:", "", "```", cli.strip(), "```", ""]
    return lines


def vectors_usage(meta, name, repo, wheel):
    return [
        "## Use", "",
        "```python", "import spacy", "", f'nlp = spacy.load("{name}")',
        "print(nlp.vocab.vectors.shape)",
        'print(nlp("می\u200cرود")[0].has_vector)   # floret hashes subwords: always True',
        "```", "",
        "To train your own pipeline against it, unpack the wheel and point spaCy at the",
        "directory:", "",
        "```bash",
        f"pip download --no-deps -d . https://huggingface.co/{repo}/resolve/main/{wheel}",
        f"python -m spacy train config.cfg --paths.vectors ./{name}",
        "```", "",
    ]


def cross_links(current, cards):
    """Every card carries the same tables so a reader landing on one tier can find the
    rest. Built from the catalogue rather than copied per card, so they cannot drift."""
    pipelines = [c for c in cards if c[1].get("pipeline")]
    vectors = [c for c in cards if not c[1].get("pipeline")]

    def label(entry, meta):
        name = f"{meta['lang']}_{meta['name']}"
        if name == current:
            return f"`{name}` (this one)"
        return f"[`{name}`](https://huggingface.co/{repo_id(entry)})"

    lines = ["## Other packages in this family", "",
             "| Pipeline | Tier | LAS | ENTS_F | Wheel |",
             "| --- | --- | ---: | ---: | ---: |"]
    for entry, meta, wheel in pipelines:
        perf = meta.get("performance", {})
        las = f"{perf['dep_las'] * 100:.2f}" if "dep_las" in perf else "-"
        ents = f"{perf['ents_f'] * 100:.2f}" if "ents_f" in perf else "-"
        lines.append(f"| {label(entry, meta)} | `{entry['tier']}` | {las} | {ents} | "
                     f"{mb(wheel):,.1f} MB |")
    lines += ["", "Tiers: " + ", ".join(f"`{t}` {b}" for t, b in TIER_BLURB.items()) + ".", ""]
    lines += ["Standalone vector tables, usable as `--paths.vectors` for your own training:",
              "", "| Vectors | Rows | Used by | Wheel |", "| --- | ---: | --- | ---: |"]
    for entry, meta, wheel in vectors:
        used = f"`{entry['tier']}` tier" if entry["tier"] != "-" else "no shipped pipeline"
        lines.append(f"| {label(entry, meta)} | {meta['vectors']['vectors']:,} | {used} | "
                     f"{mb(wheel):,.1f} MB |")
    lines += ["", f"Training scripts, configs and evaluation: <{PROJECT_URL}>.", ""]
    return lines


def build_card(entry, meta, wheel, cards):
    name = f"{meta['lang']}_{meta['name']}"
    vectors_only = not meta.get("pipeline")
    repo = repo_id(entry)
    lines = frontmatter(meta, vectors_only)
    lines += [f"# {name}", "", (meta.get("description") or "").strip(), ""]
    lines += ["## Install", ""]
    if "transformer" in meta.get("pipeline", []):
        lines += ["This pipeline runs on a transformer, so it also needs",
                  "`spacy-transformers` and a GPU for usable throughput.", ""]
        install = "pip install spacy-transformers\n"
    else:
        install = ""
    lines += ["```bash",
              f"{install}pip install https://huggingface.co/{repo}/resolve/main/{wheel.name}",
              "```", ""]
    if vectors_only:
        lines += vectors_usage(meta, name, repo, wheel.name)
        lines += vector_table(meta)
    else:
        lines += ["## Use", ""] + usage(meta, name)
        lines += accuracy(meta)
        lines += throughput(entry)
    lines += cross_links(name, cards)
    lines += ["## Sources", "", "| Source | Author | Licence |", "| --- | --- | --- |"]
    for s in meta.get("sources", []):
        url, nm = s.get("url"), s.get("name", "")
        label = f"[{nm}]({url})" if url else nm
        lines.append(f"| {label} | {s.get('author', '')} | {s.get('license', '')} |")
    lines.append("")
    notes = (meta.get("notes") or "").strip()
    if notes:
        lines += ["## Notes", "", notes, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", help="single package name, e.g. fa_core_news_trf")
    ap.add_argument("--all", action="store_true", help="write a card for every package")
    ap.add_argument("--out", help="output file, with --package")
    ap.add_argument("--out-dir", default="cards", help="output directory, with --all")
    args = ap.parse_args()

    # Every card needs every meta: the cross-link table is computed, not copied.
    cards = [(e, read_meta(e), wheel_path(e)) for e in CATALOG]

    if args.package:
        chosen = [c for c in cards if c[0]["pkg"] == args.package]
        if not chosen:
            raise SystemExit(f"unknown package {args.package}")
        entry, meta, wheel = chosen[0]
        out = Path(args.out or f"{args.out_dir}/{entry['pkg']}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_card(entry, meta, wheel, cards), encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size} bytes)")
        return
    if not args.all:
        raise SystemExit("pass --all or --package")
    for entry, meta, wheel in cards:
        out = Path(args.out_dir) / f"{entry['pkg']}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_card(entry, meta, wheel, cards), encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
