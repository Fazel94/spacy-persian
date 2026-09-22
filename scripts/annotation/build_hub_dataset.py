#!/usr/bin/env python
"""Assemble the `Phazel/fa-perdt-ner` dataset repo for the Hugging Face Hub.

    python scripts/annotation/build_hub_dataset.py --out packages_hf/fa-perdt-ner

Two configurations over the same 29,107 PerDT sentences and the same `--merge-subtokens`
tokenization:

* `silver`: PerDT's own seven-label NER layer realigned onto that tokenization
  (corpus/perdt-ner-iob, written by scripts/transfer_perdt_ner.py);
* `llm`: the four-label relabelling under annotation/GUIDELINES.md
  (annotation/data/llm/<run>/, exported to corpus/perdt-ner-iob-llm).

Every sentence is keyed by its PerDT `sent_id`, taken from the CoNLL-U files in order, so a
row can be traced back to the treebank. Both corpora are asserted to be in CoNLL-U order
with identical token strings before anything is written.

The dataset card is rendered from annotation/hub/README.md: every `{{name}}` placeholder is
filled from counts computed here, so the card cannot quote a number the files do not
contain. The build fails on an unknown or unfilled placeholder. manifest.json records the
inputs (URL + checksum of every PerDT asset, from project.yml), the guideline/prompt hash,
the git commit, and a sha256 per output file.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import srsly

SPLITS = ("train", "dev", "test")
LLM_LABELS = ("PER", "LOC", "ORG", "DAT")
SILVER_LABELS = ("PER", "LOC", "ORG", "DAT", "MON", "TIM", "PCT")
# LLM output files per split, in corpus order (same table as export_iob.py).
LLM_SOURCES = {
    "train": [f"train/shard-{i:03d}" for i in range(14)],
    "dev": ["dev-all"],
    "test": ["test-all"],
}
PIPELINE_SCRIPTS = [
    "select_sentences.py", "build_requests.py", "annotate.js", "validate_responses.py",
    "compare_to_silver.py", "export_iob.py", "sample_gold.py", "score_gold.py",
]


def conllu_sent_ids(path):
    return [line[len("# sent_id = "):].strip()
            for line in path.open(encoding="utf8") if line.startswith("# sent_id = ")]


def iob_sents(path):
    """[(tokens, tags)] from two-column IOB2."""
    sents, toks, tags = [], [], []
    for line in path.open(encoding="utf8"):
        line = line.rstrip("\n")
        if not line.strip():
            if toks:
                sents.append((toks, tags))
                toks, tags = [], []
            continue
        tok, tag = line.split("\t")
        toks.append(tok)
        tags.append(tag)
    if toks:
        sents.append((toks, tags))
    return sents


def spans_from_tags(tags):
    """IOB2 -> {(start, end, label)}. An I- without a matching B- opens a span, as spaCy does."""
    spans, start, label = set(), None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("I-") and label == tag[2:]:
            continue
        if start is not None:
            spans.add((start, i, label))
            start, label = None, None
        if tag != "O":
            start, label = i, tag[2:]
    return spans


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "p": round(p, 4), "r": round(r, 4), "f": round(f, 4)}


def agreement(silver_rows, llm_rows):
    """Exact-span agreement, silver as reference, restricted to the four shared labels."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for (_, s_tags), (_, l_tags) in zip(silver_rows, llm_rows):
        ref = {s for s in spans_from_tags(s_tags) if s[2] in LLM_LABELS}
        hyp = spans_from_tags(l_tags)
        for _, _, label in ref & hyp:
            tp[label] += 1
        for _, _, label in hyp - ref:
            fp[label] += 1
        for _, _, label in ref - hyp:
            fn[label] += 1
    out = {label: prf(tp[label], fp[label], fn[label]) for label in LLM_LABELS}
    out["micro"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return out


def label_counts(rows, labels):
    c = Counter()
    for _, tags in rows:
        for _, _, label in spans_from_tags(tags):
            c[label] += 1
    return {label: c[label] for label in labels}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render(template, values):
    used = set()

    def sub(m):
        key = m.group(1)
        if key not in values:
            raise SystemExit(f"README template: unknown placeholder {{{{{key}}}}}")
        used.add(key)
        return str(values[key])

    text = re.sub(r"\{\{(\w+)\}\}", sub, template)
    unused = set(values) - used
    if unused:
        raise SystemExit(f"README template: unused values {sorted(unused)}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conllu-dir", default="assets/ud")
    ap.add_argument("--silver-dir", default="corpus/perdt-ner-iob")
    ap.add_argument("--llm-iob-dir", default="corpus/perdt-ner-iob-llm")
    ap.add_argument("--llm-dir", default="annotation/data/llm/ner-v2.2-default")
    ap.add_argument("--guideline-version", default="2.2")
    ap.add_argument("--hub-dir", default="annotation/hub")
    ap.add_argument("--out", default="packages_hf/fa-perdt-ner")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for sub in ("silver", "llm", "annotation/scripts"):
        (out / sub).mkdir(parents=True)

    silver_stats, llm_stats, agree, split_stats = [], [], {}, {}
    prompt_hashes, models, dropped, unannotated = set(), set(), [], []

    for split in SPLITS:
        ids = conllu_sent_ids(Path(args.conllu_dir) / f"fa_perdt-ud-{split}.conllu")
        silver = iob_sents(Path(args.silver_dir) / f"{split}.txt")
        llm_iob = iob_sents(Path(args.llm_iob_dir) / f"{split}.txt")
        llm, split_dropped = [], []
        for part in LLM_SOURCES[split]:
            path = Path(args.llm_dir) / f"{part}.jsonl"
            llm += [json.loads(line) for line in path.open(encoding="utf8")]
            invalid = path.with_suffix(".invalid.jsonl")
            if invalid.exists():
                split_dropped += [json.loads(line)
                                  for line in invalid.open(encoding="utf8") if line.strip()]

        if not len(ids) == len(silver) == len(llm_iob) == len(llm):
            raise SystemExit(f"{split}: {len(ids)} sent_ids, {len(silver)} silver, "
                             f"{len(llm_iob)} llm iob, {len(llm)} llm rows")
        row_to_sent = {}
        for i, (sid, (s_toks, _), (l_toks, _), row) in enumerate(zip(ids, silver, llm_iob, llm)):
            if not s_toks == l_toks == row["tokens"]:
                raise SystemExit(f"{split}: token mismatch at sentence {i} ({sid} / {row['id']})")
            row_to_sent[row["id"]] = sid
            prompt_hashes.add(row["prompt_hash"])
            models.add(row["model"])
            if row.get("error"):
                unannotated.append(sid)
        for d in split_dropped:
            d["id"] = row_to_sent[d["id"]]  # KeyError = an invalid row for an unknown sentence
        dropped += split_dropped

        write_jsonl(out / "silver" / f"{split}.jsonl", (
            {"id": sid, "tokens": toks, "ner_tags": tags}
            for sid, (toks, tags) in zip(ids, silver)))
        write_jsonl(out / "llm" / f"{split}.jsonl", (
            {"id": sid, "tokens": toks, "ner_tags": tags,
             "entities": [{"start": e["start"], "end": e["end"], "label": e["label"]}
                          for e in row["entities"]],
             "prompt_hash": row["prompt_hash"], "error": row.get("error")}
            for sid, (toks, tags), row in zip(ids, llm_iob, llm)))
        shutil.copy(Path(args.silver_dir) / f"{split}.txt", out / "silver" / f"{split}.iob")
        shutil.copy(Path(args.llm_iob_dir) / f"{split}.txt", out / "llm" / f"{split}.iob")

        n_tok = sum(len(t) for t, _ in silver)
        sc, lc = label_counts(silver, SILVER_LABELS), label_counts(llm_iob, LLM_LABELS)
        silver_stats.append([split, len(ids), n_tok, sum(sc.values())] + [sc[l] for l in SILVER_LABELS])
        llm_stats.append([split, len(ids), n_tok, sum(lc.values())] + [lc[l] for l in LLM_LABELS])
        agree[split] = agreement(silver, llm_iob)
        split_stats[split] = {"sentences": len(ids), "tokens": n_tok,
                              "silver_entities": sc, "llm_entities": lc}

    if len(prompt_hashes) != 1:
        raise SystemExit(f"expected one prompt_hash across the run, found {sorted(prompt_hashes)}")
    prompt_hash = prompt_hashes.pop()
    write_jsonl(out / "llm" / "dropped-entities.jsonl", dropped)
    (out / "llm" / "agreement.json").write_text(
        json.dumps({"reference": "silver, restricted to PER/LOC/ORG/DAT",
                    "hypothesis": "llm", "match": "exact span (start, end, label)",
                    "splits": agree}, indent=2), encoding="utf8")

    # Annotation kit: guideline, prompt, scripts, licence.
    shutil.copy("annotation/GUIDELINES.md", out / "annotation" / "GUIDELINES.md")
    shutil.copy(f"annotation/prompts/ner-v{args.guideline_version}.md",
                out / "annotation" / f"prompt-v{args.guideline_version}.md")
    for name in PIPELINE_SCRIPTS:
        shutil.copy(Path("scripts/annotation") / name, out / "annotation" / "scripts" / name)
    shutil.copy(Path(args.hub_dir) / "LICENSE", out / "LICENSE")

    # Card.
    agree_rows = [
        [split] + [f"{agree[split][l]['f']:.3f}" for l in LLM_LABELS]
        + [f"{agree[split]['micro']['p']:.3f}", f"{agree[split]['micro']['r']:.3f}",
           f"**{agree[split]['micro']['f']:.3f}**"]
        for split in SPLITS
    ]
    totals = {k: sum(s[k] for s in split_stats.values()) for k in ("sentences", "tokens")}
    values = {
        "silver_stats": md_table(["split", "sentences", "tokens", "entities", *SILVER_LABELS],
                                 silver_stats),
        "llm_stats": md_table(["split", "sentences", "tokens", "entities", *LLM_LABELS], llm_stats),
        "agreement": md_table(["split", "PER F", "LOC F", "ORG F", "DAT F",
                               "micro P", "micro R", "micro F"], agree_rows),
        "n_sentences": f"{totals['sentences']:,}",
        "n_tokens": f"{totals['tokens']:,}",
        "n_silver_entities": f"{sum(r[3] for r in silver_stats):,}",
        "n_llm_entities": f"{sum(r[3] for r in llm_stats):,}",
        "oos_share": "{:.1%}".format(sum(sum(r[8:]) for r in silver_stats)
                                     / sum(r[3] for r in silver_stats)),
        "n_dropped": len(dropped),
        "dropped_reasons": ", ".join(f"{n} `{reason}`" for reason, n in
                                     Counter(d["reason"] for d in dropped).most_common()),
        "n_unannotated": len(unannotated),
        "unannotated_ids": ", ".join(f"`{i}`" for i in unannotated) or "none",
        "prompt_hash": prompt_hash,
        "guideline_version": args.guideline_version,
        "model_alias": ", ".join(sorted(models)),
    }
    template = (Path(args.hub_dir) / "README.md").read_text(encoding="utf8")
    (out / "README.md").write_text(render(template, values), encoding="utf8")

    # Manifest.
    project = srsly.read_yaml("project.yml")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(p for p in out.rglob("*") if p.is_file())}
    manifest = {
        "dataset": "Phazel/fa-perdt-ner",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": commit,
        "build_script": "scripts/annotation/build_hub_dataset.py",
        "source_assets": [{"dest": a["dest"], "url": a["url"], "md5": a["checksum"]}
                          for a in project["assets"] if a["dest"].startswith("assets/ud")],
        "silver": {"labels": list(SILVER_LABELS),
                   "transfer_script": "scripts/transfer_perdt_ner.py"},
        "llm": {"labels": list(LLM_LABELS), "guideline_version": args.guideline_version,
                "prompt_hash": prompt_hash, "model": sorted(models),
                "model_note": "alias as passed to the annotator; the resolved model id was "
                              "not recorded at run time",
                "unannotated_sentences": unannotated, "dropped_entities": len(dropped)},
        "splits": split_stats,
        "agreement": agree,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf8")

    for row in silver_stats:
        print(f"silver {row[0]:<5} {row[1]:>6} sents {row[2]:>7} toks {row[3]:>6} ents")
    for row in llm_stats:
        print(f"llm    {row[0]:<5} {row[1]:>6} sents {row[2]:>7} toks {row[3]:>6} ents "
              f"agreement F {agree[row[0]]['micro']['f']:.4f}")
    print(f"dropped {len(dropped)}, unannotated {unannotated}, prompt_hash {prompt_hash}")
    print(f"-> {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
