"""Write complete, convention-compliant metadata onto a trained pipeline.

Two variants, following spaCy's `[lang]_[type]_[genre]_[size]` naming
(https://spacy.io/models#conventions):

  dep  -> fa_dep_news_sm   tagger + morphologizer + trainable_lemmatizer + parser
  ent  -> fa_ent_news_sm   ner only

`core` is deliberately NOT produced. `core` means "tagger + parser + lemmatizer + NER" in
one package, and shipping one would imply the NER is of the same standard as the rest of
the pipeline. It is not: the UD components score 85-98 on edited prose while the ner
component manages 67.22 F and is trained on tweets, because no redistributably-licensed
Persian NER corpus shares a genre with the treebank. Merging them into a single `core`
artifact would hide that behind one package name.

`fa_core_news_sm` is reserved for when ../ner_dataset delivers prose-genre NER data that
survives its own ablation (see ../ner_dataset/PLAN.md §5).

Run twice: once after training (no metrics yet), then again after
`spacy benchmark accuracy` so meta.json["performance"] carries held-out test scores.

Usage:
  .venv/bin/python scripts/finalize_pipeline.py training/core/model-best training/fa_dep_news_sm \\
      --variant dep --version 3.8.0 [--metrics metrics/ud-test.json]
"""

import argparse
import json
from pathlib import Path

import spacy

PERDT = {
    "name": "UD_Persian-PerDT (PerUDT v1.0)",
    "url": "https://github.com/UniversalDependencies/UD_Persian-PerDT",
    "author": "Mohammad Sadegh Rasooli, Pegah Safari, Amirsaeid Moloodi, Alireza Nourian",
    "license": "CC BY-SA 4.0",
}
PARSTWINER = {
    "name": "ParsTwiNER",
    "url": "https://github.com/overfit-ir/parstwiner",
    "author": "MohammadMahdi Aghajani, AliAkbar Badri, Hamid Beigy et al. (Overfit-IR)",
    "license": "MIT",
}
LANG_DATA = {
    "name": "spaCy lang/fa language data (stop words originally from HAZM)",
    "url": "https://github.com/explosion/spaCy/tree/master/spacy/lang/fa",
    "author": "Explosion and spaCy contributors",
    "license": "MIT",
}

VARIANTS = {
    "dep": {
        "name": "dep_news_sm",
        "description": (
            "Persian dependency pipeline optimized for CPU. Components: tok2vec, tagger, "
            "morphologizer, trainable_lemmatizer, parser. No NER — see fa_ent_news_sm."
        ),
        # CC BY-SA 4.0 on the treebank propagates to anything derived from it.
        "license": "CC BY-SA 4.0",
        "sources": [PERDT, LANG_DATA],
        "notes": (
            "Trained on UD_Persian-PerDT, licensed CC BY-SA 4.0; this pipeline is therefore "
            "distributed under CC BY-SA 4.0 with attribution to the treebank authors. "
            "Multiword tokens (pronominal clitics, enclitic copulas) were merged with "
            "`spacy convert --merge-subtokens`, so a small number of XPOS tags are composite "
            "(e.g. N_IANM_PR_JOPER) and ~1.5% of lemmas contain a space. "
            "doc.noun_chunks under-fires on this pipeline: spacy/lang/fa/syntax_iterators.py "
            "matches ClearNLP labels that do not exist in Universal Dependencies — see "
            "docs/upstream/fa-noun-chunks.md."
        ),
        "keys": ("token_acc", "token_p", "token_r", "token_f", "tag_acc", "pos_acc",
                 "morph_acc", "lemma_acc", "dep_uas", "dep_las", "sents_p", "sents_r",
                 "sents_f", "dep_las_per_type"),
    },
    "ent": {
        "name": "ent_news_sm",
        "description": (
            "Persian named entity recognizer optimized for CPU, with its own internal "
            "tok2vec. Labels: PER, ORG, LOC, NAT, POG, EVE."
        ),
        "license": "MIT",
        "sources": [PARSTWINER, LANG_DATA],
        "notes": (
            "Trained on ParsTwiNER (MIT), a Persian Twitter corpus, because the standard "
            "Persian NER corpora (ARMAN, PEYMA, NSURL) are research-use-only and cannot be "
            "redistributed. Expect degraded accuracy on formal or edited prose: this scores "
            "67.22 F on its own in-genre test set, and EVE (30.0) and POG (41.2) are weak "
            "enough to be treated as unreliable. The component embeds its own tok2vec rather "
            "than using a Tok2VecListener, so it can be sourced into another pipeline with "
            "nlp.add_pipe('ner', source=...). A prose-genre replacement is being built in "
            "../ner_dataset."
        ),
        "keys": ("token_acc", "ents_p", "ents_r", "ents_f", "ents_per_type"),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="trained pipeline, e.g. training/core/model-best")
    ap.add_argument("output", help="destination directory")
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--version", default="3.8.0")
    ap.add_argument("--metrics", nargs="*", default=[],
                    help="benchmark accuracy JSON files to fold into performance")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    nlp = spacy.load(args.model)
    print(f"pipeline: {nlp.pipe_names}")
    if args.variant == "dep" and "ner" in nlp.pipe_names:
        raise SystemExit("refusing to publish a 'dep' pipeline that contains an ner component")
    if args.variant == "ent" and nlp.pipe_names != ["ner"]:
        raise SystemExit(f"expected exactly ['ner'], got {nlp.pipe_names}")

    # Build from a strict whitelist rather than inheriting nlp.meta["performance"], which
    # after training also carries raw *_loss values and scorer keys that do not apply to
    # this pipeline (e.g. tag_micro_f: 0.0). Published metadata should be the held-out
    # scores and nothing else.
    trained = nlp.meta.get("performance", {})
    performance = {k: trained[k] for k in spec["keys"] if trained.get(k) is not None}
    for path in args.metrics:
        p = Path(path)
        if not p.exists():
            print(f"  (no metrics at {p}, skipping — run `evaluate` first)")
            continue
        scored = json.loads(p.read_text(encoding="utf8"))
        for key in spec["keys"]:
            if scored.get(key) is not None:
                performance[key] = scored[key]
        if scored.get("speed") is not None:
            performance["speed"] = scored["speed"]

    nlp.meta.update(
        {
            "lang": "fa",
            "name": spec["name"],
            "version": args.version,
            "description": spec["description"],
            "author": "",
            "email": "",
            "url": "",
            "license": spec["license"],
            "sources": spec["sources"],
            "notes": spec["notes"],
            "performance": performance,
        }
    )

    out = Path(args.output)
    nlp.to_disk(out)
    print(f"wrote {out} as fa_{spec['name']} {args.version} ({spec['license']})")
    scalars = {k: round(v * 100, 2) for k, v in performance.items() if isinstance(v, float)}
    print(json.dumps(scalars, indent=2))


if __name__ == "__main__":
    main()
