"""Merge the separately-trained NER component into the UD pipeline and write full metadata.

The two components cannot be trained together: the UD components come from
UD_Persian-PerDT (CC BY-SA 4.0, edited prose) and `ner` comes from ParsTwiNER (MIT,
tweets). `ner` was therefore configured with its own embedded tok2vec (see
configs/fa_ner_sm.cfg) so it can be sourced into another pipeline without a dangling
Tok2VecListener — the same design as en_core_web_sm.

Run it twice: once right after training (no metrics yet), and again after
`spacy benchmark accuracy` has produced metrics/*.json so `meta.json["performance"]`
reflects the assembled pipeline on the held-out test sets.

Usage:
  .venv/bin/python scripts/assemble_core.py training/core/model-best training/ner/model-best \\
      training/fa_core_news_sm --version 3.8.0 \\
      [--ud-metrics metrics/ud-test.json] [--ner-metrics metrics/ner-test.json]
"""

import argparse
import json
from pathlib import Path

import spacy

DESCRIPTION = (
    "Persian pipeline optimized for CPU. Components: tok2vec, tagger, morphologizer, "
    "trainable_lemmatizer, parser, ner."
)

SOURCES = [
    {
        "name": "UD_Persian-PerDT (PerUDT v1.0)",
        "url": "https://github.com/UniversalDependencies/UD_Persian-PerDT",
        "author": (
            "Mohammad Sadegh Rasooli, Pegah Safari, Amirsaeid Moloodi, Alireza Nourian"
        ),
        "license": "CC BY-SA 4.0",
    },
    {
        "name": "ParsTwiNER",
        "url": "https://github.com/overfit-ir/parstwiner",
        "author": "MohammadMahdi Aghajani, AliAkbar Badri, Hamid Beigy et al. (Overfit-IR)",
        "license": "MIT",
    },
    {
        "name": "spaCy lang/fa language data (stop words originally from HAZM)",
        "url": "https://github.com/explosion/spaCy/tree/master/spacy/lang/fa",
        "author": "Explosion and spaCy contributors",
        "license": "MIT",
    },
]

# CC BY-SA 4.0 on the treebank propagates to anything derived from it.
NOTES = (
    "The tagger, morphologizer, trainable_lemmatizer and parser are trained on "
    "UD_Persian-PerDT, which is licensed CC BY-SA 4.0; this pipeline is therefore "
    "distributed under CC BY-SA 4.0 with attribution to the treebank authors. "
    "The ner component is trained on ParsTwiNER (MIT), a Twitter corpus, so entity "
    "recognition is weaker on formal/edited prose than on social media text. "
    "Multiword tokens in the treebank (pronominal clitics, enclitic copulas) were merged "
    "with `spacy convert --merge-subtokens`, so a small number of XPOS tags are composite "
    "(e.g. N_IANM_PR_JOPER) and ~1.5% of lemmas contain a space."
)


def load_metrics(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"  (no metrics at {p}, skipping)")
        return {}
    return json.loads(p.read_text(encoding="utf8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("core", help="trained UD pipeline (training/core/model-best)")
    ap.add_argument("ner", help="trained NER pipeline (training/ner/model-best)")
    ap.add_argument("output", help="destination directory for the assembled pipeline")
    ap.add_argument("--version", default="3.8.0")
    ap.add_argument("--ud-metrics", default=None)
    ap.add_argument("--ner-metrics", default=None)
    args = ap.parse_args()

    nlp = spacy.load(args.core)
    ner_nlp = spacy.load(args.ner)
    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")
    nlp.add_pipe("ner", source=ner_nlp)
    print(f"pipeline: {nlp.pipe_names}")
    print(f"ner labels: {sorted(nlp.get_pipe('ner').labels)}")

    ud = load_metrics(args.ud_metrics)
    ner = load_metrics(args.ner_metrics)
    performance = dict(nlp.meta.get("performance", {}))
    for key in ("token_acc", "token_p", "token_r", "token_f", "tag_acc", "pos_acc",
                "morph_acc", "lemma_acc", "dep_uas", "dep_las", "sents_p", "sents_r",
                "sents_f"):
        if key in ud:
            performance[key] = ud[key]
    for key in ("ents_p", "ents_r", "ents_f"):
        if key in ner:
            performance[key] = ner[key]
    if "ents_per_type" in ner:
        performance["ents_per_type"] = ner["ents_per_type"]
    if "dep_las_per_type" in ud:
        performance["dep_las_per_type"] = ud["dep_las_per_type"]

    nlp.meta.update(
        {
            "lang": "fa",
            "name": "core_news_sm",
            "version": args.version,
            "description": DESCRIPTION,
            "author": "",
            "email": "",
            "url": "",
            "license": "CC BY-SA 4.0",
            "sources": SOURCES,
            "notes": NOTES,
            "performance": performance,
        }
    )

    out = Path(args.output)
    nlp.to_disk(out)
    print(f"wrote {out}")
    print(json.dumps(performance, indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
