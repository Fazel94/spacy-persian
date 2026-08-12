"""Write complete, convention-compliant metadata onto a trained pipeline.

Three variants, following spaCy's `[lang]_[type]_[genre]_[size]` naming
(https://spacy.io/models#conventions):

  dep  -> fa_dep_news_<size>    tagger + morphologizer + trainable_lemmatizer + parser
  core -> fa_core_news_<size>   the above plus ner
  ent  -> fa_ent_news_<size>    ner only

`--size` fills the size slot: `sm` (hash embeddings only, the default) or `md` (the same
architecture plus the fa_floret static vector table). It is metadata only; which vectors a
model actually carries is decided at train time by `--paths.vectors`.

All three are built from UD_Persian-PerDT alone, including the NER, which comes from that
treebank's own `not-to-release/Dadegan with NER tag/` layer. That is what makes `core`
honest here: one corpus, one genre, one licence, one provenance chain. An earlier version of
this script refused to emit `core` because the only redistributable Persian NER corpus we
knew of was ParsTwiNER, a Twitter corpus scoring 67.22 F against 85-98 for the UD
components; merging those into one package would have hidden a genre and quality gap behind
a single name and version.

PerDT's NER labels are silver, produced by Beheshti-NER (Taher et al., 2020) with manual
corrections, so `notes` says so and the per-label scores are published as measured.

Run twice: once after training (no metrics yet), then again after
`spacy benchmark accuracy` so meta.json["performance"] carries held-out test scores.

Usage:
  .venv/bin/python scripts/finalize_pipeline.py training/core/model-best training/fa_dep_news_sm \\
      --variant dep --version 3.8.0 [--ud-metrics metrics/ud-test.json]
  .venv/bin/python scripts/finalize_pipeline.py training/fa_core_news_sm training/fa_core_news_sm \\
      --variant core --version 3.8.0 --ud-metrics metrics/core-ud-test.json --ner-metrics metrics/perdt-ner-test.json
"""

import argparse
import json
from pathlib import Path

import spacy

AUTHOR = "Kiyarash Fazeli"
EMAIL = "kiyarash@nlogn.ir"
PROJECT_URL = "https://github.com/Fazel94/spacy-persian"

PERDT = {
    "name": "UD_Persian-PerDT (PerUDT v1.0)",
    "url": "https://github.com/UniversalDependencies/UD_Persian-PerDT",
    "author": "Mohammad Sadegh Rasooli, Pegah Safari, Amirsaeid Moloodi, Alireza Nourian",
    "license": "CC BY-SA 4.0",
}
PERDT_NER = {
    "name": "UD_Persian-PerDT NER layer (not-to-release/Dadegan with NER tag/)",
    "url": "https://github.com/UniversalDependencies/UD_Persian-PerDT",
    "author": "PerDT authors, tagged with Beheshti-NER (Taher, Hoseini, Shamsfard 2020)",
    "license": "CC BY-SA 4.0",
}
LANG_DATA = {
    "name": "spaCy lang/fa language data (stop words originally from HAZM)",
    "url": "https://github.com/explosion/spaCy/tree/master/spacy/lang/fa",
    "author": "Explosion and spaCy contributors",
    "license": "MIT",
}
FLORET = {
    "name": "fa_floret static vectors (50k rows x 300d, floret mode, 400k Persian documents)",
    "url": PROJECT_URL,
    "author": "Kiyarash Fazeli",
    "license": "CC BY-SA 4.0",
}
FLORET_LG = {
    "name": "fa_floret static vectors (lg tier: 200k rows x 300d floret table trained on "
            "the full Persian Wikipedia dump, 5 epochs, via spacy-vectors-builder)",
    "url": "https://huggingface.co/Phazel/fa-floret-wiki-vectors",
    "author": "Kiyarash Fazeli",
    "license": "CC BY-SA 4.0",
}
TRANSFORMER = {
    "name": "HooshvareLab/roberta-fa-zwnj-base",
    "url": "https://huggingface.co/HooshvareLab/roberta-fa-zwnj-base",
    "author": "Hooshvare Team",
    "license": "Apache-2.0",
}

NER_NOTE = (
    "The ner component is trained on the NER layer shipped in UD_Persian-PerDT's "
    "not-to-release/ directory, so it shares the treebank's genre, tokenization and licence. "
    "Those labels are SILVER: the treebank README states they were produced by the BERT-based "
    "Beheshti-NER tagger (Taher et al., 2020) with manual corrections to extend recall. They "
    "were transferred onto this pipeline's tokenization by difflib alignment at a 99.86% "
    "transfer rate (scripts/transfer_perdt_ner.py); spans that could not be aligned exactly "
    "were dropped rather than guessed. Labels PER, LOC, ORG and DAT have 1,300 or more "
    "training examples each; MON (205), TIM (135) and PCT (121) are thin and their scores in "
    "`performance.ents_per_type` should be read before relying on them."
)
MWT_NOTE = (
    "Multiword tokens (pronominal clitics, enclitic copulas) were merged with "
    "`spacy convert --merge-subtokens`, so a small number of XPOS tags are composite "
    "(e.g. N_IANM_PR_JOPER) and ~1.5% of lemmas contain a space."
)
CHUNK_NOTE = (
    "doc.noun_chunks under-fires on this pipeline: spacy/lang/fa/syntax_iterators.py matches "
    "ClearNLP labels that do not exist in Universal Dependencies, see "
    "docs/upstream/fa-noun-chunks.md."
)
VECTORS_NOTE = (
    "This is the `md` tier: identical architecture to the `sm` pipeline plus static floret "
    "vectors (50,000 rows x 300 dimensions, minn=maxn=5, hash_count=2) trained on 400,000 "
    "Persian documents. floret hashes subwords into a fixed table, so there are no "
    "out-of-vocabulary tokens and `token.has_vector` is always True. That matters for "
    "Persian, where inconsistent ZWNJ (U+200C) usage splits one word across several surface "
    "forms (mi-ravad written joined, with ZWNJ, or with a space) that a classic word-vector "
    "table would miss."
)


def vectors_note_lg(nlp):
    """Row/dim counts come from the trained model, not a hardcoded description, because the
    lg-tier floret table is still being iterated on (unlike md's fixed, shipped table)."""
    rows, dim = nlp.vocab.vectors.shape
    return (
        f"This is the `lg` tier: identical architecture to `sm`/`md` but a larger static "
        f"floret vector table ({rows:,} rows x {dim} dimensions, minn=maxn=5, hash_count=2) "
        f"trained on the full Persian Wikipedia dump for 5 epochs via spacy-vectors-builder. "
        f"Same zero-OOV rationale as `md` (see docs/MODELS.md): floret hashes subwords into "
        f"a fixed table, so `token.has_vector` is always True despite Persian's ZWNJ "
        f"(U+200C) inconsistency."
    )


TRANSFORMER_NOTE = (
    "This is the `trf` tier: no static vectors; contextual embeddings instead come from a "
    "fine-tuned HooshvareLab/roberta-fa-zwnj-base (Apache-2.0) transformer via "
    "spacy-transformers. Not ParsBERT: its model card carries no licence. GPU is recommended "
    "for both training and inference."
)
# CC BY-SA 4.0 on the treebank propagates to anything derived from it.
PERDT_LICENSE = "CC BY-SA 4.0"
ATTRIBUTION = (
    "Trained on UD_Persian-PerDT, licensed CC BY-SA 4.0; this pipeline is therefore "
    "distributed under CC BY-SA 4.0 with attribution to the treebank authors."
)

UD_KEYS = ("token_acc", "token_p", "token_r", "token_f", "tag_acc", "pos_acc", "morph_acc",
           "lemma_acc", "dep_uas", "dep_las", "sents_p", "sents_r", "sents_f",
           "dep_las_per_type")
NER_KEYS = ("ents_p", "ents_r", "ents_f", "ents_per_type")

VARIANTS = {
    "dep": {
        "name": "dep_news_{size}",
        "description": (
            "Persian dependency pipeline optimized for CPU. Components: tok2vec, tagger, "
            "morphologizer, trainable_lemmatizer, parser. No NER, see fa_core_news_{size}."
        ),
        "license": PERDT_LICENSE,
        "sources": [PERDT, LANG_DATA],
        "notes": " ".join([ATTRIBUTION, MWT_NOTE, CHUNK_NOTE]),
        "keys": UD_KEYS,
        "require": lambda pipes: "ner" not in pipes,
        "require_msg": "a 'dep' pipeline must not contain an ner component",
    },
    "core": {
        "name": "core_news_{size}",
        "description": (
            "Persian pipeline optimized for CPU. Components: tok2vec, tagger, morphologizer, "
            "trainable_lemmatizer, parser, ner. Entity labels: PER, LOC, ORG, DAT, MON, TIM, "
            "PCT."
        ),
        "license": PERDT_LICENSE,
        "sources": [PERDT, PERDT_NER, LANG_DATA],
        "notes": " ".join([ATTRIBUTION, NER_NOTE, MWT_NOTE, CHUNK_NOTE]),
        "keys": UD_KEYS + NER_KEYS,
        "require": lambda pipes: "ner" in pipes and "parser" in pipes,
        "require_msg": "a 'core' pipeline must contain both parser and ner",
    },
    "ent": {
        "name": "ent_news_{size}",
        "description": (
            "Persian named entity recognizer optimized for CPU, with its own internal "
            "tok2vec. Labels: PER, LOC, ORG, DAT, MON, TIM, PCT."
        ),
        "license": PERDT_LICENSE,
        "sources": [PERDT_NER, LANG_DATA],
        "notes": " ".join([ATTRIBUTION, NER_NOTE]),
        "keys": ("token_acc",) + NER_KEYS,
        "require": lambda pipes: pipes == ["ner"],
        "require_msg": "an 'ent' pipeline must be exactly ['ner']",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="trained pipeline, e.g. training/core/model-best")
    ap.add_argument("output", help="destination directory")
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--version", default="3.8.0")
    ap.add_argument("--size", choices=("sm", "md", "lg", "trf"), default="sm",
                    help="size slot in the package name. 'md'/'lg' additionally record the "
                         "floret vector table as a source and append a vectors note; 'trf' "
                         "records the transformer source and appends a transformer note.")
    ap.add_argument("--ud-metrics", default=None,
                    help="benchmark accuracy JSON scored on the UD test split; supplies the "
                         "tagger/morph/lemma/parser keys only")
    ap.add_argument("--ner-metrics", default=None,
                    help="benchmark accuracy JSON scored on the NER test split; supplies the "
                         "ents_* keys only")
    ap.add_argument("--add-ner", default=None,
                    help="source the ner component from this trained pipeline first "
                         "(used to assemble 'core' from the dep model plus the ner model)")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    name = spec["name"].format(size=args.size)
    description = spec["description"].format(size=args.size)
    sources = list(spec["sources"])
    notes = spec["notes"]
    nlp = spacy.load(args.model)
    if args.size == "md":
        sources.append(FLORET)
        notes = " ".join([notes, VECTORS_NOTE])
    elif args.size == "lg":
        sources.append(FLORET_LG)
        notes = " ".join([notes, vectors_note_lg(nlp)])
    elif args.size == "trf":
        sources.append(TRANSFORMER)
        notes = " ".join([notes, TRANSFORMER_NOTE])
    if args.add_ner:
        ner_nlp = spacy.load(args.add_ner)
        if ner_nlp.pipe_names != ["ner"]:
            raise SystemExit(f"--add-ner expects a ['ner'] pipeline, got {ner_nlp.pipe_names}")
        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")
        # `ner` embeds its own tok2vec (configs/fa_ner_sm.cfg), so sourcing it here leaves no
        # dangling Tok2VecListener. See docs/MODELS.md.
        nlp.add_pipe("ner", source=ner_nlp)
        print(f"sourced ner from {args.add_ner}: {sorted(nlp.get_pipe('ner').labels)}")
    print(f"pipeline: {nlp.pipe_names}")
    if not spec["require"](list(nlp.pipe_names)):
        raise SystemExit(
            f"refusing to publish as '{args.variant}': {spec['require_msg']}"
            f" (got {list(nlp.pipe_names)})"
        )

    # Build from a strict whitelist rather than inheriting nlp.meta["performance"], which
    # after training also carries raw *_loss values and scorer keys that do not apply to
    # this pipeline (e.g. tag_micro_f: 0.0).
    #
    # Each metrics file supplies only the keys its corpus can actually evidence. Folding both
    # files over the same key set silently corrupted core's metadata: the NER corpus has no
    # gold tags, so its report carries tag_acc: 0.0, which overwrote the real 95.96, and its
    # --n-sents grouping produced a misleading sents_f.
    trained = nlp.meta.get("performance", {})
    performance = {k: trained[k] for k in spec["keys"] if trained.get(k) is not None}

    def fold(path, keys, label):
        if not path:
            return
        p = Path(path)
        if not p.exists():
            print(f"  (no {label} metrics at {p}, skipping; run the evaluate step first)")
            return
        scored = json.loads(p.read_text(encoding="utf8"))
        taken = [k for k in keys if k in spec["keys"] and scored.get(k) is not None]
        for key in taken:
            performance[key] = scored[key]
        if scored.get("speed") is not None:
            performance.setdefault("speed", scored["speed"])
        print(f"  folded {len(taken)} {label} keys from {p}")

    fold(args.ud_metrics, UD_KEYS + ("token_acc",), "UD")
    fold(args.ner_metrics, NER_KEYS, "NER")

    nlp.meta.update(
        {
            "lang": "fa",
            "name": name,
            "version": args.version,
            "description": description,
            "author": AUTHOR,
            "email": EMAIL,
            "url": PROJECT_URL,
            "license": spec["license"],
            "sources": sources,
            "notes": notes,
            "performance": performance,
        }
    )

    out = Path(args.output)
    nlp.to_disk(out)
    print(f"wrote {out} as fa_{name} {args.version} ({spec['license']})")
    scalars = {k: round(v * 100, 2) for k, v in performance.items() if isinstance(v, float)}
    print(json.dumps(scalars, indent=2))


if __name__ == "__main__":
    main()
