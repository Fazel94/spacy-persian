"""Fuse the UD annotation layer and the transferred NER layer into one DocBin.

The sm/md/lg tiers train `ner` as a separate pipeline with its own embedded tok2vec, then
source it into the dep model (project.yml `assemble-core`). That works because a hash-embed
tok2vec is cheap enough to train twice.

A transformer is not. Fine-tuning ParsBERT once per component would double GPU cost and
produce a package carrying two independent 162M-parameter encoders, and sourcing the second
one would collide on the `transformer` component name. So the trf tier trains every component
against a single shared transformer via TransformerListener, which requires a single corpus
carrying both annotation layers on the same Doc.

That fusion is exact, not approximate: `corpus/perdt-ner/` was produced by
scripts/transfer_perdt_ner.py from the same `--merge-subtokens` CoNLL-U as `corpus/merged/`,
then converted with the same `--n-sents`, so the two DocBins are token-for-token identical
(verified below and asserted at runtime). Only `doc.ents` is copied across; every other
annotation stays on the UD doc.
"""

import argparse
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Span

SPLITS = (("train", "fa_perdt-ud-train"), ("dev", "fa_perdt-ud-dev"), ("test", "fa_perdt-ud-test"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ud-dir", default="corpus/merged")
    ap.add_argument("--ner-dir", default="corpus/perdt-ner")
    ap.add_argument("--out", default="corpus/joint")
    ap.add_argument("--lang", default="fa")
    args = ap.parse_args()

    nlp = spacy.blank(args.lang)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split, ud_stem in SPLITS:
        ud_docs = list(DocBin().from_disk(Path(args.ud_dir) / f"{ud_stem}.spacy").get_docs(nlp.vocab))
        ner_docs = list(DocBin().from_disk(Path(args.ner_dir) / f"{split}.spacy").get_docs(nlp.vocab))
        if len(ud_docs) != len(ner_docs):
            raise SystemExit(
                f"{split}: {len(ud_docs)} UD docs vs {len(ner_docs)} NER docs; the two corpora "
                "were not converted from the same source with the same --n-sents"
            )

        db = DocBin(store_user_data=True)
        n_ents = 0
        for i, (ud, ner) in enumerate(zip(ud_docs, ner_docs)):
            if [t.text for t in ud] != [t.text for t in ner]:
                raise SystemExit(f"{split} doc {i}: tokenization differs between UD and NER layers")
            # Tokens are index-aligned, so rebuild by token index. Char offsets are NOT
            # safe here: the two converters can differ in trailing whitespace, which shifts
            # `char_span` off the token grid and silently yields None.
            ud.ents = [Span(ud, e.start, e.end, label=e.label_) for e in ner.ents]
            n_ents += len(ud.ents)
            db.add(ud)

        dest = out / f"{split}.spacy"
        db.to_disk(dest)
        print(f"{dest}: {len(ud_docs)} docs, {n_ents} entities")


if __name__ == "__main__":
    main()
