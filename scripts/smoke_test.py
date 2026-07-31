"""Run a trained pipeline over real Persian text and print every annotation layer.

This is the end-to-end check that the artifact actually works: tokenizer -> tagger ->
morphologizer -> lemmatizer -> parser -> noun_chunks, plus ner when the pipeline has one.
Works on fa_dep_news_sm, fa_ent_news_sm, or the two combined.

Usage: .venv/bin/python scripts/smoke_test.py training/fa_dep_news_sm
"""

import sys

import spacy
from spacy.lang.fa.examples import sentences as FA_EXAMPLES

EXTRA = [
    # ZWNJ-heavy verb forms, an enclitic pronoun, and named entities.
    "دانشگاه تهران در سال ۱۳۱۳ تأسیس شد و بزرگ‌ترین دانشگاه ایران است.",
    "کتاب‌هایش را روی میز گذاشت و به سرعت از خانه بیرون رفت.",
    "شرکت ایران خودرو اعلام کرد که تولید خود را افزایش می‌دهد.",
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "training/fa_dep_news_sm"
    nlp = spacy.load(path)
    print(f"loaded {nlp.meta['lang']}_{nlp.meta['name']} {nlp.meta['version']}")
    print(f"pipeline: {nlp.pipe_names}")

    for text in list(FA_EXAMPLES) + EXTRA:
        doc = nlp(text)
        print("\n" + "=" * 78)
        print(text)
        print(f"{'TEXT':<16}{'LEMMA':<16}{'UPOS':<7}{'TAG':<14}{'DEP':<14}HEAD")
        for t in doc:
            print(
                f"{t.text:<16}{t.lemma_:<16}{t.pos_:<7}{t.tag_:<14}"
                f"{t.dep_:<14}{t.head.text}"
            )
        if doc.has_annotation("MORPH"):
            print(f"morph[0]: {doc[0].morph}")
        if doc.has_annotation("DEP"):
            print(f"sents: {[s.text for s in doc.sents]}")
            print(f"noun_chunks: {[c.text for c in doc.noun_chunks]}")
        if "ner" in nlp.pipe_names:
            print(f"ents: {[(e.text, e.label_) for e in doc.ents]}")


if __name__ == "__main__":
    main()
