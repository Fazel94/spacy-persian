"""Run the assembled pipeline over real Persian text and print every annotation layer.

This is the end-to-end check that the artifact actually works: tokenizer -> tagger ->
morphologizer -> lemmatizer -> parser -> ner -> noun_chunks.

Usage: .venv/bin/python scripts/smoke_test.py training/fa_core_news_sm
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
    path = sys.argv[1] if len(sys.argv) > 1 else "training/fa_core_news_sm"
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
        print(f"morph[0]: {doc[0].morph}")
        print(f"sents: {[s.text for s in doc.sents]}")
        print(f"ents: {[(e.text, e.label_) for e in doc.ents]}")
        print(f"noun_chunks: {[c.text for c in doc.noun_chunks]}")


if __name__ == "__main__":
    main()
