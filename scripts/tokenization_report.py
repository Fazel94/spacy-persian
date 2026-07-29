"""Compare gold tokenisation against spaCy's rule-based `fa` tokenizer.

Answers the only question that decides `spacy convert --merge-subtokens` for Persian:
how often can the tokenizer we ship at runtime reproduce the gold token boundaries?

Usage: .venv/bin/python scripts/tokenization_report.py corpus/merged/fa_perdt-ud-dev.spacy [...]
"""

import sys
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def offsets(doc):
    return {(t.idx, t.idx + len(t.text)) for t in doc}


def report(path, nlp):
    gold_docs = list(DocBin().from_disk(path).get_docs(nlp.vocab))
    tp = gold_n = pred_n = 0
    tags = Counter()
    multi_lemma = 0
    tokens = 0
    for gold in gold_docs:
        pred = nlp.make_doc(gold.text)
        g, p = offsets(gold), offsets(pred)
        tp += len(g & p)
        gold_n += len(g)
        pred_n += len(p)
        for t in gold:
            tokens += 1
            tags[t.tag_] += 1
            if " " in t.lemma_:
                multi_lemma += 1
    precision = tp / pred_n
    recall = tp / gold_n
    f = 2 * precision * recall / (precision + recall)
    composite = {t: n for t, n in tags.items() if "_" in t and t.count("_") > 1}
    print(f"\n== {path}")
    print(f"docs {len(gold_docs)}  gold tokens {gold_n}  predicted tokens {pred_n}")
    print(f"token P {precision:.4f}  R {recall:.4f}  F {f:.4f}")
    print(f"tag types {len(tags)}")
    print(f"tags containing >1 underscore (merge artefacts): {len(composite)}"
          f" covering {sum(composite.values())} tokens"
          f" ({100 * sum(composite.values()) / tokens:.2f}%)")
    if composite:
        top = ", ".join(f"{t}={n}" for t, n in Counter(composite).most_common(8))
        print(f"  most frequent: {top}")
    print(f"lemmas containing a space (merge artefacts): {multi_lemma}"
          f" ({100 * multi_lemma / tokens:.2f}%)")


def main():
    nlp = spacy.blank("fa")
    for arg in sys.argv[1:]:
        report(Path(arg), nlp)


if __name__ == "__main__":
    main()
