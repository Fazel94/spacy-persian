"""Evidence for the upstream `spacy/lang/fa/syntax_iterators.py` bug.

The shipped Persian `noun_chunks` iterator matches these dependency labels:

    nsubj, dobj, nsubjpass, pcomp, pobj, dative, appos, attr, ROOT

`dobj`, `nsubjpass`, `pobj`, `dative` and `attr` are ClearNLP/English labels. They do not
exist in Universal Dependencies, and every Persian treebank is UD. So on a UD-trained `fa`
pipeline the iterator only ever fires on `nsubj`, `appos`, `ROOT` and `conj`.

It also expands each chunk to `word.left_edge`, which is wrong for Persian: ezafe
constructions put modifiers to the RIGHT of the head noun (`رئیس انجمن جراحان قلب ایران`),
so left-only expansion truncates the phrase to the head.

This script quantifies both problems against the dev corpus and prints a side-by-side
comparison with the proposed UD implementation (see docs/upstream/fa-noun-chunks.md).

Usage:
  .venv/bin/python scripts/check_noun_chunks.py training/core/model-best \\
      corpus/merged/fa_perdt-ud-dev.spacy
"""

import sys
from collections import Counter

import spacy
from spacy.lang.fa.syntax_iterators import noun_chunks as shipped_noun_chunks
from spacy.symbols import NOUN, PRON, PROPN
from spacy.tokens import DocBin

# Labels a UD-trained parser can actually emit, as heads of a base noun phrase.
UD_LABELS = [
    "nsubj",
    "nsubj:pass",
    "obj",
    "iobj",
    "obl",
    "obl:arg",
    "nmod",
    "appos",
    "vocative",
    "ROOT",
]
# Relations that continue a Persian noun phrase to the right of its head.
UD_POST_MODIFIERS = ["nmod", "nmod:poss", "amod", "flat", "flat:name", "flat:num",
                     "fixed", "compound", "det", "nummod"]


def proposed_noun_chunks(doclike):
    """UD-label implementation with right-side ezafe expansion."""
    doc = doclike.doc
    if not doc.has_annotation("DEP"):
        raise ValueError("requires a dependency parse")
    np_deps = {doc.vocab.strings.add(label) for label in UD_LABELS}
    np_modifs = {doc.vocab.strings.add(label) for label in UD_POST_MODIFIERS}
    np_label = doc.vocab.strings.add("NP")
    conj = doc.vocab.strings.add("conj")
    adp_pos = doc.vocab.strings.add("ADP")
    cconj_pos = doc.vocab.strings.add("CCONJ")
    prev_end = -1
    for word in doclike:
        if word.pos not in (NOUN, PROPN, PRON):
            continue
        if word.left_edge.i <= prev_end:
            continue
        head_dep = word.dep
        if head_dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            if head.dep not in np_deps:
                continue
        elif head_dep not in np_deps:
            continue
        # Expand right through ezafe / modifier chains.
        right = word
        for child in word.rights:
            if child.dep in np_modifs:
                right = child.right_edge
            else:
                break
        start, end = word.left_edge.i, max(word.i, right.i) + 1
        # Mirror fr/es: a leading preposition or coordinator is not part of the NP.
        while start < word.i and doc[start].pos in (adp_pos, cconj_pos):
            start += 1
        if end <= prev_end:
            continue
        prev_end = end - 1
        yield start, end, np_label


def spans(doc, iterator):
    return [doc[s:e].text for s, e, _ in iterator(doc)]


def main():
    model, corpus = sys.argv[1], sys.argv[2]
    nlp = spacy.load(model)
    docs = list(DocBin().from_disk(corpus).get_docs(nlp.vocab))

    shipped_total = proposed_total = 0
    shipped_tokens = proposed_tokens = 0
    dep_hist = Counter()
    for doc in docs:
        for t in doc:
            if t.pos in (NOUN, PROPN, PRON):
                dep_hist[t.dep_] += 1
        for s, e, _ in shipped_noun_chunks(doc):
            shipped_total += 1
            shipped_tokens += e - s
        for s, e, _ in proposed_noun_chunks(doc):
            proposed_total += 1
            proposed_tokens += e - s

    print(f"gold dev corpus: {len(docs)} docs")
    print(f"shipped  noun_chunks: {shipped_total:>6} chunks, "
          f"{shipped_tokens / max(shipped_total, 1):.2f} tokens/chunk")
    print(f"proposed noun_chunks: {proposed_total:>6} chunks, "
          f"{proposed_tokens / max(proposed_total, 1):.2f} tokens/chunk")
    print("\ndeprels on NOUN/PROPN/PRON tokens in dev (top 15):")
    for dep, n in dep_hist.most_common(15):
        reachable = "shipped" if dep in {"nsubj", "appos", "conj", "ROOT", "root"} else "-"
        print(f"  {dep:<14}{n:>7}  {reachable}")

    print("\nside by side on live text:")
    for text in [
        "رئیس انجمن جراحان قلب ایران تأکید کرد که امکانات پیشرفته در ایران وجود دارد.",
        "دانشگاه تهران بزرگ‌ترین دانشگاه ایران است.",
    ]:
        doc = nlp(text)
        print(f"\n  {text}")
        print(f"    shipped : {spans(doc, shipped_noun_chunks)}")
        print(f"    proposed: {spans(doc, proposed_noun_chunks)}")


if __name__ == "__main__":
    main()
