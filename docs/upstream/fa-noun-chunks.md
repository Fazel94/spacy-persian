# Upstream PR candidate: `spacy/lang/fa/syntax_iterators.py` uses non-UD dependency labels

## The bug

`spacy/lang/fa/syntax_iterators.py` matches these dependency labels as noun-phrase heads:

```python
labels = ["nsubj", "dobj", "nsubjpass", "pcomp", "pobj", "dative", "appos", "attr", "ROOT"]
```

`dobj`, `nsubjpass`, `pobj`, `dative` and `attr` are ClearNLP/English labels. They are not
Universal Dependencies relations, and **every Persian treebank is UD** (`UD_Persian-PerDT`,
`UD_Persian-Seraji`, `UD_Persian-PUD`, `UD_Persian-IPerUDT`). Any trained `fa` pipeline must
therefore emit UD labels, so five of the nine labels are dead code and `doc.noun_chunks`
silently returns bare head nouns.

Compare `spacy/lang/fr/syntax_iterators.py` and `spacy/lang/es/syntax_iterators.py`, which
use UD labels (`nsubj`, `nsubj:pass`, `obj`, `obl`, `nmod`, `appos`, `ROOT`) because their
treebanks are UD too. `spacy/lang/de` legitimately uses TIGER labels (`sb`, `oa`, `nk`, …)
because the German pipelines are trained on TIGER. Persian has no such excuse.

Second, smaller problem: the iterator yields `word.left_edge.i` → `word.i + 1`, i.e. it only
ever expands **left**. Persian noun phrases expand **right** through ezafe:
`رئیس انجمن جراحان قلب ایران` ("the head of the Iranian society of heart surgeons") is one NP
whose head is the leftmost token. Left-only expansion truncates it to `رئیس`.

## Evidence

Measured with `scripts/check_noun_chunks.py` on the `UD_Persian-PerDT` dev split (146 docs)
parsed by this project's trained pipeline:

```
shipped  noun_chunks:   1503 chunks, 1.31 tokens/chunk
proposed noun_chunks:   5300 chunks, 2.77 tokens/chunk

deprels on NOUN/PROPN/PRON tokens in dev (top 15):
  nmod             2894  -
  obl              1401  -
  nsubj            1326  shipped
  compound:lvc     1294  -
  obl:arg          1095  -
  obj               973  -
  conj              583  shipped
  flat:name         389  -
  ROOT              101  shipped
  xcomp              89  -
  appos              51  shipped
  nsubj:pass         38  -
```

1.31 tokens per chunk is the tell: the shipped iterator is returning single head nouns.
`nmod` (2894), `obl` (1401), `obl:arg` (1095) and `obj` (973) — the four most common
noun-bearing relations after `nsubj` — are all unreachable.

Live text:

```
رئیس انجمن جراحان قلب ایران تأکید کرد که امکانات پیشرفته در ایران وجود دارد.
  shipped : ['رئیس', 'امکانات']
  proposed: ['رئیس انجمن جراحان قلب ایران', 'امکانات پیشرفته', 'ایران']

دانشگاه تهران بزرگ‌ترین دانشگاه ایران است.
  shipped : ['دانشگاه']
  proposed: ['دانشگاه تهران', 'ایران']
```

## Proposed patch

Swap in UD labels, expand right through modifier chains, and drop a leading `ADP`/`CCONJ`
exactly as `fr`/`es` already do:

```diff
--- a/spacy/lang/fa/syntax_iterators.py
+++ b/spacy/lang/fa/syntax_iterators.py
@@
 def noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Tuple[int, int, int]]:
     """
     Detect base noun phrases from a dependency parse. Works on both Doc and Span.
     """
-    labels = [
-        "nsubj",
-        "dobj",
-        "nsubjpass",
-        "pcomp",
-        "pobj",
-        "dative",
-        "appos",
-        "attr",
-        "ROOT",
-    ]
+    # Persian pipelines are trained on Universal Dependencies treebanks
+    # (UD_Persian-PerDT, UD_Persian-Seraji), so these are UD relations.
+    labels = [
+        "nsubj",
+        "nsubj:pass",
+        "obj",
+        "iobj",
+        "obl",
+        "obl:arg",
+        "nmod",
+        "appos",
+        "vocative",
+        "ROOT",
+    ]
+    # Persian noun phrases grow to the right through ezafe constructions.
+    post_modifiers = [
+        "nmod",
+        "nmod:poss",
+        "amod",
+        "det",
+        "nummod",
+        "flat",
+        "flat:name",
+        "flat:num",
+        "fixed",
+        "compound",
+    ]
     doc = doclike.doc  # Ensure works on both Doc and Span.
 
     if not doc.has_annotation("DEP"):
         raise ValueError(Errors.E029)
 
     np_deps = [doc.vocab.strings.add(label) for label in labels]
+    np_modifs = {doc.vocab.strings.add(label) for label in post_modifiers}
     conj = doc.vocab.strings.add("conj")
     np_label = doc.vocab.strings.add("NP")
+    adp_pos = doc.vocab.strings.add("ADP")
+    cconj_pos = doc.vocab.strings.add("CCONJ")
     prev_end = -1
     for i, word in enumerate(doclike):
         if word.pos not in (NOUN, PROPN, PRON):
             continue
         # Prevent nested chunks from being produced
         if word.left_edge.i <= prev_end:
             continue
-        if word.dep in np_deps:
-            prev_end = word.i
-            yield word.left_edge.i, word.i + 1, np_label
-        elif word.dep == conj:
+        if word.dep == conj:
             head = word.head
             while head.dep == conj and head.head.i < head.i:
                 head = head.head
-            # If the head is an NP, and we're coordinated to it, we're an NP
-            if head.dep in np_deps:
-                prev_end = word.i
-                yield word.left_edge.i, word.i + 1, np_label
+            # If the head is an NP, and we're coordinated to it, we're an NP
+            if head.dep not in np_deps:
+                continue
+        elif word.dep not in np_deps:
+            continue
+        # Expand right through the ezafe / modifier chain.
+        right = word
+        for child in word.rights:
+            if child.dep in np_modifs:
+                right = child.right_edge
+            else:
+                break
+        start, end = word.left_edge.i, max(word.i, right.i) + 1
+        # A leading preposition or coordinator is not part of the NP.
+        while start < word.i and doc[start].pos in (adp_pos, cconj_pos):
+            start += 1
+        if end <= prev_end:
+            continue
+        prev_end = end - 1
+        yield start, end, np_label
```

The working implementation lives in `scripts/check_noun_chunks.py::proposed_noun_chunks`.

## Test to add

`spacy/tests/lang/fa/test_noun_chunks.py` currently has one test (a hand-built `Doc`, 296 B).
Add UD-label coverage:

```python
def test_fa_noun_chunks_ezafe(fa_vocab):
    # رئیس انجمن جراحان — "head of the surgeons' society"
    words = ["رئیس", "انجمن", "جراحان", "آمد"]
    heads = [3, 0, 1, 3]
    deps = ["nsubj", "nmod", "nmod", "ROOT"]
    pos = ["NOUN", "NOUN", "NOUN", "VERB"]
    doc = Doc(fa_vocab, words=words, heads=heads, deps=deps, pos=pos)
    assert [c.text for c in doc.noun_chunks] == ["رئیس انجمن جراحان"]


def test_fa_noun_chunks_drops_leading_adp(fa_vocab):
    words = ["در", "ایران", "بود"]
    heads = [1, 2, 2]
    deps = ["case", "obl", "ROOT"]
    pos = ["ADP", "NOUN", "VERB"]
    doc = Doc(fa_vocab, words=words, heads=heads, deps=deps, pos=pos)
    assert [c.text for c in doc.noun_chunks] == ["ایران"]
```

## Other `spacy/lang/fa` gaps found while building this pipeline

Ordered by how much they cost a real Persian pipeline:

1. **Clitic splitting.** The `fa` tokenizer cannot split pronominal enclitics or the enclitic
   copula (`پدرم` → `پدر` + `م`, `ساکتند` → `ساکت` + `ند`), which UD treebanks annotate as
   multiword tokens. Measured on PerDT dev: gold-vs-tokenizer token F is 0.9823 when clitics
   are kept split, and 1.49% of tokens are affected. Persian-specific `TOKENIZER_SUFFIXES`
   entries for the enclitic set would remove the need for `--merge-subtokens` and eliminate
   the composite XPOS tags it produces.
2. **`punctuation.py` defines only `TOKENIZER_SUFFIXES`** — no `TOKENIZER_PREFIXES`, no
   `TOKENIZER_INFIXES`. ZWNJ (U+200C) is handled only implicitly through the 65 KB generated
   verb-exception table. The missing infix rules bite on numerics: `spacy/lang/fa/examples.py`
   ships the sentence `دیروز علی به من ۲۰۰۰.۱﷼ پول نقد داد.`, and the trained pipeline splits
   `۲۰۰۰.۱﷼` into `۲۰۰۰` + `.` + `۱﷼`, with the stray `.` promoted to a sentence boundary —
   one input sentence comes out as three. `LIKE_NUM` in `lex_attrs.py` recognises Persian
   digits, but no tokenizer rule keeps a Persian decimal or a currency sign attached.
3. **No tokenizer tests at all** for `fa` — `spacy/tests/lang/fa/` contains only
   `test_noun_chunks.py`, guarding none of the 65 KB exception table.
4. **`spacy-lookups-data` has no `fa_license.txt`**, although `fa_source.txt` records that the
   lemma tables were "extracted from Mojgan Seraji's Persian Universal Dependencies Corpus" —
   which is CC BY-SA 4.0. Catalan ships a `ca_license.txt`; Persian should too.
5. **No `fa_lemma_lookup.json`** — only rule-mode lemmatizer assets exist, so
   `mode="lookup"` is unavailable for Persian.

Items 1–3 are self-contained code PRs. Item 4 is a licence-hygiene PR against
`spacy-lookups-data` and matters for anyone redistributing a Persian pipeline.
