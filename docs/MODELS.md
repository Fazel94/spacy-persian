# The Persian pipelines: what to build, from what, and why

## 1. What "the 4 English pipelines" actually are

They are **one pipeline design at four embedding budgets**, not four different products.
Every one of them has the same components; the only real axis of variation is where token
representations come from.

| | `en_core_web_sm` | `en_core_web_md` | `en_core_web_lg` | `en_core_web_trf` |
| --- | --- | --- | --- | --- |
| Size on disk | 12 MB | 31 MB | 382 MB | 436 MB |
| Embeddings | hash embeddings only | 685k keys / 20k vectors (300d) | 685k keys / 343k vectors (300d) | `roberta-base`, 768d contextual |
| Components | tok2vec, tagger, parser, senter, attribute_ruler, lemmatizer, ner | same | same | **transformer**, tagger, parser, attribute_ruler, lemmatizer, ner |
| `TAG_ACC` | 0.97 | 0.97 | 0.97 | 0.98 |
| `DEP_UAS` / `LAS` | 0.92 / 0.90 | 0.92 / 0.90 | 0.92 / 0.90 | 0.95 / 0.94 |
| `ENTS_F` | 0.84 | 0.85 | 0.86 | 0.90 |
| Training data | OntoNotes 5 (+ ClearNLP dep conversion, WordNet 3.0) | + Explosion vectors (OSCAR 2109 + Wikipedia + OpenSubtitles + WMT News Crawl) | same | OntoNotes 5 + roberta-base |

Read the accuracy table honestly: **static vectors buy almost nothing for tagging and
parsing** (identical to two decimals) and ~1–2 F on NER. The transformer buys ~4 LAS and
~6 NER F, at 36× the size and a GPU requirement. That ordering dictates the roadmap below.

Sources: <https://spacy.io/models/en>.

## 2. Target: the Persian pipelines

Naming follows `[lang]_[type]_[genre]_[size]` (<https://spacy.io/models#conventions>), and the
`type` slot is load-bearing: `dep` = tagger + parser + lemmatizer, `ent` = NER only,
`core` = both. Genre is `news`, after the dominant genre of UD_Persian-PerDT (its README lists
"news fiction nonfiction academic web blog", and spaCy labels comparable treebank-trained
pipelines such as `de_core_news_sm` as `news`).

| Pipeline | Components | Embeddings | Status |
| --- | --- | --- | --- |
| **`fa_dep_news_sm`** | tok2vec, tagger, morphologizer, trainable_lemmatizer, parser | hash embeddings | **built — the shipping artifact** |
| `fa_ent_news_sm` | ner (own internal tok2vec) | hash embeddings | **built — optional, separate package** |
| `fa_core_news_sm` | the two above, merged | hash embeddings | **reserved.** Blocked on prose-genre NER data from `../ner_dataset` |
| `fa_core_news_md` | + static vectors | floret, 50k rows | vectors must be trained first (CPU-days on fa Wikipedia + OSCAR) |
| `fa_core_news_lg` | same | floret, 200k rows | same as md, bigger table |
| `fa_core_news_trf` | transformer instead of tok2vec | `HooshvareLab/roberta-fa-zwnj-base` (Apache-2.0) | **not on this hardware** — 2 GB VRAM cannot fine-tune a 125M-param encoder |

**Why `dep` + `ent` rather than a single `core`.** `core` is a promise that the NER is part of
the same pipeline, built to the same standard, versioned together. Ours is not: the UD
components score 85–98 on edited prose, while the NER scores 67.22 F and is trained on tweets
because the good Persian NER corpora are research-use-only. One package name and one version
number would paper over a gap of that size. Shipping two packages makes the user opt into the
weak component knowingly, and costs nothing technically — `fa_ent_news_sm` embeds its own
tok2vec, so `nlp.add_pipe("ner", source=...)` reassembles a `core`-equivalent pipeline at
runtime (verified). `fa_core_news_sm` gets published when the NER earns the name.

One deviation from the English design, deliberate: Persian gets a **`morphologizer`**
(UPOS + morphological features) and a **`trainable_lemmatizer`** instead of English's
`attribute_ruler` + rule `lemmatizer`. Reasons:

- The English pipelines use `attribute_ruler` because OntoNotes gives them PTB `tag`s and
  they *derive* UPOS from tags by rule. UD treebanks give UPOS and FEATS as gold data —
  training a morphologizer on them is strictly more information, and Persian morphology
  (Number, Person, Tense, Mood, Voice, Polarity, PronType) is worth predicting.
- Persian rule-lemmatizer tables *do* exist in `spacy-lookups-data` (`fa_lemma_exc.json`
  1.68 MB, `fa_lemma_index.json`, `fa_lemma_rules.json`, derived from Seraji's treebank), but
  a rule lemmatizer's accuracy is unmeasurable against the corpus it was extracted from and it
  needs `token.pos` to work at all. `trainable_lemmatizer` learns edit trees from PerDT's gold
  lemmas and reports a real `lemma_acc`. PerDT yields **1,908 edit trees** with 100% lemma
  coverage — plenty.

`senter` is intentionally omitted from v1: it is a separately trained component that ships
*disabled by default* in the English pipelines, and the parser already produces sentence
boundaries. Adding it later requires only one extra training run.

## 3. Resource inventory (everything checked for license)

### 3.1 Treebanks — the tagger / morphologizer / lemmatizer / parser data

| Treebank | Sents | Tokens | License | LEMMA | FEATS | XPOS | PROPN? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **UD_Persian-PerDT** (PerUDT v1.0) | 29,107 | ~509k | **CC BY-SA 4.0** | converted + corrections | converted + corrections | manual native (34 types) | **yes** | **CHOSEN** |
| UD_Persian-Seraji | 5,997 | ~152k | CC BY-SA 4.0 | manual native | manual native | manual native (30 types) | **no** | secondary / cross-eval |
| UD_Persian-PUD | 1,000 | — | CC BY-SA 4.0 | — | — | none | — | test-only, parallel corpus |
| UD_Persian-IPerUDT | tiny | — | CC BY-SA 4.0 | — | — | none | — | grammar examples, unusable |

Measured locally with `scripts/inspect_treebanks.py` (train splits):

```
file                             sents    tokens    MWT  empty  lemma%  feats%  UPOS  XPOS  DEP
fa_perdt-ud-train.conllu         26196    452496   6508      0   100.0    57.7    16    34   34
fa_seraji-ud-train.conllu         4798    121067   1117      0   100.0    65.0    15    30   39
```

Why PerDT over Seraji:

1. **3.7× more training tokens** (452k vs 121k). At `sm` size, data is the binding constraint.
2. **Seraji has no `PROPN`** — proper nouns are tagged `NOUN`. A pipeline that cannot mark
   proper nouns is crippled for exactly the downstream tasks people use spaCy for.
3. hazm independently chose PerDT for its own spaCy dependency parser — its
  `config.cfg` names `modified_fa_perdt-ud-train.spacy` as the training set. Converging on
  the same corpus makes our numbers comparable to theirs.

Seraji's advantages (richer manual FEATS, fully manual lemmas) are real; it is the natural
cross-evaluation set and a candidate for a future concatenated-corpus run. The two use
different XPOS inventories, so naive concatenation would corrupt the `tag` label space.

### 3.2 NER — the one place with a licensing minefield

| Dataset | Labels | Size | License | Usable? |
| --- | --- | --- | --- | --- |
| **ParsTwiNER** | PER, ORG, LOC, NAT, POG, EVENT | 7,667 tweets / 233k tokens | **MIT** (verified via GitHub API on `overfit-ir/parstwiner`) | **CHOSEN** |
| ARMAN (PersianNER) | 6 classes | 250k tokens | academic research only | no |
| PEYMA | 7 classes | 302k tokens | "free for research purposes", no OSS licence | no |
| NSURL-2019 Task 7 | PEYMA tagset | ~1M tokens | no explicit licence | no |
| HooshvareLab merged ParsNER | 10 classes | ARMAN+PEYMA+WikiANN | inherits ARMAN/PEYMA restrictions; HF repo gated (401) | no |

This is not pedantry. **spaCy's own maintainers state that Persian models trained back in
2018 were never published precisely because of corpus licensing** (spaCy discussion #8233,
following PR #2797 which added only `spacy.blank("fa")` tokenizer support). Repeating that
mistake would waste the whole exercise: a pipeline you cannot legally redistribute is not a
pipeline, it is a local file.

Consequence to state plainly in the model card: **the NER component is trained on Twitter
text while the rest of the pipeline is trained on edited prose.** Expect NER to degrade on
formal news text relative to what an ARMAN/PEYMA-trained model would score. That is the
price of a redistributable artifact.

### 3.3 Vectors (for md / lg)

| Option | License | Note |
| --- | --- | --- |
| **floret vectors** trained via `spacy-vectors-builder` (MIT tooling) on fa Wikipedia + OSCAR | corpus-dependent | **recommended.** Subword + Bloom-hash embeddings: bounded table size, zero OOV |
| fastText `cc.fa.300` | CC BY-SA 3.0 | quick fallback; classic word table, large and OOV-prone |

Floret is the right call for Persian specifically. Persian surface forms explode through
suffixation *and* through inconsistent ZWNJ (U+200C) usage — the same word appears as
`می‌رود` / `میرود` / `می رود` in real text. A classic word-vector table has a miss for every
variant; floret's subword hashing covers all of them. spaCy already ships floret vectors for
Croatian, Finnish, Korean, Slovenian, Swedish and Ukrainian for the same reason.

### 3.4 Transformer encoders (for trf)

| Model | Arch | License | Note |
| --- | --- | --- | --- |
| **`HooshvareLab/roberta-fa-zwnj-base`** | RoBERTa-base | **Apache-2.0** | recommended: licensed, ZWNJ-aware, smallest of the credible options |
| `FacebookAI/xlm-roberta-base` | XLM-R base, 278M | MIT | licensed but larger |
| `PartAI/TookaBERT-Base` | BERT-base | Apache-2.0 | licensed |
| `m3hrdadfi/albert-fa-base-v2` | ALBERT-base-v2 | Apache-2.0 | licensed, smallest |
| `HooshvareLab/bert-base-parsbert-uncased` | BERT-base, ~162M | **no licence on the card** | what hazm used; redistribution risk |
| `sbunlp/fabert` | BERT-base, 124M | **no licence on the card** | redistribution risk |

All of these are BERT/RoBERTa/XLM-R/ALBERT, so all are supported by both
`spacy-transformers` and `spacy-curated-transformers` (the latter supports exactly
ALBERT / BERT / CamemBERT / RoBERTa / XLM-RoBERTa).

Note the trap: hazm's own pipelines use ParsBERT, which has **no license statement**. Copying
that choice would reintroduce the redistribution problem that sank the 2018 attempt.

## 4. What already exists (and why it is not enough)

- **No trained spaCy Persian pipeline exists.** Zero `persian` / `farsi` / `fa_` hits in
  spaCy's `website/meta/universe.json`. `spacy.load("fa_core_news_sm")` has never worked.
- **hazm** ships three *single-task* spaCy pipelines on the HF Hub:
  `hazm-parsbert-postagger` (`tag_acc` 0.9862, hazm's own EZ-augmented tagset),
  `hazm-bert-dependency-parser` (`dep_uas` 0.9246 / `dep_las` 0.8934, trained on PerDT),
  `hazm-parsbert-chunker` (`tag_acc` 0.9618). Each is `transformer + one component`,
  `version: 0.0.0`, empty `license`/`author`/`sources`, pinned to spaCy 3.6. Using all three
  means three separate BERT forward passes over the same text and no shared `Doc`.
- **hazm's training code is not reusable.** Its trainable models are pycrfsuite CRFs
  (`hazm/sequence_tagger.py`); there is no `config.cfg` or `spacy train` anywhere in the repo.
  Its `Spacy*` classes only *download* the HF pipelines above.
- **hazm's tokenizer is actively incompatible** with UD gold tokenization: `Normalizer`'s
  `AFFIX_SPACING_PATTERNS` fuse ZWNJ affixes and `WordTokenizer.join_verb_parts()` glues
  multi-word verb chains into single underscore-joined tokens. Training against PerDT with
  hazm's tokenizer would misalign tokens systematically.
- **DadmaTools** (Apache-2.0 code) emits spaCy-compatible `Doc` objects but is not a loadable
  spaCy pipeline package, and its NER wraps ARMAN/PEYMA — the restricted data again.

So what hazm genuinely contributes to this project is **one validated design decision**
(PerDT is the corpus) and **the stop-word list already vendored into `spacy/lang/fa`**.
Everything else is built with spaCy-native tooling.

## 5. Decision: train the `sm` tier first

Not md/lg: those need floret vectors trained from scratch on Wikipedia+OSCAR (CPU-days) and
buy ~0.00 tag/dep accuracy in the English reference numbers.
Not trf: 2 GB of VRAM cannot fine-tune a 125M-param encoder, and renting a GPU should wait
until the CPU pipeline proves the data plumbing is right.
`sm` is also the tier every other tier is validated against: md/lg/trf reuse the exact same
corpus conversion, config skeleton and evaluation harness.

### Tokenization decision, settled with a measurement

spaCy has no multi-word-token layer, so CoNLL-U MWT ranges (Persian pronominal clitics and
copulas: `پدرم` = `پدر` + `م`) must either be merged into one token or kept split. Measured on
the PerDT dev set with `scripts/tokenization_report.py`, comparing gold boundaries against
`spacy.blank("fa")`'s tokenizer:

| `spacy convert` mode | token P | token R | token F | XPOS types | merge artefacts |
| --- | --- | --- | --- | --- | --- |
| `--merge-subtokens` | 0.9860 | 0.9914 | **0.9887** | 68 | 34 composite tags on 1.49% of tokens; 1.49% lemmas contain a space |
| plain (clitics split) | 0.9875 | 0.9772 | 0.9823 | 35 | none |

**Chosen: `--merge-subtokens`** (also what Explosion's `tagger_parser_ud` template does).
Rationale: without it, 1.5% of gold tokens are boundaries the shipped tokenizer can never
produce, so those tokens are permanently unlearnable and unpredictable at runtime. With it,
every gold token is reachable; the cost is confined to rare composite XPOS tags such as
`N_IANM_PR_JOPER` (noun + enclitic pronoun) — which are, at least, informative — and 1.5% of
lemmas that come out as two words.

The better long-term fix is Persian clitic-splitting suffix rules in `spacy/lang/fa`, which
would be an upstream PR, not a model change. Recorded in `docs/CONTRIBUTING-GUIDE.md` §5.

### Final composition

**`fa_dep_news_sm`** — CC BY-SA 4.0, 7.5 MB wheel:

| Component | Trained on | Metric | Test score |
| --- | --- | --- | --- |
| `tok2vec` | shared, PerDT | — | — |
| `tagger` (XPOS) | PerDT, 90 labels | `tag_acc` | 95.96 |
| `morphologizer` (UPOS + FEATS) | PerDT, 298 labels | `pos_acc` / `morph_acc` | 96.24 / 96.29 |
| `trainable_lemmatizer` | PerDT, 1,908 edit trees | `lemma_acc` | 97.91 |
| `parser` | PerDT, 34 deprels | `dep_uas` / `dep_las` | 89.69 / 85.15 |

**`fa_ent_news_sm`** — MIT, 5.6 MB wheel, separate package:

| Component | Trained on | Metric | Test score |
| --- | --- | --- | --- |
| `ner` (own internal tok2vec) | ParsTwiNER, 6 labels | `ents_p/r/f` | 74.77 / 61.06 / **67.22** |

ParsTwiNER's label counts explain that last row better than any prose can:
`PER` 6258, `LOC` 5478, `ORG` 2694, `NAT` 939, **`EVE` 482, `POG` 399** over 232,917 tokens
(16,250 entities, 7.0% density). The two starved labels are exactly the two that score worst
(`EVE` 30.0, `POG` 41.2). So the corpus is not small — it is skewed and out of genre, which
are two different problems: the head labels need in-genre data, the tail labels need more data
of any kind.

Training cost on the target hardware (4-core i5-7200U, no GPU): **1h27m** for the UD
components (early-stopped at step 10,800; best checkpoint step 9,200) and ~25 min for NER
(best at step 4,000). Both runs are single-threaded, so they were run concurrently.
Inference: ~9,250 words/s.

The `--merge-subtokens` artefacts predicted above are visible in the shipped model exactly as
expected — `کتاب‌هایش` ("his/her books") comes out as one token tagged `N_IANM_PR_JOPER` with
lemma `کتاب او`. Worth knowing before you consume `token.lemma_` downstream.

Sources recorded in each `meta.json` with their licences, per
<https://spacy.io/api/data-formats#meta>. CC BY-SA 4.0 on PerDT means `fa_dep_news_sm` must
carry attribution and share-alike notice — handled in `scripts/finalize_pipeline.py`, which
also refuses to publish a `dep` pipeline that contains an `ner` component.
