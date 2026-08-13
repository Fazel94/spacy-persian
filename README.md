# Persian (Farsi) pipelines for spaCy

Trained spaCy pipelines for Persian, installable now. spaCy has never shipped an official one, and `spacy.blank("fa")` only gives you a tokenizer and stop words. Choose between `fa_core_news_sm` (full syntax + NER) or `fa_dep_news_sm` (syntax only).

```bash
pip install https://huggingface.co/Phazel/fa_core_news_sm/resolve/main/fa_core_news_sm-3.8.0-py3-none-any.whl
```

```python
>>> import spacy
>>> nlp = spacy.load("fa_core_news_sm")

>>> doc = nlp("محمدرضا شجریان در مشهد به دنیا آمد.")
>>> [(t.text, t.pos_, t.lemma_, t.dep_) for t in doc][:2]
[('محمدرضا', 'PROPN', 'محمدرضا', 'nsubj'), ('شجریان', 'PROPN', 'شجریان', 'flat:name')]
>>> doc.ents
(محمدرضا شجریان, مشهد)

>>> doc = nlp("شرکت ایران خودرو تولید را ۲۰ درصد افزایش می‌دهد.")
>>> [(e.text, e.label_) for e in doc.ents]
[('ایران خودرو', 'ORG'), ('۲۰ درصد', 'PCT')]
```

## Results

Compared against Hazm (the most-used Persian toolkit) and `en_core_web_sm` (English reference).

| Metric | **`spacy-persian`**<br>`fa_core_news_trf` | **Hazm**<br>(Persian toolkit) | `en_core_web_sm`<br>(English reference) |
|--------|:---:|:---:|:---:|
| **POS Accuracy (UPOS)** | **97.63%** | ~95.69%¹ | 97.21%² |
| **Lemma Accuracy** | **97.31%** | 89.9%¹ | — |
| **Dependency LAS** | **90.79%** | 85.6%¹ | 91.85%² |
| **NER F-score** | **82.89%** | — | 83.80%² |

> **¹** Hazm scores from its official README 
> **²** `en_core_web_sm` scores from spaCy's official model card

> **Note on comparability:** These benchmarks come from *different evaluation sets, treebanks, and test splits*.


From `spacy benchmark accuracy`, stored in `metrics/`.
| Package | Components | Licence | Score | Wheel |
| --- | --- | --- | --- | --- |
| [`fa_dep_news_sm`](https://huggingface.co/Phazel/fa_dep_news_sm) | tok2vec, tagger, morphologizer, trainable_lemmatizer, parser | CC BY-SA 4.0 | LEMMA 97.91 | 7.9 MB |
| [`fa_core_news_sm`](https://huggingface.co/Phazel/fa_core_news_sm) | the above plus ner | CC BY-SA 4.0 | ENTS_F 71.87 | 13.5 MB |
| `fa_ent_news_sm` | `ner` alone (own embedded tok2vec) | CC BY-SA 4.0 | ENTS_F 71.87 | 5.9 MB |
| `fa_dep_news_md` | same as `fa_dep_news_sm`, plus floret vectors | CC BY-SA 4.0 | LEMMA 97.96 | 62.6 MB |
| `fa_core_news_md` | same as `fa_core_news_sm`, plus floret vectors | CC BY-SA 4.0 | ENTS_F 74.71 | 68.5 MB |
| [`fa_ent_news_md`](https://huggingface.co/Phazel/fa_ent_news_md) | `ner` alone (own embedded tok2vec), plus floret vectors | CC BY-SA 4.0 | ENTS_F 74.71 | 60.6 MB |
| [`fa_dep_news_lg`](https://huggingface.co/Phazel/fa_dep_news_lg) | same as `fa_dep_news_sm`, plus full-wiki floret vectors | CC BY-SA 4.0 | LEMMA 98.08 | 229.3 MB |
| [`fa_core_news_lg`](https://huggingface.co/Phazel/fa_core_news_lg) | same as `fa_core_news_sm`, plus full-wiki floret vectors | CC BY-SA 4.0 | ENTS_F 75.94 | 235.2 MB |
| [`fa_ent_news_lg`](https://huggingface.co/Phazel/fa_ent_news_lg) | `ner` alone (own embedded tok2vec), plus full-wiki floret vectors | CC BY-SA 4.0 | ENTS_F 75.94 | 227.3 MB |
| [`fa_core_news_trf`](https://huggingface.co/Phazel/fa_core_news_trf) | transformer, tagger, morphologizer, trainable_lemmatizer, parser, ner | see §8, encoder unlicensed | ENTS_F 82.89, LAS 90.79 | 608.2 MB |

`fa_ent_news_sm`, `fa_dep_news_md` and `fa_core_news_md` are built by `project.yml` but not
published yet. Standalone floret vector packages:
[`fa_floret_400k`](https://huggingface.co/Phazel/fa_floret_400k),
[`fa_floret_full_wiki`](https://huggingface.co/Phazel/fa_floret_full_wiki).

The `md` tier adds a 50k x 300d floret vector table trained on 400k Persian documents. Its
config differs from `sm` by exactly one line (`include_static_vectors`), so the columns below
isolate what the vectors buy. Full breakdown in `docs/MODELS.md` §6.

| Metric | `sm` | `md` | `lg` | `trf` | Reference |
| --- | --- | --- | --- | --- | --- |
| `TOKEN_ACC` / `TOKEN_F` | 99.96 / 99.11 | 99.96 / 99.11 | 99.96 / 99.11 | 99.96 / 99.11 | |
| `TAG_ACC` (XPOS) | 95.96 | 96.25 | 96.55 | **97.62** | |
| `POS_ACC` (UPOS) | 96.24 | 96.64 | 96.68 | **97.63** | |
| `MORPH_ACC` | 96.29 | 96.64 | 96.70 | **97.82** | |
| `LEMMA_ACC` | 97.91 | 97.96 | **98.08** | 97.31 | |
| `SENTS_F` | 99.25 | **99.28** | 99.18 | 97.35 | |
| `DEP_UAS` | 89.69 | 90.52 | 90.96 | **93.87** | hazm+ParsBERT: 92.46 |
| `DEP_LAS` | 85.15 | 86.34 | 86.60 | **90.79** | hazm+ParsBERT: 89.34 |
| `ENTS_P` | 77.67 | 76.56 | 81.51 | **84.06** | |
| `ENTS_R` | 66.87 | 72.95 | 71.09 | **81.76** | |
| `ENTS_F` | 71.87 | 74.71 | 75.94 | **82.89** | |
| Speed (940MX, batch 32) | 10,235 words/s | 9,058 words/s | 9,215 words/s | see §Throughput | |
| Wheel size | 13.5 MB | 68.5 MB | 235 MB | 608 MB | |

`trf` leads everywhere except lemmatization and sentence segmentation, and is the only tier to
pass the hazm+ParsBERT `DEP_LAS` reference of 89.34. It needs a GPU, and its encoder states no
licence so it is not redistributable (`docs/MODELS.md` §8).

Entity scores are `fa_core_news_*` on the PerDT NER test split; per-label breakdown and
caveats are in [Named entity recognition](#named-entity-recognition).


For comparison, `en_core_web_sm` scores TAG 97, LAS 90, ENTS_F 84 on a larger, cleaner corpus.
Trained on a 4-core i5-7200U with no GPU: `sm` 1h27m syntax + 17 min NER, `md` 1h54m syntax
+ 25 min NER (the two `md` runs overlapped, so wall clock overstates each).

## Throughput

Median of repeated `nlp.pipe` passes over the 146-document PerDT test split (23,825 tokens),
timing the pipe only, warmup discarded. Reproduce with
`python scripts/benchmark_throughput.py <model> --gpu-id <n>`; raw records are in
`metrics/throughput-*.json`.

| Tier | CPU, i5-7200U | GPU, GeForce 940MX | GPU, Tesla T4 |
| --- | ---: | ---: | ---: |
| `sm` | 5,484 | 10,235 | |
| `md` | 5,408 | 9,058 | |
| `lg` | 4,715 | 9,215 | |
| `trf` | 187 | 1,158 | 8,320 |

`trf` is 29x slower than `sm` on the same CPU. The T4 and Xeon figures come from one Colab VM,
a 25x GPU speedup. The CPU tiers sit within 15% of each other, so the bottleneck is the parser
and lemmatizer, not the tok2vec lookup. Laptop spread is about 10% with thermal state. Running
`trf` on the 940MX needs a specific torch build, see `docs/MODELS.md` §9.

## Named entity recognition

Seven labels: `LOC`, `PER`, `ORG`, `DAT`, `MON`, `TIM`, `PCT`. They come from PerDT's own
`not-to-release/Dadegan with NER tag/` layer, transferred onto this pipeline's tokenization
by difflib at a 99.86% alignment rate; spans that could not be aligned exactly were dropped
rather than guessed (`scripts/transfer_perdt_ner.py`). That layer is silver: PerDT's README
states it was produced by the BERT-based Beheshti-NER tagger with manual corrections for
recall, so the `ENTS_F` numbers below partly reflect agreement with that tagger, not with
human annotation.

`ner` runs standalone with its own embedded tok2vec (`fa_ent_news_sm`, `fa_ent_news_md`), or
bundled into `fa_core_news_sm`/`fa_core_news_md` alongside the syntax pipeline. In `trf` it is
trained jointly against the shared transformer instead, so there is no standalone trf variant.

| Label | `sm` F | `md` F | `lg` F | `trf` F | Train examples |
| --- | --- | --- | --- | --- | --- |
| `LOC` | 80.24 | 84.05 | 83.66 | **87.78** | 4,954 |
| `PER` | 65.29 | 68.18 | 72.63 | **81.88** | 4,847 |
| `ORG` | 68.77 | 70.25 | 71.01 | **78.50** | 2,643 |
| `DAT` | 74.45 | 76.19 | 70.83 | **82.52** | 1,323 |
| `MON` | 73.68 | 84.21 | 88.89 | 88.89 | 205 |
| `TIM` | 66.67 | 66.67 | 61.54 | 50.00 | 135 |
| `PCT` | 57.14 | 33.33 | 57.14 | 33.33 | 121 |

`MON`, `TIM` and `PCT` have single-digit support in the test split, so their deltas are one or
two entities changing hands, not signal. `PER`, `LOC` and `ORG` carry the split. The `md` gain
over `sm` (`ENTS_F` 71.87 to 74.71) is almost entirely recall (+6.08), the lexical prior static
vectors give rare proper nouns that hash embeddings never had. `trf` adds another +6.95 F over
`lg`, again mostly recall (71.09 to 81.76), and its largest per-label gains are `PER` (+9.25)
and `DAT` (+11.69).


## Install

```bash
pip install https://huggingface.co/Phazel/fa_core_news_sm/resolve/main/fa_core_news_sm-3.8.0-py3-none-any.whl
# or, without NER:
pip install https://huggingface.co/Phazel/fa_dep_news_sm/resolve/main/fa_dep_news_sm-3.8.0-py3-none-any.whl
```

## Caveats

- **Some lemmas contain a space.** Multiword tokens were merged, so `کتاب‌هایش` is one token
  tagged `N_IANM_PR_JOPER` with lemma `کتاب او`. This affects about 1.5% of tokens.
- **`doc.noun_chunks` under-fires.** `spacy/lang/fa/syntax_iterators.py` upstream matches
  ClearNLP labels that do not exist in Universal Dependencies. Patch in
  [`docs/upstream/fa-noun-chunks.md`](docs/upstream/fa-noun-chunks.md).

## Build

Everything is reproducible from checksummed assets. Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install "spacy>=3.8,<3.9" spacy-lookups-data

.venv/bin/python -m spacy project assets      # download + checksum the corpora
.venv/bin/python -m spacy project run all     # -> fa_dep_news_sm + fa_core_news_sm
.venv/bin/python -m spacy project run ent     # -> fa_ent_news_sm, NER alone
```

| Command | What it does |
| --- | --- |
| `inspect` | annotation coverage of the treebanks (`scripts/inspect_treebanks.py`) |
| `convert-ud` | CoNLL-U to `DocBin` with `--merge-subtokens`, plus the tokenizer-agreement report |
| `transfer-ner` | align PerDT's NER layer onto that tokenization by difflib (`scripts/transfer_perdt_ner.py`) |
| `convert-ner` | transferred IOB2 to `DocBin` |
| `debug-data`, `debug-data-ner` | `spacy debug data` on both corpora before spending CPU |
| `train-dep` | tagger + morphologizer + trainable_lemmatizer + parser |
| `train-ner` | the `ner` component, with its own embedded tok2vec |
| `finalize-dep` | write `fa_dep_news_sm` metadata: sources, licence, notes (`scripts/finalize_pipeline.py`) |
| `evaluate-dep` | `spacy benchmark accuracy` on the held-out UD test split |
| `assemble-core` | source `ner` into the dep pipeline to produce `fa_core_news_sm` |
| `evaluate-core` | score the assembled pipeline on both test splits |
| `finalize-meta` | re-run finalize on both, folding test scores into `meta.json["performance"]` |
| `package` | build wheels + sdists for both |
| `smoke` | run both pipelines over Persian text and print every annotation layer |

The two training runs are single-threaded and independent, so they can run concurrently.

## Design decisions

1. `--merge-subtokens`. spaCy has no multiword-token layer, and PerDT splits pronominal clitics
   (`پدرم` into `پدر` + `م`). Measured on dev, merging gives token F 0.9887 against 0.9823 for
   the split version, costing 34 composite XPOS tags on 1.5% of tokens. Without it, 1.5% of gold
   tokens are boundaries the shipped tokenizer can never produce. See
   `scripts/tokenization_report.py`.
2. `ner` carries its own tok2vec. A `Tok2VecListener` only resolves inside the pipeline it was
   trained in, so a listener-based component cannot be sourced elsewhere.
   `configs/fa_ner_sm.cfg` embeds the tok2vec instead, as `en_core_web_sm` does.
3. `morphologizer` + `trainable_lemmatizer` instead of `attribute_ruler` + rule lemmatizer. The
   English pipelines derive UPOS from PTB tags by rule because OntoNotes has no UPOS. UD gives
   gold UPOS, FEATS and lemmas, which yields real `pos_acc`, `morph_acc` and `lemma_acc` numbers
   instead of unmeasurable rule coverage.
4. PerDT, not Seraji: 3.7x more tokens, and Seraji has no `PROPN` tag.

## Why not hazm's own models

hazm is the reference Persian NLP toolkit and publishes spaCy-format pipelines on the HF Hub,
so it was the obvious starting point. Four problems:

- Its trainable models are pycrfsuite CRFs (`hazm/sequence_tagger.py`). The repo contains no
  `config.cfg` and no `spacy train`; the `Spacy*` classes only download pretrained pipelines.
- Those pipelines are three single-task models (`transformer + tagger`, `transformer + parser`,
  `transformer + chunker`), each `version: 0.0.0` with an empty `license` field, pinned to
  spaCy 3.6. Using all three costs three ParsBERT forward passes and gives no shared `Doc`.
- Its tokenizer is incompatible with UD tokenization: the normaliser fuses ZWNJ affixes and
  `join_verb_parts()` glues multi-word verb chains into single tokens.
- Most corpora it reads (Bijankhan, Peykare, Hamshahri, raw PerDT) sit behind `peykaregan.ir`
  or `dadegan.ir` under research-only terms.

It did confirm the corpus choice. hazm's own spaCy parser was trained on
`modified_fa_perdt-ud-train.spacy`, the same treebank used here.

## More

- Pipeline inventory, corpus and licence analysis: [`docs/MODELS.md`](docs/MODELS.md)
- How spaCy models get published, and what upstream `fa` already has:
  [`docs/CONTRIBUTING-GUIDE.md`](docs/CONTRIBUTING-GUIDE.md)
- The build: [`project.yml`](project.yml)
- خلاصهٔ فارسی: [`README.fa.md`](README.fa.md)
- Language data comes from `spacy/lang/fa` upstream, whose stop word list came from hazm.
