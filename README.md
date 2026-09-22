# Persian (Farsi) pipelines for spaCy

Trained spaCy pipelines for Persian, installable with pip. spaCy has never shipped an official one, and `spacy.blank("fa")` only gives you a tokenizer and stop words. Ten packages across four tiers: `sm` (hash embeddings), `md` and `lg` (floret static vectors), `trf` (fine-tuned ParsBERT). Each of `sm`/`md`/`lg` ships as syntax only (`fa_dep_news_*`), syntax + NER (`fa_core_news_*`), or NER alone (`fa_ent_news_*`); `trf` ships as `fa_core_news_trf` only.

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

Compared against Hazm (<https://github.com/roshan-research/hazm>) and `en_core_web_sm`
(English reference).

| Metric | **`spacy-persian`**<br>`fa_core_news_trf` | **Hazm**<br>(Persian toolkit) | `en_core_web_sm`<br>(English reference) |
|--------|:---:|:---:|:---:|
| POS accuracy | **97.63%** UPOS | 98.8% own tagset¹ | 97.29% PTB XPOS² |
| Lemma accuracy | **97.31%** | 89.9%¹ | not reported² |
| Dependency UAS / LAS | **93.87% / 90.79%** | 92.30% / 89.15%¹ | 91.77% / 89.92%² |
| NER F-score | **82.89%** | not reported¹ | 84.33%² |

> **¹** Hazm's own README, <https://github.com/roshan-research/hazm#evaluation>: `POSTagger`
> 98.8% on Hazm's EZ-augmented tagset, which is not UPOS; `Lemmatizer` 89.9%;
> `SpacyDependencyParser` UAS 92.30 / LAS 89.15. It reports no NER score.
>
> **²** `en_core_web_sm` 3.8.0 `meta.json`,
> <https://github.com/explosion/spacy-models/blob/master/meta/en_core_web_sm-3.8.0.json>:
> `tag_acc` 0.9729, `dep_uas` 0.9177, `dep_las` 0.8992, `ents_f` 0.8433. That pipeline
> reports no `pos_acc` and no `lemma_acc`, because OntoNotes has no UPOS or lemma layer.
>
> **Note on comparability:** these benchmarks come from different evaluation sets, treebanks,
> and test splits.

### Packages

| Package | Components | Licence | Score | Wheel |
| --- | --- | --- | --- | --- |
| [`fa_dep_news_sm`](https://huggingface.co/Phazel/fa_dep_news_sm) | tok2vec, tagger, morphologizer, trainable_lemmatizer, parser | CC BY-SA 4.0 | LEMMA 97.91 | 7.9 MB |
| [`fa_core_news_sm`](https://huggingface.co/Phazel/fa_core_news_sm) | the above plus ner | CC BY-SA 4.0 | ENTS_F 71.87 | 13.5 MB |
| [`fa_ent_news_sm`](https://huggingface.co/Phazel/fa_ent_news_sm) | `ner` alone (own embedded tok2vec) | CC BY-SA 4.0 | ENTS_F 71.87 | 5.9 MB |
| [`fa_dep_news_md`](https://huggingface.co/Phazel/fa_dep_news_md) | same as `fa_dep_news_sm`, plus floret vectors | CC BY-SA 4.0 | LEMMA 97.96 | 62.6 MB |
| [`fa_core_news_md`](https://huggingface.co/Phazel/fa_core_news_md) | same as `fa_core_news_sm`, plus floret vectors | CC BY-SA 4.0 | ENTS_F 74.71 | 68.5 MB |
| [`fa_ent_news_md`](https://huggingface.co/Phazel/fa_ent_news_md) | `ner` alone (own embedded tok2vec), plus floret vectors | CC BY-SA 4.0 | ENTS_F 74.71 | 60.6 MB |
| [`fa_dep_news_lg`](https://huggingface.co/Phazel/fa_dep_news_lg) | same as `fa_dep_news_sm`, plus full-wiki floret vectors | CC BY-SA 4.0 | LEMMA 98.08 | 229.3 MB |
| [`fa_core_news_lg`](https://huggingface.co/Phazel/fa_core_news_lg) | same as `fa_core_news_sm`, plus full-wiki floret vectors | CC BY-SA 4.0 | ENTS_F 75.94 | 235.2 MB |
| [`fa_ent_news_lg`](https://huggingface.co/Phazel/fa_ent_news_lg) | `ner` alone (own embedded tok2vec), plus full-wiki floret vectors | CC BY-SA 4.0 | ENTS_F 75.94 | 227.3 MB |
| [`fa_core_news_trf`](https://huggingface.co/Phazel/fa_core_news_trf) | transformer, tagger, morphologizer, trainable_lemmatizer, parser, ner | encoder states no licence, §8 | ENTS_F 82.89, LAS 90.79 | 608.2 MB |

These scores are from `spacy benchmark accuracy`, stored in `metrics/`.

Raw `fa.floret` and `fa.vec` exports of the `lg` tier's 200k-row table are in
[`fa-floret-wiki-vectors`](https://huggingface.co/Phazel/fa-floret-wiki-vectors).

### Tier comparison

The `md` tier adds a 50k x 300d floret vector table trained on the first 400,000 Persian
Wikipedia articles. Its
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
| `DEP_UAS` | 89.69 | 90.52 | 90.96 | **93.87** | `hazm-bert-dependency-parser`: 92.46 |
| `DEP_LAS` | 85.15 | 86.34 | 86.60 | **90.79** | `hazm-bert-dependency-parser`: 89.34 |
| `ENTS_P` | 77.67 | 76.56 | 81.51 | **84.06** | |
| `ENTS_R` | 66.87 | 72.95 | 71.09 | **81.76** | |
| `ENTS_F` | 71.87 | 74.71 | 75.94 | **82.89** | |
| Speed (940MX, batch 32) | 10,235 words/s | 9,058 words/s | 9,215 words/s | 1,106 words/s | |
| Wheel size | 13.5 MB | 68.5 MB | 235.2 MB | 608.2 MB | |

Reference cells are [`hazm-bert-dependency-parser`](https://huggingface.co/roshan-research/hazm-bert-dependency-parser)'s
own `meta.json`, trained on the same treebank (`docs/MODELS.md` §4).

`trf` leads on every metric except lemmatization and sentence segmentation. It is also the only
tier to clear the `hazm-bert-dependency-parser` `DEP_LAS` reference of 89.34. It needs a GPU in
production (187 words/s on the laptop CPU), and its `HooshvareLab/bert-base-parsbert-uncased`
encoder states no licence, so the published wheel carries a redistribution warning in its
`meta.json` and its terms are unknown (`docs/MODELS.md` §8).

Entity scores are `fa_core_news_*` on the PerDT NER test split; per-label breakdown and
caveats are in [Named entity recognition](#named-entity-recognition).

Trained on a 4-core i5-7200U with no GPU: `sm` took 1h27m for syntax plus 17 min for NER,
`md` 1h54m plus 25 min (the two `md` runs overlapped, so wall clock overstates each), `lg`
about 2h08m plus 13 min. `trf` took 1h58m on a rented Colab T4.

### Vector packages

Standalone floret vector packages (vectors only, `pipeline: []`), usable as
`--paths.vectors` for your own training or as a plain embedding table:

```bash
# 50k rows x 300d, first 400,000 Persian Wikipedia articles (the md tier's table)
pip install https://huggingface.co/Phazel/fa_floret_400k/resolve/main/fa_floret_400k-0.1.0-py3-none-any.whl
# 50k rows x 300d, full Persian Wikipedia dump
pip install https://huggingface.co/Phazel/fa_floret_full_wiki/resolve/main/fa_floret_full_wiki-0.1.0-py3-none-any.whl
# 200k rows x 300d, full Persian Wikipedia dump, 5 epochs (the lg tier's table)
pip install https://huggingface.co/Phazel/fa-floret-wiki-vectors/resolve/main/fa_floret_wiki_200k-0.1.0-py3-none-any.whl
```

## Throughput

Median of repeated `nlp.pipe` passes over the 146-document PerDT test split (23,825 tokens),
timing the pipe only, warmup discarded. Reproduce with
`python scripts/benchmark_throughput.py <model> --gpu-id <n>`; raw records are in
`metrics/throughput-*.json`.

| Tier | CPU, i5-7200U | GPU, GeForce 940MX | CPU, Xeon @ 2.00GHz | GPU, Tesla T4 |
| --- | ---: | ---: | ---: | ---: |
| `sm` | 5,484 | 10,235 | | |
| `md` | 5,408 | 9,058 | | |
| `lg` | 4,715 | 9,215 | | |
| `trf` | 187 | 1,106 | 336 | 8,320 |

`trf` is 29x slower than `sm` on the same CPU. The Xeon and T4 columns come from one Colab VM,
a 25x GPU speedup. The CPU tiers sit within 15% of each other, so the bottleneck is the parser
and lemmatizer, not the tok2vec lookup. Laptop spread is about 10% with thermal state. Running
`trf` on the 940MX needs a `cu126` torch build, see `docs/MODELS.md` §9.

## Named entity recognition

Seven labels: `LOC`, `PER`, `ORG`, `DAT`, `MON`, `TIM`, `PCT`. They come from PerDT's own
`not-to-release/Dadegan with NER tag/` layer, transferred onto this pipeline's tokenization
by difflib at a 99.86% alignment rate (`scripts/transfer_perdt_ner.py`). Spans that could not
be aligned exactly were dropped rather than guessed. That layer is silver: PerDT's README
states it was produced by the BERT-based Beheshti-NER tagger with manual corrections for
recall, so the `ENTS_F` numbers below partly reflect agreement with that tagger, not with
human annotation.

Both that realigned layer and a four-label LLM relabelling of the same sentences
(`annotation/`, guideline v2.2) are published as
[`Phazel/fa-perdt-ner`](https://huggingface.co/datasets/Phazel/fa-perdt-ner), CC BY-SA 4.0,
keyed by PerDT `sent_id`; `spacy project run hub-dataset` rebuilds it. The relabelling is
measured against the silver layer in `docs/MODELS.md` §10 and ships no model yet.

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
# syntax + NER, 13.5 MB
pip install https://huggingface.co/Phazel/fa_core_news_sm/resolve/main/fa_core_news_sm-3.8.0-py3-none-any.whl
# syntax only, 7.9 MB
pip install https://huggingface.co/Phazel/fa_dep_news_sm/resolve/main/fa_dep_news_sm-3.8.0-py3-none-any.whl
# NER only, 5.9 MB
pip install https://huggingface.co/Phazel/fa_ent_news_sm/resolve/main/fa_ent_news_sm-3.8.0-py3-none-any.whl
```

For the vector tiers, swap `sm` for `md` or `lg` in both the repo name and the filename:

```bash
pip install https://huggingface.co/Phazel/fa_core_news_md/resolve/main/fa_core_news_md-3.8.0-py3-none-any.whl
pip install https://huggingface.co/Phazel/fa_core_news_lg/resolve/main/fa_core_news_lg-3.8.0-py3-none-any.whl
```

`fa_core_news_trf` is versioned 1.0.0, not 3.8.0, and needs `spacy-transformers`:

```bash
pip install spacy-transformers
pip install https://huggingface.co/Phazel/fa_core_news_trf/resolve/main/fa_core_news_trf-1.0.0-py3-none-any.whl
```

## Caveats

- **Some lemmas contain a space.** Multiword tokens were merged, so `کتاب‌هایش` is one token
  tagged `N_IANM_PR_JOPER` with lemma `کتاب او`. This affects about 1.5% of tokens.
- **`doc.noun_chunks` under-fires.** `spacy/lang/fa/syntax_iterators.py` upstream matches
  ClearNLP labels that do not exist in Universal Dependencies. Bug analysis and proposed
  upstream patch in [`docs/upstream/fa-noun-chunks.md`](docs/upstream/fa-noun-chunks.md).

## Build

Everything is reproducible from checksummed assets. Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt

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
| `finalize-ent` | write `fa_ent_news_sm` metadata from the standalone NER run |
| `evaluate-ent` | `spacy benchmark accuracy` for the NER-only package |
| `package-ent` | build the `fa_ent_news_sm` wheel + sdist |

The two training runs are single-threaded and independent, so they can run concurrently.

The vector and transformer tiers are separate workflows:

```bash
.venv/bin/python -m spacy project run md      # -> fa_dep_news_md, fa_core_news_md
.venv/bin/python -m spacy project run lg      # -> fa_dep_news_lg, fa_core_news_lg
.venv/bin/python -m spacy project run trf     # -> fa_core_news_trf, GPU only
```

The `md` and `lg` workflows stop at the dep and core packages. `fa_ent_news_lg` is built by
`finalize-ent-lg`, `evaluate-ent-lg` and `package-ent-lg`, which no workflow calls; run them
by name.

`md` and `lg` start by unpacking a floret vector wheel that `spacy project assets` does not
download, because it is built by this project rather than fetched. Put it in the repo root
under the exact filename `project.yml` expects (`vars.floret_wheel`, `vars.floret_lg_wheel`):

```bash
curl -L -o fa_floret-0.1.0-py3-none-any-400k-documents.whl \
  https://huggingface.co/Phazel/fa_floret_400k/resolve/main/fa_floret_400k-0.1.0-py3-none-any.whl
curl -L -o fa_floret-0.1.0-py3-none-any-full-wiki-200k-5epoch.whl \
  https://huggingface.co/Phazel/fa-floret-wiki-vectors/resolve/main/fa_floret_wiki_200k-0.1.0-py3-none-any.whl
```

`trf` additionally needs `spacy-transformers` and a real GPU (`vars.gpu_trf` is `0`); the
940MX needs a `cu126` torch build in a separate venv, see `docs/MODELS.md` §9.

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

## Why not Hazm's own models

Hazm publishes spaCy-format pipelines on the HF Hub, so it was the obvious starting point.
Four problems:

- Its trainable models are pycrfsuite CRFs (`hazm/sequence_tagger.py`). The repo contains no
  `config.cfg` and no `spacy train`; the `Spacy*` classes only download pretrained pipelines.
- Those pipelines are three single-task models (`transformer + tagger`, `transformer + parser`,
  `transformer + chunker`), each `version: 0.0.0` with an empty `license` field, pinned to
  spaCy 3.6. Using all three costs three ParsBERT forward passes and gives no shared `Doc`.
- Its tokenizer is incompatible with UD tokenization: the normaliser fuses ZWNJ affixes and
  `join_verb_parts()` glues multi-word verb chains into single tokens.
- Most corpora it reads (Bijankhan, Peykare, Hamshahri, raw PerDT) sit behind `peykaregan.ir`
  or `dadegan.ir` under research-only terms.

It did confirm the corpus choice. Hazm's own spaCy parser was trained on
`modified_fa_perdt-ud-train.spacy`, the same treebank used here.

## More

- Pipeline inventory, corpus and licence analysis: [`docs/MODELS.md`](docs/MODELS.md)
- How spaCy models get published, and what upstream `fa` already has:
  [`docs/CONTRIBUTING-GUIDE.md`](docs/CONTRIBUTING-GUIDE.md)
- The build: [`project.yml`](project.yml)
- Language data comes from `spacy/lang/fa` upstream, whose stop word list came from Hazm.
- خلاصهٔ فارسی: [`README.fa.md`](README.fa.md)
