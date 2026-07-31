# fa_core_news_sm and fa_dep_news_sm, Persian pipelines for spaCy

spaCy has no trained Persian pipeline. `spacy.load("fa_core_news_sm")` has never worked, and
`spacy.blank("fa")` gives you a tokenizer and stop words. This project trains one from
openly-licensed data so the result can be redistributed.

- Pipeline inventory and source analysis: [`docs/MODELS.md`](docs/MODELS.md)
- How spaCy models get published, and what upstream `fa` already has:
  [`docs/CONTRIBUTING-GUIDE.md`](docs/CONTRIBUTING-GUIDE.md)
- The build: [`project.yml`](project.yml)

## Two packages, one corpus

In spaCy's naming scheme `dep` = tagger + parser + lemmatizer, `core` = the same plus NER.
Both packages here are built entirely from UD_Persian-PerDT and differ only in whether NER is
included.

| Package | Components | Licence | Score | Wheel |
| --- | --- | --- | --- | --- |
| `fa_dep_news_sm` | tok2vec, tagger, morphologizer, trainable_lemmatizer, parser | CC BY-SA 4.0 | LAS 85.15, LEMMA 97.91 | 7.5 MB |
| `fa_core_news_sm` | the above plus ner | CC BY-SA 4.0 | LAS 85.15, ENTS_F 71.87 | 13 MB |

The NER is possible because the treebank ships its own entity layer in
`not-to-release/Dadegan with NER tag/`: 15,833 entities over the same 29,107 sentences, under
the same CC BY-SA 4.0. That is what makes `core` honest here, since one corpus means one genre,
one tokenization, one licence and one provenance chain. The alternative NER corpora are all
worse on at least one of those axes: ARMAN, PEYMA and NSURL are research-use-only, and
ParsTwiNER (MIT) is a Twitter corpus that costs about 23 F on prose.

Two caveats to know before relying on the entities:

- **The labels are silver.** The treebank README states they came from the BERT-based
  Beheshti-NER tagger with manual corrections for recall, so `ENTS_F 71.87` is measured against
  a silver test split and partly reflects agreement with that tagger.
- **Three labels are thin.** `MON` (205 training examples), `TIM` (135) and `PCT` (121) score
  73.7, 66.7 and 57.1. `PER`, `LOC`, `ORG` and `DAT` have 1,300 or more each.

Entity spans were transferred onto this pipeline's tokenization by difflib alignment at a 99.86%
rate; spans that could not be aligned exactly were dropped rather than guessed
(`scripts/transfer_perdt_ner.py`).

Language data comes from `spacy/lang/fa` upstream, whose stop word list came from hazm.
Everything trains on 4 CPU cores with no GPU.

## Results

Held-out test splits, from `spacy benchmark accuracy`, stored in `metrics/`. Trained on a
4-core i5-7200U: 1h27m for the UD components, 17 min for NER.

Syntax and morphology, identical in both packages since they share the same trained components:

| Metric | Score | Reference |
| --- | --- | --- |
| `TOKEN_ACC` / `TOKEN_F` | 99.96 / 99.11 | |
| `TAG_ACC` (XPOS) | 95.96 | |
| `POS_ACC` (UPOS) | 96.24 | |
| `MORPH_ACC` | 96.29 | |
| `LEMMA_ACC` | 97.91 | |
| `SENTS_F` | 99.25 | |
| `DEP_UAS` | 89.69 | hazm+ParsBERT: 92.46 |
| `DEP_LAS` | 85.15 | hazm+ParsBERT: 89.34 |
| Speed | ~9,250 words/s | |

Entities, `fa_core_news_sm` only, on the PerDT NER test split: `ENTS_P` 77.67, `ENTS_R` 66.87,
`ENTS_F` 71.87. Per label:

| Label | F | Train examples |
| --- | --- | --- |
| `LOC` | 80.24 | 4,954 |
| `DAT` | 74.45 | 1,323 |
| `MON` | 73.68 | 205 |
| `ORG` | 68.77 | 2,643 |
| `TIM` | 66.67 | 135 |
| `PER` | 65.29 | 4,847 |
| `PCT` | 57.14 | 121 |

Parsing is 4.2 LAS behind hazm's parser, which uses the same corpus and the same spaCy parser
architecture with a fine-tuned ParsBERT instead of hash embeddings. That gap is the target for
a future `trf` tier.

`PER` scoring below `LOC` and `ORG` despite having 4,847 examples is the silver labels showing
through: PerDT includes titles and honorifics inside `PER` spans inconsistently (6.24% of spans
start with one, against 1.41% in the human-annotated ParsTwiNER), so the boundaries the model
has to learn are less regular than the label count suggests.

For comparison, `en_core_web_sm` scores TAG 97, LAS 90, ENTS_F 84 on a larger, cleaner corpus.

Reproduce with `.venv/bin/python -m spacy project run all`, plus `run ent` for an NER-only
package.

## Install

```bash
.venv/bin/python -m pip install packages/fa_core_news_sm-3.8.0/dist/fa_core_news_sm-3.8.0-py3-none-any.whl
# or, without NER:
.venv/bin/python -m pip install packages/fa_dep_news_sm-3.8.0/dist/fa_dep_news_sm-3.8.0-py3-none-any.whl
```

```python
import spacy
nlp = spacy.load("fa_core_news_sm")
doc = nlp("محمدرضا شجریان در مشهد به دنیا آمد.")
print([(t.text, t.pos_, t.lemma_, t.dep_) for t in doc][:3])
# [('محمدرضا', 'PROPN', 'محمدرضا', 'nsubj'), ('شجریان', 'PROPN', 'شجریان', 'flat:name'), ...]
print(doc.ents)   # (محمدرضا شجریان, مشهد)  -> PER, LOC

doc = nlp("شرکت ایران خودرو تولید را ۲۰ درصد افزایش می‌دهد.")
print([(e.text, e.label_) for e in doc.ents])   # ۲۰ درصد -> PCT
```

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
`modified_fa_perdt-ud-train.spacy`, the same treebank used here. Full analysis in
[`docs/MODELS.md`](docs/MODELS.md) §4.

## Licensing drove most decisions here

spaCy's maintainers say the Persian models trained in 2018 were never published because of
corpus licensing (spaCy discussion #8233, after PR #2797 added `fa` tokenizer support). ARMAN,
PEYMA and NSURL are all research-use-only, and wrapping them in an Apache-2.0 toolkit does not
change that.

The way out was finding that PerDT ships its own NER layer under the treebank's CC BY-SA 4.0,
so the entire pipeline now derives from one corpus with one licence. The 2018 attempt also
failed for a second reason worth knowing if you plan to publish: honnibal asked for scripts
that could regenerate the model and got a notebook instead. `project.yml` is that script.

## Setup

```bash
# Python 3.12
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install "spacy>=3.8,<3.9" spacy-lookups-data
```

## Build

[`project.yml`](project.yml) has two workflows:

```bash
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

`finalize` runs twice because of an ordering constraint: test scores only exist after
evaluation, and evaluation needs a finalized pipeline to score. The second pass only copies
models. `scripts/finalize_pipeline.py` enforces the shape of each variant, refusing to publish
a `dep` pipeline that contains `ner` or a `core` one that does not, so the split cannot regress
unnoticed.

`--ud-metrics` and `--ner-metrics` are separate flags on purpose. Folding both reports over one
key set silently corrupted `core`'s metadata during development: the NER corpus has no gold
tags, so its report carries `tag_acc: 0.0`, which overwrote the real 95.96.

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

## Roadmap

1. A human-annotated NER test set, ~500 sentences. PerDT's entity labels and its NER test split
   are both silver, so `ENTS_F 71.87` is not yet a fact. Tracked in
   [`../ner_dataset`](../ner_dataset/PLAN.md).
2. A mixed-genre variant. Measured: this prose-trained NER scores 45.72 F on tweets, and mixing
   ParsTwiNER in recovers that to 66.49 for 0.69 F on prose. That belongs in a separate package
   rather than inside a `news` one.
3. `md` and `lg` need floret vectors trained on Persian Wikipedia and OSCAR (see
   `spacy-vectors-builder`). Floret rather than classic fastText, because inconsistent ZWNJ
   usage explodes the surface vocabulary.
4. `trf` needs a rented GPU and should use `HooshvareLab/roberta-fa-zwnj-base` (Apache-2.0)
   rather than ParsBERT, whose model card carries no licence.
5. `senter` is one extra training run.
6. Upstream PRs to `spacy/lang/fa`, see [`docs/upstream/fa-noun-chunks.md`](docs/upstream/fa-noun-chunks.md).
