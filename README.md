# Persian (Farsi) pipelines for spaCy

Trained spaCy pipelines for Persianinstallable now. spaCy has
never shipped one, and `spacy.blank("fa")` gives you a tokenizer and stop words.
This pipeline  built from UD_Persian-PerDT.
```bash
pip install https://huggingface.co/Phazel/fa_core_news_sm/resolve/main/fa_core_news_sm-any-py3-none-any.whl
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


## Results

From `spacy benchmark accuracy`, stored in `metrics/`.
| Package | Components | Licence | Score | Wheel |
| --- | --- | --- | --- | --- |
| `fa_dep_news_sm` | tok2vec, tagger, morphologizer, trainable_lemmatizer, parser | CC BY-SA 4.0 | LEMMA 97.91 | 7.5 MB |
| `fa_core_news_sm` | the above plus ner | CC BY-SA 4.0 | ENTS_F 71.87 | 13 MB |

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
`ENTS_F` 71.87.

| Label | F | Train examples |
| --- | --- | --- |
| `LOC` | 80.24 | 4,954 |
| `DAT` | 74.45 | 1,323 |
| `MON` | 73.68 | 205 |
| `ORG` | 68.77 | 2,643 |
| `TIM` | 66.67 | 135 |
| `PER` | 65.29 | 4,847 |
| `PCT` | 57.14 | 121 |


For comparison, `en_core_web_sm` scores TAG 97, LAS 90, ENTS_F 84 on a larger, cleaner corpus.
Trained on a 4-core i5-7200U with no GPU: 1h27m for the syntax components, 17 min for NER.

## Install

```bash
pip install https://huggingface.co/Phazel/fa_core_news_sm/resolve/main/fa_core_news_sm-any-py3-none-any.whl
# or, without NER:
pip install https://huggingface.co/Phazel/fa_dep_news_sm/resolve/main/fa_dep_news_sm-any-py3-none-any.whl
```

## Caveats

- **The entity labels are silver.** They come from the treebank's own
  `not-to-release/Dadegan with NER tag/` layer, which its README states was produced by the
  BERT-based Beheshti-NER tagger with manual corrections for recall. `ENTS_F 71.87` is measured
  against a silver test split and partly reflects agreement with that tagger.
- **Three entity labels are thin.** `MON` (205 training examples), `TIM` (135) and `PCT` (121)
  rest on 4 to 11 test entities each. `PER`, `LOC`, `ORG` and `DAT` have 1,300 or more.
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
4. PerDT, not Seraji: 3.7x more tokens, and Seraji has no `PROPN` tag. Entity spans were
   transferred onto this pipeline's tokenization by difflib at a 99.86% rate, and spans that
   could not be aligned exactly were dropped rather than guessed
   (`scripts/transfer_perdt_ner.py`).

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
- Language data comes from `spacy/lang/fa` upstream, whose stop word list came from hazm.
