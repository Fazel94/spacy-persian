# fa_dep_news_sm — a Persian pipeline for spaCy

There is no trained Persian pipeline for spaCy. `spacy.load("fa_core_news_sm")` has never
worked; `spacy.blank("fa")` gives you a tokenizer and stop words and nothing else. This
project builds the missing pipeline from openly-licensed data, using spaCy's own tooling, so
the result can actually be redistributed.

- **What the four English pipelines are, and what the four Persian equivalents should be:**
  [`docs/MODELS.md`](docs/MODELS.md)
- **How models get contributed/published in the spaCy ecosystem, and what upstream `fa`
  already has:** [`docs/CONTRIBUTING-GUIDE.md`](docs/CONTRIBUTING-GUIDE.md)
- **The build itself:** [`project.yml`](project.yml)

## Two packages, not one

spaCy's naming scheme encodes what a pipeline contains: `dep` = tagger + parser + lemmatizer,
`ent` = NER only, `core` = both. This project ships the first two separately and deliberately
does **not** ship a `core`:

| Package | Components | Trained on | Licence | Headline |
| --- | --- | --- | --- | --- |
| **`fa_dep_news_sm`** | tok2vec, tagger, morphologizer, trainable_lemmatizer, parser | [UD_Persian-PerDT](https://github.com/UniversalDependencies/UD_Persian-PerDT), 29,107 sentences of edited prose | CC BY-SA 4.0 | **LAS 85.15**, LEMMA 97.91 |
| `fa_ent_news_sm` (optional) | ner | [ParsTwiNER](https://github.com/overfit-ir/parstwiner), 7,667 **tweets**, 16,250 entities | MIT | **ENTS_F 67.22** |

Those two headline numbers are the whole argument. The UD components score 85–98 on edited
prose; the NER manages 67 F on a different genre entirely, because the good Persian NER
corpora (ARMAN, PEYMA, NSURL) are research-use-only and cannot be redistributed. Folding both
into one `fa_core_news_sm` would hide that gap behind a single package name and a single
version number — users would reasonably assume the NER is held to the same standard as the
parser. It is not.

So NER ships as its own opt-in package, and **`fa_core_news_sm` is reserved** for when
[`../ner_dataset`](../ner_dataset/PLAN.md) delivers prose-genre NER data that beats ParsTwiNER
on a human-annotated test set. The `ner` component already embeds its own tok2vec rather than
a `Tok2VecListener` precisely so that merge is a one-liner when the data arrives:

```python
dep = spacy.load("fa_dep_news_sm")
dep.add_pipe("ner", source=spacy.load("fa_ent_news_sm"))   # verified working
```

Language data comes from `spacy/lang/fa` upstream (its stop word list originally from hazm).
Everything trains on 4 CPU cores with no GPU.

### Results

Trained and evaluated on this laptop (4-core i5-7200U, CPU only, 1h27m for the UD
components, ~25 min for NER). Scores are on the **held-out test splits**, produced by
`spacy benchmark accuracy` and stored in `metrics/`.

**`fa_dep_news_sm`** (UD_Persian-PerDT test split):

| Metric | Score | reference |
| --- | --- | --- |
| `TOKEN_ACC` / `TOKEN_F` | 99.96 / 99.11 | |
| `TAG_ACC` (XPOS) | **95.96** | |
| `POS_ACC` (UPOS) | **96.24** | |
| `MORPH_ACC` | 96.29 | |
| `LEMMA_ACC` | **97.91** | |
| `SENTS_F` | 99.25 | |
| `DEP_UAS` | **89.69** | hazm+ParsBERT: 92.46 |
| `DEP_LAS` | **85.15** | hazm+ParsBERT: 89.34 |
| Speed | ~9,250 words/s (CPU) | |
| Wheel | 7.5 MB | `en_core_web_sm`: 12 MB |

**`fa_ent_news_sm`** (ParsTwiNER test split): `ENTS_P` 74.77 / `ENTS_R` 61.06 /
**`ENTS_F` 67.22**, 5.6 MB wheel. Per label:
`LOC` 73.9, `PER` 69.1, `NAT` 63.2, `ORG` 59.3, `POG` 41.2, `EVE` 30.0.

Read these honestly:

- **Parsing is 4.2 LAS behind hazm's parser**, which is the expected gap between a 7.5 MB
  CPU model with hash embeddings and a fine-tuned ParsBERT. Same corpus, same spaCy parser
  architecture, so the comparison is fair — and it sets the target for a future `trf` tier.
- **NER is the weak artifact, and the per-label numbers say why.** ParsTwiNER is not small
  (232,917 tokens, 16,250 entities, 7.0% density — the same order as the restricted ARMAN and
  PEYMA). But its label distribution is brutally skewed: `PER` 6258, `LOC` 5478, `ORG` 2694,
  `NAT` 939, **`EVE` 482, `POG` 399**. The two starved labels are exactly the two that score
  30.0 and 41.2. So the head labels suffer from genre mismatch and the tail labels from raw
  data starvation — two different problems needing two different fixes.
- **Everything else is competitive with the English `sm` pipeline** (`en_core_web_sm`:
  TAG 97, LAS 90, ENTS_F 84 — on a much larger and cleaner corpus).

Reproduce: `.venv/bin/python -m spacy project run all` (add `run ner` for the NER package).

### Install

```bash
.venv/bin/python -m pip install packages/fa_dep_news_sm-3.8.0/dist/fa_dep_news_sm-3.8.0-py3-none-any.whl
# optional, separate package:
.venv/bin/python -m pip install packages/fa_ent_news_sm-3.8.0/dist/fa_ent_news_sm-3.8.0-py3-none-any.whl
```

```python
import spacy
nlp = spacy.load("fa_dep_news_sm")
doc = nlp("دانشگاه تهران در سال ۱۳۱۳ تأسیس شد.")
print([(t.text, t.pos_, t.lemma_, t.dep_) for t in doc])
# ('دانشگاه', 'PROPN', 'دانشگاه', 'nsubj') ('تهران', 'PROPN', 'تهران', 'flat:name') ...

# Want entities too? Attach the NER package yourself, eyes open about its 67 F:
nlp.add_pipe("ner", source=spacy.load("fa_ent_news_sm"))
print(nlp(doc.text).ents)   # (دانشگاه تهران,)  -> ORG
```

### Why not hazm's own models

hazm is the reference Persian NLP toolkit and it *does* publish spaCy-format pipelines on the
HF Hub, so it was the obvious starting point. It does not survive contact:

- Its trainable models are pycrfsuite CRFs (`hazm/sequence_tagger.py`). There is no
  `config.cfg` and no `spacy train` anywhere in the repo — the `Spacy*` classes only download
  pretrained pipelines.
- Those pretrained pipelines are three *single-task* models (`transformer + tagger`,
  `transformer + parser`, `transformer + chunker`), each `version: 0.0.0` with an empty
  `license` field, pinned to spaCy 3.6. Using all three means three ParsBERT forward passes
  over the same text and no shared `Doc`.
- Its tokenizer is deliberately incompatible with UD tokenization: the normaliser fuses ZWNJ
  affixes and `join_verb_parts()` glues multi-word verb chains into single tokens.
- Most corpora it reads (Bijankhan, Peykare, Hamshahri, raw PerDT) are gated behind
  `peykaregan.ir` / `dadegan.ir` under research-only terms.

What hazm *does* give us: confirmation of the corpus choice — hazm's own spaCy parser was
trained on `modified_fa_perdt-ud-train.spacy`, i.e. the same treebank we use — plus the stop
word list already vendored into `spacy/lang/fa`. Full analysis in
[`docs/MODELS.md`](docs/MODELS.md) §4.

### Why licensing is the load-bearing constraint

spaCy's maintainers state that Persian models trained back in 2018 were never published
*because of corpus licensing* (spaCy discussion #8233, after PR #2797 added `fa` tokenizer
support). The standard Persian NER corpora — ARMAN, PEYMA, NSURL — are all "research use
only", and wrapping them in an Apache-2.0 toolkit does not launder that. ParsTwiNER (MIT) is
the only redistributable Persian NER corpus we could verify, which is why the NER component is
trained on tweets. That trade-off is recorded in the model's `meta.json["notes"]`.

## Setup

```bash
# Python 3.12
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install "spacy>=3.8,<3.9" spacy-lookups-data
```

## Build

Everything is driven by [`project.yml`](project.yml), which has **two independent workflows**:

```bash
.venv/bin/python -m spacy project assets      # download + checksum the corpora
.venv/bin/python -m spacy project run all     # -> fa_dep_news_sm  (the shipping artifact)
.venv/bin/python -m spacy project run ner     # -> fa_ent_news_sm  (optional)
```

| Workflow | Command | What it does |
| --- | --- | --- |
| `all` | `inspect` | annotation coverage of the treebanks (`scripts/inspect_treebanks.py`) |
| | `convert-ud` | CoNLL-U → `DocBin` with `--merge-subtokens`, plus the tokenizer-agreement report |
| | `debug-data` | `spacy debug data` before spending CPU |
| | `train-core` | tagger + morphologizer + trainable_lemmatizer + parser on PerDT |
| | `finalize` | write `fa_dep_news_sm` metadata: sources, licence, notes (`scripts/finalize_pipeline.py`) |
| | `evaluate` | `spacy benchmark accuracy` on the held-out UD test split |
| | `finalize-meta` | re-run `finalize`, folding test scores into `meta.json["performance"]` |
| | `package` | build the wheel + sdist |
| | `smoke` | run the pipeline over real Persian text and print every annotation layer |
| `ner` | `convert-ner` | unpack ParsTwiNER, IOB2 → `DocBin` |
| | `debug-data-ner`, `train-ner`, `finalize-ner`, `evaluate-ner`, `package-ner` | the same sequence for `fa_ent_news_sm` |

The two training runs are independent and can run concurrently — each is single-threaded.

`finalize` and `finalize-meta` are separate steps for an unavoidable ordering reason: test
scores can only exist after `evaluate`, and `evaluate` needs a finalized pipeline to score.
Re-running `finalize` afterwards is cheap (it only copies models).
`scripts/finalize_pipeline.py` **refuses** to publish a `dep` pipeline containing an `ner`
component, so the split cannot silently regress.

## Design decisions worth knowing before you touch anything

1. **`--merge-subtokens`.** spaCy has no multiword-token layer, and PerDT splits pronominal
   clitics (`پدرم` → `پدر` + `م`). Measured on dev: merging gives token F 0.9887 vs 0.9823 for
   the split version, at the cost of 34 composite XPOS tags on 1.5% of tokens. Merging wins
   because otherwise 1.5% of gold tokens are boundaries the shipped tokenizer can never
   produce. Numbers: `scripts/tokenization_report.py`.
2. **`ner` carries its own tok2vec.** A `Tok2VecListener` only resolves inside the pipeline it
   was trained in, so a listener-based component cannot be sourced into another pipeline.
   `configs/fa_ner_sm.cfg` embeds the tok2vec instead — the same design as `en_core_web_sm`.
3. **`morphologizer` + `trainable_lemmatizer` instead of `attribute_ruler` + rule lemmatizer.**
   The English pipelines derive UPOS from PTB tags by rule because OntoNotes has no UPOS. UD
   gives us gold UPOS, FEATS and lemmas, so we train on them and get real `pos_acc`,
   `morph_acc` and `lemma_acc` numbers instead of unmeasurable rule coverage.
4. **PerDT, not Seraji.** 3.7× more tokens, and Seraji has no `PROPN` tag at all.

## Roadmap

In value order, not difficulty order:

1. **Prose-genre NER** → [`../ner_dataset`](../ner_dataset/PLAN.md). This is what unlocks a real
   `fa_core_news_sm`. Note that PerDT's XPOS already encodes animacy on proper nouns
   (`N_ANM` 6,752 vs `N_IANM` 12,682), which is a strong free prior for PER vs LOC/ORG.
2. **`md`/`lg`** need floret vectors trained on Persian Wikipedia + OSCAR (see
   `spacy-vectors-builder`); floret rather than classic fastText because Persian's ZWNJ usage
   is inconsistent and explodes the surface vocabulary.
3. **`trf`** needs a rented GPU and should use `HooshvareLab/roberta-fa-zwnj-base`
   (Apache-2.0) rather than ParsBERT, whose model card carries no licence.
4. **`senter`** is one extra training run away.
5. **Upstream PRs** to `spacy/lang/fa` — see [`docs/upstream/fa-noun-chunks.md`](docs/upstream/fa-noun-chunks.md).

Details in [`docs/MODELS.md`](docs/MODELS.md) §2.
