---
language:
- fa
license: cc-by-sa-4.0
license_name: cc-by-sa-4.0
license_link: LICENSE
pretty_name: "fa-perdt-ner: Persian NER over the PerDT treebank, silver and LLM-relabelled"
task_categories:
- token-classification
task_ids:
- named-entity-recognition
annotations_creators:
- machine-generated
- expert-generated
language_creators:
- found
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- extended|universal_dependencies
tags:
- persian
- farsi
- ner
- named-entity-recognition
- universal-dependencies
- perdt
- spacy
- llm-annotated
configs:
- config_name: silver
  default: true
  data_files:
  - split: train
    path: silver/train.jsonl
  - split: validation
    path: silver/dev.jsonl
  - split: test
    path: silver/test.jsonl
- config_name: llm
  data_files:
  - split: train
    path: llm/train.jsonl
  - split: validation
    path: llm/dev.jsonl
  - split: test
    path: llm/test.jsonl
---

# fa-perdt-ner

Persian named-entity annotations over every sentence of the Persian Universal Dependency
Treebank ([UD_Persian-PerDT](https://github.com/UniversalDependencies/UD_Persian-PerDT),
PerUDT v1.0): {{n_sentences}} sentences, {{n_tokens}} tokens, in PerDT's own train/dev/test
split, on the tokenization a spaCy `--merge-subtokens` conversion of the treebank produces.

Two configurations, same sentences, same tokens, different labels:

| config | labels | entities | what it is |
|---|---|---|---|
| `silver` (default) | `PER LOC ORG DAT MON TIM PCT` | {{n_silver_entities}} | PerDT's own NER layer, realigned onto the UD tokenization. Produced by the PerDT authors with the Beheshti-NER tagger plus manual corrections; **silver, not gold**. |
| `llm` | `PER LOC ORG DAT` | {{n_llm_entities}} | A fresh annotation of the same sentences by a large language model, against a written guideline (v{{guideline_version}}) inferred from the silver layer. Four labels only. |

**Read this before comparing anything.** `llm` has four labels; `silver` has seven. A model
trained on one and scored on the other must be restricted to the four shared types, or every
correct `MON`/`TIM`/`PCT` span becomes a false positive. And neither configuration is human
gold: every agreement figure on this card is agreement between two annotators, not accuracy.

The text and split are PerDT's, CC BY-SA 4.0; this dataset is Adapted Material under the same
licence. See [Licence and attribution](#licence-and-attribution).

## Loading

```python
from datasets import load_dataset

silver = load_dataset("Phazel/fa-perdt-ner", "silver")   # 7 labels
llm = load_dataset("Phazel/fa-perdt-ner", "llm")         # 4 labels

row = llm["test"][0]
row["id"]        # PerDT sent_id, e.g. "test-s1"
row["tokens"]    # ["این", "موافقت‌نامه", "را", ...]
row["ner_tags"]  # ["O", "O", "O", ...]  IOB2 strings, parallel to tokens
row["entities"]  # [{"start": 3, "end": 5, "label": "ORG"}, ...]  token offsets, end exclusive
```

Splits are named `train`, `validation` (PerDT `dev`) and `test`. `ner_tags` are IOB2 strings,
not a `ClassLabel`, so the two configurations can share one loader; map them yourself if a
trainer wants integers.

Both configurations also ship as two-column IOB2 (`<config>/{train,dev,test}.iob`, tab
separated, blank line between sentences), which `spacy convert` reads directly:

```bash
python -m spacy convert llm/train.iob corpus/ --converter ner --lang fa
```

The tokenization is exactly what `spacy convert --merge-subtokens` yields from the PerDT
CoNLL-U files, so these files can be paired with a parser or tagger trained on the treebank
without re-alignment. That is how the
[`fa_core_news_*`](https://huggingface.co/Phazel/fa_core_news_sm) pipelines were built.

## Fields

`silver/*.jsonl`:

| field | type | |
|---|---|---|
| `id` | string | PerDT `sent_id` (`train-s1`, `dev-s12`, ...) |
| `tokens` | list[string] | merged-subtoken surface forms, in order |
| `ner_tags` | list[string] | IOB2 tag per token |

`llm/*.jsonl` adds:

| field | type | |
|---|---|---|
| `entities` | list[{start, end, label}] | token offsets, `end` exclusive; the spans `ner_tags` encodes |
| `prompt_hash` | string | `{{prompt_hash}}` for every row: hash of the guideline+prompt that produced it |
| `error` | string or null | `"missing"` on the {{n_unannotated}} sentence(s) the model refused to annotate: {{unannotated_ids}} |

## Statistics

`silver`:

{{silver_stats}}

`llm`:

{{llm_stats}}

Entity counts are spans, from the IOB2 tags. Token counts are the same in both configurations
by construction.

## Provenance

### Text and split

PerDT ([Rasooli et al. 2013](https://aclanthology.org/N13-1031)) is ~29k sentences of
contemporary Persian across news, academic, magazine and fiction genres. Its UD conversion
([Rasooli et al. 2022](https://aclanthology.org/2022.lrec-1.766)) is what was downloaded, at
the URLs and MD5 checksums recorded in `manifest.json`. The 89/5/5 split is PerDT's own; no
sentence was moved, dropped or deduplicated.

Tokenization: the CoNLL-U multiword tokens (pronominal clitics, enclitic copula) are merged
back into their surface form, i.e. `spacy convert --merge-subtokens`. `پدرم` is one token,
not `پدر` + `م`. This is the tokenization a spaCy `fa` tokenizer can actually produce, and the
one every `Phazel/fa_*` pipeline is trained on.

### `silver`

PerDT ships an entity layer at `not-to-release/Dadegan with NER tag/`, two-column IOB2 on
the *original* Dadegan tokenization, which matches the released UD tokenization in only
57-62% of sentences (the NER files drop some copulas and auxiliaries). The PerDT README
states this layer was produced with the BERT-based Beheshti-NER tagger
([Taher, Hoseini & Shamsfard 2020](https://arxiv.org/abs/2003.08875)) "with manual
corrections to extend recall", and was used to mark `PROPN` in the treebank.

`scripts/transfer_perdt_ner.py` (in the
[pipeline repo](https://github.com/Fazel94/spacy-persian)) aligns each sentence's two token
sequences with `difflib` and moves a span only if every one of its tokens maps and the result
stays contiguous. Anything else is dropped rather than guessed. Transfer rate: 99.86% of
spans on train, 99.74% on dev, 99.51% on test.

Known artefacts inherited from that layer, left as they are: the opening `(` of a
parenthetical blessing formula is tagged `I-PER` a few hundred times; a handful of entities
carry inconsistent boundaries for the same string. The `llm` guideline documents the ones it
found (`annotation/GUIDELINES.md`).

### `llm`

A second annotation of the same sentences, produced in September 2026 by a large language
model prompted with `annotation/prompt-v{{guideline_version}}.md`, which embeds
`annotation/GUIDELINES.md` v{{guideline_version}} in full. Pipeline, all in
`annotation/scripts/`: `select_sentences.py` → `build_requests.py` (batches of 50 sentences,
one guideline copy per batch) → `annotate.js` (calls the model, verifies the returned id set,
retries once, then bisects a failing batch) → `validate_responses.py` (re-aligns each
returned entity *string* onto the token sequence, rejects anything that does not align to
whole tokens) → `export_iob.py`.

**The guideline was inferred from the silver layer**, not written from linguistic
principles: each rule states the majority convention in the silver train split with the
counts behind it (e.g. `آقای` outside a `PER` span 140/141 times), and calls the minority out
as an error to correct. So the two configurations are not independent: `llm` is what the
silver conventions look like when applied consistently.

Three entity types were deliberately dropped. `MON`, `TIM` and `PCT` are {{oos_share}} of silver
spans and the least consistent — the guideline could not find a majority convention to
state for them. Consumers wanting those types should use `silver`.

Model provenance is incomplete and recorded honestly: `annotate.js` was run with the model
alias `{{model_alias}}` (the harness's session default) and the resolved model id, temperature
and date were not written into the output rows. Re-running the same prompt on a sample
agreed with the committed run at span F 0.95-0.98, so the committed files are the record of
origin and cannot be reproduced bit-exactly. `prompt_hash` is stable, so a re-run against a
changed guideline is detectable.

{{n_dropped}} entities the model returned were rejected by `validate_responses.py` and are
listed in `llm/dropped-entities.jsonl` with the sentence they came from: {{dropped_reasons}}.
`no-align` means the returned string does not cover whole tokens, mostly a name the model
separated from its clitic (`فاطمه` against the token `فاطمه‌ام`) or a span the model
re-spelled; `bad-label` is a label outside the four; `overlap` is a span inside another.

## Agreement between the two configurations

Exact span match, `silver` restricted to the four shared labels as the reference, from
`llm/agreement.json`:

{{agreement}}

Adjudicating every test-split disagreement span by span (LLM-judged, so an estimate) put the
`llm` annotations at 0.93-0.95 precision against silver's 0.84: silver is wrong or
mis-bounded on roughly 126 of its 783 test spans. That number is why `llm` exists; it is not
a measurement.

### Training the same model on each

Same spaCy NER architecture, same hyperparameters, same sentences; micro F over
`PER`/`LOC`/`ORG`/`DAT`, out-of-scope entities removed from both prediction and reference:

| trained on \ scored on | `silver` test | `llm` test |
|---|---|---|
| `silver` | **71.98** | 67.27 |
| `llm` | 67.59 | **79.94** |

The diagonal says the `llm` labels are more internally consistent (+7.96 F for an identical
model). The near-symmetric off-diagonal says the two conventions are genuinely different,
not one a subset of the other. `ORG` recall does *not* improve under `llm` (68.1 → 66.9): the
head-noun-plus-specifier construction (`کمیسیون آموزش`) is under-tagged in both. Details in
the pipeline repo's `docs/MODELS.md` §10.

## Limitations

- **No human gold.** Nothing on this card measures accuracy. A 200-sentence blind human
  sample over the test split is drawn and unfilled; until it is, prefer `silver` for
  reporting numbers comparable to prior PerDT work, and treat `llm` as a consistency-improved
  training set whose real recall is unknown.
- **`llm` has four labels.** A seven-label model evaluated on it, or vice versa, needs the
  label restriction described above.
- **Silver is silver.** Tagger output with corrections of unknown extent; the PerDT authors
  used it to mark `PROPN`, not as an NER benchmark.
- **One sentence in `llm` is unannotated** ({{unannotated_ids}}, a content-filter refusal);
  its `ner_tags` are all `O` and `error` is `"missing"`.
- **Genre.** Contemporary written Persian, news-heavy. No social media, no dialect.
- **Tokenization is spaCy-shaped.** Clitics are attached. Systems that split them need to
  re-align.

## Licence and attribution

**CC BY-SA 4.0**, inherited: PerDT is CC BY-SA 4.0 and this is Adapted Material, so
ShareAlike applies to both configurations and to anything trained on them that constitutes
an adaptation. Full text and notice in `LICENSE`.

Please credit the treebank:

```bibtex
@inproceedings{rasooli-etal-2022-persian,
  title     = {The {P}ersian Dependency Treebank Made Universal},
  author    = {Rasooli, Mohammad Sadegh and Safari, Pegah and Moloodi, Amirsaeid and Nourian, Alireza},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  year      = {2022},
  pages     = {7078--7087},
  url       = {https://aclanthology.org/2022.lrec-1.766}
}

@inproceedings{rasooli-etal-2013-development,
  title     = {Development of a {P}ersian Syntactic Dependency Treebank},
  author    = {Rasooli, Mohammad Sadegh and Kouhestani, Manouchehr and Moloodi, Amirsaeid},
  booktitle = {Proceedings of NAACL-HLT 2013},
  year      = {2013},
  url       = {https://aclanthology.org/N13-1031}
}
```

and, for the `silver` layer's tagger:

```bibtex
@article{taher-etal-2020-beheshti,
  title   = {Beheshti-{NER}: {P}ersian named entity recognition Using {BERT}},
  author  = {Taher, Ehsan and Hoseini, Seyed Abbas and Shamsfard, Mehrnoush},
  journal = {arXiv preprint arXiv:2003.08875},
  year    = {2020}
}
```

If you use the `llm` configuration or the annotation kit, cite this dataset:

```bibtex
@misc{fazeli-2026-fa-perdt-ner,
  title  = {fa-perdt-ner: {P}ersian NER over the {PerDT} treebank, silver and {LLM}-relabelled},
  author = {Fazeli, Kiyarash},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Phazel/fa-perdt-ner}
}
```

Source, build script and the full annotation history:
<https://github.com/Fazel94/spacy-persian> (`annotation/`, `scripts/annotation/`).
`manifest.json` in this repo records the commit each file was built from.
