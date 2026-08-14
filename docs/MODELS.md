# The Persian pipelines: what to build, from what, and why

## 1. What the four English pipelines are

One pipeline design at four embedding budgets. Every one has the same components; the axis of
variation is where token representations come from.

| | `en_core_web_sm` | `en_core_web_md` | `en_core_web_lg` | `en_core_web_trf` |
| --- | --- | --- | --- | --- |
| Size on disk | 12 MB | 31 MB | 382 MB | 436 MB |
| Embeddings | hash embeddings only | 685k keys / 20k vectors (300d) | 685k keys / 343k vectors (300d) | `roberta-base`, 768d contextual |
| Components | tok2vec, tagger, parser, senter, attribute_ruler, lemmatizer, ner | same | same | transformer, tagger, parser, attribute_ruler, lemmatizer, ner |
| `TAG_ACC` | 0.97 | 0.97 | 0.97 | 0.98 |
| `DEP_UAS` / `LAS` | 0.92 / 0.90 | 0.92 / 0.90 | 0.92 / 0.90 | 0.95 / 0.94 |
| `ENTS_F` | 0.84 | 0.85 | 0.86 | 0.90 |
| Training data | OntoNotes 5 (+ ClearNLP dep conversion, WordNet 3.0) | + Explosion vectors (OSCAR 2109 + Wikipedia + OpenSubtitles + WMT News Crawl) | same | OntoNotes 5 + roberta-base |

Static vectors buy nothing measurable for tagging and parsing (identical to two decimals) and
1 to 2 F on NER. The transformer buys about 4 LAS and 6 NER F, at 36x the size and a GPU
requirement. That ordering sets the roadmap below.

Source: <https://spacy.io/models/en>.

That last point does **not** transfer to Persian. The `sm` -> `md` step measured on this
project buys +1.19 LAS and +2.85 NER F (§6), where English gets ~0.00 LAS. Two reasons: PerDT
is roughly a tenth the size of OntoNotes, so hash embeddings have far less signal to learn a
lexicon from, and floret's subword hashing gives 0% OOV on a language whose ZWNJ variation
(می‌رود / میرود / می رود) fragments any fixed word-key table. English `md` uses 20k classic
word vectors and hits OOV constantly. Do not use the English row as the Persian prior.

## 2. Target: the Persian pipelines

Naming follows `[lang]_[type]_[genre]_[size]` (<https://spacy.io/models#conventions>). The
`type` slot carries real information: `dep` = tagger + parser + lemmatizer, `ent` = NER only,
`core` = both. Genre is `news`, after the dominant genre of UD_Persian-PerDT, whose README
lists "news fiction nonfiction academic web blog". spaCy labels comparable treebank-trained
pipelines such as `de_core_news_sm` as `news`.

| Pipeline | Components | Embeddings | Status |
| --- | --- | --- | --- |
| `fa_dep_news_sm` | tok2vec, tagger, morphologizer, trainable_lemmatizer, parser | hash embeddings | built, shipping |
| `fa_core_news_sm` | the above plus ner | hash embeddings | built, shipping |
| `fa_ent_news_sm` | ner (own internal tok2vec) | hash embeddings | built, optional |
| `fa_core_web_sm` | same as core, mixed-genre training data | hash embeddings | not built; would add ParsTwiNER to cover social media |
| `fa_dep_news_md` | same as `fa_dep_news_sm` | floret, 50k rows / 300d | built, shipping |
| `fa_core_news_md` | same as `fa_core_news_sm` | floret, 50k rows / 300d | built, shipping |
| `fa_dep_news_lg` | same as `fa_dep_news_sm` | floret, 200k rows / 300d, full-wiki 5 epochs | built, shipping |
| `fa_core_news_lg` | same as `fa_core_news_sm` | floret, 200k rows / 300d, full-wiki 5 epochs | built, shipping |
| `fa_ent_news_lg` | ner (own internal tok2vec) | floret, 200k rows / 300d, full-wiki 5 epochs | built, optional |
| `fa_core_news_trf` | transformer instead of tok2vec | `HooshvareLab/bert-base-parsbert-uncased`, fine-tuned | built on a rented Colab T4 (not on this hardware: 2 GB VRAM cannot fine-tune a 125M-param encoder), shipping with a redistribution caveat because that encoder's card states no licence; §3.4 and §8 |

### Why `core` is honest here

`core` promises that the NER belongs to the same pipeline, built to the same standard and
versioned together. That holds because PerDT ships its own entity layer in
`not-to-release/Dadegan with NER tag/`: 15,833 entities over the same 29,107 sentences, under
the same CC BY-SA 4.0, in the same genre, aligned to the same tokenization.

An earlier revision of this project refused to ship `core`, and was right to given what it knew
then. The only redistributable Persian NER corpus found at that point was ParsTwiNER, a Twitter
corpus scoring 67.22 F against 85-98 for the UD components, and folding that into one package
would have hidden a genre and quality gap behind a single name and version number. Finding the
treebank's own layer removed the objection rather than answering it.

`fa_dep_news_sm` still ships alongside `core`, for users who want a 7.5 MB syntax-only model or
who would rather not depend on silver entity labels.

### Why `morphologizer` + `trainable_lemmatizer`

The English pipelines use `attribute_ruler` because OntoNotes gives them PTB tags and they
derive UPOS by rule. UD treebanks give UPOS and FEATS as gold data, so a morphologizer trained
on them has strictly more information, and Persian morphology (Number, Person, Tense, Mood,
Voice, Polarity, PronType) is worth predicting.

Persian rule-lemmatizer tables do exist in `spacy-lookups-data` (`fa_lemma_exc.json` 1.68 MB,
`fa_lemma_index.json`, `fa_lemma_rules.json`, derived from Seraji's treebank), but a rule
lemmatizer's accuracy cannot be measured against the corpus it was extracted from, and it needs
`token.pos` to run at all. `trainable_lemmatizer` learns edit trees from PerDT's gold lemmas and
reports a real `lemma_acc`. PerDT yields 1,908 edit trees at 100% lemma coverage.

`senter` is omitted from v1. It is a separately trained component that ships disabled by default
in the English pipelines, and the parser already produces sentence boundaries. Adding it later
takes one training run.

## 3. Resource inventory, with licences

### 3.1 Treebanks, for tagger / morphologizer / lemmatizer / parser

| Treebank | Sents | Tokens | License | LEMMA | FEATS | XPOS | PROPN | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UD_Persian-PerDT (PerUDT v1.0) | 29,107 | ~509k | CC BY-SA 4.0 | converted + corrections | converted + corrections | manual native (34 types) | yes | chosen |
| UD_Persian-Seraji | 5,997 | ~152k | CC BY-SA 4.0 | manual native | manual native | manual native (30 types) | no | secondary, cross-eval |
| UD_Persian-PUD | 1,000 | | CC BY-SA 4.0 | | | none | | test-only, parallel corpus |
| UD_Persian-IPerUDT | tiny | | CC BY-SA 4.0 | | | none | | grammar examples, unusable |

Measured locally with `scripts/inspect_treebanks.py` (train splits):

```
file                             sents    tokens    MWT  empty  lemma%  feats%  UPOS  XPOS  DEP
fa_perdt-ud-train.conllu         26196    452496   6508      0   100.0    57.7    16    34   34
fa_seraji-ud-train.conllu         4798    121067   1117      0   100.0    65.0    15    30   39
```

PerDT over Seraji for three reasons. It has 3.7x more training tokens (452k against 121k), and
at `sm` size data is the binding constraint. Seraji has no `PROPN`: proper nouns are tagged
`NOUN`, which breaks the downstream tasks people use spaCy for. And hazm independently chose
PerDT for its own spaCy dependency parser, whose `config.cfg` names
`modified_fa_perdt-ud-train.spacy`, which makes the numbers comparable.

Seraji's richer manual FEATS and fully manual lemmas make it the natural cross-evaluation set
and a candidate for a future concatenated-corpus run. The two use different XPOS inventories,
so naive concatenation would corrupt the `tag` label space.

### 3.2 NER

| Dataset | Labels | Size | License | Usable |
| --- | --- | --- | --- | --- |
| PerDT's own NER layer | PER, LOC, ORG, DAT, MON, TIM, PCT | 29,107 sentences / 484k tokens / 15,833 entities | CC BY-SA 4.0, same as the treebank | chosen |
| ParsTwiNER | PER, ORG, LOC, NAT, POG, EVENT | 7,667 tweets / 233k tokens / 16,250 entities | MIT (verified via GitHub API on `overfit-ir/parstwiner`) | usable, wrong genre |
| ARMAN (PersianNER) | 6 classes | 250k tokens | academic research only | no |
| PEYMA | 7 classes | 302k tokens | "free for research purposes", no OSS licence | no |
| NSURL-2019 Task 7 | PEYMA tagset | ~1M tokens | no explicit licence | no |
| HooshvareLab merged ParsNER | 10 classes | ARMAN+PEYMA+WikiANN | inherits ARMAN/PEYMA restrictions; HF repo gated (401) | no |

spaCy's maintainers state that the Persian models trained in 2018 were never published because
of corpus licensing (spaCy discussion #8233, following PR #2797, which added only
`spacy.blank("fa")` tokenizer support).

The PerDT layer lives in `not-to-release/Dadegan with NER tag/{train,dev,test}_with_NER_tag.txt`
as two-column IOB2. In UD convention `not-to-release/` means "excluded from the official UD
release build", normally working and source data, and the directory is public on GitHub under
the repo's `LICENSE.txt`. Confirm that reading with the PerDT authors before publishing anything
derived from it, since redistribution rights are the whole point.

Three properties of that layer decide how it gets used:

1. **The labels are silver.** The treebank README states they came from the BERT-based
   Beheshti-NER tagger (Taher et al., 2020) with manual corrections to extend recall. The
   published `ents_f` is therefore measured against a silver test split and partly reflects
   agreement with that tagger. A human-annotated test set is the outstanding work.
2. **Tokenization differs.** The NER files use the original Dadegan tokenization, which matches
   the released UD tokenization exactly in only 57 to 62% of sentences: the NER files drop some
   copulas and auxiliaries, and at least one honorific is corrupted (`ص` written as `،`).
   Entities sit on content words present in both, so `scripts/transfer_perdt_ner.py` aligns them
   with difflib, transferring 99.86% of train entities, 99.74% of dev and 99.51% of test. Spans
   whose tokens do not all map contiguously are dropped rather than guessed.
3. **The label sets do not line up with ParsTwiNER.** PerDT has `DAT`, `MON`, `TIM` and `PCT`;
   ParsTwiNER has `NAT`, `EVE` and `POG`. The intersection is `PER`, `LOC`, `ORG`. Since spaCy
   assigns one label per token, concatenating the two raw would teach the model that dates are
   `O` in half the corpus. Any mixed-genre variant has to solve that first.

Genre matters more than any of this. Measured with one config on the three shared labels, a
PerDT-trained NER scores 72.80 F on prose and 45.72 on tweets, while a ParsTwiNER-trained one
scores 68.59 on tweets and 55.59 on prose. Training on both gives 72.11 and 66.49, so mixing
costs under 1 F on prose and 2 F on tweets while recovering roughly 20 F off-genre. That is the
argument for a future `fa_core_web_sm`, and the reason the current `news` packages stay
prose-only.

### 3.3 Vectors, for md and lg

| Option | License | Note |
| --- | --- | --- |
| floret vectors trained via `spacy-vectors-builder` (MIT tooling) on fa Wikipedia + OSCAR | corpus-dependent | recommended: subword + Bloom-hash embeddings, bounded table size, zero OOV |
| fastText `cc.fa.300` | CC BY-SA 3.0 | fallback; classic word table, large and OOV-prone |

Persian surface forms multiply through suffixation and through inconsistent ZWNJ (U+200C)
usage: the same word appears as `می‌رود`, `میرود` and `می رود` in real text. A classic
word-vector table misses every variant it did not see, while floret's subword hashing covers
them. spaCy ships floret vectors for Croatian, Finnish, Korean, Slovenian, Swedish and
Ukrainian for the same reason.

### 3.4 Transformer encoders, for trf

| Model | Arch | License | Note |
| --- | --- | --- | --- |
| `HooshvareLab/roberta-fa-zwnj-base` | RoBERTa-base | Apache-2.0 | recommended: licensed, ZWNJ-aware, smallest credible option |
| `FacebookAI/xlm-roberta-base` | XLM-R base, 278M | MIT | licensed but larger |
| `PartAI/TookaBERT-Base` | BERT-base | Apache-2.0 | licensed |
| `m3hrdadfi/albert-fa-base-v2` | ALBERT-base-v2 | Apache-2.0 | licensed, smallest |
| `HooshvareLab/bert-base-parsbert-uncased` | BERT-base, ~162M | no licence on the card | what hazm used; redistribution risk |
| `sbunlp/fabert` | BERT-base, 124M | no licence on the card | redistribution risk |

All are BERT, RoBERTa, XLM-R or ALBERT, so all work with `spacy-transformers` and with
`spacy-curated-transformers`, which supports exactly ALBERT, BERT, CamemBERT, RoBERTa and
XLM-RoBERTa.

hazm's pipelines use ParsBERT, which carries no licence statement. Copying that choice would
reintroduce the redistribution problem that stopped the 2018 attempt.

## 4. What already exists, and why it is not enough

No trained spaCy Persian pipeline exists. There are zero `persian`, `farsi` or `fa_` hits in
spaCy's `website/meta/universe.json`, and `spacy.load("fa_core_news_sm")` has never worked.

hazm ships three single-task spaCy pipelines on the HF Hub: `hazm-parsbert-postagger`
(`tag_acc` 0.9862, hazm's own EZ-augmented tagset), `hazm-bert-dependency-parser` (`dep_uas`
0.9246, `dep_las` 0.8934, trained on PerDT) and `hazm-parsbert-chunker` (`tag_acc` 0.9618).
Each is `transformer + one component`, `version: 0.0.0`, with empty `license`, `author` and
`sources`, pinned to spaCy 3.6. Using all three costs three BERT forward passes over the same
text and gives no shared `Doc`.

hazm's training code is not reusable: its trainable models are pycrfsuite CRFs
(`hazm/sequence_tagger.py`), and the repo contains no `config.cfg` or `spacy train`. Its
`Spacy*` classes only download the HF pipelines above.

hazm's tokenizer is incompatible with UD gold tokenization. `Normalizer`'s
`AFFIX_SPACING_PATTERNS` fuse ZWNJ affixes, and `WordTokenizer.join_verb_parts()` glues
multi-word verb chains into single underscore-joined tokens, so training against PerDT with it
would misalign tokens systematically.

DadmaTools (Apache-2.0 code) emits spaCy-compatible `Doc` objects but is not a loadable spaCy
pipeline package, and its NER wraps ARMAN and PEYMA.

What hazm contributes here is one validated design decision, that PerDT is the corpus, and the
stop-word list already vendored into `spacy/lang/fa`.

## 5. Decision: train the `sm` tier first

Not md or lg: they need floret vectors trained from scratch on Wikipedia and OSCAR, costing
CPU-days, and the English reference numbers show no tag or dep accuracy gain. Not trf: 2 GB of
VRAM cannot fine-tune a 125M-param encoder, and renting a GPU should wait until the CPU
pipeline proves the data plumbing. `sm` is also the tier the others are validated against,
since md, lg and trf reuse the same corpus conversion, config skeleton and evaluation harness.

### Tokenization, settled by measurement

spaCy has no multi-word-token layer, so CoNLL-U MWT ranges (Persian pronominal clitics and
copulas, `پدرم` = `پدر` + `م`) must either be merged into one token or kept split. Measured on
the PerDT dev set with `scripts/tokenization_report.py`, comparing gold boundaries against
`spacy.blank("fa")`'s tokenizer:

| `spacy convert` mode | token P | token R | token F | XPOS types | merge artefacts |
| --- | --- | --- | --- | --- | --- |
| `--merge-subtokens` | 0.9860 | 0.9914 | 0.9887 | 68 | 34 composite tags on 1.49% of tokens; 1.49% lemmas contain a space |
| plain (clitics split) | 0.9875 | 0.9772 | 0.9823 | 35 | none |

Chosen: `--merge-subtokens`, which is also what Explosion's `tagger_parser_ud` template does.
Without it, 1.5% of gold tokens are boundaries the shipped tokenizer can never produce, so
those tokens are permanently unlearnable and unpredictable at runtime. With it, every gold
token is reachable, and the cost is limited to rare composite XPOS tags such as
`N_IANM_PR_JOPER` (noun + enclitic pronoun) and 1.5% of lemmas that come out as two words.

The better long-term fix is Persian clitic-splitting suffix rules in `spacy/lang/fa`, which is
an upstream PR rather than a model change. Recorded in `docs/CONTRIBUTING-GUIDE.md` §5.

### Final composition

Shared trained components, in both `fa_dep_news_sm` (7.5 MB) and `fa_core_news_sm` (13 MB),
both CC BY-SA 4.0:

| Component | Trained on | Metric | Test score |
| --- | --- | --- | --- |
| `tok2vec` | shared, PerDT | | |
| `tagger` (XPOS) | PerDT, 90 labels | `tag_acc` | 95.96 |
| `morphologizer` (UPOS + FEATS) | PerDT, 298 labels | `pos_acc` / `morph_acc` | 96.24 / 96.29 |
| `trainable_lemmatizer` | PerDT, 1,908 edit trees | `lemma_acc` | 97.91 |
| `parser` | PerDT, 34 deprels | `dep_uas` / `dep_las` | 89.69 / 85.15 |

`fa_core_news_sm` adds, and `fa_ent_news_sm` ships alone:

| Component | Trained on | Metric | Test score |
| --- | --- | --- | --- |
| `ner` (own internal tok2vec) | PerDT NER layer, 7 labels | `ents_p/r/f` | 77.67 / 66.87 / 71.87 |

Per label, F against training examples: `LOC` 80.24 (4,954), `DAT` 74.45 (1,323), `MON` 73.68
(205), `ORG` 68.77 (2,643), `TIM` 66.67 (135), `PER` 65.29 (4,847), `PCT` 57.14 (121).

`PER` scoring below `LOC` and `ORG` on nearly the same amount of data is the silver labels
showing through. PerDT includes titles and honorifics inside `PER` spans inconsistently: 6.24% of
its `PER` spans start with one (`دکتر`, `مهندس`, `آقای`), against 1.41% in the human-annotated
ParsTwiNER, so the boundaries are less regular than the count suggests. `MON`, `TIM` and `PCT`
are thin enough that their scores rest on 4 to 11 test entities each and should be treated as
indicative only. Persian money, times and percentages are regular enough that an `EntityRuler`
may beat the statistical model for those three.

Training cost on a 4-core i5-7200U with no GPU: 1h27m for the UD components (early-stopped at
step 10,800, best checkpoint step 9,200) and 17 min for NER (best at step 6,000). Both runs are
single-threaded and can run concurrently. Inference runs at about 9,250 words/s.

The `--merge-subtokens` artefacts show up in the shipped model as predicted: `کتاب‌هایش`
("his/her books") is one token tagged `N_IANM_PR_JOPER` with lemma `کتاب او`. Check this before
consuming `token.lemma_` downstream.

Sources are recorded in each `meta.json` with their licences, per
<https://spacy.io/api/data-formats#meta>, including the treebank's NER layer as its own entry
crediting Beheshti-NER. CC BY-SA 4.0 on PerDT means every package carries attribution and a
share-alike notice, handled in `scripts/finalize_pipeline.py`, which also enforces the shape of
each variant: it refuses to publish a `dep` pipeline containing `ner`, or a `core` one without it.

## 6. The `md` tier: floret static vectors

Built after the `sm` tier, from `fa_floret`: 50,000 rows x 300d, floret mode, `minn=maxn=5`,
`hash_count=2`, trained on 400,000 Persian documents. The wheel is a vectors-only pipeline;
`scripts/unpack_vectors.py` unwraps it into a directory `--paths.vectors` can read, so nothing
needs pip-installing to train against it.

`configs/fa_dep_news_md.cfg` and `configs/fa_ner_md.cfg` are their `sm` counterparts with one
line changed, `include_static_vectors = false -> true`. Same seed, same corpus, same widths,
same batcher, same patience. The deltas below are therefore attributable to the vector table
and nothing else. Reproduce with `spacy project run md`, or the table alone with
`python scripts/compare_tiers.py`.

### UD_Persian-PerDT test split

| Metric | `sm` | `md` | Delta |
| --- | --- | --- | --- |
| `TAG_ACC` | 95.96 | 96.25 | +0.29 |
| `POS_ACC` | 96.24 | 96.64 | +0.40 |
| `MORPH_ACC` | 96.29 | 96.64 | +0.35 |
| `LEMMA_ACC` | 97.91 | 97.96 | +0.05 |
| `SENTS_F` | 99.25 | 99.28 | +0.03 |
| `DEP_UAS` | 89.69 | 90.52 | +0.83 |
| `DEP_LAS` | 85.15 | 86.34 | +1.19 |
| Speed (dep) | 12,505 w/s | 10,493 w/s | -16.1% |

### PerDT NER test split, `fa_core_news_md`

| Metric | `sm` | `md` | Delta |
| --- | --- | --- | --- |
| `ENTS_P` | 77.67 | 76.56 | -1.10 |
| `ENTS_R` | 66.87 | 72.95 | +6.08 |
| `ENTS_F` | 71.87 | 74.71 | +2.85 |

Almost all of the NER gain is recall. That is the expected shape of a fix for a coverage
problem: hash embeddings had no lexical prior for rare proper nouns, so the `sm` model
declined to tag them. Precision slips ~1 point because the model now guesses more.

| Label | Gold in test | `sm` F | `md` F | Delta |
| --- | --- | --- | --- | --- |
| `PER` | 297 | 65.29 | 68.18 | +2.89 |
| `LOC` | 273 | 80.24 | 84.05 | +3.81 |
| `ORG` | 144 | 68.77 | 70.25 | +1.48 |
| `DAT` | 69 | 74.45 | 76.19 | +1.74 |
| `MON` | 10 | 73.68 | 84.21 | +10.53 |
| `TIM` | 9 | 66.67 | 66.67 | +0.00 |
| `PCT` | 4 | 57.14 | 33.33 | -23.81 |

Read the bottom three rows as noise, not signal. `PCT` has four gold entities in the whole
test split, so its -23.81 F is one entity changing hands; `MON`'s +10.53 is likewise one of
ten. The three labels with real support (`PER`, `LOC`, `ORG`, 714 entities between them) all
improve, which is the finding.

### Cost

The vectors dominate the artifact: `fa_dep_news_md` is a 62 MB wheel against 7.5 MB for `sm`,
`fa_core_news_md` 68 MB against 13 MB. Inference is ~16% slower across all three pipelines,
a uniform hit consistent with the extra 300d concatenation per token rather than anything
component-specific. Training cost was comparable to `sm` (early stop at step 12,400 of 20,000,
best checkpoint near 10,800).

Whether that trade is worth it depends on deployment. For a 1.19 LAS and 2.85 NER F gain, a
9x larger download and 16% slower parse is a good deal on a server and a bad one in a browser
or a Lambda cold start. Both tiers ship; pick per target.

## 7. The `lg` tier: bigger floret table, full pipeline

Built after `md`, from a new `fa_floret` table: 200,000 rows x 300d, floret mode,
`minn=maxn=5`, `hash_count=2`, trained on the full Persian Wikipedia dump for 5 epochs (4x
the rows of `md`'s 50k-row table trained on 400k documents). Raw `.floret`/`.vec` and the
packaged spaCy wheel are at <https://huggingface.co/Phazel/fa-floret-wiki-vectors>. Unpacked
the same way as `md` via `scripts/unpack_vectors.py`, into `assets/vectors/fa_floret_lg`.

`configs/fa_ner_lg.cfg` and `configs/fa_dep_news_lg.cfg` are `fa_ner_md.cfg`/
`fa_dep_news_md.cfg` unchanged except `--paths.vectors`. Same seed, same corpus, same
architecture as `sm`/`md` throughout, so the deltas below are attributable to the vector
table alone. Reproduce with `spacy project run lg`, or the tables alone with
`python scripts/compare_tiers.py`.

### UD test split, `fa_dep_news_lg` / `fa_core_news_lg`

| Metric | `sm` | `md` | `lg` | Delta (lg vs sm) | Delta (lg vs md) |
| --- | --- | --- | --- | --- | --- |
| `TAG_ACC` | 95.96 | 96.25 | 96.55 | +0.59 | +0.30 |
| `POS_ACC` | 96.24 | 96.64 | 96.68 | +0.44 | +0.04 |
| `MORPH_ACC` | 96.29 | 96.64 | 96.70 | +0.41 | +0.06 |
| `LEMMA_ACC` | 97.91 | 97.96 | 98.08 | +0.17 | +0.12 |
| `DEP_UAS` | 89.69 | 90.52 | 90.96 | +1.27 | +0.44 |
| `DEP_LAS` | 85.15 | 86.34 | 86.60 | +1.45 | +0.26 |

`lg` beats `md` on every UD metric, the same monotonic pattern as `md` beating `sm` in §6.
The bigger, less collision-prone floret table keeps paying off, though the `md`-to-`lg`
gains (4x the vector rows) are smaller than the `sm`-to-`md` gains (going from none to 50k
rows): diminishing returns, as expected.

### PerDT NER test split, `fa_ent_news_lg` (identical `ner` component embedded in `fa_core_news_lg`)

| Metric | `sm` | `md` | `lg` | Delta (lg vs sm) | Delta (lg vs md) |
| --- | --- | --- | --- | --- | --- |
| `ENTS_P` | 77.67 | 76.56 | 81.51 | +3.84 | +4.95 |
| `ENTS_R` | 66.87 | 72.95 | 71.09 | +4.22 | -1.86 |
| `ENTS_F` | 71.87 | 74.71 | 75.94 | +4.08 | +1.23 |

`lg` beats both `sm` and `md` on `ENTS_F`, and unlike `md`'s recall-only gain over `sm`, `lg`
improves precision too (+3.84 over `sm`, whereas `md` cost -1.10). Consistent with a bigger,
less collision-prone floret table giving both better recall on rare proper nouns and fewer
false positives from hash collisions.

| Label | Gold in test | `sm` F | `md` F | `lg` F | Delta (lg vs sm) |
| --- | --- | --- | --- | --- | --- |
| `PER` | 297 | 65.29 | 68.18 | 72.63 | +7.33 |
| `LOC` | 273 | 80.24 | 84.05 | 83.66 | +3.42 |
| `ORG` | 144 | 68.77 | 70.25 | 71.01 | +2.24 |
| `DAT` | 69 | 74.45 | 76.19 | 70.83 | -3.62 |
| `MON` | 10 | 73.68 | 84.21 | 88.89 | +15.20 |
| `TIM` | 9 | 66.67 | 66.67 | 61.54 | -5.13 |
| `PCT` | 4 | 57.14 | 33.33 | 57.14 | +0.00 |

`PER`, `LOC` and `ORG` (714 entities, the labels with real support) all improve over both
smaller tiers. `DAT` and `TIM` regress a few points against `md`; `MON`/`TIM`/`PCT` swings are
one-or-two-entity noise, same caveat as §6.

### Cost

The bigger table dominates the artifact even more than `md`'s did: the 200k x 300d float32
vector table is ~240 MB uncompressed, so `fa_dep_news_lg` is a 219 MB wheel (vs 7.5 MB `sm`,
60 MB `md`), `fa_core_news_lg` 225 MB (vs 13 MB `sm`, 66 MB `md`), and `fa_ent_news_lg` alone
217 MB (vs 5.6 MB `sm`, 58 MB `md`). Training cost roughly doubled `md`'s: `dep_lg` ran to
early stop at step 12,000 of 20,000 over ~2h08m CPU wall time (vs `dep_md`'s single-digit
minutes territory implied by its architecture-identical config; `lg`'s extra time is
entirely the larger embedding table's per-step cost, not more steps). `ner_lg` early-stopped
at step 7,200, ~13 min, in line with `sm`/`md`.

`words/s` from `spacy benchmark accuracy` were noisier at this tier than `sm`-vs-`md`: dep/core
throughput dropped as expected (9,387 / 6,655 words/s vs `sm`'s 12,505 / 8,834, `md`'s
10,493 / 7,269 words/s; the larger table costs real lookup time), but the standalone `ent_lg` run
showed 15,614 words/s, higher than `sm`/`md`'s ent runs despite an identical `ner`
architecture and the same larger table. That figure was single-run CPU contention noise on
shared hardware, not a real speedup. Those numbers are superseded by §9, which times
`nlp.pipe` alone instead of reading a scoring-contaminated figure off the benchmark command.

For a 4x download over `md` (and up to 39x over `sm`) buying +1.45 DEP_LAS / +1.23 ENTS_F
over `md` (+1.45 DEP_LAS / +4.08 ENTS_F over `sm`), `lg` is a server/offline-batch pipeline,
not something to ship to a browser or a cold-start function. All three variants (`dep`,
`ent`, `core`) are built and evaluated at this tier, same as `md`.

## 8. The `trf` tier: one fine-tuned ParsBERT

`configs/fa_core_news_trf.cfg` replaces the static-vector tok2vec with
`HooshvareLab/bert-base-parsbert-uncased`, fine-tuned during training. Trained on a rented
Colab T4 in 1h58m: 3000 steps, no early stop, the full learning-rate anneal.

### One corpus, because a transformer cannot be trained twice

The `sm`/`md`/`lg` tiers train `ner` as its own pipeline with its own embedded tok2vec and
then source it into the dep model. That is affordable because a hash-embed tok2vec is cheap.
A 162M-parameter encoder is not: fine-tuning it once per component would double GPU cost and
put two encoders in one wheel, and sourcing the second would collide on the `transformer`
component name.

So every component listens to a single shared transformer through a `TransformerListener`,
which requires one corpus carrying both the UD and NER annotation layers on the same `Doc`.
`scripts/merge_joint_corpus.py` builds it. The fusion is exact rather than approximate:
`corpus/perdt-ner/` was converted from the same `--merge-subtokens` CoNLL-U as
`corpus/merged/` with the same `--n-sents`, so the two DocBins are token-for-token identical.
The script asserts that per document and copies only `doc.ents` across. Char offsets are not
usable for the copy, because the two converters differ in trailing whitespace, which shifts
`char_span` off the token grid and returns None; the transfer goes by token index.

### Results against `lg`

| Metric | `lg` | `trf` | Delta |
| --- | ---: | ---: | ---: |
| `TAG_ACC` | 96.55 | 97.62 | +1.07 |
| `POS_ACC` | 96.68 | 97.63 | +0.95 |
| `MORPH_ACC` | 96.70 | 97.82 | +1.12 |
| `LEMMA_ACC` | 98.08 | 97.31 | -0.77 |
| `DEP_UAS` | 90.96 | 93.87 | +2.91 |
| `DEP_LAS` | 86.60 | 90.79 | +4.19 |
| `SENTS_F` | 99.18 | 97.35 | -1.83 |
| `ENTS_F` | 75.94 | 82.89 | +6.95 |

The parser gain is the headline: `DEP_LAS` 90.79 passes the hazm+ParsBERT reference of 89.34,
which no CPU tier reached. NER gains 6.95 F, almost all of it recall (71.09 to 81.76) at
higher precision, which is what a pretrained encoder buys on the difflib-transferred layer.

Two metrics regress. `SENTS_F` drops 1.83, most likely because `strided_spans` at
`window = 128, stride = 96` leaves 32 tokens of overlap, so tokens near a span edge see
truncated right context where the CPU tiers' tok2vec sees the whole doc. `LEMMA_ACC` drops
0.77 and is the one metric where a static-vector tier wins: `trainable_lemmatizer` reads a
single `reduce_mean`-pooled vector per token, while `lg` runs an edit-tree lemmatizer over
floret subwords that model Persian orthography directly. Neither is a training-length
problem; see TODO.md for the evidence that more steps do not help.

### Cost, and the licence problem

608 MB wheel, 2.6x `lg` and 45x `sm`. 187 words/s on the laptop CPU against `sm`'s 5,484
(§9), so this tier needs a GPU in production rather than merely benefiting from one.

ParsBERT's model card states no licence. §3.4 picked `HooshvareLab/roberta-fa-zwnj-base`
(Apache-2.0) for exactly this reason, and the published wheel therefore embeds weights whose
redistribution terms are unknown. `scripts/finalize_pipeline.py` reads the encoder name out
of the trained config and writes a redistribution warning into `meta.json` when the encoder
has no licence, so the artifact carries the caveat. Retraining on the Apache-2.0 encoder is a
one-line change to `name` in the config.

## 9. Throughput

Measured with `scripts/benchmark_throughput.py`, which times `nlp.pipe` and nothing else.
The `words/s` printed by `spacy benchmark accuracy` runs the Scorer's per-token alignment
inside the timed region, which is why the §7 numbers disagree with these and why one of them
was impossible.

Median of repeated passes over the 146-document PerDT test split (23,825 tokens), batch 32,
warmup discarded. Raw records in `metrics/throughput-*.json`.

| Tier | CPU, i5-7200U | GPU, GeForce 940MX | CPU, Xeon @ 2.00GHz | GPU, Tesla T4 |
| --- | ---: | ---: | ---: | ---: |
| `sm` | 5,484 | 10,235 | | |
| `md` | 5,408 | 9,058 | | |
| `lg` | 4,715 | 9,215 | | |
| `trf` | 187 | 1,106 | 336 | 8,320 |

The CPU tiers sit within about 15% of each other, less than their vector-table sizes suggest,
so the tok2vec lookup is not the bottleneck; the parser and lemmatizer are. Run-to-run spread
on the laptop is roughly 10% either way with thermal state, and a background rsync halved
every number, so treat small differences as noise.

`trf` is 29x slower than `sm` on the same CPU. The T4 and Xeon columns come from the same Colab
VM, giving a clean 25x GPU speedup for the transformer.

`trf` on the 940MX needs a `cu126` torch build. sm_50 kernels were dropped from the `cu128` and
`cu129` wheels at torch 2.8, which is what `pip install torch` resolves to. `.venv-trf-gpu` pins
`torch==2.7.1+cu126`, separate from `.venv` because torch's pinned `nvidia-*` wheels downgrade
the CUDA libraries cupy uses there from 12.9 to 12.6. Batch 32 fits in 2 GB.
