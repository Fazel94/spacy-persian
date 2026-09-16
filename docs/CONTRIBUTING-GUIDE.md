# Getting a Persian pipeline into the spaCy ecosystem

Sourced from spaCy's primary docs and repos, URLs inline. Where a contribution goes decides
how it has to be shaped, so read this before writing code.

## 0. Two separate contribution surfaces

| Surface | What it is | Where it lives | How you contribute |
| --- | --- | --- | --- |
| Language data (`fa`) | Hand-written rules: tokenizer exceptions, stop words, `LIKE_NUM`, punctuation, noun-chunk iterator | `spacy/lang/fa/*.py` inside the spaCy repo | Normal PR to `explosion/spaCy` |
| Trained pipeline (`fa_dep_news_sm`) | Statistical weights + `config.cfg` + `meta.json`, shipped as a pip wheel | `explosion/spacy-models` releases | You cannot. Publish it yourself (PyPI or HF Hub) and get it listed in spaCy Universe |

`spacy/lang/fa` already exists upstream. No trained `fa` pipeline did when this started; the
ten packages under `Phazel/` on the HF Hub are the output of this project. So this is a
publishing project, with optional upstream PRs for the language-data gaps found along the way.

## 1. Policy, verbatim

From <https://github.com/explosion/spaCy/blob/master/CONTRIBUTING.md>:

- No CLA. There is no contributor licence agreement section; contribution is an ordinary
  GitHub PR governed by the Contributor Covenant Code of Conduct v1.4.
- Inclusion philosophy:
  > "Our philosophy is to prefer a smaller core library. […] If you're looking to implement a
  > new spaCy feature, starting with a custom component package is usually the best strategy.
  > […] And if it works well, we can always integrate it into the core library later."
- The sanctioned route for non-trivial additions is the "Publishing spaCy extensions and
  plugins" section:
  > "An extension or plugin should add substantial functionality, be well-documented and
  > open-source. It should be available for users to download and install as a Python package
  > – for example via PyPi."
  > "Once your extension is published, you can open a PR to suggest it for the Universe page."
- `CONTRIBUTING.md` never mentions submitting trained models. Neither does
  <https://github.com/explosion/spacy-models>, whose README only documents `compatibility.json`
  as "the source of spaCy's internal compatibility check, performed when you run the download
  command". There is no PR template, issue label, or documented process for a third party to
  land a pipeline in `spacy download`. So the `spacy download` index is an Explosion-only
  release channel. (`[INFERENCE]` from the absence of any process across all three sources.)
- `https://spacy.io/usage/adding-languages` no longer exists and redirects to
  <https://spacy.io/usage/linguistic-features#language-data>. The `BaseDefaults` and `Language`
  contract is at <https://spacy.io/api/language>.

### Code rules for a `spacy/lang/fa` PR

- `black` formatting, `flake8` clean, type hints where practical.
- Tests go in `spacy/tests/lang/fa/`; regression tests use `@pytest.mark.issue(N)`.
- Touching `.pyx` means the reviewer must rebuild: `python setup.py build_ext --inplace`.
- Language data must be data and rules only: no model downloads, no network, no heavy deps.

## 2. Three publishing routes for the trained pipeline

1. Hugging Face Hub, using Explosion's own tool
   (<https://github.com/explosion/spacy-huggingface-hub>). It needs its own virtualenv:
   version 0.0.10 pins `typer<0.8`, and typer 0.7 against click 8.4 makes every `spacy`
   command die with `TypeError: Secondary flag is not valid for non-boolean flag`, so
   installing it into the training venv breaks training.
   ```bash
   /home/fazel/anaconda3/envs/p12/bin/python -m venv .venv-publish
   .venv-publish/bin/python -m pip install "spacy>=3.8,<3.9" spacy-huggingface-hub "click<8.2"
   ```
   Publishing needs an HF **write** token (`hf_...`); the account password will not work.
   Strip the proxy for the upload: `huggingface.co` is reachable through it, but multi-MB LFS
   blobs die mid-transfer (`RemoteProtocolError: Server disconnected without sending a
   response`) regardless of retries.
   ```bash
   export HF_TOKEN=$(rbw get api/huggingface_token)
   NO_PROXY="*" no_proxy="*" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
     ALL_PROXY="" all_proxy="" \
     .venv-publish/bin/python -m spacy huggingface-hub push \
       packages/fa_dep_news_sm-3.8.0/dist/fa_dep_news_sm-3.8.0-py3-none-any.whl -V
   ```
   Omit `--org` to publish into the personal namespace. Then upload a second copy under the
   real versioned filename, because `spacy huggingface-hub push` renames the artifact to
   `<name>-any-py3-none-any.whl`, and `any` is not a valid PEP 440 version, so current pip
   rejects it with `Invalid wheel filename (invalid version)`:
   ```python
   from huggingface_hub import HfApi
   HfApi().upload_file(
       path_or_fileobj="packages/fa_dep_news_sm-3.8.0/dist/fa_dep_news_sm-3.8.0-py3-none-any.whl",
       path_in_repo="fa_dep_news_sm-3.8.0-py3-none-any.whl",
       repo_id="Phazel/fa_dep_news_sm",
   )
   ```
   Link the versioned copy; leave the `-any-` file in place, unlinked. Users then
   `pip install https://huggingface.co/Phazel/fa_dep_news_sm/resolve/main/fa_dep_news_sm-3.8.0-py3-none-any.whl`.
2. PyPI or a self-hosted wheel: `spacy package … --build sdist,wheel` then `twine upload`, or
   attach the wheel to a GitHub Release. See <https://spacy.io/api/cli#package>.
3. spaCy Universe, which lists the package on spacy.io but hosts nothing. Per
   <https://github.com/explosion/spaCy/blob/master/website/UNIVERSE.md>: fork `explosion/spaCy`,
   append an entry to `website/meta/universe.json`, open a PR. Required fields: `id`, `title`,
   `slogan`, `description`, `github`, `pip`, `code_example`, `code_language`, `url`, `thumb`,
   `image`, `author`, `author_links`, `category`, `tags`. The package must be open-source with
   a user-friendly license and "at least somewhat documented".

Route taken here: HF Hub for the artifacts, all ten published under `Phazel/`. The Universe
PR is still open work.

## 3. Naming and versioning

From <https://spacy.io/models#conventions> and the `spacy-models` README:

```
[lang]_[type]_[genre]_[size]        e.g. fa_dep_news_sm
```

| Slot | Allowed values | Meaning |
| --- | --- | --- |
| `type` | `core` | tagger + parser + lemmatizer + NER |
| | `dep` | tagger + parser + lemmatizer, no NER |
| | `ent` | NER only |
| | `sent` | sentence segmentation only |
| `genre` | `news`, `web`, `wiki` | text domain of the training corpus |
| `size` | `sm` | no static vectors |
| | `md` | ~20k unique vectors / ~500k keys (or 50k floret rows) |
| | `lg` | ~500k vectors (or 200k floret rows) |
| | `trf` | transformer, no static vectors |

Version numbers are `a.b.c` = spaCy-major, spaCy-minor, model-revision. Trained against spaCy
3.8, so the package version starts at `3.8.0`; retraining on new data bumps `c`. Avoid
`0.1.0`-style versions, which is how hazm's HF pipelines ended up with `version: 0.0.0` and
`spacy_version: >=3.6.0,<3.7.0`.

### `meta.json` fields that matter (<https://spacy.io/api/data-formats#meta>)

`lang`, `name`, `version`, `spacy_version` (range), `description`, `author`, `email`, `url`,
`license`, `sources` (list of `{name, url, author, license}`, which is where treebank
provenance and its CC BY-SA obligation is recorded), `requirements` (extra pip deps injected
into the generated `setup.cfg`), `vectors`, `pipeline`, `labels`, `performance` (auto-filled by
`spacy train`), `speed`, `spacy_git_version`.

In v3, `meta.json` "isn't used to construct the language class and pipeline anymore".
`config.cfg` is the single source of truth for loading; `meta.json` is metadata and packaging.

## 4. Canonical training workflow

Template: <https://github.com/explosion/projects/tree/v3/pipelines/tagger_parser_ud>
(`project.yml` + `configs/default.cfg`). The directory layout `assets / corpus / configs /
training / metrics / packages` is the convention, which this repo follows.

```bash
# 1. assets: clone the UD treebank
git clone https://github.com/UniversalDependencies/UD_Persian-PerDT assets/UD_Persian-PerDT

# 2. corpus: CoNLL-U -> binary DocBin
python -m spacy convert assets/.../fa_perdt-ud-train.conllu corpus/ \
    --converter conllu --n-sents 10 --merge-subtokens

# 3. config
python -m spacy init config configs/default.cfg --lang fa \
    --pipeline tagger,morphologizer,trainable_lemmatizer,parser --optimize efficiency
python -m spacy init fill-config partial.cfg configs/default.cfg   # completes defaults

# 4. (md/lg only) vectors
python -m spacy init vectors fa cc.fa.300.vec vectors/ --mode floret --prune 200000

# 5. train
python -m spacy train configs/default.cfg --output training/ --gpu-id -1 \
    --paths.train corpus/train.spacy --paths.dev corpus/dev.spacy

# 6. evaluate on held-out test
python -m spacy benchmark accuracy training/model-best corpus/test.spacy --output metrics/test.json
#   ('spacy evaluate' is now an alias for 'benchmark accuracy')

# 7. package
python -m spacy package training/fa_dep_news_sm packages --name dep_news_sm --version 3.8.0 --build sdist,wheel
```

`spacy assemble` builds a pipeline from a config without training, which suits a rule-only
artifact (tokenizer + lookup lemmatizer + stop words).

### Flags that matter on `spacy convert`

- `--converter conllu`, explicit rather than `auto`.
- `--n-sents N`, how many sentences per `Doc`. 10 is the Explosion default and gives the parser
  and senter cross-sentence context.
- `--merge-subtokens`. spaCy has no multi-word-token layer, and Persian UD treebanks use MWT
  ranges for clitics (`پدرم` = `پدر` + `م`), so this flag decides whether gold tokens are
  clitic-split or fused. See `docs/MODELS.md` §5.
- `--morphology`, which appends morph features to the tag. Only for pipelines without a
  morphologizer.

### How the size tiers differ mechanically

| Tier | Embedding source | Config difference |
| --- | --- | --- |
| `sm` | hash embeddings only | `spacy.MultiHashEmbed` with `include_static_vectors = false`, `spacy.MaxoutWindowEncoder` width 96, every other component a `spacy.Tok2VecListener` on the shared `tok2vec` |
| `md` / `lg` | + static vectors | same CNN, `include_static_vectors = true`, `[initialize] vectors = <dir built by init vectors>`; `lg` has a bigger table |
| `trf` | contextual | replace `tok2vec` with a `transformer` component; other components listen via `TransformerListener` instead of `Tok2VecListener` |

`spacy pretrain` (Tok2Vec LM-style pretraining on raw text into `[initialize] init_tok2vec`) is
orthogonal to the tier and optional.

## 5. Upstream `spacy/lang/fa`: current state and gaps

Contents of <https://github.com/explosion/spaCy/tree/master/spacy/lang/fa>:

| File | Size | What it provides |
| --- | --- | --- |
| `__init__.py` | 1.3 KB | `PersianDefaults`: tokenizer exceptions, `TOKENIZER_SUFFIXES`, `LEX_ATTRS`, `SYNTAX_ITERATORS`, `STOP_WORDS`, `writing_system = {"direction": "rtl", "has_case": False, "has_letters": True}`; registers a `lemmatizer` factory defaulting to `mode="rule"` |
| `tokenizer_exceptions.py` | 64.9 KB | Generated compound-verb and enclitic exception table |
| `generate_verbs_exc.py` | 14.8 KB | Dev script that generates the above |
| `stop_words.py` | 3.8 KB | ~500 stop words; the comment says "Stop words from HAZM package" |
| `lex_attrs.py` | 1.4 KB | `LIKE_NUM` only (Persian numerals plus `ام` and `ین` suffixes) |
| `punctuation.py` | 508 B | `TOKENIZER_SUFFIXES` only, no prefixes and no infixes |
| `syntax_iterators.py` | 1.6 KB | `noun_chunks()`, which needs a trained parser |

Tests: `spacy/tests/lang/fa/` has only `test_noun_chunks.py`. No tokenizer, lemmatizer or
stop-word tests.

`spacy-lookups-data` (MIT) already ships Persian rule-lemmatizer tables:

| File | Size |
| --- | --- |
| `fa_lemma_exc.json` | 1.68 MB |
| `fa_lemma_index.json` | 171 KB |
| `fa_lemma_rules.json` | 882 B |
| `fa_source.txt` | "extracted from Mojgan Seraji's Persian Universal Dependencies Corpus" |

Missing there: `fa_lemma_lookup.json` (no lookup-mode table), `fa_lexeme_norm.json`, and
`fa_license.txt` (Catalan has one; Persian's Seraji-derived provenance is undocumented).

Gaps blocking a full `fa_core_news_*`:

1. No trained artefacts: no weights, no `config.cfg`, no `meta.json`, no entry in
   `compatibility.json`. `spacy.load("fa_core_news_sm")` fails today.
2. No word vectors for `fa`, so `md` and `lg` need vectors built from scratch.
3. No NER data or labels anywhere in `spacy/lang/fa`, so `core` needs an external corpus.
4. The rule lemmatizer needs `token.pos` from a tagger or morphologizer, which is
   chicken-and-egg until the tagger exists. `trainable_lemmatizer` avoids this by learning edit
   trees and needing no tables.
5. No Persian-specific prefix or infix rules. ZWNJ (U+200C) is handled only implicitly through
   the verb-exception table.
6. No tokenizer test coverage for a 65 KB exception table.

Items 5 and 6 are upstream PR material, along with the `noun_chunks` bug in
`docs/upstream/fa-noun-chunks.md`. Items 1 to 4 are this project's job.

## 6. Prior art

hazm publishes spaCy-format Persian pipelines on the HF Hub, each a single-task pipeline built
on ParsBERT, with the `license` field left empty:

| HF repo | Pipeline | Reported | Trained on (`[paths]` in its `config.cfg`) |
| --- | --- | --- | --- |
| `roshan-research/hazm-parsbert-postagger` | `transformer, tagger` | `tag_acc` 0.9862 | `data_train_98_rs10.spacy` (hazm's own EZ-augmented tagset: `NOUN,EZ`, `ADJ,EZ`, …) |
| `roshan-research/hazm-bert-dependency-parser` | `transformer, parser` | `dep_uas` 0.9246 / `dep_las` 0.8934 | `modified_fa_perdt-ud-train.spacy`, i.e. UD_Persian-PerDT |
| `roshan-research/hazm-parsbert-chunker` | `transformer, tagger` (IOB chunk tags) | `tag_acc` 0.9618 | hazm chunk data |

All three use `spacy-transformers` `TransformerModel.v3` on
`HooshvareLab/bert-base-parsbert-uncased`, `strided_spans` window 128 and stride 96, with
`spacy_version >= 3.6.0,<3.7.0`.

Problems this project avoids:

- Three separate pipelines rather than one, so users pay for three BERT forward passes and
  cannot share a `Doc`.
- `version: 0.0.0`, empty `license`, `author` and `sources`, and `name: "pipeline"`, which
  breaks the conventions in §3 and leaves redistribution rights unclear.
- Transformer-only, with no CPU-friendly tier.
- No lemmatizer, morphologizer, NER or vectors.
- Pinned to spaCy 3.6, so unusable on 3.8 without retraining.

The reusable finding is hazm's parser recipe, UD_Persian-PerDT plus spaCy's
`TransitionBasedParser`, which independently confirms the corpus choice in `docs/MODELS.md`.
