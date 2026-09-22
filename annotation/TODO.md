# Open work on the NER annotation kit — v3 agenda

Written 2026-09-20. Current state: guideline **v2.2**, prompt `prompts/ner-v2.2.md`
(`prompt_hash 3c5f435761a2`), outputs in `data/llm/ner-v2.2-default/`.

## Context for whoever picks this up

The kit relabels PerDT's silver NER layer with an LLM, using a guideline inferred from that
same layer. `GUIDELINES.md` states each convention with the train-split counts behind it, so
rules are falsifiable against the corpus rather than argued. The pipeline is five scripts in
`../scripts/annotation/`: select → build_requests → annotate.js → validate_responses →
compare_to_silver. `work/` holds raw responses and the response cache and is gitignored.

Coverage so far: **22,300 of 26,196 train sentences** (shards 000-010 complete, shard-011
first 300) and all **1,455 test sentences**; `pool-500` and `select-100` are test subsets
kept from guideline development.

Measured against silver: micro F **0.742** on train 000-010, **0.754** on test-all.
Adjudicating every test-split disagreement span by span put the annotations at **0.934-0.946
precision** against silver's **0.839** — silver is wrong or mis-bounded on 126 of its 783
test spans. That adjudication is LLM-judged, so it is an estimate, not a measurement.

### Three process rules learned the hard way

1. **Check the base rate before writing a rule.** v2 claimed a bare `امام`/`حضرت` denoting a
   specific person is `PER`, generalising from 11 instances of `امام (`. With no name
   following, silver says `O` 128:12 for `امام` and 81:3 for `حضرت`. That one rule caused 7
   of the 19 real errors on the full test split. v2.1 claimed `دنیا` behaves like `جهان`;
   silver says the opposite (`دنیا` `O` 240:20, `جهان` `LOC` 202:87). Both were caught only
   by measuring after shipping.
2. **A guideline edit invalidates every cache entry**, by design — `prompt_hash` is hashed
   into every `cache_key`. Batch v3 changes into one edit; do not trickle them.
3. **Rate limits are about concurrency, not volume.** A 24-way run got the account limited
   for ~3 h. Batch 50 at concurrency 2 runs clean and sends the 11 KB guideline once per 50
   sentences instead of once per 20. Keep concurrency at 2.

## Blockers, before anything is published

- [x] **Licence — applied.** CC BY-SA 4.0, inherited from PerDT. `../LICENSE` clause 2 now
      names `annotation/data/`, `corpus/perdt-ner-iob/`, `corpus/perdt-ner-iob-llm/` and the
      published dataset; the dataset repo carries `LICENSE` (attribution notice + full legal
      code, source `hub/LICENSE`) and the card states it.
- [ ] **Courtesy notice to the PerDT authors — drafted, not sent.** Text in
      `hub/perdt-authors-notice.md`, addressed to the contacts in the treebank README
      (rasooli@seas.upenn.edu, pegh.safari@gmail.com). It states the licence reading, the
      `not-to-release/` interpretation, and offers to pull `silver` on request. Send it;
      record the date and any reply here.
- [x] **Provenance — as good as it gets for v2.2.** `manifest.json` in the dataset repo
      records the build commit, the PerDT asset URLs + MD5s, `prompt_hash`, the model
      *alias*, unannotated ids, and a sha256 per file. The resolved model id for the v2.2 run
      is not recoverable; the card says so. For v3, write model id, temperature and date
      into every output row from `annotate.js`.
- [ ] **No human gold — the sample is drawn and waiting.** `human/gold-200.iob` is a blind
      worksheet over 200 test sentences (3,667 tokens), stratified 30 both-empty / 50
      agreeing / 120 disagreeing, with `human/README.md` for the protocol and
      `../scripts/annotation/score_gold.py` to score it. Filling it in measures real recall
      for both annotators, grades the LLM judge every precision estimate here depends on,
      and — with a second annotator — yields the missing IAA number. 2-4 hours of work.

## v3 guideline changes

Batch all of these into one edit, then re-run everything.

- [ ] **Metonymic industry and place names.** `هالیوود` is `ORG` in silver, `LOC` in the v2.2
      output; the guideline decides neither (`train:22049`). Same class: `تهران` as a
      government already resolved by rule 16, so mirror that wording.
- [ ] **Places embedded in event names.** Rule 31 makes events `O`, which swallowed `آسیا`
      inside `لیگ قهرمانان آسیا`. Decide whether a place inside a discarded event name still
      gets labelled.
- [ ] **`DAT` versus the excluded `TIM` class.** `فردا`, `امشب`, `فردا عصر` are labelled
      `DAT` under rule 25 while silver calls them `TIM`, and rule 30 says times of day are
      `O`. 3 spans of 822 on the test split, but the boundary is genuinely unstated.
- [ ] **Rule 35 (epithet = `PER`) is a taste call, not evidence.** It is the only deliberate
      deviation with no corpus count behind it and it still produces disagreements
      (`امیرکبیر فوتبال ایران`). Either find the counts or drop it.
- [ ] **Coordination.** Rule 3 splits `تیم‌های راه‌آهن و نفت تهران` into two spans; silver
      merges it. The rule is right, but the guideline should say explicitly that a shared
      head noun (`تیم‌های`) is not part of either span.

## Known compliance failures — no rule change needed

The rule exists and is simply not followed often enough. If v3's prompt work does not move
these, the fix is a second recall-oriented pass, not more prompt text.

- [ ] `ORG` head noun plus specifier under-tagged: `کمیسیون آموزش`, `کمیسیون نفت`,
      `آموزش و پرورش`, `برق منطقه‌ای` (rule 20).
- [ ] Future relative dates missed: `ماه آینده`, `12 سال آینده` (rule 25 lists
      `هفتهٔ آینده`).
- [ ] Bare demonyms and literary works occasionally tagged: `ایرانی` (rule 4), `سعدی‌نامه`
      (rule 5).

## Pipeline work

- [ ] **Clitic-attached names are dropped silently.** `فاطمه` against the token `فاطمه‌ام`,
      `زینبم` — `validate_responses.py` records these as `no-align` and discards them. A
      `partial-token` reason plus prefix matching would recover them. Currently ~0.5 % of
      entities.
- [ ] **One sentence is unannotated.** `train:16860` (H5N1 virus strains) trips a content
      filter; bisection recovered the other 19 in its batch. Either leave it labelled
      `error: missing` or annotate it by hand.
- [ ] **Adjudication is ad hoc.** The scripts that dump disagreements and score judge
      verdicts live in `/tmp` and are rewritten each time. If adjudicated precision is going
      to be quoted, the script belongs in `scripts/annotation/`.

## Stand up Potato for the human annotation pass

Hand-editing IOB in a text editor is the weak link in the gold pass. Persian is
bidirectional, so a tag column drifts visually away from its token, editors silently convert
TAB to spaces or "fix" the text, and nothing stops an annotator corrupting the token column.
A browser tool removes the whole class of problem and brings agreement metrics with it.

Potato (`potato-annotation` on PyPI, 2.9.3, GPL-3.0, Python >=3.7) is the fit: span
labelling is a first-class scheme, `examples/span/` ships an NER template, and it serves on
the LAN by default so a local annotator just opens a link.

- [ ] **Install into its own venv, never `../.venv`.** The precedent is in `.omp/AGENTS.md`:
      `spacy-huggingface-hub` pulled `typer<0.8` into the training venv and broke every
      `spacy` command. Use `.venv-annotate/`, add it to `.gitignore`, then
      `potato start <config.yaml> -p 8000`. Clone the repo separately if the `examples/`
      templates are wanted; the wheel does not carry them. Find the LAN address with
      `hostname -I`.
- [ ] **Write `scripts/annotation/to_potato.py`.** Potato reads JSON, JSONL or CSV with an
      `id` and a `text` field, named through `item_properties: {id_key, text_key}`. Emit
      `text` as `" ".join(tokens)` and nothing else, so a character offset maps back to a
      token index by cumulative token length plus one per space. Any other rendering breaks
      the round trip.
- [ ] **Settle right-to-left display in the pilot, before converting all 200.** Potato
      allows raw HTML in the text field, so `<div dir="rtl">…</div>` is the obvious fix —
      but if the exported span offsets are then measured against the string *including* the
      markup, every offset shifts and the mapping above is wrong. Check what the export
      actually contains. If offsets do shift, style the container through a custom layout or
      CSS instead of inline HTML, and leave `text` as bare tokens.
- [ ] **Write `scripts/annotation/from_potato.py`.** Convert the export back to a worksheet
      that `score_gold.py --check` accepts. It MUST reject any span whose boundaries do not
      fall on token boundaries: a mouse selection can easily land mid-token, and Persian
      ZWNJ inside words such as `احمدی‌نژاد` makes that likely rather than rare.
- [ ] **Prove the round trip on 10 sentences.** Feed the v2.2 LLM spans in as if they were
      annotations, export, convert back, and require span-for-span equality with
      `gold-200.llm.iob`. Until that passes byte-clean, no annotator time goes in.
- [ ] **Use two projects, not one.** A blind project over `gold-200.jsonl` with no
      suggestions produces the measurement of record; a separate review project may
      pre-load the existing labels. Potato has LLM label suggestions and an adjudication
      workflow, which fit the review pass exactly — and would quietly destroy the blind one.
- [ ] **Take the agreement metrics from Potato rather than reimplementing them.** It
      computes Krippendorff's alpha and tracks per-annotator accuracy against gold
      standards, which is strictly more than `score_gold.py --second` does. Keep
      `score_gold.py` for scoring silver and the LLM against the finished gold, since that
      is not something Potato knows about.
- [ ] **Restrict sign-up.** `user_config: {allow_all_users: False, authorized_users: [...]}`
      for a known annotator list, or `login: {type: url_direct, url_argument: ...}` for a
      one-click link. The default is open self-registration, which is wrong for a LAN
      service left running.

Acceptance: the 10-sentence pilot round-trips with identical spans, Persian renders
right-to-left in the browser, and the exported worksheet passes `--check`.

## Remaining annotation

All three splits are relabelled under v2.2: **26,196 train** (13,644 entities, 1
unannotated — `train:16860`, content refusal), **1,456 dev** (731), **1,455 test** (769).
Micro F against silver: train 0.742, dev 0.748, test 0.754. Exported to
`corpus/perdt-ner-iob-llm/{train,dev,test}.txt` by `../scripts/annotation/export_iob.py`.

- [ ] Decide whether v3 re-runs the whole corpus (~29k sentences ≈ 580 calls at batch 50,
      ~45 min at concurrency 2) or only the shards where v3's rules can fire.

## Ship it upstream

Do this only once the annotation is final — every guideline change re-runs the corpus and
moves these numbers.

- [x] **Wire the new corpus into `project.yml`.** Done: `convert-ner-llm`, `train-ner-llm`
      and `evaluate-ent-llm` were added alongside the existing commands, plus an `ner-llm`
      workflow. `configs/fa_ner_sm.cfg` needed no change, as predicted — the label set is
      read from the data and the trained model carries exactly `DAT, LOC, ORG, PER`.
      `evaluate-ent-llm` fills the whole 2x2 through the new
      `../scripts/eval_ner_restricted.py`, which drops out-of-scope entities from both the
      prediction and the reference so the four-label micro average is like-for-like.
- [x] **Add `export_iob.py` as a project command.** Done: `export-ner-llm` heads the `ner-llm`
      workflow, with `annotation/data/llm/ner-v2.2-default/` (a directory dep — spaCy hashes
      it, it does not glob), the three silver IOB files and the script as deps. Verified: its
      output is byte-identical to the hand-run export.
- [x] **Push to Gitea.** Done at `eaa04fe`, GitHub too. `pass` failed that day with
      `decryption failed: No secret key` even though commit signing worked, so the push used
      the `rbw` fallback from `.omp/AGENTS.md` via a one-off `git -c credential...helper`.

## Publish the dataset as its own repo

Published 2026-09-22: <https://huggingface.co/datasets/Phazel/fa-perdt-ner>, two configs
(`silver`, 7 labels, default; `llm`, 4 labels), keyed by PerDT `sent_id`. Built by
`spacy project run hub-dataset` (`../scripts/annotation/build_hub_dataset.py`) from the
card template in `hub/README.md`; every number on the card is computed from the shipped
files, and the build asserts both corpora against the CoNLL-U sentence order. Upload
recipe is in the command's help text. Verified after upload: both configs `load_dataset`
from the Hub with the declared splits.

- [x] Home: `datasets/Phazel/fa-perdt-ner`. Not mirrored to Gitea: the source of every
      file is in this repo and `manifest.json` names the commit.
- [x] What ships: IOB2 + JSONL (spans, `prompt_hash`, `error`) per split for both configs,
      `dropped-entities.jsonl`, `agreement.json`, `GUIDELINES.md`, the v2.2 prompt, the
      eight pipeline scripts, LICENSE, manifest. Response cache stays out.
- [x] Datasheet: provenance, tokenization, label sets and why MON/TIM/PCT were dropped,
      guideline derivation, model/prompt provenance and its gap, agreement table, the 2x2,
      no-human-gold limitation, citations for PerDT (LREC 2022 + NAACL 2013) and
      Beheshti-NER.
- [x] Four-label set named in the first table and in a bold warning before any comparison.
- [ ] When v3 lands: bump `--llm-dir`/`--guideline-version`, rebuild, re-upload; the card
      will re-render with the new counts and `prompt_hash`.

## Evaluate the existing models on the new labels

**Read this before comparing anything.** The published NER models are trained on **7
labels**; this dataset has **4**. A raw `ents_f` comparison is meaningless — the old model
is scored on entities the new data deliberately does not contain. Compare per-type F on the
four shared labels only, and report `ents_f` restricted to those types, never the headline
number. Second caveat: neither test set is human gold, so every figure below is agreement
between two annotators, not accuracy.

Baselines on the **silver** test split, from `../metrics/`:

| model | `ents_f` (7 labels) | PER | LOC | ORG | DAT |
|---|---|---|---|---|---|
| `fa_ent_news_sm` (`perdt-ner-test.json`) | 0.7187 | .653 | .802 | .688 | .745 |
| `fa_ent_news_md` (`ent-md-test.json`) | 0.7471 | .682 | .841 | .703 | .762 |

- [x] **Score the shipped models against the new test split.** `fa_ent_news_sm` scores
      **67.27** four-label micro F on the LLM test split against **71.98** on the silver one
      (`../metrics/ent-sm-on-llm-test.json`, `ent-sm-on-silver-test.json`). Note both
      numbers are restricted to the four shared labels, so neither is the published 71.87
      seven-label figure. The prediction expressed above was half right: `DAT` moved most
      (74.45 to 68.42) and `ORG` second (68.77 to 62.04), but `PER` moved nearly as much
      (65.29 to 57.25). md/lg/trf are not scored yet; the sm cell is what the retraining
      comparison needs.
- [x] **Fill the 2x2.** In `../docs/MODELS.md` §10. Diagonal 71.98 (silver model, silver
      test) vs 79.94 (LLM model, LLM test); off-diagonal 67.27 and 67.59 — near-symmetric,
      0.32 F apart, i.e. each annotation looks about equally foreign from the other side.

## Retrain sm on the new data and compare

- [x] **Train.** Ran as given, CPU. Early stop at step 6,000 of 20,000 (best at 4,400, dev
      ENTS_F 77.43), 19m18s wall — slower than the ~12 min in the timing note, same order.
      Output in `../training/perdt-ner-llm/model-best`.
- [ ] **Evaluate on both test splits** and finalize with
      `scripts/finalize_pipeline.py --variant ent --ner-metrics …` as `evaluate-ent` does,
      so the packaged meta carries the right numbers. Evaluation is done (all four cells in
      `../metrics/ent-{sm,llm}-on-{silver,llm}-test.json`); the finalize step is deliberately
      not run, since nothing is packaged until the human-gold blocker closes.
- [x] **Compare against the table above**, per label, four labels only. **The hypothesis
      holds**: 79.94 on the LLM test split against the old model's 71.98 on the silver one,
      +7.96 F for the same architecture on the same sentences. Per-label deltas, each model
      on its own split: `PER` +10.21, `LOC` +7.02, `ORG` +5.09, `DAT` +4.88. This measures
      consistency, not correctness — a self-consistent but wrong convention scores the same.
- [x] **Watch recall on ORG.** **It did not improve, as feared.** `ORG` F rises 68.77 to
      73.86, but entirely on precision (69.50 to 82.41); recall falls 68.06 to 66.92. By the
      criterion set here, that makes the head-noun-plus-specifier compliance gap (rule 20)
      the cause, and it is a guideline problem, not a training one. The retrained model is
      reproducing a hole in its own training labels.
- [ ] **Do not publish a wheel from this** until the human-gold blocker is closed. A model
      trained on LLM labels and evaluated against LLM labels can look excellent while being
      worse in the field; the 200 human-annotated sentences are the only thing that can tell
      the difference. Still open, and `ner-llm` has no packaging step for this reason.
