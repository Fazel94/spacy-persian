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

- [ ] **Licence — decided, not yet applied.** The source is CC BY-SA 4.0 (PerDT /
      UD_Persian-PerDT, NER layer under `not-to-release/Dadegan with NER tag/`), so
      share-alike carries: these annotations are Adapted Material and ship as
      **CC BY-SA 4.0** with attribution to PerDT. Apply it in three places — extend
      `../LICENSE` clause 2 to name `annotation/data/` and `corpus/perdt-ner-iob-llm/`, put
      a LICENSE and attribution notice in the standalone dataset repo, and state it in the
      datasheet. `../docs/MODELS.md` still advises confirming with the PerDT authors before
      publishing anything derived from the NER layer; that courtesy notice is separate from
      the licence question and should go out with the release.
- [ ] **Provenance.** Every output row records `"model": "default"` — an alias, not a
      resolved model id — with no timestamp, temperature or script revision. Re-runs of the
      same prompt agree at F 0.95-0.98, so the committed files are the record of origin and
      cannot be reproduced bit-exact. Add a sidecar manifest per output tree.
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

- [ ] **Wire the new corpus into `project.yml`.** Add `convert-ner-llm`, `train-ner-llm` and
      `evaluate-ent-llm` alongside the existing commands rather than editing them in place,
      so both datasets stay trainable and comparable from one checkout. They mirror
      `convert-ner` (line 191), `train-ner` (line 174) and `evaluate-ent`, swapping
      `corpus/perdt-ner-iob` for `corpus/perdt-ner-iob-llm`, `corpus/perdt-ner` for
      `corpus/perdt-ner-llm`, and `training/perdt-ner` for `training/perdt-ner-llm`. Add an
      `ner-llm` workflow. `configs/fa_ner_sm.cfg` needs no change: the label set is read
      from the data.
- [ ] **Add `export_iob.py` as a project command** so the corpus is regenerated by
      `spacy project run`, not by hand, with `annotation/data/llm/ner-v2.2-default/**` as
      deps and the three IOB files as outputs.
- [ ] **Push to Gitea** once the above runs clean:
      `NO_PROXY="*" no_proxy="*" http_proxy="" https_proxy="" git push gitea main`
      (credential helper and `pass`/`rbw` fallback documented in `.omp/AGENTS.md`).

## Publish the dataset as its own repo

Licence is settled: **CC BY-SA 4.0**, inherited from PerDT (see the licence blocker above).
The models repo keeps its own licence; the dataset stands alone.

- [ ] **Decide the home.** HF Hub `datasets/Phazel/<name>` is the natural one — the account
      already hosts the model wheels, and `spacy-huggingface-hub` notes in `.omp/AGENTS.md`
      apply (strip the proxy for uploads; token in `rbw get "api/huggingface_token"`).
      Mirror to Gitea for provenance.
- [ ] **Decide what ships.** Recommended: the three IOB files, the JSONL with spans and
      `prompt_hash`, `GUIDELINES.md`, `prompts/ner-v2.2.md`, the five pipeline scripts, and
      the agreement JSONs. The response cache stays out; it is 1,600 opaque blobs.
- [ ] **Write a datasheet** — provenance (PerDT sentences, `--merge-subtokens`
      tokenization), label set and why MON/TIM/PCT were dropped, how the guideline was
      derived, the model and prompt that produced the labels, measured agreement with the
      silver layer, and the honest limitation that no human gold exists.
- [ ] **Name the four-label set explicitly** in the README. Consumers will otherwise assume
      PerDT's seven and silently get zero recall on MON/TIM/PCT.

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

- [ ] **Score the shipped models against the new test split.**
      `spacy benchmark accuracy training/fa_ent_news_sm corpus/perdt-ner-llm/test.spacy
      --output metrics/ent-sm-on-llm-test.json`, same for md/lg/trf. This measures how far
      the current models sit from the new conventions; expect ORG and DAT to move most,
      since that is where guideline and silver disagree hardest.
- [ ] **Fill the 2x2.** Old model and new model, each scored on the silver test split and on
      the LLM test split. Only the diagonal is flattering to either side; the off-diagonal
      cells are the interesting ones, and the four-cell table is what belongs in
      `docs/MODELS.md`, not a single number.

## Retrain sm on the new data and compare

- [ ] **Train.** `spacy train configs/fa_ner_sm.cfg --output training/perdt-ner-llm
      --paths.train corpus/perdt-ner-llm/train.spacy --paths.dev corpus/perdt-ner-llm/dev.spacy
      --gpu-id -1`. The sm NER tier is CPU-bound and took ~12 min in the timing test recorded
      in `.omp/AGENTS.md`; the GPU note there applies to `dep`, not `ner`.
- [ ] **Evaluate on both test splits** and finalize with
      `scripts/finalize_pipeline.py --variant ent --ner-metrics …` as `evaluate-ent` does,
      so the packaged meta carries the right numbers.
- [ ] **Compare against the table above**, per label, four labels only. The hypothesis worth
      testing: the new labels are more internally consistent, so the same architecture on the
      same sentences should fit them better — a higher score on the LLM test split than the
      old model reaches on the silver one. If that does not happen, the annotation is not
      actually cleaner than what it replaced, whatever the adjudication said.
- [ ] **Watch recall on ORG.** It is the weakest label in the annotation (train R .628) and
      the weakest in the old model (F .688). If the retrained model does not improve there,
      the head-noun compliance gap listed above is the cause and it is a guideline problem,
      not a training one.
- [ ] **Do not publish a wheel from this** until the human-gold blocker is closed. A model
      trained on LLM labels and evaluated against LLM labels can look excellent while being
      worse in the field; the 200 human-annotated sentences are the only thing that can tell
      the difference.
