# Human gold sample — 200 sentences

This is the only thing that can settle whether the LLM annotations are actually better than
PerDT's silver layer. Every figure quoted so far (LLM ~0.94 precision, silver ~0.84, a 16%
silver error rate) comes from an LLM judge belonging to the annotator's own model family,
and neither side's recall is measured at all, because entities that *both* annotators miss
are invisible to a pairwise comparison. A human pass fixes both problems at once.

Hand the annotator `ANNOTATOR-GUIDE.md`, or `ANNOTATOR-GUIDE.fa.md` for the Persian
version. Those explain the label set with worked right-and-wrong examples and how to edit
the files, and they are meant to be read once before starting. This README is the operator's
view: which file is which, and what each one measures.

## Files

| file | what it is |
|---|---|
| `gold-200.iob` | **the blind worksheet.** Two columns, `token<TAB>tag`, every tag pre-filled `O`, a `# <id>` header before each sentence, blank line between sentences. Edit column 2. |
| `gold-200.review.tsv` | **the review sheet.** Four columns: `token`, silver tag, LLM tag, verdict — the verdict column seeded with the LLM tag. Every token where the two annotators differ is marked `*` at the end of the line. 208 such tokens across the 200 sentences. Edit column 4. |
| `gold-200.silver.iob` | the same sentences pre-filled with the silver layer alone, 251 spans. For reading or correcting PerDT's labels directly. |
| `gold-200.llm.iob` | the same sentences pre-filled with the LLM annotation alone, 243 spans. |
| `gold-200.jsonl` | the sentences as JSON, ids and tokens only. For tooling, not for annotating. |
| `gold-200.key.jsonl` | silver spans, LLM spans, tokens, stratum. **Do not open before finishing `gold-200.iob`,** if you are doing the blind pass. |

Regenerate all of them with `python scripts/annotation/sample_gold.py` (seeded, byte-identical
re-runs). Score with `python scripts/annotation/score_gold.py`, adding `--column 4` for a
corrected review sheet.

## Blind pass or review pass — they do not measure the same thing

**Reviewing is faster and worth less.** Correcting a pre-filled sheet anchors you: you will
accept labels you would never have produced yourself, and you will find fewer of the
annotator's errors than exist. The effect is strongest exactly where it hurts — the spans
both annotators already agree on, which the review sheet shows as unmarked and invites you
to skim.

So:

- **`gold-200.iob`, blind, is the measurement of record.** It is the only version whose
  numbers can be quoted as precision and recall for silver and for the LLM. 2-4 hours.
- **`gold-200.review.tsv` is triage.** Use it to adjudicate the 208 disputed tokens quickly,
  to sanity-check the guideline, or to produce a corrected corpus. Perhaps 45 minutes.
  Numbers from it are biased toward whichever labels were pre-filled and must be reported
  as "reviewed", never as gold.

If you only have time for one, do the review pass and say so in the write-up. If you have
time for both, do the blind pass first — once you have seen the pre-filled labels you cannot
unsee them, and that sentence is no longer usable as blind gold.

## How to annotate

1. Read `../GUIDELINES.md` first, all of it. It is the definition of correct here; where
   your intuition and the guideline disagree, the guideline wins, and if that feels wrong
   often enough, say so — that is a finding about the guideline, not a mistake by you.
2. Work through `gold-200.iob` top to bottom, changing `O` to `B-<LABEL>` on the first token
   of an entity and `I-<LABEL>` on each following token. Labels are `PER`, `LOC`, `ORG`,
   `DAT` and nothing else.
3. **Do not look at the key, the silver corpus, or any model output while you work.** The
   whole value of this sample is that it was produced without anchoring on either annotator.
4. Leave the token column untouched. The scorer joins on it and will refuse a worksheet
   whose tokens moved.
5. Sentences with no entities stay entirely `O`. Roughly 15% of the sample is like that by
   construction, and they matter: they are the only place where an entity both annotators
   missed can show up.

Expect 2–4 hours. 200 sentences, 3,667 tokens.

## Inter-annotator agreement

If a second person is available, have them annotate the same file independently into
`gold-200.b.iob`, then:

```bash
python scripts/annotation/score_gold.py --gold annotation/human/gold-200.iob \
    --second annotation/human/gold-200.b.iob
```

This is the number that says whether the guideline is teachable. Without it, "the guideline
is good" remains an opinion. Two annotators on all 200 is ideal; 100 shared is enough.

## Scoring, once the worksheet is done

```bash
python scripts/annotation/score_gold.py            # silver and LLM, both against the human
python scripts/annotation/score_gold.py --judge /path/to/adjudicated.json   # grade the judge
```

The scorer prints each annotator twice, and **only the reweighted block means anything**.
The sample is deliberately unbalanced — 120 of the 229 disagreeing sentences but only 30 of
the 916 where both annotators found nothing — so raw sample counts describe the sample, not
the corpus. Each sentence carries weight `stratum_size / stratum_sampled`.

| stratum | in the test split | sampled | weight |
|---|---|---|---|
| both annotators found nothing | 916 | 30 | 30.53 |
| both found exactly the same spans | 310 | 50 | 6.20 |
| they disagree | 229 | 120 | 1.91 |

The reweighting is verified, not assumed. Feeding the scorer a worksheet fabricated to equal
the LLM output returns exactly 1.000 for the LLM, and for silver it returns a reweighted
micro F of **0.769** against the true silver-vs-LLM figure of **0.754** on the full 1,455
sentence split — 1.5 points, while the raw sample counts say 0.615, off by fourteen. Per
label the estimator lands at PER .779/.734, LOC .835/.827, ORG .670/.679, DAT .701/.706
(estimated/true).

## What the answer decides

- **Which annotator is better, and by how much**, with real recall for both — the claim the
  whole dataset rests on.
- **Whether the LLM judge can be trusted.** `--judge` grades its verdicts against the human.
  If judge accuracy is poor, every adjudicated number in `../TODO.md` and the commit history
  has to be restated.
- **Whether a model may be published.** A model trained on LLM labels and evaluated against
  LLM labels can look excellent while being worse in the field. Until this file is filled
  in, that risk is unquantified and no wheel ships.
