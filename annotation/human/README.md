# Human gold sample — 200 sentences

This is the only thing that can settle whether the LLM annotations are actually better than
PerDT's silver layer. Every figure quoted so far (LLM ~0.94 precision, silver ~0.84, a 16%
silver error rate) comes from an LLM judge belonging to the annotator's own model family,
and neither side's recall is measured at all, because entities that *both* annotators miss
are invisible to a pairwise comparison. A human pass fixes both problems at once.

## Files

| file | what it is |
|---|---|
| `gold-200.iob` | **the worksheet.** Two columns, `token<TAB>tag`, every tag pre-filled `O`, a `# <id>` header before each sentence, blank line between sentences. Edit column 2. |
| `gold-200.jsonl` | the same 200 sentences as JSON, ids and tokens only. For tooling, not for annotating. |
| `gold-200.key.jsonl` | silver spans, LLM spans, stratum. **Do not open until the worksheet is finished.** |

Regenerate the sample with `python scripts/annotation/sample_gold.py` (seeded, byte-identical
re-runs). Score it with `python scripts/annotation/score_gold.py`.

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
