# Review brief — how to read and fill `gold-200.review.tsv`

This is the short instruction set for the review pass. Read `ANNOTATOR-GUIDE.md` first for
what the four labels mean; this file only covers the review sheet itself. The Persian
version of this brief is `REVIEW-BRIEF.fa.md`.

No programming is needed for any of this. You edit one column in a text file and send it
back. Nothing has to be installed, and no command has to be run.

## Opening the file

The file is plain text. Open it in a **plain text editor**, not in Excel:

- Windows — Notepad++ or Visual Studio Code, both free.
- macOS — Visual Studio Code, or TextEdit switched to plain-text mode.
- Linux — Visual Studio Code, gedit, Kate, or whatever you already use.

Avoid Excel and Google Sheets. They rewrite the file on save: quotation marks appear around
cells, tab characters turn into commas, Persian text gets reordered, and anything that looks
like a number can be converted to a date. A single such save destroys the file.

If a spreadsheet is genuinely the only way you can work, use LibreOffice Calc, and on
opening set the character set to Unicode UTF-8, tick Tab as the only separator, select all
columns and set the column type to Text. On saving, keep the format as Text CSV and set the
field delimiter back to Tab. Then say so when you send the file back, so the result gets
checked more carefully.

## What the columns are

Every row is one token — one word or punctuation mark. Blocks are separated by a blank line
and each begins with a header line holding the sentence id, such as `# test:965`. Columns
are separated by a single TAB character, which looks like a wide gap.

| column | content | edit it? |
|---|---|---|
| 1 | the token | never |
| 2 | what the PerDT corpus says — the existing labels | never |
| 3 | what the machine annotation says, guideline v2.2 | never |
| 4 | **your verdict — the only column that counts** | yes, this one |
| 5 | a `*` when columns 2 and 3 disagree on this token | never |

## Read this before you start

Column 4 arrives pre-filled with the machine's answer. Doing nothing therefore accepts the
machine on all 200 sentences. Treat the pre-fill as a starting point, never as a
recommendation: on every marked row, decide for yourself which of the two is right, or write
a third answer when both are wrong.

## How much work this is

The sheet holds 200 sentences. Of those, 120 carry at least one disagreement, and the total
comes to 152 decisions, never more than 4 in a single sentence. They break down as:

| how many | situation | what you are deciding |
|---|---|---|
| 61 | only the corpus has a span | did the machine miss it, or is the corpus wrong? |
| 53 | only the machine has a span | did the machine invent it, or did the corpus miss it? |
| 33 | both found it, boundaries differ | which boundary does the guideline call for? |
| 5 | same span, different label | which label is right? |

Budget about 45 minutes.

## The four shapes, from the real file

Both agree, nothing marked, in `test:299`:

| token | corpus | machine | verdict |
|---|---|---|---|
| `آخوند` | `B-PER` | `B-PER` | `B-PER` |
| `خراسانی` | `I-PER` | `I-PER` | `I-PER` |

Only the machine found it, in `test:965`. The corpus missed a gallery:

| token | corpus | machine | verdict | |
|---|---|---|---|---|
| `نگارخانهٔ` | `O` | `B-ORG` | `B-ORG` | `*` |
| `ایوان` | `O` | `I-ORG` | `I-ORG` | `*` |

Only the corpus has it, in `test:862`. The machine missed a commission:

| token | corpus | machine | verdict | |
|---|---|---|---|---|
| `کمیسیون` | `B-ORG` | `O` | `O` | `*` |
| `نفت` | `I-ORG` | `O` | `O` | `*` |

Boundaries differ, in `test:49`. This one is a real judgement call — is
`تیم کانوپولو بانوان ایران` a single organisation, or an organisation followed by
`[ایران]` as a place? Rule 20 of the guideline says one span:

| token | corpus | machine | verdict | |
|---|---|---|---|---|
| `تیم` | `B-ORG` | `B-ORG` | `B-ORG` | |
| `کانوپولو` | `I-ORG` | `I-ORG` | `I-ORG` | |
| `بانوان` | `O` | `I-ORG` | `I-ORG` | `*` |
| `ایران` | `B-LOC` | `I-ORG` | `I-ORG` | `*` |

## The marked rows are not the whole job

Where columns 2 and 3 agree there is no `*`, and nobody has ever checked those spans — the
two annotators can be wrong together. Two cases to watch for as you read each sentence:

- Case one — an entity that neither column marked. Add it in column 4.
- Case two — something both columns marked that is not an entity under the guideline. Set
  column 4 to `O`.

These are the most valuable findings in the whole exercise, because no automatic comparison
between the two annotators can ever surface them.

## Writing tags

Valid tags are exactly `O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `B-DAT`
and `I-DAT`. The first token of an entity takes `B-`, every following token takes `I-`.

Two adjacent entities of the same label each need `B-`, otherwise they merge into one. The
phrase `ایران و عراق` is `B-LOC`, `O`, `B-LOC` — writing `I-LOC` on the second would claim
that the three tokens are a single place name.

Type the tags in capital letters exactly as written above. Common slips that make a row
unreadable: lowercase such as `b-per`, a missing hyphen such as `BPER`, a space instead of
the hyphen, and a stray space left after the tag.

## Things that quietly break the file

- Do not add or delete any row, and do not reorder sentences.
- Do not edit the token column, even to fix an obvious typo in the original text.
- Do not delete the `# test:NNN` header lines or the blank lines between sentences.
- Do not replace the TAB characters with spaces. If your editor does this automatically,
  it usually has a setting called "insert spaces" or "expand tabs" that can be turned off —
  or use a different editor from the list above.
- Do not turn on autocorrect, spell-check fixes, or any "fix bidirectional text" feature.
  These rewrite Persian silently.
- Save in UTF-8, which every editor listed above does by default.

## What to send back

Send the edited `gold-200.review.tsv`, plus a plain list of the sentence ids where the
guideline genuinely did not decide the case — a simple text file or an email body is fine.
That list is as valuable as the annotations: it is the only evidence of where the guideline
is unclear, and nobody but the person doing this work can produce it.

Your file gets checked automatically when it arrives, so a broken tag or a missing row is
caught and can be fixed. You do not need to verify anything yourself.

## For whoever receives the file

The structural check, which prints no scores and so can also be handed to the annotator if
they are comfortable with a terminal:

```bash
python scripts/annotation/score_gold.py --gold annotation/human/gold-200.review.tsv \
    --column 4 --check
```

It reports unparseable tags, missing sentences and spans running off the end of a sentence.
It cannot detect an `I-` that should have been a `B-`, so coordinations still need an eye.

## What this produces, and what it does not

The result is a **reviewed** corpus, not gold. Whoever fills this sheet has seen both
annotators' labels and will unavoidably anchor on them, so the output cannot be quoted as
an unbiased precision or recall figure for either annotator.

For that, the blind worksheet `gold-200.iob` is the instrument, and it must be filled in
**first** — a sentence that has been reviewed can never be blind again.
