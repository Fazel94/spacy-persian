# Annotator guide — Persian named entities

This guide is for a human marking named entities in Persian sentences. It teaches the file
format first, then every label with worked right-and-wrong examples, then the traps that
cause most mistakes.

The authority is `../GUIDELINES.md`. That document is terse and states each convention with
the corpus counts behind it. This one is longer, assumes nothing, and exists to be read
once before you start. Where the two appear to disagree, `../GUIDELINES.md` wins — and tell
whoever gave you the task, because that means one of the two needs fixing.

## Labels, and only these four

| label | covers | example |
|---|---|---|
| `PER` | people, real or fictional, including prophets and imams named as people | `علی`, `احمدی‌نژاد`, `اوباما`, `پیامبر` |
| `LOC` | countries, cities, provinces, regions, waters, planets | `ایران`, `تهران`, `خاورمیانه`, `خلیج فارس` |
| `ORG` | companies, ministries, parties, clubs, media, universities, armed groups | `گوگل`, `سازمان ملل`, `استقلال`, `دانشگاه تهران` |
| `DAT` | dates, years, named periods, relative dates | `سال 1974`, `امروز`, `ماه رمضان`, `قرن بیستم` |

Everything else is `O`, which means "not an entity". Money, percentages, clock times,
products, events, sects and ideologies are all `O` here. They are not mistakes in the text;
they are simply outside this label set.

## Part one — how to edit the file

### What the file looks like

Each sentence is a block. The block starts with a comment line holding the sentence id,
then one line per token, then a blank line before the next block. Each token line has the
token, a single TAB character, and the tag.

Shown as a table, the first block of the worksheet begins like this:

| token | tag |
|---|---|
| `در` | `O` |
| `صبح` | `O` |
| `بهاری` | `O` |
| `جمعه` | `O` |

In the actual file that is `در<TAB>O`, one pair per line, under a header line reading
`# test:492`.

### What B- and I- mean

Tags follow the IOB2 scheme. The first token of an entity gets `B-` and the rest get `I-`.

Take the three tokens `دانشگاه`, `علوم`, `پزشکی` forming one organisation:

| token | tag | why |
|---|---|---|
| `دانشگاه` | `B-ORG` | first token of the entity |
| `علوم` | `I-ORG` | continues the same entity |
| `پزشکی` | `I-ORG` | continues the same entity |

Two entities of the same label that touch each other must each start with `B-`, otherwise
they merge into one. This matters for coordination:

| token | tag | why |
|---|---|---|
| `ایران` | `B-LOC` | first entity |
| `و` | `O` | the conjunction is never inside a span |
| `عراق` | `B-LOC` | second entity, so `B-` again, not `I-` |

### Rules for editing

- Change the tag column only. Never touch the token column, never add or delete a token
  line, never reorder blocks, never delete the `# test:NNN` header lines or the blank lines
  between blocks.
- Use a real TAB between columns. If your editor converts tabs to spaces, turn that off
  before you start. In VS Code this is `"editor.insertSpaces": false` for the file type.
- Save as UTF-8 without a byte-order mark, which is the default nearly everywhere.
- Turn off any autocorrect, spell-fix or bidirectional-text "helper" your editor offers.
  These silently rewrite Persian text and corrupt the token column.
- Leave a sentence entirely `O` when it has no entity. That is a real answer, not a skip,
  and roughly one sentence in seven is like that on purpose.

### Tags you may write

Only these nine strings are valid: `O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`,
`I-ORG`, `B-DAT`, `I-DAT`. Anything else fails the checker. Watch for lowercase (`b-per`),
a missing hyphen (`BPER`), and a trailing space after the tag.

### Mistakes the checker catches, and ones it cannot

Run this whenever you like, as often as you like. It prints no scores, so it cannot
influence your judgement:

```bash
python scripts/annotation/score_gold.py --gold annotation/human/gold-200.iob --check
```

It catches unparseable tags, missing sentences, and spans running off the end. It cannot
catch an `I-` tag that should have been `B-`, because both parse. Check coordinations by
eye.

### The four-column review sheet

If you were given `gold-200.review.tsv` instead, the columns are token, the existing corpus
label, the machine label, and your verdict — with a `*` in a fifth column on every token
where the two disagree. Edit the fourth column only; leave the `*` markers alone. Score it
by adding `--column 4` to the command above.

## Part two — the labels in detail

### Decide in this order

- Step one — ask whether the phrase names one specific thing. Generic nouns are `O`.
  `شهر` alone names no city; `شهر تهران` names one.
- Step two — pick the longest span that is itself a name. Prefer `[دانشگاه تهران]` over
  `[تهران]` inside it. Never mark one entity inside another.
- Step three — decide the label from what the thing is, not from the words around it.
- Step four — when genuinely torn, choose the reading that a reader of the sentence would
  take, mark the sentence id in a notes file, and move on. Do not stall.

### PER, people

Correct annotations:

- Correct — `[احمدی‌نژاد]` in `آقای احمدی‌نژاد گفت`, because secular titles such as `آقای`,
  `دکتر`, `استاد` and `ارتشبد` stay outside the span.
- Correct — `[امام صادق علیه‌السلام]` as one span, because the spelled-out blessing directly
  after the name is part of it.
- Correct — `[امام حسین]` in `امام حسین ( ع )`, with the parenthesis and the letter left as
  `O`. The abbreviated parenthetical is never inside a span.
- Correct — `[احمد بن محمد بن عیسی]` as a single span, since a patronymic chain is one name.
- Correct — `[پیامبر]` and `[رسول خدا]`, which name one specific person.

Incorrect annotations:

- Wrong — `[آقای احمدی‌نژاد]`, which swallows the honorific.
- Wrong — `[حضرت]` on its own in `حضرت فرمودند`. A religious title with no name after it is
  `O`, and the corpus is emphatic: bare `امام` is `O` 128 times against 12, bare `حضرت` 81
  against 3.
- Wrong — `[ائمهٔ اطهار]`, `[اهل‌بیت]`, `[معصومین]`, which refer to several people
  collectively rather than naming one.
- Wrong — `[خدا]`, `[الله]`, `[یزدان]`, `[شیطان]`, `[جبرئیل]`, all `O` in this label set.
- Wrong — `[رئیس جمهور احمدی‌نژاد]`, since role nouns sit outside the span.

### LOC, places

Correct annotations:

- Correct — `[شهر تهران]`, `[استان گلستان]`, `[کشور روسیه]`, each including the geographic
  head noun because a name follows it directly.
- Correct — `[خلیج فارس]`, `[خاورمیانه]`, `[مریخ]`, `[غرب تهران]`.
- Correct — `[ایران]` only, in `مردم ایران` and in `ملت ایران`.
- Correct — `[تهران]` in `تهران اعلام کرد`. A capital standing for its government stays
  `LOC`.
- Correct — `[جهان]` when it means the world as a place, as in `بزرگ‌ترین تالاب جهان`.

Incorrect annotations:

- Wrong — `[کشور]` or `[شهر]` alone with no name attached.
- Wrong — `[دنیا]` in any sense. This is the one pair that trips everyone: the corpus tags
  `جهان` as `LOC` 202 times against 87, and tags `دنیا` as `O` 240 times against 20. Same
  meaning in English, opposite treatment here.
- Wrong — `[غرب]` alone, including `غرب` meaning "the West" as a bloc.
- Wrong — `[زمین]` unless it means the planet, as in `کرهٔ زمین`.
- Wrong — `[ایرانی]`, `[آمریکایی]`, `[ترکمن]`, `[قوم مایا]`. Demonyms, nationalities and
  peoples are `O`.

### ORG, organisations

Correct annotations:

- Correct — `[دانشگاه تهران]`, `[وزارت امور خارجه]`, `[شرکت مخابرات ایران]`,
  `[تیم ملی بوکس]`, `[کلینیک نور]`. The head noun belongs inside once a name or specifier
  follows it.
- Correct — `[سازمان ملل]`, `[بانک مرکزی]`, `[مجلس شورای اسلامی]`, `[صدا و سیما]`,
  `[طالبان]`, `[ساواک]`.
- Correct — `[جمهوری اسلامی ایران]` as one `ORG`, not an `ORG` plus a `LOC`.
- Correct — `[مهر]` in `به گزارش خبرنگار مهر`, since a cited news agency is an organisation.
- Correct — `[راه‌آهن]` and `[نفت تهران]` as two spans in `تیم‌های راه‌آهن و نفت تهران`. The
  shared head noun `تیم‌های` stays outside both.

Incorrect annotations:

- Wrong — `[دولت]`, `[مجلس]`, `[شهرداری]`, `[فدراسیون]`, `[تیم ملی]`, `[وزارت]`, `[سفارت]`
  standing alone. A head noun with no name after it is `O`, however clearly context points
  at one institution.
- Wrong — `[شیعه]`, `[سنی]`, `[معتزله]`, `[صوفیه]`, `[مارکسیسم]`. Denominations, schools of
  thought and movements are `O`; a named party such as `[حزب توده]` is still `ORG`.
- Wrong — `[بنز]`, `[فتوشاپ]`, `[ویندوز]`. Products are `O`; the company is `ORG` only when
  the sentence names the company.
- Wrong — `[جام جهانی]`, `[المپیک]`, `[نمایشگاه CES]`. Events are `O`; the body that runs or
  plays in them is `ORG` when named.

### DAT, dates

Correct annotations:

- Correct — `[سال 1974]`, `[ماه رمضان]`, `[روز جمعه]`, `[سی‌ام اکتبر سال 1910]`, including
  the head noun when one is present.
- Correct — `[امروز]`, `[فردا]`, `[امسال]`, `[سال گذشته]`, `[هفتهٔ آینده]`, `[این هفته]`.
- Correct — `[10 سال پیش]`, `[سه دهه پیش]`, `[یک ماه قبل]`, durations anchored to now.
- Correct — `[از بهمن 1343 تا خرداد 1357]` as ONE span including both `از` and `تا`. A range
  is never two spans.
- Correct — `[سال 1251 شمسی]` and `[438 میلادی]`, with the calendar word inside.
- Correct — `[قرن بیستم]`, `[قرن سوم هجری]`.
- Correct — `[ماه رمضان]`, `[سال نو چینی]`, `[نوروز امسال]`, where a festival carries a date
  head noun or a year.

Incorrect annotations:

- Wrong — `[به مدت سه سال]`. An unanchored duration is `O`.
- Wrong — `[هر روز]`, `[هر شب]`, `[روزانه]`. Habits are not dates.
- Wrong — `[این روزها]`, `[این ماه]`, `[این سال]`. The corpus treats these as vague, `O` in
  every one of the 24, 6 and 3 occurrences respectively — while `این هفته` is `DAT`. Yes,
  this is inconsistent; follow it anyway.
- Wrong — `[نوروز]` or `[عاشورا]` alone, and `[تشریفات نوروز]`.
- Wrong — `[ساعت ۸ صبح]` and any clock time, `[۲۰ درصد]`, `[هزار تومان]`.
- Wrong — `[هخامنشی]`, `[ساسانی]`, `[امپراطوری پارس]`. Dynasties and empires as periods are
  `O`.

## Part three — the traps

Ranked by how often they were got wrong in earlier passes over this corpus.

| trap | rule |
|---|---|
| Bare head nouns tagged `ORG` | `شهرداری`, `فدراسیون`, `مجلس` alone are `O`; with a name they are `ORG` |
| Word `دنیا` tagged `LOC` | always `O`, unlike `جهان` |
| Bare religious titles tagged `PER` | `حضرت`, `امام` alone are `O`; `امام صادق` is `PER` |
| Compass words tagged `LOC` | `غرب` alone is `O`, `غرب تهران` is `LOC` |
| Habits tagged `DAT` | `هر روز` is `O`, `امروز` is `DAT` |
| Date ranges split in two | `از … تا …` is one span |
| Honorifics pulled into `PER` | `آقای`, `دکتر`, `( ع )` stay outside |
| Coordination merged into one span | `ایران و عراق` is two spans, each starting `B-` |
| Events tagged `ORG` | `جام جهانی` is `O` |
| Demonyms tagged `LOC` | `ایرانی` is `O` |

## Before you hand the file back

- Check that the structural check passes with no error.
- Check that no token line lost its tab.
- Check every coordination you marked, since `B-` versus `I-` there is invisible to the
  checker.
- Send along the notes file listing sentence ids you found genuinely ambiguous. Those are
  the most valuable output of this whole exercise: they are where the guideline is unclear,
  and nobody else can find them.
