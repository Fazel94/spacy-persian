# Persian NER annotation guidelines, v2.2

Labels: `PER`, `LOC`, `ORG`, `DAT`. Everything else is `O`. Spans are contiguous token
sequences on the PerDT `--merge-subtokens` tokenization; an entity text must be reproducible
by joining consecutive tokens with single spaces.

These conventions were inferred from PerDT's silver NER layer (Beheshti-NER output with manual
corrections, `corpus/perdt-ner-iob/train.txt`, 26,196 sentences) so that new annotations agree
with the bulk of the existing data. Where the silver layer is inconsistent, the majority rule
is stated and the minority is called out as an error to correct. Counts are train-split.

## General

1. Annotate every mention, including repeats within a sentence.
2. Do not nest or overlap. The longest span that is itself a name wins:
   `دانشگاه تهران` is one `ORG`, not `ORG` around a `LOC`; `شرکت مخابرات ایران` is one `ORG`.
3. Coordinated names are separate spans: `ایران و عراق` is `LOC` + `LOC`.
4. Nationality and language adjectives and demonyms are `O`: `ایرانی` (181/181 `O`),
   `آمریکایی`, `ایرانیان`, `مسلمانان`, `فارسی`, `انگلیسی`.
5. Religions, ideologies, scriptures and literary works are `O`: `اسلام`, `قرآن`, `شاهنامه`.
6. Generic role nouns are `O` and sit outside the span: `رئیس جمهور ایران` labels only
   `ایران` (`LOC`); `وزیر` and `رئیس` precede `ORG` spans but are never inside them.

## PER

7. Personal names, including religious and historical figures: `علی`, `احمدی‌نژاد`,
   `داریوش`, `اوباما`, `پیامبر`, `رسول خدا`, `امام حسین`.
8. Secular honorifics and titles are `O`, outside the span: `آقای` (140/141 `O`), `دکتر`
   (104/107 `O`), `آیت‌الله`, `استاد`, `ارتشبد`. `آقای احمدی‌نژاد` → `[احمدی‌نژاد]`.
9. Religious titles fused with the name are inside the span: `امام` (217 `B-PER`), `حضرت`
   (129 `B-PER`), `سید`: `[امام صادق]`, `[حضرت علی]`, `[سید محمدحسن ابوترابی‌فرد]`.
10. Blessing formulas: the spelled-out `علیه‌السلام` directly after a name is inside the span
    (88/98): `[امام صادق علیه‌السلام]`. The abbreviated parenthetical `( ع )` / `( ص )` is
    entirely `O`: `[امام حسین] ( ع )`. (The silver layer tags the opening `(` as `I-PER` 295
    times; that is a tokenization artefact, correct it.)
11. `خدا` and `الله` alone are `O` (520/566); `رسول خدا` as a name of the Prophet is `PER`.
12. Patronymic chains are one span: `[احمد بن محمد بن عیسی]`.

## LOC

13. Countries, cities, provinces, regions, continents, bodies of water, planets:
    `ایران`, `تهران`, `خاورمیانه`, `اروپا`, `خلیج فارس`, `مریخ`.
14. Geographic head nouns are INSIDE the span when directly followed by the name:
    `[شهر تهران]` (88 vs 4), `[استان گلستان]` (69 vs 0), `[منطقهٔ ...]` (32 vs 4),
    `[کشور روسیه]` (27 vs 5). Alone, without a name, they are `O` (`کشور` alone 440 `O`).
15. `جهان` ("the world") is `LOC` when it denotes the world as a place (`در جهان`,
    `بزرگ‌ترین تالاب جهان`), 202 vs 87. `زمین` is `LOC` only as the planet (`کرهٔ زمین`),
    otherwise `O` (250 `O`).
16. Metonymic country/capital names acting as governments stay `LOC`: `تهران اعلام کرد`.
17. `مردم ایران`, `ملت ایران`: only `ایران` is labelled.
18. Compass terms are `O` unless they are part of a named region: `غرب` alone `O`
    (84 vs 15), and `غرب` meaning "the West" as a civilisation or political bloc is `O`
    too (`مدل‌برداری از غرب`, `پوشش‌های مبتذل غرب`). `[غرب تهران]`, `[خاورمیانه]` are `LOC`.

## ORG

19. Companies, institutions, ministries, parties, armed groups, sports clubs, media outlets,
    banks, universities, international bodies: `گوگل`, `طالبان`, `استقلال`, `سازمان ملل`,
    `بانک مرکزی`, `مجلس شورای اسلامی`, `صدا و سیما`, `اینترپل`, `ساواک`.
20. Organisational head nouns are INSIDE the span when followed by a specifier:
    `[دولت ایران]`, `[وزارت امور خارجه]`, `[شرکت مخابرات ایران]`, `[تیم استقلال]`,
    `[سفارت ایران]`, `[ارتش دریایی ایران]`, `[دانشگاه تهران]`, `[تیم ملی بوکس]`.
    A head noun with no specifier is `O`, even when it clearly denotes one institution in
    context: `دولت` alone 221 `O`, `تیم` 129 `O`, `مجلس` 132 `O`, and likewise bare
    `شهرداری`, `فدراسیون`, `وزارت`, `سفارت`, `دانشگاه`, `تیم ملی`. "The municipality
    collects its dues" labels nothing; `[شهرداری تهران]` is `ORG`.
21. A country name inside such a phrase belongs to the `ORG` span, not a separate `LOC`
    (103 `ORG` spans end in `ایران`).
22. Political entities named as states are `ORG`: `جمهوری اسلامی`, `جمهوری اسلامی ایران`,
    `رژیم صهیونیستی`, `رژیم ایران`.
23. News agencies cited as sources are `ORG`: `به گزارش خبرنگار [مهر]`.

## DAT

24. Absolute dates and years, including the head noun `سال`/`ماه`/`روز` when present:
    `[سال 1974]`, `[ماه رمضان]`, `[روز جمعه]`, `[سی‌ام اکتبر سال 1910]`,
    `[سال هزار و هشتصد و یازده]`, `[ماه ژوئن سال 2011]`.
25. Relative date expressions are `DAT`: `امروز` (149), `فردا`, `دیروز`, `امسال`,
    `[سال گذشته]`, `[سال جاری]`, `[هفتهٔ آینده]`, `[این هفته]` (10 vs 0), `[روز گذشته]`.
    The vague deictics `این روزها` (0/24), `این ماه` (0/6) and `این سال` (0/3) are `O`:
    the silver layer treats them as "nowadays", not as a dated period.
26. Durations anchored to now are `DAT`: `[10 سال پیش]`, `[سه دهه پیش]`, `[یک ماه قبل]`.
    Unanchored durations are `O`: `به مدت سه سال`.
27. A date range with `از ... تا/الی ...` is ONE span including `از` and `تا`:
    `[از بهمن 1343 تا خرداد 1357]`, `[از سال 1375 تا سال 1386]`, `[از 420 تا 438 میلادی]`.
    `[از آغاز امسال] تاکنون`: `تاکنون` is outside.
28. Calendar suffixes are inside: `[سال 1251 شمسی]`, `[438 میلادی]`; the abbreviated
    `ه . ش` after a year is outside.
29. Eras and dynasties as periods are `O` (`هخامنشی`, `ساسانی`), and so are historic
    empires named after them (`امپراطوری پارس`); named festivals `نوروز`, `عاشورا` are
    `O` on their own (`نوروز` 30 `O` vs 3 `DAT`, `عاشورا` 41/0), including attributive
    use (`تشریفات نوروز`). A festival phrase headed by `سال`/`ماه`/`روز`, or carrying a
    year, is `DAT` under rule 24: `[ماه رمضان]`, `[سال نو چینی]`, `[نوروز امسال]`.
30. Times of day, clock times, money and percentages are `O` in this label set.

## Cases v1 left open, decided in v2

31. Events are `O`. The label set has no EVENT class, and an event is not the organisation
    that runs it: `نمایشگاه CES`, `جام جهانی`, `المپیک`, `لیگ قهرمانان آسیا` are all `O`.
    The organising or competing body is still `ORG` when named: `[فدراسیون فوتبال]`.
32. Habitual and frequency expressions are `O`: `هر روز`, `هر شب`, `هر هفته`, `روزانه`,
    `همه‌روزه`. Only absolute dates (rule 24), relative dates (rule 25) and now-anchored
    durations (rule 26) are `DAT`. `هر روز` is a habit; `امروز` is a date.
33. A religious title is part of a `PER` span only when a personal name follows it
    (rule 9): `[امام صادق]`, `[حضرت علی]`, `[امام خمینی]`. Standing alone it is `O`,
    whatever it refers to in context: `امام` with no name following is `O` 128 times
    against 12 `PER`, `حضرت` 81 against 3. So `حضرت فرمودند`, `آن حضرت`, `به محضر حضرت`
    and `امام ( ع ) فرمود` label nothing. Collective references are `O` as well:
    `ائمهٔ اطهار`, `اهل‌بیت`, `معصومین`. `پیامبر` and `رسول خدا` stay `PER` (rule 7):
    they are used as the name of one person, not as a title before a name.
34. Peoples, tribes and ethnic groups are `O`, consistent with rule 4: `قوم مایا`,
    `ترکمن`, `اعراب`, `کردها`. A place named after such a group is still `LOC`.
35. A personal name used as an epithet is still `PER`: `[امیرکبیر] فوتبال ایران`. The name
    is the entity; the metaphor does not change that. (The silver layer usually leaves
    these unlabelled; this is a deliberate deviation.)
36. `دنیا` is `O`, and does NOT follow rule 15 the way `جهان` does. The two words diverge
    sharply in the silver layer: `جهان` is `LOC` 202 times against 87 `O`, while `دنیا` is
    `O` 240 times against 20 `LOC`. So `در سراسر دنیا` and `قفس دنیا` label nothing, while
    `در جهان` is `LOC`.
37. Century expressions are `DAT`, including the head noun and any calendar suffix:
    `[قرن سوم هجری]`, `[قرن بیستم]`, `[اوایل قرن نوزدهم]`.
38. Brands, products, software and vehicle models are `O`: `بنز`, `فتوشاپ`, `ویندوز`,
    `فایرفاکس`. The company behind them is `ORG` when the sentence names the company
    (`[شرکت مایکروسافت]`), not when it names the product.
39. Deities, angels and mythological beings are `O`, extending rule 11: `خدا`, `الله`,
    `یزدان`, `شیطان`, `ابلیس`, `جبرئیل`. Prophets and imams named as people stay `PER`.
40. Religious denominations, theological and philosophical schools, and political
    movements are `O`, extending rule 5: `شیعه` (41 `O`), `سنی` (28), `معتزله` (5),
    `اشاعره`, `صوفیه`, `مارکسیسم` (4). A named organisation belonging to one is still
    `ORG` (`[حزب توده]`).

## Changelog

v2, derived from adjudicating 500 annotated sentences against the silver layer
(`annotation/data/llm/ner-v1-default/pool-500.agreement.json`):

- Rules 31–37 added. 31–34 close gaps that produced most of the real annotation errors
  under v1 (events tagged `ORG`, habituals tagged `DAT`, collective religious references
  tagged `PER`, ethnonyms tagged `LOC`). 35–37 decide cases where v1 was silent.
- Rules 18, 20 and 29 were already correct but were violated repeatedly, so each now names
  the exact forms that were mislabelled: `غرب` as "the West", bare `شهرداری`/`فدراسیون`/
  `مجلس`/`تیم ملی`, and bare `نوروز`.
- Rules 1–17, 19, 21–28, 30 are unchanged from v1. Numbering is stable across versions.
- A first v2 draft over-corrected: it dropped `[کلینیک نور]` (`ORG` with a name, rule 20).
  Rules 25 and 29 were re-checked against train-split counts (`این روزها` 0/24 `O`,
  `نوروز` 30 `O` vs 3 `DAT`) and now state both directions explicitly.

v2.1, after annotating the full 1455-sentence test split and adjudicating every
disagreement:

- Rule 33 is reverted and rewritten. v2 claimed a bare `امام`/`حضرت` denoting a specific
  person is `PER`, generalising from 11 instances of `امام (`. The base rate contradicts
  it: with no personal name following, `امام` is `O` 128 times against 12 `PER` and
  `حضرت` 81 against 3. That single rule caused 7 of the 19 real annotation errors left on
  the full split.
- Rules 38 (brands and products) and 39 (deities and mythological beings) added; they
  cover 5 more of those 19 errors (`بنز`, `فتوشاپ`, `جاماسپ`, `شیطان`, `یزدان`).

v2.2, after relabelling the first 2,000 train sentences (largely literary and memoir prose,
a genre the test split barely contains):

- Rule 36 is reversed. v2.1 claimed `دنیا` behaves like `جهان`; the silver layer says the
  opposite, `دنیا` `O` 240 against 20 `LOC` while `جهان` is `LOC` 202 against 87 `O`.
- Rule 40 added: denominations, theological schools and political movements are `O`
  (`شیعه` 41 `O`, `سنی` 28, `معتزله` 5, `مارکسیسم` 4). `معتزله` tagged `ORG` was an error
  with no rule to prevent it.
