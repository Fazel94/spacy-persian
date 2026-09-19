<!-- Frozen rendering of prompts/ner-v1.md with GUIDELINES.md v1 substituted.
     prompt_hash f30dcc84d745. This exact text produced annotation/data/llm/ner-v1-default/.
     GUIDELINES.md has since moved to v2; this file is the only copy of the v1 guideline. -->

You are annotating Persian sentences for named entities. Labels: PER, LOC, ORG, DAT.
Follow the guideline below exactly; it was derived from the corpus and overrides your own
intuition where they differ.

Output rules:
- Return one item per input id, every id, no extra ids.
- For each entity return `text` copied character-for-character from the sentence (same
  spaces, same ZWNJ U+200C, same digits) and `label`.
- List entities in sentence order. Repeat mentions are separate entities.
- A sentence with no entity gets an empty `entities` list.
- Never nest or overlap spans. Never include honorifics آقای/دکتر, the parenthetical ( ع ) /
  ( ص ), or role nouns like رئیس/وزیر.

<guideline>
# Persian NER annotation guidelines, v1

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
18. Compass regions are `O` unless part of a named region: `غرب` alone `O` (84 vs 15);
    `[غرب تهران]`, `[خاورمیانه]` are `LOC`.

## ORG

19. Companies, institutions, ministries, parties, armed groups, sports clubs, media outlets,
    banks, universities, international bodies: `گوگل`, `طالبان`, `استقلال`, `سازمان ملل`,
    `بانک مرکزی`, `مجلس شورای اسلامی`, `صدا و سیما`, `اینترپل`, `ساواک`.
20. Organisational head nouns are INSIDE the span when followed by a specifier:
    `[دولت ایران]`, `[وزارت امور خارجه]`, `[شرکت مخابرات ایران]`, `[تیم استقلال]`,
    `[سفارت ایران]`, `[ارتش دریایی ایران]`, `[دانشگاه تهران]`. Alone they are `O`
    (`دولت` alone 221 `O`, `تیم` 129 `O`, `مجلس` 132 `O`).
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
    `[سال گذشته]`, `[سال جاری]`, `[هفتهٔ آینده]`, `[این هفته]`, `[روز گذشته]`.
26. Durations anchored to now are `DAT`: `[10 سال پیش]`, `[سه دهه پیش]`, `[یک ماه قبل]`.
    Unanchored durations are `O`: `به مدت سه سال`.
27. A date range with `از ... تا/الی ...` is ONE span including `از` and `تا`:
    `[از بهمن 1343 تا خرداد 1357]`, `[از سال 1375 تا سال 1386]`, `[از 420 تا 438 میلادی]`.
    `[از آغاز امسال] تاکنون`: `تاکنون` is outside.
28. Calendar suffixes are inside: `[سال 1251 شمسی]`, `[438 میلادی]`; the abbreviated
    `ه . ش` after a year is outside.
29. Eras and dynasties as periods are `O` (`هخامنشی`, `ساسانی`); named festivals `نوروز`,
    `عاشورا` are `O` unless used as a date (`در نوروز امسال` → whole phrase `DAT`).
30. Times of day, clock times, money and percentages are `O` in this label set.
</guideline>
