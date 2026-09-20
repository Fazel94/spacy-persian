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
- `label` must be exactly one of PER, LOC, ORG, DAT. Never emit O; to mark something as not
  an entity, leave it out.

These are the mistakes made most often on this corpus. Check every candidate span against
this list before returning it, in both directions — the rule tells you what to drop AND what
to keep:
- An organisational head noun with NO proper name after it is not an entity: شهرداری،
  فدراسیون، مجلس، دولت، وزارت، سفارت، دانشگاه، تیم، تیم ملی. As soon as a name or
  specifier follows, the whole phrase is ORG: شهرداری تهران، دولت ایران، تیم ملی بوکس،
  کلینیک نور، وزارت امور خارجه، دانشگاه تهران.
- A bare compass term is not an entity, including غرب meaning "the West" as a civilisation
  or bloc. غرب تهران is LOC.
- Events are not entities: جام جهانی، المپیک، نمایشگاه CES، لیگ قهرمانان آسیا. The body
  that runs or plays in them is ORG when named.
- Habituals are not dates: هر روز، هر شب، هر هفته، روزانه. Nor are the vague deictics این
  روزها، این ماه، این سال. But امروز، دیروز، این هفته، سال گذشته، همین حالا-type relative
  dates ARE dates, and so are قرن بیستم، قرن سوم هجری.
- A festival name alone is not a date: نوروز، عاشورا، تشریفات نوروز. With a date head noun
  or a year it is: ماه رمضان، سال نو چینی، نوروز امسال.
- A religious title counts only when a personal name follows it: امام صادق، حضرت علی are
  persons. Alone it is nothing, however specific the reference: حضرت فرمودند، آن حضرت، امام
  ( ع ) فرمود tag nothing. Collective references tag nothing either: ائمهٔ اطهار، اهل‌بیت،
  معصومین. پیامبر and رسول خدا are persons.
- Brands, products and software are not organisations: بنز، فتوشاپ، ویندوز، فایرفاکس. The
  company is ORG only when the sentence names the company: شرکت مایکروسافت.
- Deities and mythological beings are not persons: خدا، الله، یزدان، شیطان، جبرئیل.
- Peoples and ethnic groups are not places: قوم مایا، ترکمن، اعراب. Denominations,
  theological schools and political movements are not organisations: شیعه، سنی، معتزله،
  صوفیه، مارکسیسم.
- دنیا is never a place, however it is used: در سراسر دنیا، قفس دنیا tag nothing. جهان
  usually is one: در جهان، بزرگ‌ترین تالاب جهان are LOC.

<guideline>
{{GUIDELINES}}
</guideline>
