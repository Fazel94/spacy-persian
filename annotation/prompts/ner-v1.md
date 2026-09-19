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
{{GUIDELINES}}
</guideline>
