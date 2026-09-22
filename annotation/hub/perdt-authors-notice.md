# Courtesy notice to the PerDT authors

To: rasooli@seas.upenn.edu, pegh.safari@gmail.com (contacts in the UD_Persian-PerDT README)
Subject: Redistribution of the PerDT NER layer on the Hugging Face Hub (CC BY-SA 4.0)

---

Dear Dr. Rasooli and Ms. Safari,

I am writing to let you know that I have redistributed a derivative of the Persian Universal
Dependency Treebank (PerUDT v1.0) on the Hugging Face Hub, and to give you the chance to
object or correct anything before it circulates further:

    https://huggingface.co/datasets/Phazel/fa-perdt-ner

What it contains:

1. `silver`: the named-entity layer from your repository's
   `not-to-release/Dadegan with NER tag/` directory, realigned onto the released UD
   tokenization (multiword tokens merged). All 29,107 sentences, your train/dev/test split,
   your seven labels. 99.5-99.9% of spans transferred; the rest were dropped, not guessed.
2. `llm`: a re-annotation of the same sentences with four labels (PER, LOC, ORG, DAT) by a
   large language model, against a written guideline that was itself inferred from your
   entity layer. It is clearly marked as machine-produced with no human gold.

I have read the licence as follows, and would be grateful for a correction if it is wrong:
the treebank is CC BY-SA 4.0, so both configurations are Adapted Material and are published
under CC BY-SA 4.0 with attribution to PerDT. The dataset's LICENSE and README credit the
treebank and cite Rasooli et al. (LREC 2022), Rasooli, Kouhestani and Moloodi (NAACL 2013),
and Taher, Hoseini and Shamsfard (2020) for the Beheshti-NER tagger your README names as the
source of the entity layer.

One point I want to be explicit about: the entity files sit under `not-to-release/`. I have
taken that in the UD sense, "excluded from the official release build", since the directory
is public in the repository under its LICENSE.txt. If you intended it to mean the layer
should not be redistributed, please tell me and I will take the `silver` configuration down
immediately; the `llm` configuration can stand on its own if you prefer.

The same data underlies the `Phazel/fa_core_news_*` spaCy pipelines, which carry the same
attribution.

Thank you for making the treebank available. Please reply if you would like any change to
the attribution, the description of your work, or the dataset's existence.

Kind regards,
Kiyarash Fazeli
https://github.com/Fazel94/spacy-persian
