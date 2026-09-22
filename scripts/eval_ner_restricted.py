"""Score an NER pipeline on a corpus, restricted to a shared label set.

The PerDT silver NER layer carries seven labels (PER, LOC, ORG, DAT, MON, TIM, PCT);
the LLM-relabelled corpus in `corpus/perdt-ner-iob-llm/` carries four (PER, LOC, ORG,
DAT). A raw `ents_f` from `spacy benchmark accuracy` is therefore not comparable across
the two: a seven-label model scored on four-label data is punished for every MON/TIM/PCT
span it correctly finds, because the reference deliberately does not contain them.

This drops entities outside `--labels` from BOTH the prediction and the reference before
scoring, so the number is a like-for-like micro average over the shared types. Per-type
rows carry gold support, which `spacy benchmark accuracy` omits and which is needed to
tell a real difference from noise on a thin label.
"""

from pathlib import Path

import spacy
import srsly
import typer
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Corpus, Example

DEFAULT_LABELS = "PER,LOC,ORG,DAT"


def restrict(doc: Doc, keep: set[str]) -> Doc:
    doc.ents = [e for e in doc.ents if e.label_ in keep]
    return doc


def main(
    model: str = typer.Argument(..., help="pipeline to score"),
    corpus_path: Path = typer.Argument(..., help=".spacy DocBin to score against"),
    labels: str = typer.Option(DEFAULT_LABELS, help="comma-separated labels to keep"),
    output: Path = typer.Option(None, help="write JSON scores here"),
    gpu_id: int = typer.Option(-1, "--gpu-id"),
):
    if gpu_id >= 0:
        spacy.require_gpu(gpu_id)
    keep = {x.strip() for x in labels.split(",") if x.strip()}
    nlp = spacy.load(model)

    examples = []
    dropped_pred = dropped_gold = 0
    for eg in Corpus(corpus_path, gold_preproc=False)(nlp):
        # eg.predicted carries the gold tokenization with no annotation; running the
        # pipeline over it is what nlp.evaluate does, and it keeps alignment exact.
        predicted = nlp(eg.predicted)
        dropped_pred += sum(1 for e in predicted.ents if e.label_ not in keep)
        dropped_gold += sum(1 for e in eg.reference.ents if e.label_ not in keep)
        examples.append(
            Example(restrict(predicted, keep), restrict(eg.reference, keep))
        )

    scores = Scorer.score_spans(examples, "ents")
    support: dict[str, int] = {}
    for eg in examples:
        for ent in eg.reference.ents:
            support[ent.label_] = support.get(ent.label_, 0) + 1

    result = {
        "model": str(model),
        "corpus": str(corpus_path),
        "labels": sorted(keep),
        "ents_p": scores["ents_p"],
        "ents_r": scores["ents_r"],
        "ents_f": scores["ents_f"],
        "ents_per_type": {
            label: {**vals, "support": support.get(label, 0)}
            for label, vals in sorted(scores["ents_per_type"].items())
        },
        "gold_entities": sum(support.values()),
        "dropped_out_of_scope": {"predicted": dropped_pred, "reference": dropped_gold},
    }

    def pct(x):
        return "   n/a" if x is None else f"{100 * x:6.2f}"

    print(f"\n{model}  vs  {corpus_path}   [{','.join(sorted(keep))}]")
    print(f"{'':10}{'P':>7}{'R':>7}{'F':>7}{'gold':>8}")
    print(f"{'ALL':10}{pct(scores['ents_p'])}{pct(scores['ents_r'])}"
          f"{pct(scores['ents_f'])}{sum(support.values()):8d}")
    for label, vals in result["ents_per_type"].items():
        print(f"{label:10}{pct(vals['p'])}{pct(vals['r'])}{pct(vals['f'])}"
              f"{vals['support']:8d}")
    print(f"out-of-scope dropped: {dropped_pred} predicted, {dropped_gold} reference")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        srsly.write_json(output, result)
        print(f"wrote {output}")


if __name__ == "__main__":
    typer.run(main)
