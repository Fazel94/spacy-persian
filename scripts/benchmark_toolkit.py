"""Time other Persian NLP toolkits on the same input as scripts/benchmark_throughput.py.

Each toolkit runs in its own env under .venv-bench/, never in .venv, so its pins cannot reach
the training environment. The spaCy side exports the input once; each toolkit then runs its
full pipeline over it:

    python scripts/benchmark_toolkit.py export
    .venv-bench/udpipe/bin/python scripts/benchmark_toolkit.py run udpipe --model <file>.udpipe
    .venv-bench/stanza/bin/python scripts/benchmark_toolkit.py run stanza --model <dir>
    .venv-bench/dadmatools/bin/python scripts/benchmark_toolkit.py run dadmatools --model <dir>

The protocol matches benchmark_throughput.py: raw document texts in, one discarded warmup
pass, the median of --runs timed passes, CPU only. Words/s divides by the PerDT gold token
count of the documents processed, not by the toolkit's own token count, because every
tokenizer splits the same text differently.
"""

import argparse
import json
import os
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

TEXTS = "metrics/ud-test-texts.json"


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def export(args):
    import spacy
    from spacy.tokens import DocBin

    docs = list(DocBin().from_disk(args.corpus).get_docs(spacy.blank("fa").vocab))
    payload = {
        "corpus": args.corpus,
        "texts": [d.text for d in docs],
        "words": [len(d) for d in docs],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"wrote {out}: {len(docs)} docs, {sum(payload['words'])} words")


def udpipe(args):
    from ufal.udpipe import Model, Pipeline, ProcessingError

    model = Model.load(args.model)
    if model is None:
        raise SystemExit(f"cannot load UDPipe model {args.model}")
    pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    error = ProcessingError()

    # `model` as a default argument keeps it alive: the SWIG Pipeline holds a raw pointer to
    # it, and collecting it when this function returns makes the first process() segfault.
    def process(texts, model=model):
        for text in texts:
            pipeline.process(text, error)
            if error.occurred():
                raise RuntimeError(error.message)

    info = {"version": version("ufal.udpipe"), "model_file": Path(args.model).name,
            "calls": "one per document"}
    return process, ["tokenizer", "tagger (UPOS, XPOS, FEATS, lemma)", "parser"], info


def stanza(args):
    import stanza as st

    processors = args.processors or "tokenize,mwt,pos,lemma,depparse"
    nlp = st.Pipeline("fa", dir=args.model, processors=processors, use_gpu=False,
                      download_method=st.DownloadMethod.REUSE_RESOURCES,
                      logging_level="WARN")

    def process(texts):
        nlp.bulk_process(texts)

    info = {"version": st.__version__, "calls": "one bulk_process per pass"}
    return process, processors.split(","), info


def dadmatools(args):
    import dadmatools.pipeline.language as language

    processors = args.processors or "tok,lem,pos,dep"
    nlp = language.Pipeline(processors, gpu=False, cache_dir=args.model)

    def process(texts):
        for text in texts:
            nlp(text)

    info = {"version": version("dadmatools"), "calls": "one per document"}
    return process, processors.split(","), info


RUNNERS = {"udpipe": udpipe, "stanza": stanza, "dadmatools": dadmatools}


def run(args):
    payload = json.loads(Path(args.texts).read_text(encoding="utf8"))
    texts, words = payload["texts"], payload["words"]
    if args.limit:
        texts, words = texts[:args.limit], words[:args.limit]
    n_words = sum(words)

    process, components, info = RUNNERS[args.tool](args)
    process(texts[:args.warmup_docs])

    wps = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        process(texts)
        wps.append(n_words / (time.perf_counter() - t0))

    record = {
        "model": f"{args.tool} {info.pop('version')}",
        "pipeline": components,
        "device": f"cpu ({cpu_model()})",
        "docs": len(texts),
        "words": n_words,
        "runs": [round(w, 1) for w in wps],
        "wps_median": round(statistics.median(wps), 1),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        **info,
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    out = Path(args.output or f"metrics/throughput-toolkits-{args.tool}-cpu.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("export", help="write the test texts and gold word counts (needs spaCy)")
    ex.add_argument("--corpus", default="corpus/merged/fa_perdt-ud-test.spacy")
    ex.add_argument("--output", default=TEXTS)

    rn = sub.add_parser("run", help="time one toolkit on the exported texts")
    rn.add_argument("tool", choices=sorted(RUNNERS))
    rn.add_argument("--texts", default=TEXTS)
    rn.add_argument("--model", help="UDPipe model file, Stanza resources directory, or "
                                     "DadmaTools cache directory")
    rn.add_argument("--processors", help="Stanza or DadmaTools processors to run")
    rn.add_argument("--runs", type=int, default=3)
    rn.add_argument("--warmup-docs", type=int, default=32,
                    help="docs in the discarded warmup pass; 32 matches benchmark_throughput.py")
    rn.add_argument("--limit", type=int, default=0, help="cap number of docs (0 = all)")
    rn.add_argument("--output", help="JSON record path")

    args = ap.parse_args()
    export(args) if args.command == "export" else run(args)


if __name__ == "__main__":
    main()
