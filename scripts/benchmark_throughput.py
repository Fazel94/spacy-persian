"""Measure inference throughput (words/second) for a pipeline, on CPU or GPU.

`spacy benchmark accuracy` prints a speed number, but it is scoring-contaminated: the
Scorer's per-token alignment and per-type bookkeeping run inside the timed region, which
matters a lot for the cheap CPU tiers and understates them. This times `nlp.pipe` only.

Reported figure is the median of `--runs` passes over the same texts, after a discarded
warmup pass. Median rather than mean because the first CUDA kernel launches, cuBLAS
autotuning and any page-cache miss produce outliers that a mean would smear into the result.

Batch size matters far more for the trf tier than the CPU tiers (a transformer amortizes a
GEMM over the batch; a hash-embed tok2vec barely cares), so it is a parameter and gets
recorded in the output rather than being left implicit.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def gpu_model():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="installed package name or path to a pipeline")
    ap.add_argument("--corpus", default="corpus/merged/fa_perdt-ud-test.spacy",
                    help="DocBin whose raw texts are used as input")
    ap.add_argument("--gpu-id", type=int, default=-1, help="-1 for CPU")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap number of docs (0 = all)")
    ap.add_argument("--output", default=None, help="write a JSON record here")
    args = ap.parse_args()

    if args.gpu_id >= 0:
        # require_gpu, not prefer_gpu: a silent fall back to CPU would be reported as a GPU
        # number, which is exactly the measurement error this script exists to avoid.
        spacy.require_gpu(args.gpu_id)
        device = f"gpu:{args.gpu_id} ({gpu_model()})"
    else:
        device = f"cpu ({cpu_model()})"

    nlp = spacy.load(args.model)
    vocab_docs = list(DocBin().from_disk(args.corpus).get_docs(spacy.blank("fa").vocab))
    if args.limit:
        vocab_docs = vocab_docs[:args.limit]
    texts = [d.text for d in vocab_docs]
    n_words = sum(len(d) for d in vocab_docs)

    # Warmup: first pass pays for lazy CUDA context creation, cuBLAS handles and any
    # transformer weight transfer. Timing it would misattribute setup cost to throughput.
    for _ in nlp.pipe(texts[:args.batch_size], batch_size=args.batch_size):
        pass

    wps = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        for _ in nlp.pipe(texts, batch_size=args.batch_size):
            pass
        elapsed = time.perf_counter() - t0
        wps.append(n_words / elapsed)

    median = statistics.median(wps)
    record = {
        "model": args.model,
        "pipeline": list(nlp.pipe_names),
        "device": device,
        "batch_size": args.batch_size,
        "docs": len(texts),
        "words": n_words,
        "runs": [round(w, 1) for w in wps],
        "wps_median": round(median, 1),
        "spacy_version": spacy.__version__,
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
