"""Table the sm vs md test-set deltas.

Both tiers are trained from the same corpus, the same seed and the same architecture; the
only difference is `include_static_vectors`. So the delta printed here is attributable to the
fa_floret vector table and nothing else.

Reads the `spacy benchmark accuracy` reports written by the `evaluate-*` targets. Missing
files are reported rather than fatal, so this is runnable mid-build.

Usage:
  python scripts/compare_tiers.py [--metrics-dir metrics]
"""

import argparse
import json
from pathlib import Path

# (label, sm report, md report)
PAIRS = [
    ("dep pipeline, UD test", "ud-test.json", "md-ud-test.json"),
    ("core pipeline, UD test", "core-ud-test.json", "md-core-ud-test.json"),
    ("core pipeline, NER test", "perdt-ner-test.json", "md-perdt-ner-test.json"),
]

SCALARS = [
    ("tag_acc", "TAG_ACC"),
    ("pos_acc", "POS_ACC"),
    ("morph_acc", "MORPH_ACC"),
    ("lemma_acc", "LEMMA_ACC"),
    ("dep_uas", "DEP_UAS"),
    ("dep_las", "DEP_LAS"),
    ("sents_f", "SENTS_F"),
    ("ents_p", "ENTS_P"),
    ("ents_r", "ENTS_R"),
    ("ents_f", "ENTS_F"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def table(title, sm, md, rows):
    print(f"\n## {title}\n")
    print(f"| {'metric':<12} | {'sm':>7} | {'md':>7} | {'delta':>7} |")
    print(f"| {'-' * 12} | {'-' * 7} | {'-' * 7} | {'-' * 7} |")
    for key, label in rows:
        a, b = sm.get(key), md.get(key)
        if a is None and b is None:
            continue
        # The NER report scores tag_acc 0.0 because its corpus has no gold tags.
        if a == 0.0 and b == 0.0:
            continue
        cells = [f"{v * 100:.2f}" if isinstance(v, float) else "-" for v in (a, b)]
        delta = f"{(b - a) * 100:+.2f}" if isinstance(a, float) and isinstance(b, float) else "-"
        print(f"| {label:<12} | {cells[0]:>7} | {cells[1]:>7} | {delta:>7} |")
    for key, label in (("speed", "words/s"),):
        a, b = sm.get(key), md.get(key)
        if isinstance(a, float) and isinstance(b, float):
            print(f"| {label:<12} | {a:>7.0f} | {b:>7.0f} | {b / a - 1:>+6.1%} |")


def per_type(title, sm, md):
    a, b = sm.get("ents_per_type"), md.get("ents_per_type")
    if not a or not b:
        return

    def pct(v):
        return f"{v * 100:.2f}" if v is not None else "-"

    print(f"\n### {title}, per label\n")
    print(f"| {'label':<6} | {'sm F':>7} | {'md F':>7} | {'delta':>7} |")
    print(f"| {'-' * 6} | {'-' * 7} | {'-' * 7} | {'-' * 7} |")
    for label in sorted(set(a) | set(b), key=lambda k: -b.get(k, {}).get("f", 0)):
        fa, fb = a.get(label, {}).get("f"), b.get(label, {}).get("f")
        delta = f"{(fb - fa) * 100:+.2f}" if fa is not None and fb is not None else "-"
        print(f"| {label:<6} | {pct(fa):>7} | {pct(fb):>7} | {delta:>7} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    args = ap.parse_args()

    print("# sm vs md (fa_floret 400k static vectors)")
    print("\nSame corpus, same seed, same architecture. Only difference:")
    print("`include_static_vectors = false -> true`.")

    for title, sm_name, md_name in PAIRS:
        sm = load(args.metrics_dir / sm_name)
        md = load(args.metrics_dir / md_name)
        if sm is None or md is None:
            missing = [n for n, d in ((sm_name, sm), (md_name, md)) if d is None]
            print(f"\n## {title}\n\n  (skipped, missing {', '.join(missing)})")
            continue
        table(title, sm, md, SCALARS)
        per_type(title, sm, md)


if __name__ == "__main__":
    main()
