"""Table the sm vs md vs lg test-set deltas.

All tiers are trained from the same corpus, the same seed and the same architecture; the
only difference is the static vector table (none for sm, fa_floret 50k rows for md, fa_floret
200k rows for lg) via `include_static_vectors`. So the delta printed here is attributable to
the vector table and nothing else.

Reads the `spacy benchmark accuracy` reports written by the `evaluate-*` targets. Missing
files are reported rather than fatal, so this is runnable mid-build.

Usage:
  python scripts/compare_tiers.py [--metrics-dir metrics]
"""

import argparse
import json
from pathlib import Path

# (label, {tier_label: report_filename})
GROUPS = [
    (
        "dep pipeline, UD test",
        {"sm": "ud-test.json", "md": "md-ud-test.json", "lg": "lg-ud-test.json"},
    ),
    (
        "core pipeline, UD test",
        {
            "sm": "core-ud-test.json",
            "md": "md-core-ud-test.json",
            "lg": "lg-core-ud-test.json",
        },
    ),
    (
        "ent NER test",
        {
            "sm": "perdt-ner-test.json",
            "md": "md-perdt-ner-test.json",
            "lg": "lg-perdt-ner-test.json",
        },
    ),
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


def table(title, tiers, rows):
    """tiers: list of (label, data-dict-or-None), first tier is the baseline for deltas."""
    labels = [label for label, _ in tiers]
    base_label, base = tiers[0]
    print(f"\n## {title}\n")
    header = " | ".join(f"{label:>7}" for label in labels)
    delta_header = " | ".join(f"{'d(' + label + ')':>9}" for label, _ in tiers[1:])
    print(f"| {'metric':<12} | {header} | {delta_header} |")
    sep = " | ".join("-" * 7 for _ in labels)
    delta_sep = " | ".join("-" * 9 for _ in tiers[1:])
    print(f"| {'-' * 12} | {sep} | {delta_sep} |")
    for key, label in rows:
        values = [d.get(key) if d is not None else None for _, d in tiers]
        if all(v is None for v in values):
            continue
        # The NER report scores tag_acc 0.0 because its corpus has no gold tags.
        if all(v == 0.0 for v in values):
            continue
        cells = [f"{v * 100:.2f}" if isinstance(v, float) else "-" for v in values]
        deltas = []
        for v in values[1:]:
            a, b = values[0], v
            deltas.append(
                f"{(b - a) * 100:+.2f}" if isinstance(a, float) and isinstance(b, float) else "-"
            )
        row = " | ".join(f"{c:>7}" for c in cells)
        drow = " | ".join(f"{d:>9}" for d in deltas)
        print(f"| {label:<12} | {row} | {drow} |")
    speeds = [d.get("speed") if d is not None else None for _, d in tiers]
    if isinstance(speeds[0], float):
        cells = [f"{s:.0f}" if isinstance(s, float) else "-" for s in speeds]
        deltas = [
            f"{s / speeds[0] - 1:+.1%}" if isinstance(s, float) else "-" for s in speeds[1:]
        ]
        row = " | ".join(f"{c:>7}" for c in cells)
        drow = " | ".join(f"{d:>9}" for d in deltas)
        print(f"| {'words/s':<12} | {row} | {drow} |")


def per_type(title, tiers):
    per_types = [(label, (d or {}).get("ents_per_type")) for label, d in tiers]
    if not any(pt for _, pt in per_types):
        return
    labels = [label for label, _ in tiers]

    def pct(v):
        return f"{v * 100:.2f}" if v is not None else "-"

    all_labels = set()
    for _, pt in per_types:
        if pt:
            all_labels |= set(pt)

    print(f"\n### {title}, per label\n")
    header = " | ".join(f"{label + ' F':>7}" for label in labels)
    delta_header = " | ".join(f"{'d(' + label + ')':>9}" for label in labels[1:])
    print(f"| {'label':<6} | {header} | {delta_header} |")
    sep = " | ".join("-" * 7 for _ in labels)
    delta_sep = " | ".join("-" * 9 for _ in labels[1:])
    print(f"| {'-' * 6} | {sep} | {delta_sep} |")

    def sort_key(entity_label):
        last_pt = per_types[-1][1] or {}
        return -last_pt.get(entity_label, {}).get("f", 0)

    for entity_label in sorted(all_labels, key=sort_key):
        fs = [(pt or {}).get(entity_label, {}).get("f") for _, pt in per_types]
        cells = [pct(f) for f in fs]
        deltas = []
        for f in fs[1:]:
            a = fs[0]
            deltas.append(f"{(f - a) * 100:+.2f}" if a is not None and f is not None else "-")
        row = " | ".join(f"{c:>7}" for c in cells)
        drow = " | ".join(f"{d:>9}" for d in deltas)
        print(f"| {entity_label:<6} | {row} | {drow} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    args = ap.parse_args()

    print("# sm vs md vs lg (fa_floret static vectors)")
    print("\nSame corpus, same seed, same architecture per group. Only difference:")
    print("`include_static_vectors = false -> true`, and which floret table (md: 50k rows,")
    print("400k documents; lg: 200k rows, full Persian Wikipedia, 5 epochs).")

    for title, reports in GROUPS:
        tiers = []
        missing = []
        for label, fname in reports.items():
            data = load(args.metrics_dir / fname)
            if data is None:
                missing.append(fname)
            tiers.append((label, data))
        if tiers[0][1] is None:
            print(f"\n## {title}\n\n  (skipped, missing baseline {reports[list(reports)[0]]})")
            continue
        table(title, tiers, SCALARS)
        per_type(title, tiers)
        if missing:
            print(f"\n  (missing: {', '.join(missing)})")


if __name__ == "__main__":
    main()
