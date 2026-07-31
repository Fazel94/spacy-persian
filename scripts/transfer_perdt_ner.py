"""Transfer PerDT's NER layer onto the tokenization `spacy convert --merge-subtokens` produces.

The NER files use the original Dadegan tokenization, which matches the released UD tokenization
in only 57 to 62% of sentences: the NER files drop some copulas and auxiliaries, and at least
one honorific is corrupted (`ص` written as `،`). Entities sit on content words present in both,
so difflib aligns them. A span transfers only if every one of its tokens maps and the result
stays contiguous; anything else is dropped rather than guessed.

Measured transfer rate: 99.86% (train), 99.74% (dev), 99.51% (test). See PLAN.md §1.

Usage:
  python scripts/transfer_perdt_ner.py --conllu-dir assets/ud --ner-dir assets/ud-ner \\
      --out corpus/perdt-ner-iob

With --out, writes IOB2 files on the merged tokenization to DIR/{split}.txt.
Without it, only reports the transfer rate.
"""

import argparse
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

SPLITS = ("train", "dev", "test")


def merged_tokens(conllu_path):
    """Tokens as --merge-subtokens yields them: multiword-token surface forms, not subtokens."""
    sents, cur, skip_to = [], [], 0
    for line in conllu_path.open(encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if not line:
            if cur:
                sents.append(cur)
                cur, skip_to = [], 0
            continue
        c = line.split("\t")
        if "." in c[0]:
            continue
        if "-" in c[0]:
            skip_to = int(c[0].split("-")[1])
            cur.append(c[1])
            continue
        if skip_to and int(c[0]) <= skip_to:
            continue
        cur.append(c[1])
    if cur:
        sents.append(cur)
    return sents


def iob_sents(path):
    sents, cur = [], []
    for line in path.open(encoding="utf8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        p = line.split("\t")
        if len(p) < 2:
            p = line.split()
        if len(p) < 2:
            continue
        cur.append((p[0], p[-1]))
    if cur:
        sents.append(cur)
    return sents


def spans_from_iob(tagged):
    """[(start, end, label)] over the source token indices."""
    spans, i = [], 0
    while i < len(tagged):
        tag = tagged[i][1]
        if tag.startswith("B-"):
            label, j = tag[2:], i + 1
            while j < len(tagged) and tagged[j][1] == f"I-{label}":
                j += 1
            spans.append((i, j, label))
            i = j
        else:
            i += 1
    return spans


def index_map(src, dst):
    """src index -> dst index for tokens difflib considers equal."""
    out = {}
    matcher = SequenceMatcher(a=src, b=dst, autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            out[a + k] = b + k
    return out


def transfer_split(conllu, ner_file):
    """Return (tokens, tags) per sentence on the merged tokenization, plus counters."""
    gold = merged_tokens(conllu)
    ner = iob_sents(ner_file)
    moved, lost = Counter(), Counter()
    result = []
    for i in range(min(len(gold), len(ner))):
        dst = gold[i]
        src = [t for t, _ in ner[i]]
        m = index_map(src, dst)
        tags = ["O"] * len(dst)
        for start, end, label in spans_from_iob(ner[i]):
            idx = [m[k] for k in range(start, end) if k in m]
            contiguous = idx and idx == list(range(idx[0], idx[0] + len(idx)))
            if len(idx) == end - start and contiguous:
                tags[idx[0]] = f"B-{label}"
                for k in idx[1:]:
                    tags[k] = f"I-{label}"
                moved[label] += 1
            else:
                lost[label] += 1
        result.append((dst, tags))
    return result, moved, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu-dir", default="assets/ud",
                    help="directory holding fa_perdt-ud-{split}.conllu")
    ap.add_argument("--ner-dir", default="assets/ud-ner",
                    help="directory holding {split}_with_NER_tag.txt")
    ap.add_argument("--out", default=None, help="write IOB2 files here")
    args = ap.parse_args()

    conllu_dir, nerdir = Path(args.conllu_dir), Path(args.ner_dir)
    for d in (conllu_dir, nerdir):
        if not d.is_dir():
            raise SystemExit(f"not found: {d}")
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    grand_moved, grand_lost = Counter(), Counter()
    for split in SPLITS:
        sents, moved, lost = transfer_split(
            conllu_dir / f"fa_perdt-ud-{split}.conllu",
            nerdir / f"{split}_with_NER_tag.txt",
        )
        m, dropped = sum(moved.values()), sum(lost.values())
        grand_moved += moved
        grand_lost += lost
        print(f"{split:<6} entities {m + dropped:>6}  transferred {m:>6}"
              f" ({100 * m / max(m + dropped, 1):.2f}%)  dropped {dropped}")
        if out:
            path = out / f"{split}.txt"
            with path.open("w", encoding="utf8") as fh:
                for toks, tags in sents:
                    for tok, tag in zip(toks, tags):
                        fh.write(f"{tok}\t{tag}\n")
                    fh.write("\n")
            print(f"       wrote {path}")

    print("\nper label:")
    for label in sorted(grand_moved | grand_lost, key=lambda k: -grand_moved[k]):
        m, dropped = grand_moved[label], grand_lost[label]
        print(f"  {label:<6} {m:>6}/{m + dropped:<6} ({100 * m / max(m + dropped, 1):5.1f}%)")


if __name__ == "__main__":
    main()
