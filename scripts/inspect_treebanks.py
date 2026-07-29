"""Report annotation coverage of the UD Persian treebanks in assets/ud/.

Usage: .venv/bin/python scripts/inspect_treebanks.py [assets/ud]
"""

import sys
from collections import Counter
from pathlib import Path


def read_conllu(path):
    sent = []
    for line in path.open(encoding="utf8"):
        line = line.rstrip("\n")
        if not line:
            if sent:
                yield sent
                sent = []
            continue
        if line.startswith("#"):
            continue
        sent.append(line.split("\t"))
    if sent:
        yield sent


def stats(path):
    s = {
        "sents": 0,
        "toks": 0,
        "mwt": 0,
        "empty": 0,
        "lemma": 0,
        "feats": 0,
        "nonproj_root": 0,
    }
    upos, xpos, dep = Counter(), Counter(), Counter()
    for sent in read_conllu(path):
        s["sents"] += 1
        roots = 0
        for c in sent:
            if "-" in c[0]:
                s["mwt"] += 1
                continue
            if "." in c[0]:
                s["empty"] += 1
                continue
            s["toks"] += 1
            upos[c[3]] += 1
            xpos[c[4]] += 1
            dep[c[7]] += 1
            if c[2] not in ("_", ""):
                s["lemma"] += 1
            if c[5] != "_":
                s["feats"] += 1
            if c[7] == "root":
                roots += 1
        if roots != 1:
            s["nonproj_root"] += 1
    return s, upos, xpos, dep


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/ud")
    agg = {}
    print(
        f"{'file':<30}{'sents':>8}{'tokens':>10}{'MWT':>7}{'empty':>7}"
        f"{'lemma%':>8}{'feats%':>8}{'UPOS':>6}{'XPOS':>6}{'DEP':>5}"
    )
    for path in sorted(root.glob("*.conllu")):
        s, upos, xpos, dep = stats(path)
        agg[path.name] = (s, upos, xpos, dep)
        print(
            f"{path.name:<30}{s['sents']:>8}{s['toks']:>10}{s['mwt']:>7}{s['empty']:>7}"
            f"{100 * s['lemma'] / s['toks']:>8.1f}{100 * s['feats'] / s['toks']:>8.1f}"
            f"{len(upos):>6}{len(xpos):>6}{len(dep):>5}"
        )

    for name, (s, upos, xpos, dep) in agg.items():
        if not name.endswith("train.conllu"):
            continue
        print(f"\n=== {name} ===")
        print(f"UPOS ({len(upos)}): {' '.join(sorted(upos))}")
        print(f"XPOS ({len(xpos)}): {' '.join(sorted(xpos))}")
        print(f"DEPREL ({len(dep)}): {' '.join(sorted(dep))}")
        print(f"sentences with != 1 root: {s['nonproj_root']}")


if __name__ == "__main__":
    main()
