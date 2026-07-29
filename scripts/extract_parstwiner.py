"""Unpack the ParsTwiNER release zip into flat IOB2 files.

The archive carries macOS resource-fork junk (`__MACOSX/._*`) that would confuse
`spacy convert`, so only the three real splits are extracted.

Usage: .venv/bin/python scripts/extract_parstwiner.py assets/ner/ParsTwiNER_corpus_v1.0.0.zip assets/ner
"""

import sys
import zipfile
from pathlib import Path

SPLITS = ("train.txt", "dev.txt", "test.txt")


def main():
    archive = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        missing = [s for s in SPLITS if s not in names]
        if missing:
            raise SystemExit(f"{archive}: missing expected members {missing}")
        for split in SPLITS:
            zf.extract(split, dest)
            path = dest / split
            sents = path.read_text(encoding="utf8").strip().split("\n\n")
            tokens = sum(1 for line in path.open(encoding="utf8") if line.strip())
            print(f"{path}: {len(sents)} sentences, {tokens} tokens")


if __name__ == "__main__":
    main()
