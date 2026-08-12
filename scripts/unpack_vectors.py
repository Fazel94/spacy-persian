"""Unpack a spaCy vectors-only wheel into a plain model directory.

`spacy train --paths.vectors` wants a directory it can `spacy.load()`. The fa_floret wheel
already contains exactly that (an empty pipeline carrying only vocab/vectors), it is just
buried under the wheel's package layout, so this unwraps it rather than pip-installing a
package whose only job is to hold a 57 MB array.

Usage:
  python scripts/unpack_vectors.py fa_floret-0.1.0-py3-none-any-400k-documents.whl \\
      assets/vectors/fa_floret_400k
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(args.wheel) as z:
            z.extractall(tmp)
        # The model directory is the one holding config.cfg, e.g. fa_floret/fa_floret-0.1.0/.
        models = sorted(p.parent for p in tmp.rglob("config.cfg"))
        if len(models) != 1:
            sys.exit(f"expected exactly one config.cfg in {args.wheel}, found {len(models)}")
        if args.output.exists():
            shutil.rmtree(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(models[0]), str(args.output))

    import spacy

    nlp = spacy.load(args.output)
    vectors = nlp.vocab.vectors
    if vectors.shape[0] == 0:
        sys.exit(f"{args.output} has no vectors")
    print(
        f"{args.output}: mode={vectors.mode} shape={vectors.shape} "
        f"n_keys={vectors.n_keys} pipeline={nlp.pipe_names}"
    )


if __name__ == "__main__":
    main()
