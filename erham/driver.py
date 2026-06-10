"""Top-level driver -- the Fortran PROGRAM ERHAM.

Reads an input file, optionally performs the least-squares fit, computes the
derived parameters and writes the predictions.  Usage mirrors the original
interactive prompts but also accepts file names on the command line.
"""

from __future__ import annotations

import sys
from datetime import datetime

from .iterate import iter_fit
from .reader import read_input
from .workspace import Workspace

_BANNER = "program ERHAM V16g-R3 20 may 2013"
_MONTHS = [" jan", " feb", " mar", " apr", " may", " jun",
           " jul", " aug", " sep", " oct", " nov", " dec"]


def _timestamp() -> str:
    n = datetime.now()
    return (f"{_BANNER}  ***  date and time:{n.day:3d}{_MONTHS[n.month - 1]}"
            f"{n.year:5d}{n.hour:5d}:{n.minute:2d}:{n.second:2d}\n")


def run(filein: str, fileout: str, catalog_inputs=None) -> None:
    with open(filein, "r") as f:
        lines = f.read().splitlines()

    out = open(fileout, "w")
    try:
        out.write(_timestamp() + "\n")
        model = read_input(lines, out)
        ws = Workspace()

        if model.NIT > 0:
            iter_fit(model, ws, out)

        from .derived import ftest
        from .predict import predic
        ftest(model, ws, out)
        predic(model, ws, out, catalog_inputs)

        out.write(_timestamp())
    finally:
        out.close()


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2:
        filein, fileout = argv[0], argv[1]
    else:
        filein = input("Enter input file name !\n").strip()
        fileout = input("Enter output file name !\n").strip()
    print(_BANNER)
    run(filein, fileout)


if __name__ == "__main__":
    main()
