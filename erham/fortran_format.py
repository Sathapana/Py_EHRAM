"""Helpers to reproduce Fortran edit descriptors (F, I, D, A) for output."""

from __future__ import annotations

import math


def fI(n: int, w: int) -> str:
    """Fortran Iw."""
    s = f"{int(n):d}"
    if len(s) > w:
        return "*" * w
    return s.rjust(w)


def fF(x: float, w: int, d: int) -> str:
    """Fortran Fw.d (fixed point)."""
    s = f"{x:.{d}f}"
    # Fortran prints -0.000... as -0.000; Python agrees.  Avoid "-0" with no dec.
    if len(s) > w:
        return "*" * w
    return s.rjust(w)


def fD(x: float, w: int, d: int) -> str:
    """Fortran Dw.d (normalised mantissa 0.xxxx with a D exponent)."""
    if x == 0.0 or not math.isfinite(x):
        mant = 0.0
        exp = 0
    else:
        exp = math.floor(math.log10(abs(x))) + 1
        mant = x / (10.0 ** exp)
    s = f"{mant:.{d}f}"
    # handle rounding that pushes |mant| up to 1.0
    if abs(float(s)) >= 1.0:
        mant /= 10.0
        exp += 1
        s = f"{mant:.{d}f}"
    body = f"{s}D{exp:+03d}"
    if len(body) > w:
        return "*" * w
    return body.rjust(w)


def fA(s: str, w: int) -> str:
    """Fortran Aw for character data (right-justified, truncated to w)."""
    s = str(s)
    if len(s) >= w:
        return s[:w]
    return s.rjust(w)
