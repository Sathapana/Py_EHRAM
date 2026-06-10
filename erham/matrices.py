"""Matrix-element building blocks: ASYMRO, TRIG, DMAT, BMAT, INDMAT.

These are direct translations of the corresponding Fortran subroutines.
All matrices are the 1-based padded arrays held on the :class:`Workspace`.
The Hamiltonian ``HC`` is stored as a *lower triangle* (row >= col), exactly
as the Fortran code does.
"""

from __future__ import annotations

import math

from .workspace import NL, NU, SQRT2, Workspace


# ---------------------------------------------------------------------------
# ASYMRO -- asymmetric rigid rotor (A-reduction) in the symmetric-rotor basis
# ---------------------------------------------------------------------------
def asymro(b, j: int, j21: int, rjj1: float, ws: Workspace) -> None:
    """Set the rigid-rotor part of the Hamiltonian.

    ``b`` is a 1-based length-15 vector: b[1..3] = A,B,C rotational constants,
    b[4..8] = quartic, b[9..15] = sextic distortion constants.  Writes the
    real diagonal ``HC[k,k]`` and second sub-diagonal ``HC[k+2,k]``.
    """
    em3 = 1e-3
    j1 = j + 1
    qtm = rjj1 * em3
    b1 = (b[2] + b[3]) / 2.0
    a1 = rjj1 * (b1 + qtm * (-b[4] + qtm * b[9]))
    a2 = b[1] - b1 + qtm * (-b[5] + qtm * b[10])
    a3 = em3 * (-b[6] + qtm * b[11])
    a4 = em3 * em3 * b[12]
    a5 = (b1 - b[3]) / 2.0 + qtm * (-b[7] + qtm * b[13])
    a6 = em3 * (-b[8] + qtm * b[14])
    a7 = em3 * em3 * b[15]
    for k in range(1, j21 + 1):
        rk2 = (k - j1) * (k - j1)
        ws.HC[k, k] = a1 + rk2 * (a2 + rk2 * (a3 + rk2 * a4))
    for k in range(1, j21 - 1):
        rk2 = (k - j) * (k - j)
        ws.HC[k + 2, k] = (ws.EJ[k - j1 + NL] * ws.EJ[k - j + NL]
                           * (a5 + (1.0 + rk2) * a6 + (1.0 + rk2 * (6.0 + rk2)) * a7))


# ---------------------------------------------------------------------------
# TRIG -- trigonometric factor of a tunneling matrix element (or its rho deriv)
# ---------------------------------------------------------------------------
def trig(kp: int, km: int, meg: int, q1: float, q2: float, s2: float,
         phi1, phi2, nod: int, mq: int) -> float:
    """Reproduce the FUNCTION TRIG, including its computed-GOTO dispatch.

    ``nod = -1`` evaluates the matrix element; ``nod = +1`` evaluates the
    derivative with respect to a rho parameter.
    """
    if q1 == 0.0 and q2 == 0.0:
        return 1.0
    phi1q = phi1[kp + NU] * q1 + phi2[km + NU] * q2
    phi2q = phi1[kp + NU] * q2 + phi2[km + NU] * q1
    if nod == 1:
        der1 = -kp * q1
        der2 = -kp * q2
    else:
        der1 = 1.0
        der2 = 1.0
    fac = 2.0
    if meg == -1:
        fac = nod * fac
    if mq == 1:
        ix = min(1, int((q1 + q2) * (q1 - q2))) + 2
    else:
        ix = 2
    selector = ix + meg * nod
    # Computed GOTO (20,10,40,30): an out-of-range selector falls through to
    # the statement following the GOTO, which is label 10.
    if selector == 1:        # -> 20
        s = fac * (math.cos(phi1q) * der1)
    elif selector == 3:      # -> 40
        s = -fac * (math.sin(phi1q) * der1)
    elif selector == 4:      # -> 30 then 40
        s = -fac * (s2 * math.sin(phi2q) * der2 + math.sin(phi1q) * der1)
    else:                    # selector == 2 or out of range -> 10 then 20
        s = fac * (s2 * math.cos(phi2q) * der2 + math.cos(phi1q) * der1)
    return s


# ---------------------------------------------------------------------------
# DMAT -- build the reduced rotation matrix d^(J+1) from d^J
# ---------------------------------------------------------------------------
def dmat(j: int, dc, off: int, D, DD, m: int) -> None:
    """Recursion for the Wigner reduced rotation matrix.

    ``dc`` is the workspace DC vector and ``off`` selects the rotor
    (0 -> DC[1..4], 4 -> DC[5..8]).  ``D`` is updated in place; for ``m == 1``
    the result is copied from ``DD`` (equivalent-rotor shortcut).
    """
    j1 = j + 1

    if m == 1:
        for k in range(NL - j1, NL + j1 + 1):
            for kq in range(NL - j1, NL + j1 + 1):
                D[kq, k] = DD[kq, k]
        return

    def F1(L, K):
        return math.sqrt((L + K) * (L + K + 1))

    def F2(L, K):
        return math.sqrt(2 * (L - K + 1) * (L + K + 1))

    c1, c2, c3, c4 = dc[off + 1], dc[off + 2], dc[off + 3], dc[off + 4]
    Bsc = _dmat_scratch()
    bj = (j + j1) * (j1 + j1)
    for k in range(0, j1 + 1):
        fa = F1(j, -k)
        fb = F2(j, k)
        fc = F1(j, k)
        L = NL + k
        for kq in range(-k, k + 1):
            lq = NL + kq
            br = fa * D[lq + 1, L + 1] * c2 - fb * D[lq + 1, L] * c4 + fc * D[lq + 1, L - 1] * c3
            bq = fa * D[lq, L + 1] * c4 + fb * D[lq, L] * c1 - fc * D[lq, L - 1] * c4
            bp = fa * D[lq - 1, L + 1] * c3 + fb * D[lq - 1, L] * c4 + fc * D[lq - 1, L - 1] * c2
            Bsc[lq, L] = F1(j, -kq) * br + F2(j, kq) * bq + F1(j, kq) * bp
    j2 = NL + NL
    for L in range(NL, NL + j1 + 1):
        for lq in range(j2 - L, L + 1):
            D[lq, L] = Bsc[lq, L] / bj
            D[j2 - L, j2 - lq] = D[lq, L]
            D[L, lq] = Bsc[lq, L] * (1 - 2 * ((L - lq) % 2)) / bj
            D[j2 - lq, j2 - L] = D[L, lq]


_SCRATCH = None


def _dmat_scratch():
    """Lazily-allocated scratch matrix for DMAT (the unnamed COMMON B)."""
    global _SCRATCH
    if _SCRATCH is None:
        import numpy as np
        _SCRATCH = np.zeros((NU + 1, NU + 1))
    else:
        _SCRATCH[:] = 0.0
    return _SCRATCH


# ---------------------------------------------------------------------------
# BMAT -- combine the two rotor d-matrices into the B matrix
# ---------------------------------------------------------------------------
def bmat(j: int, delalp: float, m: int, ws: Workspace) -> None:
    """Build ``ws.BC`` from ``ws.D1`` and ``ws.D2`` for symmetry case ``m``."""
    j1 = NL - j
    j2 = NL + j
    f = {}
    if m in (1, 3):
        for k in range(j1, j2 + 1):
            f[k] = 1 - 2 * (abs(k - NL) % 2)
    else:  # m in (2, 4)
        for k in range(j1, j2 + 1):
            f[k] = 1.0

    D1, D2, BC = ws.D1, ws.D2, ws.BC
    if m in (1, 2):  # B is real
        for k1 in range(j1, j2 + 1):
            for k2 in range(j1, j2 + 1):
                s = 0.0
                for k in range(j1, j2 + 1):
                    s += f[k] * D1[k1, k] * D2[k2, k]
                BC[k1, k2] = (s / 2.0) + 0j
    else:            # B is complex (m in {3,4})
        dalp = 1j * delalp
        for k1 in range(j1, j2 + 1):
            for k2 in range(j1, j2 + 1):
                sc = 0j
                for k in range(j1, j2 + 1):
                    sc += f[k] * D1[k1, k] * D2[k2, k] * complex(math.cos((k - NL) * delalp),
                                                                 math.sin((k - NL) * delalp))
                BC[k1, k2] = sc / 2.0


# ---------------------------------------------------------------------------
# INDMAT -- initialise D matrices, phase tables and the tunneling-energy matrix
# ---------------------------------------------------------------------------
def indmat(model, iv: int, is1: int, is2: int, pin1: float, pin2: float,
           ws: Workspace) -> None:
    """Set up per-symmetry-block scratch (the Fortran SUBROUTINE INDMAT)."""
    A = model.A[:, iv]
    INPAR = model.INPAR[:, iv]
    nte = model.NTE[iv]
    mq = model.MQ

    ws.EV[: NU + 1, : NU + 1] = 0.0
    ws.D1[: NU + 1, : NU + 1] = 0.0
    ws.D2[: NU + 1, : NU + 1] = 0.0
    ws.D1[NL, NL] = 1.0
    ws.D2[NL, NL] = 1.0

    dc = ws.DC
    dc[1] = math.cos(A[3])
    dc[2] = (1.0 + dc[1]) / 2.0
    dc[3] = (1.0 - dc[1]) / 2.0
    dc[4] = math.sin(A[3]) / SQRT2
    dc[5] = math.cos(A[4])
    dc[6] = (1.0 + dc[5]) / 2.0
    dc[7] = (1.0 - dc[5]) / 2.0
    dc[8] = math.sin(A[4]) / SQRT2

    ws.JDM = 0
    ss1 = is1 * 2
    ss2 = is2 * 2
    for k in range(-NU + 1, NU):
        ws.PHI1[k + NU] = (ss1 - A[1] * k) * pin1
        ws.PHI2[k + NU] = (ss2 - A[2] * k) * pin2

    for i in range(1, nte + 1):
        iq = INPAR[i] // 256
        q2 = (iq // 64) % 16 - 8
        q1 = iq // 1024 - 8
        aval = A[21 + i]
        for k2 in range(-NL + 1, NL):
            for k1 in range(-NL + 1, NL):
                ws.EV[k1 + NL, k2 + NL] += aval * trig(
                    2 * k1, 2 * k2, +1, q1, q2, 1.0, ws.PHI1, ws.PHI2, -1, mq)
