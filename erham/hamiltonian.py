"""Effective rotational Hamiltonian assembly (HAMILT) and the even/odd-Ka
symmetrisation helper (EVENODD).

The custom complex-Hermitian eigensolver SEIGCX (Householder + bisection +
inverse iteration) is replaced by :func:`numpy.linalg.eigh`, which returns the
same eigenvalues (ascending) and an orthonormal eigenbasis.  For non-degenerate
levels the line strengths used downstream are phase-invariant, so predictions
are reproduced; the symmetry block bookkeeping that produces the ``-`` level
labels is translated literally and does not depend on eigenvector phase.

The expensive vibrational (tunneling-energy) contribution -- a triple loop in
the Fortran -- is recast as a few numpy matrix products.
"""

from __future__ import annotations

import numpy as np

from .matrices import asymro, bmat, dmat, trig
from .workspace import NL, NU, SQRT2, Workspace

# MCODE(10,2): symmetry-case selector, Fortran column-major.
_MCODE_C1 = [2, 2, 3, 1, 1, 3, 3, 3, 1, 2]
_MCODE_C2 = [0, 0, 4, 1, 1, 4, 4, 1, 3, 0]


def mcode(iscd: int, nc: int) -> int:
    row = iscd + 6                      # 1..10
    col = (nc + 3) // 2                 # 1 for NC=-1, 2 for NC=+1
    return _MCODE_C1[row - 1] if col == 1 else _MCODE_C2[row - 1]


# MSYM(10,3): symmetrisation selector by (ISCD+6, ISZ), Fortran column-major.
_MSYM = [
    [3, 2, 4, 4, 2, 3, 0, 3, 1, 0],
    [3, 2, 1, 3, 0, 0, 0, 3, 0, 0],
    [0, 0, 2, 1, 0, 0, 0, 0, 1, 0],
]


def compute_jsym(iscd: int, nc: int, n1: int, is1: int, is2: int) -> int:
    """Symmetrisation code JSYM for a symmetry block (shared by ITER/PREDIC)."""
    if is1 == is2:
        isz = 1 if is1 == 0 else 2
    else:
        isz = 3 if (is1 + is2) % n1 == 0 else 4
    jsym = 0
    if isz < 4:
        jsym = _MSYM[isz - 1][iscd + 6 - 1]
    if not (nc == 1 or jsym == 0 or abs(iscd) < 2 or abs(iscd) > 3
            or iscd == -3 or jsym == 4):
        jsym = 4 - jsym
    return jsym


# ---------------------------------------------------------------------------
# Diagonalise an n x n Hermitian block stored in the lower triangle of Hsrc
# (1-based), writing eigenvalues to Eout[e_off+1..] and eigenvectors to the
# columns UCout[1..n, col_off+1..].  Replaces the SEIGCX calls.
# ---------------------------------------------------------------------------
def diag_block(Hsrc, n, Eout, e_off, UCout, col_off):
    if n <= 0:
        return
    L = np.array(Hsrc[1:n + 1, 1:n + 1])          # copy of the n x n block
    Hm = np.tril(L)                               # keep lower triangle + diag
    Hm += np.tril(L, -1).conj().T                 # mirror to upper triangle
    d = np.diag(Hm).real.copy()
    np.fill_diagonal(Hm, d)                       # force real diagonal
    w, v = np.linalg.eigh(Hm)
    Eout[e_off + 1: e_off + n + 1] = w
    UCout[1:n + 1, col_off + 1: col_off + n + 1] = v


def build_hermitian(Hsrc, n):
    """Return the full n x n Hermitian matrix from the lower triangle of Hsrc."""
    L = np.array(Hsrc[1:n + 1, 1:n + 1])
    Hm = np.tril(L)
    Hm += np.tril(L, -1).conj().T
    d = np.diag(Hm).real.copy()
    np.fill_diagonal(Hm, d)
    return Hm


def herm_add(ws, j21, SC):
    """Add a Hermitian contribution whose pre-symmetrised array is ``SC``.

    Reproduces the Fortran accumulation
        KQ<K : HC[K,KQ]  += conj(SC[KQ,K])
        KQ>K : HC[KQ,K]  += SC[KQ,K]
        KQ=K : HC[K,K]   += 2 Re SC[K,K]
    onto the lower triangle of ``ws.HC``.
    """
    lower = np.tril(SC, -1) + np.triu(SC, 1).conj().T
    block = ws.HC[1:j21 + 1, 1:j21 + 1]
    block += lower
    block[np.diag_indices(j21)] += 2.0 * np.real(np.diag(SC))


def vib_sandwich(ws, j, j21, inner, a5, a6, mscm, nc):
    """Vectorised tunneling-energy contribution (the HAMILT/DERIV DO-50 loop).

    ``inner`` is the torsional matrix (EV in HAMILT, the derivative matrix in
    DERIV) as a full 1-based padded array.
    """
    n1 = NL - j
    n2 = NL + j
    sl = slice(n1, n2 + 1)
    D1b = ws.D1[sl, sl]
    D2b = ws.D2[sl, sl]
    kvals = np.arange(n1, n2 + 1)
    cidx = kvals if nc == 1 else (2 * NL - kvals)
    INNp = inner[sl][:, cidx]
    Mm = ws.BC[sl, sl] * INNp
    Gmat = D1b.T @ (Mm @ D2b)
    e1 = np.exp(1j * a5 * (kvals - NL))
    e2 = np.exp(-1j * a6 * (kvals - NL))
    f2 = 1 - 2 * (np.abs(NL - kvals) % 2) * mscm
    SC = e1[:, None] * Gmat * (f2 * e2)[None, :]
    herm_add(ws, j21, SC)


# ---------------------------------------------------------------------------
# EVENODD -- split a Hermitian matrix into even/odd-Ka sub-blocks.  Translated
# literally from the Fortran (scalar, in-place) so the basis bookkeeping is
# preserved exactly; only the two SEIGCX calls become diag_block.
# ---------------------------------------------------------------------------
def evenodd(ws: Workspace, m: int, e_off: int, col_off: int):
    HC = ws.HC
    UC = ws.UC
    E = ws.E
    m1 = (m + 1) // 2
    m2 = (m % 2) * (m // 2)
    m3 = 1 - (m % 2)
    m5 = m - m1
    m4 = m // 2 - m2

    for k in range(1, m + 1):
        kk = (k - m3) // 2
        for kq in range(k, m + 1, 2):
            kk += 1
            HC[kk, k] = HC[kq, k]
    for k in range(2 - m3, m + 1, 2):
        kk = (k - m3) // 2 + 1
        for kq in range(kk, m1 + 1):
            HC[kq + m2, kk + m2] = HC[kq, k]
            HC[kq + m4, kk + m4] = HC[kq, k + 1]

    diag_block(HC, m1, E, e_off, UC, col_off)

    for i in range(1, m1 + 1):
        ci = col_off + i
        L = m + m3
        if m % 2 == 1:
            UC[L, ci] = UC[m1, ci]
        for k in range(m5, 0, -1):
            L -= 1
            UC[L, ci] = 0.0
            L -= 1
            UC[L, ci] = UC[k, ci]

    for kq in range(1, m5 + 1):
        for k in range(1, kq + 1):
            HC[kq, k] = HC[m1 + kq, m1 + k]

    diag_block(HC, m5, E, e_off + m1, UC, col_off + m1)

    for i in range(m1 + 1, m + 1):
        ci = col_off + i
        L = m + m3
        if m % 2 == 1:
            UC[L, ci] = 0.0
        for k in range(m5, 0, -1):
            L -= 1
            UC[L, ci] = UC[k, ci]
            L -= 1
            UC[L, ci] = 0.0


# ---------------------------------------------------------------------------
# HAMILT -- build and diagonalise the effective rotational Hamiltonian for one
# vibrational state and one J.  Returns (j1, j21, rjj1, lbl).
# ---------------------------------------------------------------------------
def hamilt(model, iv: int, j: int, jsym: int, ws: Workspace):
    A = model.A[:, iv]
    INPAR = model.INPAR[:, iv]
    nc = model.NC
    iscd = model.ISCD
    mq = model.MQ
    ntup = model.NTUP[iv]
    nte = model.NTE[iv]

    pi = np.pi
    LA = nc == 1
    LB = nc == -1 and (A[5] == 0.0 or A[5] == pi)
    LC = nc == -1 and (A[5] != 0.0 and A[5] != pi)

    j1 = j + 1
    j21 = j + j1
    rjj1 = float(j * j1)

    ws.clear_H(j21)
    for k in range(-j, j + 1):
        ws.EJ[k + NL] = np.sqrt((j - k) * (j + k + 1))

    # rigid-rotor (A-reduction) part
    asymro(A[6:22], j, j21, rjj1, ws)   # A[7..21] -> b[1..15]

    msc = mcode(iscd, nc)
    mscm = msc % 2
    n1 = NL - j
    n2 = NL + j

    # build the reduced rotation matrices up to this J
    if ws.JDM != j:
        for jj in range(ws.JDM, j):
            dmat(jj, ws.DC, 0, ws.D1, ws.D1, 0)
            dmat(jj, ws.DC, 4, ws.D2, ws.D1, mq)
        ws.JDM = j

    bmat(j, A[6] - A[5], msc, ws)
    a5 = A[5]
    a6 = A[6]

    # --- vibrational (tunneling-energy) contribution, vectorised ---
    vib_sandwich(ws, j, j21, ws.EV, a5, a6, mscm, nc)

    # --- higher-order tunneling / distortion contributions (scalar) ---
    m1 = nte + 1
    iq = INPAR[nte + 1] // 256
    m2 = nte
    for i in range(nte + 1, ntup + 2):
        inp1 = INPAR[i] // 256
        if inp1 == iq:
            m2 = i
            continue
        _tun_group(ws, A, INPAR, iq, m1, m2, j, j1, rjj1, nc, iscd, mq)
        m1 = i
        iq = inp1
        m2 = i

    # --- labels and symmetrisation ---
    lbl = [' '] * (j21 + 1)

    if jsym == 0:
        diag_block(ws.HC, j21, ws.E, 0, ws.UC, 0)
        return j1, j21, rjj1, lbl

    ll1 = (j + (j % 2)) // 2
    ll2 = j1 * (1 - (j % 2))
    ll3 = ll2 + ll1
    if jsym == 4:
        ll2 = (ll1 + 1) * (1 - (j % 2))
        ll3 = ll1
        lj = (j % 2) == 0
        if (LB and lj) or (LC and not lj):
            ll2 = ll2 + ll1
        if (LC and lj) or (LA and not lj):
            ll3 = 2 * ll3
        ll3 = ll2 + ll3

    if jsym in (1, 2, 4):
        _wang_symmetrise(ws, j, j1, j21, jsym)
    elif jsym == 3:
        evenodd(ws, j21, 0, 0)

    # mark antisymmetric labels
    for i in range(1, ll1 + 1):
        lbl[i + ll3] = '-'
        lbl[i + ll2] = '-'

    # sort eigenvalues ascending, permuting eigenvectors and labels
    E = ws.E
    UC = ws.UC
    for i in range(1, j21 + 1):
        kq = i
        s = E[kq]
        for k in range(i, j21 + 1):
            if E[k] < s:
                kq = k
                s = E[kq]
        E[kq] = E[i]
        E[i] = s
        lbl[kq], lbl[i] = lbl[i], lbl[kq]
        if kq != i:
            tmp = UC[1:j21 + 1, kq].copy()
            UC[1:j21 + 1, kq] = UC[1:j21 + 1, i]
            UC[1:j21 + 1, i] = tmp

    return j1, j21, rjj1, lbl


def _tun_group(ws, A, INPAR, iq, m1, m2, j, j1, rjj1, nc, iscd, mq):
    """Add the contribution of one group of tunneling/distortion parameters."""
    kap = iq % 16
    meg_raw = (iq // 16) % 4
    img = 2 - (meg_raw % 2)
    meg = meg_raw + img - 3
    q2 = (iq // 64) % 16 - 8
    q1 = (iq // 1024) % 16 - 8
    s2 = meg * (1 - 2 * (kap % 2))
    if abs(iscd) == 4:
        s2 = meg
    if iscd == 2 and img == 2 and nc == -1:
        s2 = -s2
    if iscd == 3 and img == 2 and nc == +1:
        s2 = -s2
    EJ = ws.EJ
    HC = ws.HC
    for k in range(-j, j - kap + 1):
        kq = k + kap
        kk = k + kq
        skk = float(k)
        skq = float(kq)
        s = 0.0
        for m in range(m1, m2 + 1):
            jp = (INPAR[m] // 16) % 16
            kp = INPAR[m] % 16
            s += A[21 + m] * rjj1 ** (jp // 2) * (skk ** kp + skq ** kp)
        for k1 in range(0, kap):
            s = s * EJ[k + k1 + NL]
        val = s * trig(kk, nc * kk, meg, q1, q2, s2, ws.PHI1, ws.PHI2, -1, mq) / 2.0
        if img == 1:
            HC[kq + j1, k + j1] += val
        else:
            HC[kq + j1, k + j1] += 1j * val


def _wang_symmetrise(ws, j, j1, j21, jsym):
    """Wang folding into (e,o) blocks for JSYM in {1,2,4}."""
    HC = ws.HC
    UC = ws.UC
    j2 = 2 * j1
    sk = [0.0] * (j + 2)
    for k in range(1, j + 1):
        sk[k] = 1.0
    if jsym == 2:
        fj = 1 - 2 * (j % 2)
        for k in range(1, j + 1):
            sk[k] = fj
            fj = -fj

    for kq in range(1, j + 1):
        for k in range(1, kq + 1):
            sc = HC[kq, k] + np.conj(HC[j2 - k, j2 - kq]) * sk[k] * sk[kq]
            qc = HC[j2 - kq, k] * sk[kq] + np.conj(HC[j2 - k, kq]) * sk[k]
            HC[kq, k] = (sc + qc) / 2.0
            HC[j2 - k, j2 - kq] = np.conj(sc - qc) / 2.0
        sc = HC[j1, kq] + np.conj(HC[j2 - kq, j1]) * sk[kq]
        HC[j1, kq] = sc / SQRT2

    if jsym != 4:
        diag_block(HC, j1, ws.E, 0, UC, 0)
    else:
        if j1 > 0:
            evenodd(ws, j1, 0, 0)

    for i in range(1, j1 + 1):
        for k in range(1, j + 1):
            UC[k, i] = UC[k, i] / SQRT2
            UC[j2 - k, i] = UC[k, i] * sk[k]

    for kq in range(1, j + 1):
        for k in range(1, kq + 1):
            HC[kq, k] = HC[j1 + kq, j1 + k]

    if jsym != 4:
        diag_block(HC, j, ws.E, j1, UC, j1)
    else:
        if j > 0:
            evenodd(ws, j, j1, j1)

    for i in range(j1 + 1, j21 + 1):
        for k in range(1, j + 1):
            UC[j1 + k, i] = -UC[k, i] / SQRT2 * sk[j1 - k]
        for k in range(1, j + 1):
            UC[k, i] = -UC[j2 - k, i] * sk[k]
        UC[j1, i] = 0.0
