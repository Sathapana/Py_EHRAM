"""DERIV -- derivatives of the rotational energy levels with respect to the
spectroscopic parameters.

Each parameter type builds the corresponding ``dH/dp`` matrix in ``ws.HC`` and
then projects it onto the eigenbasis (``ws.UC`` from the preceding HAMILT call)
to obtain ``dE_i/dp`` in ``ws.E``.  The matrix "sandwich" transforms are
expressed as numpy matrix products (with row/column shifts realising the
angular-momentum ladder operators that appear in the beta/alpha derivatives).
"""

from __future__ import annotations

import math

import numpy as np

from .hamiltonian import build_hermitian, herm_add, mcode, vib_sandwich
from .matrices import asymro, trig
from .workspace import NL, NU, Workspace


def deriv(model, iv: int, j: int, j1: int, j21: int, rjj1: float, ip: int,
          ws: Workspace) -> None:
    A = model.A[:, iv]
    INPAR = model.INPAR[:, iv]
    nc = model.NC
    iscd = model.ISCD
    mq = model.MQ
    ntup = model.NTUP[iv]
    nte = model.NTE[iv]
    pin1 = math.pi / model.N1
    pin2 = math.pi / model.N2

    msc = mcode(iscd, nc)
    mscm = msc % 2
    n1 = NL - j
    n2 = NL + j
    a5 = A[5]
    a6 = A[6]

    ws.clear_H(j21)

    if 7 <= ip <= 21:
        # derivative w.r.t. a rotational / distortion constant
        e15 = np.zeros(16)
        e15[ip - 6] = 1.0
        asymro(e15, j, j21, rjj1, ws)

    elif ip in (1, 2):
        _deriv_rho(model, A, INPAR, ws, ip, j, j21, nte, mq, nc, a5, a6, mscm,
                   pin1, pin2)
        _deriv_rho_tun(A, INPAR, ws, ip, j, j + 1, rjj1,
                       model.NTUP[iv], nte, mq, nc, iscd, pin1, pin2)

    elif ip >= 22:
        if ip <= 21 + nte:
            _deriv_tun_energy(INPAR, ws, ip, j, j21, mq, nc, a5, a6, mscm)
        else:
            _deriv_tun_higher(INPAR, ws, ip, j, j1, rjj1, mq, nc, iscd)

    elif ip in (3, 4):
        if ip == 3:
            _deriv_beta1(ws, j, j21, nc, a5, a6, mscm)
            if mq == 1:
                _deriv_beta2(ws, j, j21, nc, a5, a6, mscm)
        else:
            _deriv_beta2(ws, j, j21, nc, a5, a6, mscm)

    elif ip in (5, 6):
        if ip == 5:
            _deriv_alpha1(ws, j, j21, nc, a5, a6, mscm, msc)
            if mq == 1:
                sfac = math.copysign(1.0, a6 * a5)
                _deriv_alpha2(ws, j, j21, nc, a5, a6, mscm, msc, sfac)
        else:
            _deriv_alpha2(ws, j, j21, nc, a5, a6, mscm, msc, 1.0)

    # --- project dH onto the eigenbasis:  dE_i = <i| dH |i> ---
    Hm = build_hermitian(ws.HC, j21)
    U = ws.UC[1:j21 + 1, 1:j21 + 1]
    ws.E[1:j21 + 1] = np.real(np.einsum('ki,kl,li->i', U.conj(), Hm, U))


# ---------------------------------------------------------------------------
def _deriv_rho(model, A, INPAR, ws, ip, j, j21, nte, mq, nc, a5, a6, mscm,
               pin1, pin2):
    EW = np.zeros((NU + 1, NU + 1))
    for i in range(1, nte + 1):
        iq = INPAR[i] // 256
        q2 = (iq // 64) % 16 - 8
        q1 = iq // 1024 - 8
        qa = q1 + (q1 - q2) * (1 - mq)
        qb = q1 + q2 - qa
        a21i = A[21 + i]
        for k2 in range(-j, j + 1):
            kk2 = k2 + NL
            for k1 in range(-j, j + 1):
                kk1 = k1 + NL
                if ip == 1:
                    EW[kk1, kk2] += a21i * trig(2 * k1, 2 * k2, 1, q1, q2, 1.0,
                                                ws.PHI1, ws.PHI2, 1, mq) * pin1
                if ip == mq:
                    EW[kk1, kk2] += a21i * trig(2 * k2, 2 * k1, 1, qa, qb, 1.0,
                                                ws.PHI2, ws.PHI1, 1, mq) * pin2
    vib_sandwich(ws, j, j21, EW, a5, a6, mscm, nc)


def _deriv_rho_tun(A, INPAR, ws, ip, j, j1, rjj1, ntup, nte, mq, nc, iscd,
                   pin1, pin2):
    """Rho-derivative of the higher-order tunneling / distortion terms
    (DERIV CASE(1:2), DO 100)."""
    EJ = ws.EJ
    HC = ws.HC
    m1 = nte + 1
    iq = INPAR[nte + 1] // 256
    m2 = nte
    for i in range(nte + 1, ntup + 2):
        inp1 = INPAR[i] // 256
        if inp1 == iq:
            m2 = i
            continue
        kap = iq % 16
        meg_raw = (iq // 16) % 4
        img = 2 - (meg_raw % 2)
        meg = meg_raw + img - 3
        q2 = (iq // 64) % 16 - 8
        q1 = (iq // 1024) % 16 - 8
        qa = q1 + (q1 - q2) * (1 - mq)
        qb = q1 + q2 - qa
        s2 = meg * (1 - 2 * (kap % 2))
        if abs(iscd) == 4:
            s2 = meg
        if iscd == 2 and img == 2 and nc == -1:
            s2 = -s2
        if iscd == 3 and img == 2 and nc == +1:
            s2 = -s2
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
            val = 0.0
            if ip == 1:
                val += s * trig(kk, nc * kk, meg, q1, q2, s2,
                                ws.PHI1, ws.PHI2, 1, mq) * pin1 / 2.0
            if ip == mq:
                val += s * trig(nc * kk, kk, meg, qa, qb, s2,
                                ws.PHI2, ws.PHI1, 1, mq) * pin2 / 2.0
            if img == 1:
                HC[kq + j1, k + j1] += val
            else:
                HC[kq + j1, k + j1] += 1j * val
        m1 = i
        iq = inp1
        m2 = i


def _deriv_tun_energy(INPAR, ws, ip, j, j21, mq, nc, a5, a6, mscm):
    iq = INPAR[ip - 21] // 256
    q2 = (iq // 64) % 16 - 8
    q1 = iq // 1024 - 8
    EW = np.zeros((NU + 1, NU + 1))
    for k2 in range(-j, j + 1):
        for k1 in range(-j, j + 1):
            EW[k1 + NL, k2 + NL] = trig(2 * k1, 2 * k2, 1, q1, q2, 1.0,
                                        ws.PHI1, ws.PHI2, -1, mq)
    vib_sandwich(ws, j, j21, EW, a5, a6, mscm, nc)


def _deriv_tun_higher(INPAR, ws, ip, j, j1, rjj1, mq, nc, iscd):
    code = INPAR[ip - 21]
    iq = code // 256
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
    jp = (code // 16) % 16
    kp = code % 16
    EJ = ws.EJ
    HC = ws.HC
    for k in range(-j, j - kap + 1):
        kq = k + kap
        kk = k + kq
        skk = float(k)
        skq = float(kq)
        s = rjj1 ** (jp // 2) * (skk ** kp + skq ** kp)
        for k1 in range(0, kap):
            s = s * EJ[k + k1 + NL]
        val = s * trig(kk, nc * kk, meg, q1, q2, s2, ws.PHI1, ws.PHI2, -1, mq) / 2.0
        if img == 1:
            HC[kq + j1, k + j1] += val
        else:
            HC[kq + j1, k + j1] += 1j * val


# ---------------------------------------------------------------------------
# Helpers shared by the beta/alpha derivatives
# ---------------------------------------------------------------------------
def _block_setup(ws, j, nc):
    n1 = NL - j
    n2 = NL + j
    sl = slice(n1, n2 + 1)
    kvals = np.arange(n1, n2 + 1)
    cidx = kvals if nc == 1 else (2 * NL - kvals)
    P = ws.EV[sl][:, cidx]
    D1b = ws.D1[sl, sl]
    D2b = ws.D2[sl, sl]
    EJa = ws.EJ[sl]                 # EJ[a]
    EJja = ws.EJ[sl][::-1]          # EJ[JN-a]
    return n1, n2, sl, kvals, P, D1b, D2b, EJa, EJja


def _phases(kvals, a5, a6, mscm):
    e1 = np.exp(1j * a5 * (kvals - NL))
    e2 = np.exp(-1j * a6 * (kvals - NL))
    f2 = 1 - 2 * (np.abs(NL - kvals) % 2) * mscm
    return e1, e2, f2


def _deriv_beta1(ws, j, j21, nc, a5, a6, mscm):
    n1, n2, sl, kvals, P, D1b, D2b, EJa, EJja = _block_setup(ws, j, nc)
    e1, e2, f2 = _phases(kvals, a5, a6, mscm)
    M = ws.BC[sl, sl] * P
    ECmat = M @ D2b
    Mm1 = ws.BC[n1 - 1:n2, sl] * P              # BC[a-1, b]
    Mp1 = ws.BC[n1 + 1:n2 + 2, sl] * P          # BC[a+1, b]
    S1Cmat = EJja[:, None] * (Mm1 @ D2b) - EJa[:, None] * (Mp1 @ D2b)
    D1p = ws.D1[n1 + 1:n2 + 2, sl]              # D1[a+1, KQ]
    D1m = ws.D1[n1 - 1:n2, sl]                  # D1[a-1, KQ]
    W = EJa[:, None] * D1p - EJja[:, None] * D1m
    raw = -(W.T @ ECmat) + (D1b.T @ S1Cmat)
    SC = e1[:, None] * raw * (f2 * e2)[None, :] * 0.5
    herm_add(ws, j21, SC)


def _deriv_beta2(ws, j, j21, nc, a5, a6, mscm):
    n1, n2, sl, kvals, P, D1b, D2b, EJa, EJja = _block_setup(ws, j, nc)
    e1, e2, f2 = _phases(kvals, a5, a6, mscm)
    M = ws.BC[sl, sl] * P
    SCmat2 = M.T @ D1b
    Mbm1 = ws.BC[sl, n1 - 1:n2] * P             # BC[a, b-1]
    Mbp1 = ws.BC[sl, n1 + 1:n2 + 2] * P         # BC[a, b+1]
    S1Cmat2 = EJja[:, None] * (Mbm1.T @ D1b) - EJa[:, None] * (Mbp1.T @ D1b)
    D2p = ws.D2[n1 + 1:n2 + 2, sl]              # D2[b+1, KQ]
    D2m = ws.D2[n1 - 1:n2, sl]                  # D2[b-1, KQ]
    W2 = EJa[:, None] * D2p - EJja[:, None] * D2m
    raw = -(W2.T @ SCmat2) + (D2b.T @ S1Cmat2)
    SC = (e2 * f2)[:, None] * raw * e1[None, :] * 0.5
    herm_add(ws, j21, SC)


def _alpha_common(ws, j, nc, a5, a6, msc):
    n1, n2, sl, kvals, P, D1b, D2b, EJa, EJja = _block_setup(ws, j, nc)
    nb = kvals.size
    if msc >= 3:
        if msc == 3:
            Evec = (kvals - NL) * (1 - 2 * (np.abs(kvals - NL) % 2))
        else:
            Evec = (kvals - NL).astype(float)
        expk = np.exp((kvals - NL) * 1j * (a6 - a5))
        Q = Evec * expk
        EXmat = 0.5j * ((D1b * Q[None, :]) @ D2b.T)
    else:
        EXmat = np.zeros((nb, nb), dtype=complex)
    M = ws.BC[sl, sl] * P
    B1C = M @ D2b                  # the BC-based EC
    ECa = (EXmat * P) @ D2b        # the EX-based EC
    SAmat = D1b.T @ ECa
    SBmat = D1b.T @ B1C
    return n1, n2, kvals, SAmat, SBmat


def _deriv_alpha1(ws, j, j21, nc, a5, a6, mscm, msc):
    n1, n2, kvals, SAmat, SBmat = _alpha_common(ws, j, nc, a5, a6, msc)
    e1, e2, f2 = _phases(kvals, a5, a6, mscm)
    kqfac = 1j * (kvals - NL)
    raw = SBmat * kqfac[:, None] - SAmat
    SC = e1[:, None] * raw * (f2 * e2)[None, :]
    herm_add(ws, j21, SC)


def _deriv_alpha2(ws, j, j21, nc, a5, a6, mscm, msc, sfac):
    n1, n2, kvals, SAmat, SBmat = _alpha_common(ws, j, nc, a5, a6, msc)
    e1, e2, f2 = _phases(kvals, a5, a6, mscm)
    kfac = -1j * (kvals - NL)
    raw = SBmat * kfac[None, :] + SAmat
    SC = e1[:, None] * raw * (f2 * sfac * e2)[None, :]
    herm_add(ws, j21, SC)
