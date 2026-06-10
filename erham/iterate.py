"""ITER -- non-linear least-squares refinement of the spectroscopic parameters.

For every iteration the routine walks the sorted list of energy levels, builds
each rotational sub-block once with HAMILT, accumulates calculated transition
frequencies and (scaled) derivatives, forms the normal equations through the
SVD solver, prints the parameter table and applies the corrections.  The final
variance-covariance matrix is stored on the workspace for FTEST / PREDIC.
"""

from __future__ import annotations

import math

import numpy as np

from .deriv import deriv
from .fortran_format import fD, fF, fI
from .hamiltonian import compute_jsym, hamilt
from .leastsq import leasqu1
from .matrices import indmat
from .workspace import Workspace

MS1, MS2, MS4 = 8, 128, 16384
MS3 = MS1 * MS1 * MS2

TT = ["", "RHO1", "RHO2", "BETA1", "BETA2", "ALPHA1", "ALPHA2", "A", "B", "C",
      "DELTA J", "DELTA JK", "DELTA K", "DDELTA J", "DDELTA K", "PHI J",
      "PHI JK", "PHI KJ", "PHI K", "PPHI J", "PPHI JK", "PPHI K"]


def _tt12(name: str) -> str:
    """A character*8 name printed with the Fortran A12 descriptor."""
    return name[:8].ljust(8).rjust(12)


def iter_fit(model, ws: Workspace, out) -> None:
    pi = math.pi
    lp = model.LP
    ntra = model.NTRA
    ntr = 2 * ntra + 1

    DER = np.zeros((ntra + 1, 81))
    CALC = np.zeros(ntra + 1)
    AUX = np.zeros(ntra + 1)
    STER = np.zeros(lp + 1)
    CORR = np.zeros(2 * lp + 1)
    corr_mat = np.zeros((lp, lp))

    for it in range(1, model.NIT + 1):
        print(f"CYCLE {it}")
        out.write(f"\n{' ' * 20}{'*' * 14}\n{' ' * 20}*  CYCLE{fI(it, 3)}  *\n"
                  f"{' ' * 20}{'*' * 14}\n\n")
        DER[:] = 0.0
        CALC[:] = 0.0

        _accumulate(model, ws, CALC, DER, ntr, lp)

        # form the weighted residuals and the Jacobian (handling blends)
        nt = _residuals(model, out, CALC, AUX, DER, lp)

        jac = DER[1:nt + 1, 1:lp + 1].copy()
        res = CALC[1:nt + 1].copy()
        result = leasqu1(jac, res, out)

        changes = np.zeros(lp + 1)
        prec = np.zeros(lp + 1)
        for k in range(1, lp + 1):
            sc = model.SCP[k]
            changes[k] = result.changes[k - 1] * sc
            prec[k] = result.precision[k - 1] * sc
            STER[k] = result.ster[k - 1] * sc
        CORR[1:lp + 1] = changes[1:lp + 1]
        CORR[lp + 1:2 * lp + 1] = prec[1:lp + 1]
        corr_mat = result.corr

        out.write(f"{fI(nt, 5)} NORMAL EQUATIONS\n\n")
        print("STANDARD DEVIATION", result.std_dev)
        out.write(" STATE       OLD PARAMETER                STANDARD ERROR"
                  "      CHANGE         PREC       SCALE FAC\n\n")

        _print_and_update(model, out, CORR, STER, prec, lp)
        _print_correlation(out, corr_mat, lp)

    # build the variance-covariance matrix for FTEST / PREDIC
    for k in range(1, lp + 1):
        for l in range(k, lp + 1):
            v = result.cov[k - 1, l - 1] * model.SCP[k] * model.SCP[l]
            ws.VCM[k, l] = v
            ws.VCM[l, k] = v
        ws.VCM[k, k] = STER[k] * STER[k]


def _accumulate(model, ws, CALC, DER, ntr, lp):
    ilev = model.ILEV
    ivr = model.IVR
    scp = model.SCP
    m1 = 1
    iq = ilev[1, 1]
    ix1 = -1
    m2 = 0
    for i in range(1, ntr + 1):
        inn = ilev[1, i]
        if inn == iq:
            m2 = i
            continue
        ix = iq // MS2
        j = iq % MS2
        iv = ix % MS1
        is2 = (ix // MS1) % MS1
        is1 = iq // MS3
        jsym = compute_jsym(model.ISCD, model.NC, model.N1, is1, is2)
        if ix != ix1:
            indmat(model, iv, is1, is2, math.pi / model.N1, math.pi / model.N2, ws)
            ix1 = ix
        j1, j21, rjj1, lbl = hamilt(model, iv, j, jsym, ws)

        for mm in range(m1, m2 + 1):
            lcode = ilev[2, mm]
            nq = lcode // MS4
            l = lcode % MS4 - MS4 // 2
            il = abs(l)
            CALC[il] += ws.E[nq] * (l // il)

        kp1 = -1
        for ip in range(1, 22 + model.NTUP[iv]):
            if ivr[ip, iv] == 0:
                continue
            kp = ivr[ip, iv]
            s = scp[kp]
            if ip == 6 and kp == kp1:
                s = math.copysign(s, model.A[5, 1] * model.A[6, 1])
            deriv(model, iv, j, j1, j21, rjj1, ip, ws)
            for mm in range(m1, m2 + 1):
                lcode = ilev[2, mm]
                nq = lcode // MS4
                l = lcode % MS4 - MS4 // 2
                il = abs(l)
                DER[il, kp] += ws.E[nq] * (l // il) * s
            kp1 = kp

        m1 = i
        iq = inn
        m2 = i


def _residuals(model, out, CALC, AUX, DER, lp):
    ntra = model.NTRA
    frq = model.FRQ
    wt = model.WT
    bl = model.BL
    itra = model.ITRA
    nt = 0
    ibl = 1
    for i in range(1, ntra + 1):
        CALC[i] = frq[i] - CALC[i]
        AUX[i] = CALC[i] * wt[i]
        if wt[i] * bl[i] >= 0.0:
            _write_trans(out, i, itra, frq[i], bl[i], wt[i], CALC[i], AUX[i], None)
            if wt[i] > 0.0 and bl[i] == 0.0:
                nt += 1
                CALC[nt] = AUX[i]
                for k in range(1, lp + 1):
                    DER[nt, k] = DER[i, k] * wt[i]
            else:
                if wt[i] * bl[i] > 0.0:
                    continue
        else:
            nt += 1
            s = 0.0
            for ib in range(ibl, i + 1):
                s += AUX[ib] * abs(bl[ib])
            CALC[nt] = s
            _write_trans(out, i, itra, frq[i], bl[i], wt[i], CALC[i], AUX[i], s)
            for k in range(1, lp + 1):
                s = 0.0
                for ib in range(ibl, i + 1):
                    s += DER[ib, k] * wt[ib] * abs(bl[ib])
                DER[nt, k] = s
        ibl = i + 1
    return nt


def _write_trans(out, i, itra, frq, ble, wt, calc, aux, blend):
    # FORMAT(I3,I2,I1,2(I3,2I4),F13.4,F6.2,F8.2,F14.4,2F10.4)
    line = (fI(i, 3) + fI(itra[1, i], 2) + fI(itra[2, i], 1)
            + fI(itra[3, i], 3) + fI(itra[4, i], 4) + fI(itra[5, i], 4)
            + fI(itra[6, i], 3) + fI(itra[7, i], 4) + fI(itra[8, i], 4)
            + fF(frq, 13, 4) + fF(ble, 6, 2) + fF(wt, 8, 2)
            + fF(calc, 14, 4) + fF(aux, 10, 4))
    if blend is not None:
        line += fF(blend, 10, 4)
    out.write(line + "\n")


def _print_and_update(model, out, CORR, STER, prec, lp):
    pi = math.pi
    A = model.A
    ivr = model.IVR
    scp = model.SCP
    s1 = math.copysign(1.0, A[5, 1] * A[6, 1])
    l1 = -1
    for i in range(1, 7):
        s = 1.0 if i <= 2 else 180.0 / pi
        l = ivr[i, 1]
        if l == 0:
            out.write(f"{' ' * 8}{_tt12(TT[i])}{fD(A[i, 1] * s, 20, 12)}\n")
        else:
            if l1 == l and i == 6:
                CORR[l] *= s1
            out.write(f"{fI(l, 8)}{_tt12(TT[i])}{fD(A[i, 1] * s, 20, 12)}"
                      f"{fD(STER[l] * s, 15, 5)}{fD(CORR[l] * s, 15, 5)}"
                      f"{fD(CORR[l + lp] * s, 15, 5)}{fD(scp[l], 12, 3)}\n")
            A[i, 1] += CORR[l]
        l1 = l

    if model.MQ == 1:
        A[2, 1] = A[1, 1]
        A[4, 1] = A[3, 1]
        A[6, 1] = A[5, 1] * s1

    for iv in range(1, model.NIV + 1):
        out.write("\n")
        for k in range(1, 7):
            A[k, iv] = A[k, 1]
        for i in range(7, 22 + model.NTUP[iv]):
            l = ivr[i, iv]
            if i > 21:
                code = model.INPAR[i - 21, iv]
                iqc = code // 256
                kap = iqc % 16
                meg_raw = (iqc // 16) % 4
                img = 2 - (meg_raw % 2)
                imgs = 2 * (meg_raw % 2) - 1
                meg = meg_raw + img - 3
                iq2 = (iqc // 64) % 16 - 8
                iq1 = (iqc // 1024) % 16 - 8
                jp = (code // 16) % 16
                kp = code % 16
                tag = (fI(iq1, 2) + fI(iq2, 2) + fI(meg, 2) + fI(imgs * kap, 2)
                       + fI(jp, 2) + fI(kp, 2))
                if l > 0:
                    out.write(f"{fI(iv, 4)}{fI(l, 4)}{tag}"
                              f"{fD(A[i, iv], 20, 12)}{fD(STER[l], 15, 5)}"
                              f"{fD(CORR[l], 15, 5)}{fD(CORR[l + lp], 15, 5)}"
                              f"{fD(scp[l], 12, 3)}\n")
                    A[i, iv] += CORR[l]
                else:
                    out.write(f"{fI(iv, 4)}    {tag}{fD(A[i, iv], 20, 12)}\n")
            else:
                if l > 0:
                    out.write(f"{fI(iv, 4)}{fI(l, 4)}{_tt12(TT[i])}"
                              f"{fD(A[i, iv], 20, 12)}{fD(STER[l], 15, 5)}"
                              f"{fD(CORR[l], 15, 5)}{fD(CORR[l + lp], 15, 5)}"
                              f"{fD(scp[l], 12, 3)}\n")
                    A[i, iv] += CORR[l]
                else:
                    out.write(f"{fI(iv, 4)}    {_tt12(TT[i])}{fD(A[i, iv], 20, 12)}\n")


def _print_correlation(out, corr_mat, lp):
    out.write("\n CORRELATION MATRIX\n")
    for k in range(1, lp + 1, 12):
        j1 = min(lp, k + 11)
        out.write("\n PAR" + "".join(fI(j, 7) for j in range(k, j1 + 1)) + "\n")
        out.write("\n")
        for i in range(k, lp + 1):
            j2 = min(j1, i)
            cells = []
            for j in range(k, j2 + 1):
                cells.append(fI(int(corr_mat[i - 1, j - 1] * 1e5 + 0.5), 7))
            out.write(f"{fI(i, 4)}  " + "".join(cells) + "\n")
