"""FTEST / DERPAR -- derived molecular parameters and their standard errors.

Computes the internal and reduced moments of inertia, the rho-axis direction
angles, the internal-rotation constants (F-numbers), the angle between the two
internal-rotation axes and the torsional energy differences, propagating the
fit covariance matrix to standard errors.  A faithful translation of the
Fortran SUBROUTINE FTEST / DERPAR.
"""

from __future__ import annotations

import math

import numpy as np

from .fortran_format import fF
from .workspace import C0, MAXVAR, Workspace

CON = 505379.05
PI = math.pi
DEGRAD = 180.0 / PI


def ftest(model, ws: Workspace, out) -> None:
    nit = model.NIT
    mq = model.MQ
    nc = model.NC
    niv = model.NIV
    lp = model.LP
    A = model.A
    IVR = model.IVR
    INPAR = model.INPAR
    NTE = model.NTE

    rho = np.zeros((7, 3))
    B = np.zeros(4)
    iq1 = np.zeros(11, dtype=int)
    iq2 = np.zeros(11, dtype=int)
    ivx = np.zeros((7, 10), dtype=int)
    eps = np.zeros((7, 10))

    ntee = 0
    i2 = 0
    for i in range(4, 7):
        for k in range(1, 3):
            i2 += 1
            rho[i, k] = A[i2, 1]
        B[i - 3] = A[i + 3, 1]
    if nc == 1:
        rho[6, 2] += PI

    for iv in range(1, niv + 1):
        ntee += NTE[iv]
        for i in range(1, NTE[iv] + 1):
            kcode = INPAR[i, iv] // 256
            iq2[i] = (kcode // 64) % 16 - 8
            iq1[i] = kcode // 1024 - 8
            ivx[iv, i] = IVR[21 + i, iv]
            eps[iv, i] = A[21 + i, iv]
    if nc == -1:
        rho[5, 2] = PI - rho[5, 2]

    # covariance of the relevant parameters
    i2dim = (mq + 1) * 3 + ntee
    C = np.zeros((i2dim + 1, i2dim + 1))
    if nit != 0:
        D1 = np.zeros((i2dim + 1, MAXVAR + 1))
        L = 0
        slot_N = [0] * (i2dim + 1)
        for i in range(1, 10):
            if mq == 1 and i in (2, 4, 6):
                continue
            N = IVR[i, 1]
            L += 1
            slot_N[L] = N
            if N != 0:
                D1[L, 1:lp + 1] = ws.VCM[N, 1:lp + 1]
        for iv in range(1, niv + 1):
            for ip in range(1, NTE[iv] + 1):
                N = IVR[21 + ip, iv]
                if N > 0:
                    L += 1
                    slot_N[L] = N
                    D1[L, 1:lp + 1] = ws.VCM[N, 1:lp + 1]
        I1 = 0
        for i in range(1, 10):
            if mq == 1 and i in (2, 4, 6):
                continue
            N = IVR[i, 1]
            I1 += 1
            if N != 0:
                C[1:L + 1, I1] = D1[1:L + 1, N]
        for iv in range(1, niv + 1):
            for ip in range(1, NTE[iv] + 1):
                N = IVR[21 + ip, iv]
                if N > 0:
                    I1 += 1
                    C[1:L + 1, I1] = D1[1:L + 1, N]

    res, ster = _derpar(NTE, ntee, model.N1, model.N2, mq, nc, B, rho,
                        iq1, iq2, ivx, eps, C, niv)
    _write_derived(out, res, ster)


def _derpar(NTE, ntee, n1, n2, mq, nc, B, rho, iq1, iq2, ivx, eps, C, niv):
    DD = np.zeros((28, 31))
    bin_ = np.zeros(4)
    din = np.zeros(4)
    bitau = np.zeros(3)
    ritau = np.zeros(4)
    dirc = np.zeros((8, 3))
    F = np.zeros(4)
    U = np.zeros((4, 4))
    res = np.zeros(31)
    ster = np.zeros(28)

    for i in range(1, 4):
        bin_[i] = CON / B[i]
        din[i] = -bin_[i] / B[i]

    ll = 5
    for k in range(1, 3):
        s = math.sin(rho[5, k])
        rho[2, k] = math.cos(rho[6, k]) * s
        rho[3, k] = math.sin(rho[6, k]) * s
        rho[1, k] = math.cos(rho[5, k])
        si = 0.0
        sr = 0.0
        for i in range(1, 4):
            rho[i, k] = rho[i, k] * rho[4, k]
            sq = rho[i, k] * rho[i, k]
            si += bin_[i] * bin_[i] * sq
            sr += bin_[i] * sq
        bitau[k] = math.sqrt(si)
        res[k] = bitau[k]
        if rho[4, k] == 0.0:
            bitau[k] = 1.0
        ritau[k] = bitau[k] - sr
        res[k + 2] = ritau[k]
        for i in range(1, 4):
            dirc[i, k] = bin_[i] * rho[i, k] / bitau[k]
            dirc[i + 3, k] = math.acos(dirc[i, k])
            ll += 1
            res[ll] = dirc[i + 3, k]
            DD[k, i + 6] = dirc[i, k] * rho[i, k]
            DD[k + 2, i + 6] = DD[k, i + 6] - rho[i, k] * rho[i, k]
            DD[k, i + (k - 1) * 3] = bin_[i] * dirc[i, k]
            DD[k + 2, i + (k - 1) * 3] = DD[k, i + (k - 1) * 3] - 2 * bin_[i] * rho[i, k]
        if rho[4, k] == 0.0:
            dirc[7, k] = 1.0
        else:
            dirc[7, k] = math.atan(dirc[3, k] / dirc[2, k])
        res[11 + k] = dirc[7, k]

    sr = 0.0
    omeg = 0.0
    for i in range(1, 4):
        omeg += dirc[i, 1] * dirc[i, 2]
        s = rho[i, 1] * rho[i, 2]
        sr += bin_[i] * s
        DD[5, i + 6] = -s
        DD[5, i] = -rho[i, 2] * bin_[i]
        DD[5, i + 3] = -rho[i, 1] * bin_[i]
    ritau[3] = -sr
    res[5] = ritau[3]
    res[14] = math.acos(omeg)

    for k in range(1, 3):
        for i in range(1, 4):
            DD[2 + k * 3 + i, i + 6] = rho[i, k] / bitau[k]
            DD[2 + k * 3 + i, i + (k - 1) * 3] = bin_[i] / bitau[k]
            for j in range(1, 10):
                DD[2 + k * 3 + i, j] = DD[2 + k * 3 + i, j] - dirc[i, k] * DD[k, j] / bitau[k]
        sr = 1.0 - dirc[1, k] * dirc[1, k]
        for j in range(1, 10):
            DD[11 + k, j] = (dirc[2, k] * DD[5 + k * 3, j] - dirc[3, k] * DD[4 + k * 3, j]) / sr

    sr = math.sin(res[14])
    for j in range(1, 10):
        for i in range(1, 4):
            DD[14, j] = DD[14, j] + dirc[i, 2] * DD[5 + i, j] + dirc[i, 1] * DD[8 + i, j]
        DD[14, j] = -DD[14, j] / sr

    for k in range(1, 3):
        for i in range(1, 4):
            sr = math.sqrt(abs(1.0 - dirc[i, k] * dirc[i, k]))
            for j in range(1, 10):
                DD[2 + k * 3 + i, j] = -DD[2 + k * 3 + i, j] / sr

    for k in range(1, 18):
        for i in range(1, 4):
            DD[k, i + 6] = DD[k, i + 6] * din[i]

    d = ritau[1] * ritau[2] - ritau[3] * ritau[3]
    F[1] = CON * ritau[2] / d
    F[2] = CON * ritau[1] / d
    F[3] = -CON * ritau[3] / d
    for i in range(1, 4):
        res[14 + i] = F[i]
    U[1, 1] = -F[1] * F[1]
    U[1, 2] = -F[3] * F[3]
    U[2, 1] = U[1, 2]
    U[2, 2] = -F[2] * F[2]
    U[3, 1] = -F[1] * F[3]
    U[1, 3] = 2.0 * U[3, 1]
    U[3, 2] = -F[2] * F[3]
    U[2, 3] = 2.0 * U[3, 2]
    U[3, 3] = -F[1] * F[2] - F[3] * F[3]
    for i in range(1, 4):
        for k in range(1, 4):
            U[k, i] = U[k, i] / CON
    for i in range(1, 4):
        for j in range(1, 10):
            s = 0.0
            for k in range(1, 4):
                s += U[i, k] * DD[k + 2, j]
            DD[i + 14, j] = s

    for k in range(1, 3):
        if rho[4, k] == 0.0:
            res[14 + k] = 0.0
            res[2 + k] = 0.0
        for i in range(1, 4):
            U[i, 1] = 0.0 if rho[4, k] == 0.0 else rho[i, k] / rho[4, k]
        srr = 1.0
        if k == 2:
            srr = float(nc)
        U[2, 2] = rho[1, k] * math.cos(rho[6, k]) * srr
        U[3, 2] = rho[1, k] * math.sin(rho[6, k]) * srr
        U[1, 2] = -rho[4, k] * math.sin(rho[5, k]) * srr
        U[2, 3] = -rho[3, k]
        U[3, 3] = rho[2, k]
        U[1, 3] = 0.0
        for j in range(1, 18):
            tmp = [0.0, 0.0, 0.0, 0.0]
            for i in range(1, 4):
                s = 0.0
                for l in range(1, 4):
                    s += DD[j, l + 3 * (k - 1)] * U[l, i]
                tmp[i] = s
            for i in range(1, 4):
                DD[j, i + 3 * (k - 1)] = tmp[i]

    for j in range(1, 18):
        s = DD[j, 2]
        DD[j, 2] = DD[j, 4]
        DD[j, 4] = DD[j, 5]
        DD[j, 5] = DD[j, 3]
        DD[j, 3] = s

    pin1 = 2.0 * PI / n1
    pin2 = 2.0 * PI / n2
    L = 17
    k2 = 9
    for iv in range(1, niv + 1):
        k3 = k2
        for i in range(1, n1 // 2 + 2):
            pin1i = pin1 * (i - 1)
            j2 = n2
            if i == 1 or 2 * (i - 1) == n1:
                j2 = n2 // 2 + 1
            j1 = 1
            if mq == 1:
                j1 = i
            for j in range(j1, j2 + 1):
                if i * j == 1:
                    continue
                pin2j = pin2 * (j - 1)
                L += 1
                res[L] = 0.0
                k3 = k2
                for k in range(1, NTE[iv] + 1):
                    s1 = 2.0 * (math.cos(pin1i * iq1[k] + pin2j * iq2[k]) - 1.0)
                    DD[L, k + k3] = s1
                    s2 = 2.0 * (math.cos(pin1i * iq2[k] + pin2j * iq1[k]) - 1.0)
                    if mq == 1 and abs(iq2[k]) != iq1[k]:
                        s1 += s2
                        DD[L, k + k3] += s2
                    if ivx[iv, k] == 0:
                        k3 -= 1
                    res[L] += eps[iv, k] * s1
        k2 = k3 + NTE[iv]

    i2 = (mq + 1) * 3 + ntee
    if mq == 1:
        sgn = math.copysign(1.0, rho[6, 1] * rho[6, 2])
        for i in range(1, L + 1):
            DD[i, 1] = DD[i, 1] + DD[i, 2]
            DD[i, 2] = DD[i, 3] + DD[i, 4]
            DD[i, 3] = DD[i, 5] + DD[i, 6] * sgn
            for k in range(4, i2 + 1):
                DD[i, k] = DD[i, k + 3]

    AB = np.zeros(28)
    for i in range(1, L + 1):
        for k in range(1, i2 + 1):
            s = 0.0
            for j in range(1, i2 + 1):
                s += DD[i, j] * C[j, k]
            AB[k] = s
        aii = 0.0
        for k in range(1, L + 1):
            s = 0.0
            for j in range(1, i2 + 1):
                s += AB[j] * DD[k, j]
            if k == i:
                aii = s
        ster[i] = math.sqrt(abs(aii))

    return res, ster


def _write_derived(out, res, ster):
    out.write("\nDERIVED PARAMETERS" + " " * 20 + "VALUE" + " " * 7 + "STD ERROR\n\n")
    out.write("INTERNAL MOMENTS OF INERTIA (u*A**2)\n")
    for k in (1, 2):
        out.write(" " * 35 + fF(res[k], 12, 7) + fF(ster[k], 12, 7) + "\n")
    out.write("REDUCED INTERNAL MOMENTS OF INERTIA (u*A**2)\n")
    for k in (3, 4, 5):
        out.write(" " * 35 + fF(res[k], 12, 7) + fF(ster[k], 12, 7) + "\n")
    labels = ["(A,1)", "(B,1)", "(C,1)", "(A,2)", "(B,2)", "(C,2)"]
    for n, lab in enumerate(labels):
        k = 6 + n
        prefix = "ANGLES  " if n == 0 else "        "
        out.write(prefix + lab + " " * 22 + fF(res[k] * DEGRAD, 12, 7)
                  + fF(ster[k] * DEGRAD, 12, 7) + "\n")
    out.write("ALPHAQ1" + " " * 28 + fF(res[12] * DEGRAD, 12, 7)
              + fF(ster[12] * DEGRAD, 12, 7) + "\n")
    out.write("ALPHAQ2" + " " * 28 + fF(res[13] * DEGRAD, 12, 7)
              + fF(ster[13] * DEGRAD, 12, 7) + "\n")
    out.write("OMEGA" + " " * 30 + fF(res[14] * DEGRAD, 12, 7)
              + fF(ster[14] * DEGRAD, 12, 7) + "\n")
    out.write("F-NUMBERS (F1,F2,F') (MHz)\n")
    for k in (15, 16, 17):
        out.write(" " * 35 + fF(res[k], 12, 2) + fF(ster[k], 12, 2) + "\n")
    out.write("F-NUMBERS (F1,F2,F') (CM-1)\n")
    for k in (15, 16, 17):
        out.write(" " * 35 + fF(res[k] / C0, 12, 7) + fF(ster[k] / C0, 12, 7) + "\n")
    out.write("TORSIONAL ENERGY DIFFERENCES (MHz)\n")
    k = 18
    while k <= 30 and (res[k] != 0.0 or ster[k] != 0.0):
        out.write(fF(res[k], 47, 4) + fF(ster[k], 12, 4) + "\n")
        k += 1
