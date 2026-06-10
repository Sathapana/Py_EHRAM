"""PREDIC / ORDER -- predict energy levels, transition frequencies, line
strengths, relative intensities, the partition function and (optionally) a JPL
catalog file.
"""

from __future__ import annotations

import math

import numpy as np

from .deriv import deriv
from .fortran_format import fD, fF, fI
from .hamiltonian import compute_jsym, hamilt
from .matrices import indmat
from .workspace import C0, BOLTZ, PLANCK, XINTF, Workspace

MS1 = 8


def predic(model, ws: Workspace, out, catalog_inputs=None) -> None:
    pi = math.pi
    pin1 = pi / model.N1
    pin2 = pi / model.N2
    fac = -PLANCK * 1e6 / (model.TEMP * BOLTZ)
    dip = model.DIP
    lp = model.LP

    def f1(jj, k):
        return math.sqrt((jj - k) * (jj + k + 1)) / 2.0

    def f2(jj, k):
        return math.sqrt((jj + k) * (jj + k + 1))

    trans = []        # all predicted transitions (for ORDER)

    print("PREDICTIONS")
    out.write("\n PREDICTIONS\n")

    for iv in range(1, model.NIV + 1):
        fiv = model.FMIN[iv]
        fav = model.FMAX[iv]
        emin = 0.0
        nsig = model.NSIG[iv]
        sst = np.zeros((model.JMAX[iv] + 2, nsig + 1, 3))

        for isb in range(1, nsig + 1):
            is1 = model.ISIG[1, isb, iv]
            is2 = model.ISIG[2, isb, iv]
            isg1 = model.ISIG[3, isb, iv]
            isg2 = model.ISIG[4, isb, iv]
            sst1 = 0.0
            sst2 = 0.0
            print(f" IS1 {is1}  IS2 {is2}  IV {iv}")
            out.write(f"\n VIBRATIONAL STATE{fI(iv, 3)}      IS1{fI(is1, 2)}"
                      f"    IS2{fI(is2, 2)}\n")

            jsym = compute_jsym(model.ISCD, model.NC, model.N1, is1, is2)
            indmat(model, iv, is1, is2, pin1, pin2, ws)

            prev = None          # (EE, LB, UC, DER) of J-1
            for j in range(model.JMIN[iv], model.JMAX[iv] + 1):
                j1, j21, rjj1, lbl = hamilt(model, iv, j, jsym, ws)
                _write_levels(out, j, j21, ws.E, lbl)

                lb = [0] * (j21 + 1)
                for i in range(1, j21 + 1):
                    lb[i] = -i if lbl[i] == '-' else i
                ee = ws.E[1:j21 + 1].copy()
                uc = ws.UC[1:j21 + 1, 1:j21 + 1].copy()

                der = None
                if model.NUNC != 0:
                    der = np.zeros((j21 + 1, lp + 1))
                    for ip in range(1, 22 + model.NTUP[iv]):
                        if model.IVR[ip, iv] == 0:
                            continue
                        kp = model.IVR[ip, iv]
                        deriv(model, iv, j, j1, j21, rjj1, ip, ws)
                        der[1:j21 + 1, kp] = ws.E[1:j21 + 1]

                sj = 4 * j
                th = model.THRES[iv]

                # ---- R-transitions (J <-> J-1) ----
                if j != model.JMIN[iv] and prev is not None:
                    out.write(f"   R-TRANSITIONS{fI(j, 12)}{fI(j - 1, 12)}\n")
                    ee_low, lb_low, uc_low, der_low = prev
                    nlow = 2 * (j - 1) + 1
                    for i in range(1, nlow + 1):
                        isp = isg1 if lb_low[i] >= 0 else isg2
                        ss = math.exp(fac * ee_low[i - 1]) * isp * XINTF
                        for mm in range(1, j21 + 1):
                            de = ee[mm - 1] - ee_low[i - 1]
                            if abs(de) < fiv or abs(de) > fav:
                                continue
                            s1c = s2c = s3c = 0j
                            for kq in range(1, j21 - 1):
                                k = kq - j
                                ulow = uc_low[kq - 1, i - 1]
                                s1c += np.conj(uc[kq - 1, mm - 1]) * f2(j, -k) * ulow
                                s2c += np.conj(uc[kq + 1, mm - 1]) * f2(j, k) * ulow
                                s3c += (np.conj(uc[kq, mm - 1])
                                        * math.sqrt((j - k) * (j + k)) * ulow)
                            s3c *= 2.0
                            sa, sb, sc, sm = _strengths_r(s1c, s2c, s3c, sj, dip)
                            sbb = sm * ss * (1.0 - math.exp(fac * de)) * de
                            if sbb > th:
                                up = max(ee_low[i - 1], ee[mm - 1]) / C0
                                unc = _uncert(model, ws, der, der_low, mm, i, lp)
                                _write_tr(out, j, lb[mm], j - 1, lb_low[i], de,
                                          isp, sa, sb, sc, sbb, unc)
                                trans.append((8 * is1 + is2, iv, j, lb[mm],
                                              j - 1, lb_low[i], de, isp, sa, sb,
                                              sc, sbb, unc, up))

                # ---- partition-function bookkeeping ----
                ss = math.exp(fac * ee[j21 - 1]) * j21
                emin = min(emin, ee[0])
                if lb[j21] > 0:
                    sst1 += ss
                else:
                    sst2 += ss

                # ---- Q-transitions ----
                if j != 0:
                    out.write(f"   Q-TRANSITIONS{fI(j, 12)}\n")
                    rq = j21 / rjj1
                    for i in range(1, j21):
                        ssj = math.exp(fac * ee[i - 1])
                        isp = isg1
                        if lb[i] > 0:
                            sst1 += ssj * j21
                        else:
                            sst2 += ssj * j21
                            isp = isg2
                        ssw = ssj * XINTF * isp
                        for mm in range(i + 1, j21 + 1):
                            de = ee[mm - 1] - ee[i - 1]
                            if de < fiv or de > fav:
                                continue
                            s3c = np.conj(uc[0, i - 1]) * (-j) * uc[0, mm - 1]
                            s1c = s2c = 0j
                            for kq in range(2, j21 + 1):
                                k = kq - j1
                                s1c += np.conj(uc[kq - 2, i - 1]) * f1(j, -k) * uc[kq - 1, mm - 1]
                                s2c += np.conj(uc[kq - 1, i - 1]) * f1(j, -k) * uc[kq - 2, mm - 1]
                                s3c += np.conj(uc[kq - 1, i - 1]) * k * uc[kq - 1, mm - 1]
                            sa, sb, sc, sm = _strengths_q(s1c, s2c, s3c, rq, sj, dip)
                            sbb = sm * ssw * (1.0 - math.exp(fac * de)) * de
                            if sbb > th:
                                up = ee[mm - 1] / C0
                                unc = _uncert(model, ws, der, der, mm, i, lp)
                                _write_tr(out, j, lb[mm], j, lb[i], de, isp,
                                          sa, sb, sc, sbb, unc)
                                trans.append((8 * is1 + is2, iv, j, lb[mm], j,
                                              lb[i], de, isp, sa, sb, sc, sbb,
                                              unc, up))

                prev = (ee, lb, uc, der)
                sst[j + 1, isb, 1] = sst1 * isg1
                sst[j + 1, isb, 2] = sst2 * isg2

        if model.JMIN[iv] == 0:
            _write_sum_of_states(out, model, iv, sst, emin, fac)

    order(model, out, trans, catalog_inputs)


def _strengths_r(s1c, s2c, s3c, sj, dip):
    sm = ((np.real(np.conj(s3c) * (s1c - s2c)) * dip[2]
           - np.imag(np.conj(s3c) * (s1c + s2c)) * dip[3]) * dip[1]
          + np.imag(np.conj(s2c) * s1c) * dip[2] * dip[3])
    sc = (np.conj(s1c + s2c) * (s1c + s2c) / sj).real
    sb = (np.conj(s1c - s2c) * (s1c - s2c) / sj).real
    sa = (np.conj(s3c) * s3c / sj).real
    sm = sm / sj + sa * dip[1] ** 2 + sb * dip[2] ** 2 + sc * dip[3] ** 2
    return sa, sb, sc, sm


def _strengths_q(s1c, s2c, s3c, rq, sj, dip):
    sm = ((np.real(np.conj(s3c) * (s1c + s2c)) * dip[2]
           - np.imag(np.conj(s3c) * (s1c - s2c)) * dip[3]) * dip[1]
          + np.imag(np.conj(s2c) * s1c) * dip[2] * dip[3])
    sc = (np.conj(s1c - s2c) * (s1c - s2c) * rq).real
    sb = (np.conj(s1c + s2c) * (s1c + s2c) * rq).real
    sa = (np.conj(s3c) * s3c * rq).real
    sm = sm / sj + sa * dip[1] ** 2 + sb * dip[2] ** 2 + sc * dip[3] ** 2
    return sa, sb, sc, sm


def _uncert(model, ws, der_up, der_low, mm, i, lp):
    if model.NUNC == 0 or der_up is None:
        return 0.0
    dnu = der_up[mm, 1:lp + 1] - der_low[i, 1:lp + 1]
    vcm = ws.VCM[1:lp + 1, 1:lp + 1]
    return float(np.sqrt(dnu @ vcm @ dnu))


def _write_levels(out, j, j21, E, lbl):
    out.write(f" J ={fI(j, 3)}\n")
    parts = []
    for k in range(1, j21 + 1):
        parts.append(fF(E[k], 14, 3) + (lbl[k] if lbl[k] != ' ' else ' '))
        if k % 8 == 0 and k < j21:
            parts.append("\n")
    out.write("".join(parts) + "\n")


def _write_tr(out, ju, nu, jl, nl, de, isp, sa, sb, sc, sbb, unc):
    # FORMAT(4I4,F15.4,I4,4F10.5,D15.4)
    line = (fI(ju, 4) + fI(nu, 4) + fI(jl, 4) + fI(nl, 4) + fF(de, 15, 4)
            + fI(isp, 4) + fF(sa, 10, 5) + fF(sb, 10, 5) + fF(sc, 10, 5)
            + fF(sbb, 10, 5))
    if unc != 0.0:
        line += fD(unc, 15, 4)
    out.write(line + "\n")


def _write_sum_of_states(out, model, iv, sst, emin, fac):
    nsig = model.NSIG[iv]
    out.write("\n SUM OF STATES\n\n")
    hdr = "IS1,IS2"
    for isb in range(1, nsig + 1):                    # each block printed twice
        entry = fI(model.ISIG[1, isb, iv], 3) + fI(model.ISIG[2, isb, iv], 1) + " " * 7
        hdr += entry + entry
    out.write(hdr + "\n")
    for j in range(model.JMIN[iv] + 1, model.JMAX[iv] + 2):
        row = "".join(fF(sst[j, isb, 1], 12, 3) + fF(sst[j, isb, 2], 12, 3)
                      for isb in range(1, nsig + 1))
        out.write(row + "\n")
    ss = math.exp(-fac * emin)
    out.write("\n")
    j = model.JMAX[iv] + 1
    row = "".join(fF(sst[j, isb, 1] * ss, 12, 3) + fF(sst[j, isb, 2] * ss, 12, 3)
                  for isb in range(1, nsig + 1))
    out.write(row + "\n")


# ---------------------------------------------------------------------------
# ORDER -- sort the predicted transitions and print them (and the catalog)
# ---------------------------------------------------------------------------
def order(model, out, trans, catalog_inputs):
    iout2 = None
    icatid = q = logstr0 = logstr1 = None
    if model.IFPR == 4:
        if catalog_inputs is None:
            catalog_inputs = _prompt_catalog()
        filecat, icatid, q, logstr0, logstr1 = catalog_inputs
        iout2 = open(filecat, "w")
        out.write(f"\n\n{'*' * 80}\n CATALOG ENTRY INFORMATION\n"
                  f" Catalog ID{fI(icatid, 21)}\n PARTITION FUNCTION"
                  f"{fD(q, 22, 8)}\n 1st INT CUTOFF (LOGSTR0){fD(logstr0, 16, 8)}\n"
                  f" 2nd INT CUTOFF (LOGSTR1){fD(logstr1, 16, 8)}\n{'*' * 80}\n\n")

    # normalise direction of each transition (positive frequency)
    rows = []
    for (isc, iv, j, n, j1, n1, fr, isp, s1, s2, s3, bl, un, up) in trans:
        if n * n1 == 0:
            continue
        if fr <= 0.0:
            j, j1 = j1, j
            n, n1 = n1, n
            fr = -fr
        rows.append([isc, iv, j, n, j1, n1, fr, isp, s1, s2, s3, bl, un, up])

    if not rows:
        return

    rows.sort(key=lambda r: r[6])           # by frequency (stable)
    out.write("\n TRANSITIONS ORDERED BY FREQUENCY\n\n")

    for r in rows:
        (isc, iv, j, n, j1, n1, fr, isp, s1, s2, s3, bl, un, up) = r
        is1 = isc // 8
        is2 = isc % 8
        ka1 = abs(n) // 2
        kc1 = (2 * j + 2 - abs(n)) // 2
        ka2 = abs(n1) // 2
        kc2 = (2 * j1 + 2 - abs(n1)) // 2
        if iout2 is not None:
            el = up - fr / C0
            igup = min(isp * (2 * j + 1), 999)
            blm = -99.0 if bl == 0.0 else math.log10(bl / q)
            thrlog = math.log10(10 ** logstr0 + (fr / 300000.0) ** 2 * 10 ** logstr1)
            if blm >= thrlog:
                unm = min(un, 999.9999)
                iout2.write(
                    fF(fr, 13, 4) + fF(unm, 8, 4) + fF(blm, 8, 4) + fI(3, 2)
                    + fF(el, 10, 4) + fI(igup, 3) + fI(icatid, 7) + fI(1404, 4)
                    + fI(j, 2) + fI(ka1, 2) + fI(kc1, 2) + fI(is1, 2) + "    "
                    + fI(j1, 2) + fI(ka2, 2) + fI(kc2, 2) + fI(is2, 2) + "\n")
        out.write(
            fI(is1, 2) + fI(is2, 1)
            + fI(iv, 3) + fI(j, 4) + fI(n, 4) + fI(ka1, 4) + fI(kc1, 4)
            + fI(iv, 3) + fI(j1, 4) + fI(n1, 4) + fI(ka2, 4) + fI(kc2, 4)
            + fF(fr, 12, 3) + fF(un, 9, 3) + fI(isp, 4)
            + fF(s1, 8, 4) + fF(s2, 8, 4) + fF(s3, 8, 4) + fF(bl, 8, 4)
            + fF(up, 10, 3) + "\n")

    if iout2 is not None:
        iout2.close()


def _prompt_catalog():
    filecat = input("Enter catalog filename !\n").strip()
    icatid = int(input("Enter catalog ID !\n"))
    q = float(input("Enter Partition function !\n"))
    logstr0 = float(input("Enter first LOG Intensity Cutoff (LOGSTR0) !\n"))
    logstr1 = float(input("Enter second LOG Intensity Cutoff (LOGSTR1) !\n"))
    return filecat, icatid, q, logstr0, logstr1
