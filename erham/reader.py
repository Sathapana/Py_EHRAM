"""INPUT -- read and validate the ERHAM input file, echoing it to the output.

The reader reproduces Fortran list-directed (``READ *``) semantics: each read
starts a fresh record and consumes as many whitespace/comma separated tokens as
requested, spanning records when necessary and discarding the remainder of the
last record (this is what lets the input carry trailing comments).
"""

from __future__ import annotations

import math
import re

from .fortran_format import fD, fF, fI
from .workspace import JMX, MAXSTATE, NTRX, Model

MS1, MS2, MS4 = 8, 128, 16384
PI = math.pi


class FortranReader:
    """List-directed record reader over a list of text lines."""

    def __init__(self, lines):
        self.lines = lines
        self.li = 0

    def _next_tokens(self):
        line = self.lines[self.li]
        self.li += 1
        return [t for t in re.split(r"[\s,]+", line.strip()) if t != ""]

    def read(self, n):
        toks = []
        while len(toks) < n:
            if self.li >= len(self.lines):
                raise EOFError("unexpected end of input")
            toks.extend(self._next_tokens())
        return toks[:n]


def _i(tok):
    return int(tok)


def _f(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def read_input(lines, out) -> Model:
    m = Model()
    rdr = FortranReader(lines)

    # ---- Input 1: title (FORMAT 1X,10A8) ----
    title_line = lines[0].rstrip("\n")
    m.title = title_line[1:81]
    rdr.li = 1
    out.write(" " + m.title.ljust(80).rstrip() + "\n")

    # ---- Input 2: control parameters ----
    t = rdr.read(10)
    m.ISCD = _i(t[0]); m.NC = _i(t[1]); m.NIV = _i(t[2]); m.NIT = _i(t[3])
    m.IFPR = _i(t[4]); m.NUNC = _i(t[5])
    m.DIP[1] = _f(t[6]); m.DIP[2] = _f(t[7]); m.DIP[3] = _f(t[8])
    m.TEMP = _f(t[9])
    m.KVQ = 0

    if (min(5 + m.ISCD, 4 - m.ISCD, m.NIV - 1, 6 - m.NIV, m.NIT, m.IFPR,
            4 - m.IFPR) < 0 or abs(m.NC) != 1):
        out.write(f" INPUT ERROR\n ISCD= {m.ISCD} NC= {m.NC} NIV= {m.NIV}"
                  f" NIT= {m.NIT} IFPR= {m.IFPR}\n")
        raise ValueError("input error in control parameters")

    if abs(m.ISCD) > 3:
        m.NC = -1
    m.MQ = 3 - min(max(abs(m.ISCD), 1), 2)
    lisc = (m.ISCD * m.NC == 3 or m.ISCD * m.NC == -2) and m.ISCD > 0
    wtwt = "NON-" if m.MQ == 2 else "    "
    if m.NIT <= 0:
        m.NUNC = 0

    _echo_control(out, m, wtwt)

    lxy = 1 if m.ISCD in (0, -3, -5) else 0

    # ---- Input 3a ----
    t = rdr.read(7)
    N1 = _i(t[0]); RHO1 = _f(t[1]); IVRHO1 = _i(t[2]); BETA1 = _f(t[3])
    IVBET1 = _i(t[4]); ALPHA1 = _f(t[5]); IVALP1 = _i(t[6])
    if IVRHO1 != 1:
        IVRHO1 = 0
    if IVBET1 != 1:
        IVBET1 = 0
    if IVALP1 != 1:
        IVALP1 = 0
    if m.ISCD < 0 and lxy == 0:
        ALPHA1 = 0.0
        IVALP1 = 0
    if lxy == 1:
        BETA1 = 90.0
        IVBET1 = 0

    # ---- Input 3b (only if non-equivalent rotors) ----
    if m.MQ == 2:
        t = rdr.read(7)
        N2 = _i(t[0]); RHO2 = _f(t[1]); IVRHO2 = _i(t[2]); BETA2 = _f(t[3])
        IVBET2 = _i(t[4]); ALPHA2 = _f(t[5]); IVALP2 = _i(t[6])
        if IVRHO2 < 1 or IVRHO2 > 2:
            IVRHO2 = 0
        if IVBET2 < 1 or IVBET2 > 2:
            IVBET2 = 0
        if IVALP2 < 1 or IVALP2 > 2:
            IVALP2 = 0
        if m.ISCD == -1:
            ALPHA2 = ALPHA1
            IVALP2 = IVALP1
        elif m.ISCD == 0:
            BETA2 = BETA1
            IVBET2 = IVBET1
    else:
        N2 = N1
        RHO2 = RHO1
        IVRHO2 = 0
        BETA2 = BETA1
        IVBET2 = 0
        ALPHA2 = ALPHA1
        if lisc or m.ISCD == -3:
            ALPHA2 = -ALPHA2
        IVALP2 = 0

    m.N1 = N1
    m.N2 = N2
    if min(N1, N2) < 1:
        out.write(f" INPUT ERROR   N1= {N1}   N2= {N2}\n")
        raise ValueError("input error: periodicity")

    _echo_rotor(out, N1, N2, RHO1, RHO2, IVRHO1, IVRHO2, BETA1, BETA2,
                IVBET1, IVBET2, ALPHA1, ALPHA2, IVALP1, IVALP2)

    BETA1 *= PI / 180.0
    BETA2 *= PI / 180.0
    ALPHA1 *= PI / 180.0
    ALPHA2 *= PI / 180.0

    # working copies of per-parameter scaling factors
    scc = [[0.0] * (MAXSTATE + 1) for _ in range(61)]

    m.NTRA = 0          # LM
    LL = 0
    for iv in range(1, m.NIV + 1):
        for kp in range(1, 7):
            m.IVR[kp, iv] = -1
        _read_state_block(rdr, out, m, iv, scc, lxy)
        LL = _read_transitions(rdr, out, m, iv, LL)

    out.write(f"\n{fI(m.NTRA, 4)} TRANSITIONS,{fI(LL, 4)} WITH NON-ZERO WEIGHT\n")

    _sort_levels(m)

    # internal-rotation parameters live in column 1
    m.A[1, 1] = RHO1; m.A[2, 1] = RHO2; m.A[3, 1] = BETA1
    m.A[4, 1] = BETA2; m.A[5, 1] = ALPHA1; m.A[6, 1] = ALPHA2
    m.IVR[1, 1] = IVRHO1; m.IVR[2, 1] = IVRHO2; m.IVR[3, 1] = IVBET1
    m.IVR[4, 1] = IVBET2; m.IVR[5, 1] = IVALP1; m.IVR[6, 1] = IVALP2
    scc[1][1] = 1e-3
    scc[2][1] = 1e-3
    for iv in range(1, m.NIV + 1):
        for kp in range(3, 22):
            scc[kp][iv] = 1.0

    _resolve_variables(m, scc)
    return m


# ---------------------------------------------------------------------------
def _read_state_block(rdr, out, m, iv, scc, lxy):
    # ---- Input 4 ----
    t = rdr.read(35)
    m.JMIN[iv] = _i(t[0])
    m.JMAX[iv] = min(_i(t[1]), JMX)
    m.FMIN[iv] = _f(t[2]); m.FMAX[iv] = _f(t[3]); m.THRES[iv] = _f(t[4])
    for k in range(15):
        m.A[7 + k, iv] = _f(t[5 + k])
    for k in range(15):
        m.IVR[7 + k, iv] = _i(t[20 + k])
    _echo_state(out, m, iv)

    # ---- Input 5: tunneling parameters ----
    spar = [0.0] * 40
    igr = [0] * 40
    scp_tmp = [0.0] * 40
    inpar = [0] * 40
    i = 0
    ii = 0
    while True:
        t = rdr.read(9)
        IQ1 = _i(t[0]); IQ2 = _i(t[1]); MEG = _i(t[2]); KAP = _i(t[3])
        JP = _i(t[4]); KP = _i(t[5]); PAR = _f(t[6]); IVAR = _i(t[7])
        SCPP = _f(t[8])
        if MEG == 0:
            break
        MEG = 1 if MEG > 0 else -1
        IMGS = 1 if KAP >= 0 else -1
        IMG = 0
        if KAP != 0 and m.ISCD > 0:
            IMG = (1 - IMGS) // 2
        KAP = abs(KAP)
        L = KAP % 2
        IRO = JP + KP + KAP
        IQ = IQ1 + IQ2
        IQQ = IQ1 - IQ2

        reject = False
        if min(7 - IQ1, 7 - abs(IQ2), IQ, JP, KP, 14 - KAP, m.ISCD * IMG) < 0:
            reject = True
        elif m.ISCD * IMG < 0 or KAP + 1 - IMG == 0:
            reject = True
        else:
            if m.ISCD * m.NC * IMG == 3 or m.ISCD * m.NC * IMG == -2:
                L = 1 - L
            if abs(m.ISCD) == 4:
                L = 0
            if m.MQ == 1:
                ok = (min(IQQ, m.N1 - IQ1 + m.NC * IQ2) >= 0 and
                      (1 + IQQ) * (L * (IQ - 1) + 1)
                      + MEG * (IQQ - 1) * (L * (IQ + 1) - 1) > 0)
            else:
                ok = IQ > 0 or (IQ == 0 and IQQ >= 1 - MEG)
            reject = not ok

        if reject:
            _echo_tun_error(out, IQ1, IQ2, MEG, IMGS * KAP, JP, KP, PAR, IVAR, SCPP)

        if abs(IVAR) != 1:
            IVAR = 0
        if i >= 37:
            out.write(f" TOO MANY TUNNELING PARAMETERS FOR STATE IV = {iv}\n")
            raise ValueError("too many tunneling parameters")
        i += 1
        IRO = min(IRO, 1)
        ii += 1 - IRO
        inpar[i] = (((((IRO * 16 + 8 + IQ1) * 16 + 8 + IQ2) * 4 + MEG + 2 - IMG)
                     * 16 + KAP) * 16 + JP) * 16 + KP
        spar[i] = PAR
        igr[i] = IVAR
        scp_tmp[i] = SCPP
        _echo_tun(out, IQ1, IQ2, MEG, IMGS * KAP, JP, KP, PAR, IVAR, SCPP)

    # sort the parameters by their packed code (ascending)
    for kp in range(1, i + 1):
        jp = kp
        iq = inpar[jp]
        for k in range(kp, i + 1):
            if inpar[k] < iq:
                jp = k
                iq = inpar[jp]
        spar[jp], spar[kp] = spar[kp], spar[jp]
        scp_tmp[jp], scp_tmp[kp] = scp_tmp[kp], scp_tmp[jp]
        igr[jp], igr[kp] = igr[kp], igr[jp]
        inpar[jp], inpar[kp] = inpar[kp], inpar[jp]
    for kp in range(1, i + 1):
        m.INPAR[kp, iv] = inpar[kp]
        m.A[kp + 21, iv] = spar[kp]
        scc[kp + 21][iv] = scp_tmp[kp]
        m.IVR[kp + 21, iv] = igr[kp]
    m.INPAR[i + 1, iv] = 0
    m.NTUP[iv] = i
    m.NTE[iv] = ii

    # ---- Input 6: symmetry blocks ----
    nsig = 0
    while True:
        t = rdr.read(4)
        IS1 = _i(t[0]); IS2 = _i(t[1]); ISP1 = _i(t[2]); ISP2 = _i(t[3])
        if IS1 < 0:
            break
        if m.MQ == 1:
            ok = min(IS2 - IS1, m.N1 // 2 - IS1, m.N1 - IS1 - IS2,
                     m.N1 // 2 + IS1 * (1 + IS2) - IS2) >= 0
        else:
            ok = (min(m.N1 // 2 - IS1, IS2, m.N2 // 2 - IS2) >= 0 or
                  ((IS1 == 0 or IS1 == m.N1 // 2) and IS2 < m.N2))
        if not ok:
            out.write(f"INPUT ERROR{fI(IS1, 4)}{fI(IS2, 4)}      INPUT DELETED\n")
            continue
        nsig += 1
        m.ISIG[1, nsig, iv] = IS1
        m.ISIG[2, nsig, iv] = IS2
        m.ISIG[3, nsig, iv] = ISP1
        m.ISIG[4, nsig, iv] = ISP2
        out.write(f"        SYMMETRY BLOCK   {IS1}{IS2}  SPIN WEIGHTS"
                  f"{fI(ISP1, 4)}{fI(ISP2, 4)}\n")
        m.NSIG[iv] = nsig


def _read_transitions(rdr, out, m, iv, LL):
    BLS = 0.0
    IBL = m.NTRA + 1
    FREQQ = -1.0
    WTT = 0.0
    while True:
        t = rdr.read(9)
        IS1 = _i(t[0]); IS2 = _i(t[1]); JQ = _i(t[2]); NQ = _i(t[3])
        J = _i(t[4]); NN = _i(t[5]); FREQ = _f(t[6]); BLE = _f(t[7]); ER = _f(t[8])
        IVQ = iv
        IV1 = iv
        if IS1 < 0:
            break
        bad = max(NN - 2 * J - 1, NQ - 2 * JQ - 1, abs(JQ - J) - 1, JQ - JMX,
                  J - JMX, -JQ, -NQ, -J, -NN, -IS2, IVQ - m.NIV, IV1 - m.NIV) > 0
        accept = False
        if not bad:
            if m.MQ == 1:
                if max(IS1 - m.N1 // 2, IS1 + IS2 - m.N1, IS1 - IS2,
                       IS2 - m.N1 // 2 - IS1 * (1 + IS2)) <= 0:
                    accept = True
            else:
                if min(IS1 - m.N1 // 2, IS2 - m.N2 // 2) >= 0:
                    accept = True
                elif (IS1 == 0 or IS1 == m.N1 // 2) and IS2 < m.N2:
                    accept = True
        if not accept:
            ER = abs(ER)
            out.write(f" INPUT ERROR{fI(IS1, 4)}{fI(IS2, 4)}{fI(JQ, 4)}"
                      f"{fI(NQ, 4)}{fI(J, 4)}{fI(NN, 4)}{fF(FREQ, 12, 4)}"
                      f"{fF(BLE, 8, 3)}{fF(ER, 8, 3)}    TRANSITION DELETED\n")
            continue

        m.NTRA += 1
        lm = m.NTRA
        m.ITRA[1, lm] = IS1; m.ITRA[2, lm] = IS2; m.ITRA[3, lm] = IVQ
        m.ITRA[4, lm] = JQ; m.ITRA[5, lm] = NQ; m.ITRA[6, lm] = IV1
        m.ITRA[7, lm] = J; m.ITRA[8, lm] = NN
        m.FRQ[lm] = FREQ
        m.WT[lm] = 0.0
        if ER != 0.0:
            LL += 1
            m.WT[lm] = 1.0 / ER
        m.BL[lm] = BLE

        # blend bookkeeping (screen-input style, IIN = 5)
        if BLE == 0.0:
            if BLS > 0.0:
                for ib in range(IBL, lm + 1):
                    m.BL[ib] = 0.0
                BLS = 0.0
            IBL = lm + 1
        else:
            if IBL == lm:
                FREQQ = FREQ
                WTT = m.WT[lm]
            m.FRQ[lm] = FREQQ
            m.WT[lm] = WTT
            BLS += abs(BLE)
            if BLE < 0.0:
                for ib in range(IBL, lm + 1):
                    m.BL[ib] = m.BL[ib] / BLS
                BLS = 0.0
                IBL = lm + 1

        # energy-level encoding
        k = (IS1 * MS1 + IS2) * MS1
        m.ILEV[1, 2 * lm - 1] = (k + IVQ) * MS2 + JQ
        m.ILEV[2, 2 * lm - 1] = NQ * MS4 + MS4 // 2 + lm
        m.ILEV[1, 2 * lm] = (k + IV1) * MS2 + J
        m.ILEV[2, 2 * lm] = NN * MS4 + MS4 // 2 - lm

        out.write(f"{fI(IS1, 4)}{fI(IS2, 4)}{fI(IVQ, 4)}{fI(JQ, 4)}{fI(NQ, 4)}"
                  f"{fI(IV1, 4)}{fI(J, 4)}{fI(NN, 4)}{fF(FREQ, 15, 4)}"
                  f"{fF(BLE, 8, 3)}{fF(ER, 8, 3)}\n")
        if lm >= NTRX:
            break
    return LL


def _sort_levels(m):
    n = 2 * m.NTRA
    ilev = m.ILEV
    for kp in range(1, n + 1):
        jp = kp
        iq = ilev[1, jp]
        nn = ilev[2, jp]
        for k in range(kp, n + 1):
            if ilev[1, k] < iq or (ilev[1, k] == iq and ilev[2, k] < nn):
                jp = k
            nn = ilev[2, jp]
            iq = ilev[1, jp]
        ilev[1, jp], ilev[1, kp] = ilev[1, kp], ilev[1, jp]
        ilev[2, jp], ilev[2, kp] = ilev[2, kp], ilev[2, jp]
    ilev[1, 2 * m.NTRA + 1] = 0


def _resolve_variables(m, scc):
    # rotational / distortion constants: cap variation flag at 1
    lp = 0
    for kp in range(7, 22):
        for iv in range(1, m.NIV + 1):
            m.IVR[kp, iv] = min(m.IVR[kp, iv], 1)
        lp = min(lp, m.IVR[kp, 1])
    if lp < 0:
        raise ValueError("illegal input of variable parameters")

    lp = 0
    for iv in range(1, m.NIV + 1):
        kp = 1
        while kp <= 21 + m.NTUP[iv]:
            v = m.IVR[kp, iv]
            if v > 0:
                if v == 1:
                    lp += 1
                m.IVR[kp, iv] = lp
                m.SCP[lp] = scc[kp][iv]
            elif v < 0:
                if kp <= 21:
                    src = -v
                    m.IVR[kp, iv] = m.IVR[src, iv]
                    m.A[kp, iv] = m.A[src, iv]
                else:
                    iq = m.INPAR[kp - 21, iv] % 16384
                    iq = abs((iq // 1024) % 16 - (iq // 64) % 16) * 245760
                    redo = False
                    for jp in range(22, 22 + m.NTUP[iv]):
                        if abs(m.INPAR[kp - 21, iv] - m.INPAR[jp - 21, iv]) == iq:
                            if jp < kp:
                                m.IVR[kp, iv] = m.IVR[jp, iv]
                                m.A[kp, iv] = m.A[jp, iv]
                            elif jp > kp:
                                m.IVR[jp, iv] = -1
                                m.IVR[kp, iv] = 1
                                redo = True
                                break
                    if redo:
                        continue
            kp += 1

    m.LP = lp
    if lp <= 0:
        m.NIT = 0
        m.NUNC = 0
    if lp > 80:
        raise ValueError(f"too many variable parameters: {lp}")


# ---------------------------------------------------------------------------
# echo helpers (FORMAT statements 901, 911, 907, 902, 903)
# ---------------------------------------------------------------------------
def _echo_control(out, m, wtwt):
    out.write(f"\n **** CALCULATION FOR {wtwt}EQUIVALENT MOTIONS ****\n\n")
    out.write(f" SYMMETRY PARAMETER (ISCD){fI(m.ISCD, 12)}\n")
    out.write(f" DIRECTION COSINE PARAMETER (NC){fI(m.NC, 6)}\n")
    out.write(f" NUMBER OF VIBRATIONAL STATES (NIV){fI(m.NIV, 3)}\n")
    out.write(f" NUMBER OF ITERATIONS (NIT){fI(m.NIT, 11)}\n")
    out.write(f" PRINT OPTION PARAMETER (IFPR){fI(m.IFPR, 8)}\n")
    out.write(f" UNCERTAINTY PARAMETER (NUNC){fI(m.NUNC, 9)}\n")
    out.write(f" DIPOLE MOMENT COMPONENTS (DIP) {fF(m.DIP[1], 6, 3)}"
              f"{fF(m.DIP[2], 6, 3)}{fF(m.DIP[3], 6, 3)}\n")
    out.write(f" TEMPERATURE/KELVIN (TEMP){fF(m.TEMP, 12, 2)}\n")
    out.write(f" VIB STATE READ (KVQ,1=YES,ELSE=NO){fI(m.KVQ, 3)}\n")


def _echo_rotor(out, N1, N2, RHO1, RHO2, IVRHO1, IVRHO2, BETA1, BETA2,
                IVBET1, IVBET2, ALPHA1, ALPHA2, IVALP1, IVALP2):
    out.write(f"\n PERIODICITY OF INTERNAL ROTOR (N){fI(N1, 13)}{fI(N2, 14)}\n")
    out.write(f" RHO PARAMETER{fF(RHO1, 33, 8)}{fF(RHO2, 14, 8)}\n")
    out.write(f" VARIATION PARAMETER FOR RHO, IVRHO{fI(IVRHO1, 12)}{fI(IVRHO2, 14)}\n")
    out.write(f" RHO AXIS ANGLE, BETA{fF(BETA1, 26, 8)}{fF(BETA2, 14, 8)}\n")
    out.write(f" VARIATION PARAMETER FOR BETA, IVBET{fI(IVBET1, 11)}{fI(IVBET2, 14)}\n")
    out.write(f" RHO AXIS ANGLE, ALPHA{fF(ALPHA1, 25, 8)}{fF(ALPHA2, 14, 8)}\n")
    out.write(f" VARIATION PARAMETER FOR ALPHA, IVALP{fI(IVALP1, 10)}{fI(IVALP2, 14)}\n")


def _echo_state(out, m, iv):
    out.write(f"\nVIBRATIONAL STATE{fI(iv, 4)}\n")
    out.write(f" JMIN{fI(m.JMIN[iv], 4)}  JMAX{fI(m.JMAX[iv], 4)}      FMIN"
              f"{fF(m.FMIN[iv], 10, 1)}      FMAX{fF(m.FMAX[iv], 10, 1)}"
              f"  INTENSITY CUTOFF{fF(m.THRES[iv], 6, 2)}\n")
    out.write(f"{fF(m.A[7, iv], 15, 8)}{fF(m.A[8, iv], 15, 8)}{fF(m.A[9, iv], 15, 8)}\n")
    for row in range(2):
        vals = "".join(fF(m.A[10 + row * 5 + k, iv], 15, 8) for k in range(5))
        out.write(vals + "\n")
    out.write(f"{fF(m.A[20, iv], 15, 8)}{fF(m.A[21, iv], 15, 8)}\n")
    out.write("".join(fI(m.IVR[7 + k, iv], 4) for k in range(15)) + "\n")


def _echo_tun(out, IQ1, IQ2, MEG, KAP, JP, KP, PAR, IVAR, SCPP):
    out.write(f"{fI(IQ1, 4)}{fI(IQ2, 4)}{fI(MEG, 4)}{fI(KAP, 4)}{fI(JP, 4)}"
              f"{fI(KP, 4)}{fF(PAR, 15, 8)}{fI(IVAR, 4)}{fD(SCPP, 12, 3)}\n")


def _echo_tun_error(out, IQ1, IQ2, MEG, KAP, JP, KP, PAR, IVAR, SCPP):
    out.write(f" INPUT ERROR{fI(IQ1, 4)}{fI(IQ2, 4)}{fI(MEG, 4)}{fI(KAP, 4)}"
              f"{fI(JP, 4)}{fI(KP, 4)}{fF(PAR, 15, 8)}{fI(IVAR, 4)}"
              f"{fD(SCPP, 12, 3)}  INPUT DELETED\n")
