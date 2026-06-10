"""Shared state and data model for the ERHAM port.

The original Fortran kept large scratch matrices in COMMON blocks (with
EQUIVALENCE between real and complex views) and passed the spectroscopic
parameters around as plain arrays.  To keep the translation faithful and to
minimise off-by-one mistakes, every array here is allocated *1-based*: an
array that Fortran declares as ``X(243)`` becomes a numpy array of length
244 whose element 0 is never used.  Index expressions such as ``EJ(K+NL)``
therefore port verbatim to ``ws.EJ[k + NL]``.

Two containers are defined:

``Workspace``
    The transient matrices that the Fortran code kept in COMMON and shared
    between INDMAT / HAMILT / DERIV / PREDIC.  Complex matrices are stored
    directly as ``complex128`` arrays (no EQUIVALENCE trick needed).

``Model``
    The parsed input: spectroscopic parameters, tunneling-parameter codes,
    variation flags, transitions and prediction settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Global dimensions (identical to the Fortran DATA statements)
# ---------------------------------------------------------------------------
NU = 243            # matrix dimension (basis size for J up to 120)
NL = (NU + 1) // 2  # = 122, index of K = 0 in a padded vector
JMX = 120           # maximum J quantum number
NTRX = 8191         # maximum number of transitions
MAXVAR = 80         # maximum number of variable parameters
MAXPAR = 60         # rows of the A / IVR arrays (21 + up to 37 tunneling + slack)
MAXTUN = 38         # columns reserved for INPAR (37 tunneling params + terminator)
MAXSTATE = 6        # maximum number of vibrational states

# Physical constants (from the Fortran DATA statements)
PI = 3.141592653589793
C0 = 29979.2458              # speed of light, MHz * cm
PLANCK = 6.6260755e-34
BOLTZ = 1.380658e-23
XINTF = 4.16231e-5           # intensity prefactor
CON = 505379.05              # h / (8 pi^2) conversion, MHz * u * A^2
DEGRAD = 180.0 / PI

SQRT2 = 1.4142135623730951


def vec(n: int, dtype=float) -> np.ndarray:
    """A 1-based vector with valid indices 1..n (element 0 unused)."""
    return np.zeros(n + 1, dtype=dtype)


def mat(n: int, m: int, dtype=float) -> np.ndarray:
    """A 1-based matrix with valid indices [1..n, 1..m]."""
    return np.zeros((n + 1, m + 1), dtype=dtype)


class Workspace:
    """Scratch matrices shared between the Hamiltonian routines.

    Mirrors the COMMON block

        B(2,243,243), D1(243,243), D2(243,243), PHI1(485), PHI2(485),
        EV(243,243), U(2,243,243), E(243), EJ(243), H(2,243,243), EW(243,243)

    with the complex views BC = B, UC = U, HC = H.
    """

    def __init__(self) -> None:
        n = NU
        # Wigner reduced rotation matrices d^J (real), built recursively in DMAT
        self.D1 = mat(n, n)
        self.D2 = mat(n, n)
        # Phase-angle tables (length 2*NU-1 = 485 in Fortran); pad to 2*NU
        self.PHI1 = vec(2 * NU)
        self.PHI2 = vec(2 * NU)
        # Tunneling-energy matrix built by INDMAT
        self.EV = mat(n, n)
        # B matrix from BMAT (complex)
        self.BC = mat(n, n, dtype=complex)
        # Hamiltonian (complex Hermitian) and its eigenvectors
        self.HC = mat(n, n, dtype=complex)
        self.UC = mat(n, n, dtype=complex)
        # Eigenvalues / derivative values, and angular-momentum factors
        self.E = vec(n)
        self.EJ = vec(n)
        # sqrt((J-K)(J+K+1)) ladder factors, set up in HAMILT
        # Direction-cosine helper (DC(8)) produced by INDMAT
        self.DC = vec(8)
        # Generic complex scratch used by DERIV (the EX/EW equivalence)
        self.EW = mat(n, n, dtype=complex)
        # Variance-covariance matrix from the fit (LP x LP, 1-based), used by
        # FTEST (derived-parameter errors) and PREDIC (frequency uncertainties)
        self.VCM = mat(MAXVAR, MAXVAR)
        # Running J for which D1/D2 have been accumulated (the Fortran JDM)
        self.JDM = 0

    def clear_H(self, j21: int) -> None:
        """Zero the lower triangle of HC for a J block of dimension j21."""
        self.HC[: j21 + 1, : j21 + 1] = 0.0


@dataclass
class Model:
    """Parsed spectroscopic model and fit/prediction configuration."""

    title: str = ""

    # --- control parameters (Input 2) ---
    ISCD: int = 0
    NC: int = 0
    NIV: int = 1
    NIT: int = 0
    IFPR: int = 0
    NUNC: int = 0
    KVQ: int = 0
    DIP: np.ndarray = field(default_factory=lambda: vec(3))
    TEMP: float = 0.0

    # --- derived symmetry / rotor info ---
    MQ: int = 1            # 1 = equivalent rotors, 2 = non-equivalent
    N1: int = 1
    N2: int = 1

    # --- spectroscopic parameter arrays (1-based) ---
    # A[i, iv]: i = 1..6 internal-rotation params, 7..21 rot/CD constants,
    #           22.. tunneling parameters.  iv = 1..NIV.
    A: np.ndarray = field(default_factory=lambda: mat(MAXPAR, MAXSTATE))
    IVR: np.ndarray = field(default_factory=lambda: np.zeros((MAXPAR + 1, MAXSTATE + 1), dtype=int))
    INPAR: np.ndarray = field(default_factory=lambda: np.zeros((MAXTUN + 1, MAXSTATE + 1), dtype=int))
    SCP: np.ndarray = field(default_factory=lambda: vec(MAXVAR))

    NTUP: np.ndarray = field(default_factory=lambda: np.zeros(MAXSTATE + 1, dtype=int))
    NTE: np.ndarray = field(default_factory=lambda: np.zeros(MAXSTATE + 1, dtype=int))

    # --- prediction ranges (per state) ---
    JMIN: np.ndarray = field(default_factory=lambda: np.zeros(MAXSTATE + 1, dtype=int))
    JMAX: np.ndarray = field(default_factory=lambda: np.zeros(MAXSTATE + 1, dtype=int))
    FMIN: np.ndarray = field(default_factory=lambda: vec(MAXSTATE))
    FMAX: np.ndarray = field(default_factory=lambda: vec(MAXSTATE))
    THRES: np.ndarray = field(default_factory=lambda: vec(MAXSTATE))

    # --- symmetry blocks for predictions ---
    NSIG: np.ndarray = field(default_factory=lambda: np.zeros(MAXSTATE + 1, dtype=int))
    ISIG: np.ndarray = field(default_factory=lambda: np.zeros((5, 11, MAXSTATE + 1), dtype=int))

    # --- transitions ---
    NTRA: int = 0
    FRQ: np.ndarray = field(default_factory=lambda: vec(NTRX))
    BL: np.ndarray = field(default_factory=lambda: vec(NTRX))
    WT: np.ndarray = field(default_factory=lambda: vec(NTRX))
    ITRA: np.ndarray = field(default_factory=lambda: np.zeros((9, NTRX + 1), dtype=int))
    ILEV: np.ndarray = field(default_factory=lambda: np.zeros((3, 2 * NTRX + 2), dtype=int))

    LP: int = 0     # number of variable parameters in the fit
