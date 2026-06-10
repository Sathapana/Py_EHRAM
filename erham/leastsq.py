"""Linear least-squares via SVD -- the LEASQU1 routine.

The original used the IMSL singular-value-decomposition chain
(LSVDF/LSVDB/LSVG2/VHS12/DROTG).  Here numpy provides the SVD and the
statistical post-processing (parameter changes, precision estimate, standard
errors, correlation and covariance matrices) reproduces the LEASQU1 formulas
for the regular least-squares case (LSMD = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fortran_format import fD, fI


@dataclass
class LsqResult:
    changes: np.ndarray      # parameter changes, length N (scaled space)
    precision: np.ndarray    # precision estimate, length N
    ster: np.ndarray         # standard errors, length N (scaled space)
    corr: np.ndarray         # correlation matrix, N x N
    cov: np.ndarray          # covariance matrix (scaled space), N x N
    sigma2: float            # variance
    ss: float                # sum of squared residuals
    std_dev: float           # sqrt(sigma2)


def leasqu1(jac: np.ndarray, res: np.ndarray, out) -> LsqResult:
    """Solve ``jac @ x ~= res`` (regular least squares).

    ``jac`` is M x N (M observations, N parameters), ``res`` length M.
    Prints the singular-value block to ``out`` (Fortran unit 6).
    """
    m, n = jac.shape
    ss = float(res @ res)

    # economy SVD:  jac = U @ diag(s) @ Vt
    U, s, Vt = np.linalg.svd(jac, full_matrices=False)
    V = Vt.T                                   # right singular vectors (columns)

    out.write("\n SINGULAR VALUES AND RIGHT SINGULAR VECTORS FOR JACOBIAN"
              "MATRIX\n\n")
    for i in range(n):
        _write_sv_row(out, s[i], V[:, i])

    sigma2 = ss / (m - n)
    std_dev = float(np.sqrt(sigma2))
    nn = n                                      # all singular values used (LSMD=0)
    out.write(f"\n{fI(nn, 3)} SINGULAR VALUES USED OUT OF{fI(n, 3)}, "
              f"STANDARD DEVIATION ={fD(std_dev, 16, 8)}\n")

    b = U.T @ res                              # U^T . res
    changes = V @ (b / s)                      # parameter changes

    Vs = V / s[None, :]                        # V(K,I)/s(I)
    cov = sigma2 * (Vs @ Vs.T)                 # scaled-space covariance
    ster = np.sqrt(np.diag(cov))
    corr = cov / np.outer(ster, ster)

    denom = (V ** 2) @ (s ** 2)                # sum_I s_I^2 V(K,I)^2
    precision = np.sqrt(sigma2 * m / denom) / n

    return LsqResult(changes, precision, ster, corr, cov, sigma2, ss, std_dev)


def _write_sv_row(out, sval: float, vec: np.ndarray) -> None:
    """FORMAT(D12.4,12F10.5/(12X,12F10.5))."""
    n = vec.size
    parts = [fD(sval, 12, 4)]
    for k in range(n):
        parts.append(f"{vec[k]:10.5f}")
        if (k + 1) % 12 == 0 and k + 1 < n:
            parts.append("\n" + " " * 12)
    out.write("".join(parts) + "\n")
