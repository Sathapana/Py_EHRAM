"""Finite-difference validation of DERIV against HAMILT eigenvalues."""
import numpy as np

from erham.reader import read_input
from erham.hamiltonian import compute_jsym, hamilt
from erham.deriv import deriv
from erham.matrices import indmat
from erham.workspace import Workspace
import io

with open(r"C:\dev\ERHAM\ac10x-r3.in") as f:
    lines = f.read().splitlines()
model = read_input(lines, io.StringIO())
ws = Workspace()

iv, is1, is2, J = 1, 0, 1, 5     # a 01 block (jsym=0), J=5
jsym = compute_jsym(model.ISCD, model.NC, model.N1, is1, is2)


def eig(j):
    indmat(model, iv, is1, is2, np.pi / model.N1, np.pi / model.N2, ws)
    _, j21, _, _ = hamilt(model, iv, j, jsym, ws)
    return ws.E[1:j21 + 1].copy(), j21


def analytic(ip, j):
    indmat(model, iv, is1, is2, np.pi / model.N1, np.pi / model.N2, ws)
    j1, j21, rjj1, lbl = hamilt(model, iv, j, jsym, ws)
    deriv(model, iv, j, j1, j21, rjj1, ip, ws)
    return ws.E[1:j21 + 1].copy()


# for MQ=1 the slaved partner (rho2/beta2/alpha2) must move with the variable
PARTNER = {1: 2, 3: 4, 5: 6} if model.MQ == 1 else {}

print(f"jsym={jsym}  (is1,is2)=({is1},{is2})  J={J}")
for ip in [1, 3, 7, 8, 18, 22, 25]:
    base = model.A[ip, iv]
    part = PARTNER.get(ip)
    basep = model.A[part, iv] if part else None
    delta = 1e-5 * max(abs(base), 1.0)
    e0, j21 = eig(J)
    ana = analytic(ip, J)
    model.A[ip, iv] = base + delta
    if part:
        model.A[part, iv] = basep + delta
    ep, _ = eig(J)
    model.A[ip, iv] = base - delta
    if part:
        model.A[part, iv] = basep - delta
    em, _ = eig(J)
    model.A[ip, iv] = base
    if part:
        model.A[part, iv] = basep
    fd = (ep - em) / (2 * delta)
    err = np.max(np.abs(ana - fd))
    rel = err / max(np.max(np.abs(fd)), 1e-30)
    print(f"IP={ip:2d}  base={base: .6e}  max|ana-fd|={err:.3e}  rel={rel:.2e}"
          f"   ana[0]={ana[0]: .6e} fd[0]={fd[0]: .6e}")
