# ERHAM (Python port)

Python translation of P. Groner's Fortran program **ERHAM** — *Effective
Rotational Hamiltonian for molecules with two periodic large-amplitude motions*
(version V16g-R3, 20 May 2013).

> Reference: P. Groner, *J. Chem. Phys.* **107** (1997) 4483–4498.

ERHAM fits spectroscopic constants to observed rotational transition
frequencies of molecules with two internal rotors and predicts their spectra
(line positions, intensities, partition function, and an optional JPL catalog
file).

## Documentation

- **[Tutorial](docs/tutorial.md)** — hands-on walkthrough: fit and predict the
  acetone spectrum and learn to read every part of the output. Start here.
- **[Architecture & reference](docs/architecture.md)** — the pipeline, design
  diagrams, the bit-packed parameter encoding, the symmetry logic, and a
  per-module reference.
- **`erham.txt`** (project root) — the original Fortran user manual; the
  authoritative input-format specification.

## Requirements

- Python 3.9+
- numpy

## Usage

```bash
# interactive (prompts for input/output file names, like the original)
python run_erham.py

# non-interactive
python run_erham.py ac10x-r3.in ac10x-r3.out

# or as a module
python -m erham ac10x-r3.in ac10x-r3.out
```

The input file format is identical to the original Fortran program (see
`erham.txt`). If the input requests a JPL catalog (print option `IFPR = 4`),
the program prompts for the catalog file name, catalog ID, partition function,
and the two log-intensity cutoffs — exactly as the Fortran version does.

To drive it from Python instead of the CLI:

```python
from erham.driver import run
run("ac10x-r3.in", "ac10x-r3.out",
    catalog_inputs=("cat.txt", 58999, 0.260395e6, -10.0, -10.0))
```

## What changed from the Fortran

The translation is faithful: the bit-packed tunneling-parameter codes, the
1-based index conventions, the symmetry bookkeeping (Wang folding, even/odd-Ka
splitting and the level `-` labels) and all output formats are reproduced.
The example `ac10x-r3.in` reproduces `ac10x-r3.out` and `ac10x-r3-catalog.txt`:
fitted parameters, standard errors, derived parameters, predicted levels and
transitions, intensities, uncertainties and the catalog all agree to the
printed precision.

Two numerical kernels are replaced by their NumPy/LAPACK equivalents, which is
the point of the refactor:

| Fortran | Python |
|---------|--------|
| `SEIGCX` + `HOUSCX`/`BISECT`/`INVITR`/`REVCX` (complex-Hermitian eigensolver) | `numpy.linalg.eigh` |
| `LEASQU1` + `LSVDF`/`LSVDB`/`LSVG2`/`VHS12`/`DROTG` (IMSL SVD) | `numpy.linalg.svd` |

Eigenvalues and singular values are identical; the line strengths used for the
predictions are phase-invariant. The expensive "sandwich" transforms in the
Hamiltonian and its derivatives are expressed as NumPy matrix products.

## Module layout

| Module | Fortran routine(s) |
|--------|--------------------|
| `workspace.py` | COMMON blocks, parameter arrays (the shared state) |
| `reader.py` | `INPUT` |
| `matrices.py` | `ASYMRO`, `TRIG`, `DMAT`, `BMAT`, `INDMAT` |
| `hamiltonian.py` | `HAMILT`, `EVENODD`, symmetry helpers |
| `deriv.py` | `DERIV` |
| `leastsq.py` | `LEASQU1` (via SVD) |
| `iterate.py` | `ITER` |
| `derived.py` | `FTEST`, `DERPAR` |
| `predict.py` | `PREDIC`, `ORDER` |
| `driver.py` | `PROGRAM ERHAM` |
| `fortran_format.py` | Fortran `F`/`I`/`D`/`A` edit descriptors |
