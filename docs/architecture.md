# ERHAM — Architecture & Reference

Detailed documentation of the Python port of **ERHAM** (*Effective Rotational
Hamiltonian for molecules with two periodic large-amplitude motions*, P. Groner,
version V16g-R3). This document explains how the package is structured, how data
flows through it, the core concepts inherited from the Fortran original, and what
each module does.

> [!NOTE]
> This is an architecture document. For a quick start and installation, see the
> [README](../README.md). For the scientific theory, see P. Groner,
> *J. Chem. Phys.* **107** (1997) 4483, and the user manual `erham.txt`.

---

## Table of contents

- [What ERHAM does](#what-erham-does)
- [The processing pipeline](#the-processing-pipeline)
- [Package architecture](#package-architecture)
- [Core concepts](#core-concepts)
  - [The shared workspace and 1-based arrays](#the-shared-workspace-and-1-based-arrays)
  - [Bit-packed tunneling parameters](#bit-packed-tunneling-parameters)
  - [Symmetry: ISCD, NC, MQ and JSYM](#symmetry-iscd-nc-mq-and-jsym)
  - [Numerical substitutions](#numerical-substitutions)
- [Module reference](#module-reference)
- [The effective Hamiltonian](#the-effective-hamiltonian)
- [The least-squares fit](#the-least-squares-fit)
- [Predictions and the catalog](#predictions-and-the-catalog)
- [Input file format](#input-file-format)
- [Output sections](#output-sections)
- [Validation](#validation)
- [Performance and extension notes](#performance-and-extension-notes)
- [Glossary](#glossary)

---

## What ERHAM does

ERHAM models the rotational spectrum of a molecule that has **two periodic
large-amplitude motions** (typically two internal-rotor methyl tops, as in
acetone or dimethyl ether). Each rotational level is split into torsional
sub-levels by tunneling between equivalent minima. ERHAM:

1. **Fits** spectroscopic constants (rotational, centrifugal-distortion and
   tunneling parameters) to a list of observed transition frequencies by
   non-linear least squares.
2. **Predicts** the spectrum — energy levels, transition frequencies, line
   strengths, relative intensities, one-sigma uncertainties and the partition
   function — and can emit a **JPL catalog** file.

The central object is the *effective rotational Hamiltonian*: a complex
Hermitian matrix per `(vibrational state, J, symmetry block)` whose eigenvalues
are the energy levels and whose eigenvectors give the transition moments.

---

## The processing pipeline

The driver reproduces the Fortran `PROGRAM ERHAM`: read input, optionally fit,
compute derived parameters, predict, and order/output.

```text
 input file (FILEIN)                                   output file (FILEOUT)
       │                                                        ▲
       ▼                                                        │
  read_input ──► Model ──► iter_fit ──► ftest ──► predic ─────► order
   (INPUT)       (params,   (ITER)      (FTEST)   (PREDIC)      (ORDER)
                 trans.,      │           │          │             │
                 settings)    │           │          │             └─► JPL catalog
                              ▼           ▼          ▼                  (FILECAT)
                        least-squares  derived   levels, line
                        refinement     molecular strengths,
                        (NIT cycles)   constants intensities,
                                       + errors  uncertainties,
                                                 partition fn
```

Each stage writes its section to the output stream as it runs, exactly mirroring
the order of the Fortran `WRITE` statements, so the produced file matches the
original byte layout (to printed precision).

---

## Package architecture

The modules form a clean dependency stack. Foundational modules (state and
formatting) have no internal dependencies; the physics builds on the matrix
kernels; the high-level routines orchestrate.

```text
                          driver.py            ← PROGRAM ERHAM (entry point)
                ┌────────────┼───────────┬──────────────┐
                ▼            ▼            ▼              ▼
            reader.py   iterate.py    derived.py     predict.py
            (INPUT)      (ITER)        (FTEST/        (PREDIC/
                │          │            DERPAR)        ORDER)
                │          ├─ leastsq.py                 │
                │          │  (LEASQU1 via SVD)          │
                │          └────────┬──────────┬─────────┘
                │                   ▼          ▼
                │              deriv.py   hamiltonian.py
                │              (DERIV)     (HAMILT, EVENODD)
                │                   └─────┬──────┘
                │                         ▼
                │                    matrices.py        ← ASYMRO, TRIG,
                │                         │                DMAT, BMAT, INDMAT
                ▼                         ▼
         fortran_format.py          workspace.py        ← Workspace, Model,
         (F/I/D/A descriptors)      (shared state)         constants
```

| Layer | Modules | Role |
|-------|---------|------|
| Foundation | `workspace.py`, `fortran_format.py` | Shared state/constants; output formatting |
| Matrix kernels | `matrices.py` | Wigner d-matrices, trig factors, rigid rotor |
| Physics | `hamiltonian.py`, `deriv.py` | Hamiltonian and its parameter derivatives |
| Numerics | `leastsq.py` | SVD least-squares solver |
| Orchestration | `reader.py`, `iterate.py`, `derived.py`, `predict.py` | I/O, fit, derived params, predictions |
| Entry point | `driver.py` | Wires the pipeline together |

---

## Core concepts

These four ideas are inherited from the Fortran design and are essential to
reading the code.

### The shared workspace and 1-based arrays

The Fortran code kept its large scratch matrices in `COMMON` blocks and reused
them across subroutines (with `EQUIVALENCE` aliasing real and complex views).
The Python port reproduces this with a single [`Workspace`](../erham/workspace.py)
object passed by reference. Complex matrices are stored directly as
`complex128` arrays — no aliasing trick is needed.

```text
 Workspace  (mirrors the Fortran COMMON block)
 ┌────────────────────────────────────────────────────────────┐
 │  D1, D2      Wigner reduced rotation matrices  (built once   │
 │              per J via the JDM cache)                        │
 │  PHI1, PHI2  torsional phase-angle tables                    │
 │  EV          tunneling-energy matrix (from INDMAT)           │
 │  BC          B matrix (from BMAT)                            │
 │  HC          the Hamiltonian / a derivative matrix (complex) │
 │  UC          eigenvectors (complex)                          │
 │  E           eigenvalues / per-level derivatives             │
 │  EJ          sqrt((J-k)(J+k+1)) ladder factors               │
 │  VCM         post-fit variance-covariance matrix             │
 │  JDM         highest J for which D1/D2 are built (cache key)  │
 └────────────────────────────────────────────────────────────┘
        ▲ written by INDMAT/HAMILT     ▲ read by DERIV/PREDIC
```

Every array is allocated **1-based**: a Fortran `X(243)` becomes a NumPy array of
length 244 whose element `0` is never used. This lets index expressions port
verbatim — Fortran `EJ(K+NL)` becomes `ws.EJ[k + NL]` with `NL = 122` marking
`K = 0`. The helpers `vec(n)` and `mat(n, m)` build these padded arrays.

> [!TIP]
> When reading the physics modules, remember that loop bounds and indices are
> deliberately Fortran-style (`for k in range(n1, n2 + 1)`) to keep a
> line-by-line correspondence with the original.

### Bit-packed tunneling parameters

Each tunneling/distortion parameter is described by seven small integers packed
into a single integer `INPAR` value. The reader builds it; the Hamiltonian and
derivative routines decode it.

```text
INPAR  (one packed integer per tunneling parameter)

  field:  IRO   8+IQ1     8+IQ2    MEG/IMG    KAP       JP        KP
        ┌─────┬─────────┬─────────┬────────┬────────┬────────┬────────┐
        │ 1 b │   4 b   │   4 b   │  2 b   │  4 b   │  4 b   │  4 b   │
        └─────┴─────────┴─────────┴────────┴────────┴────────┴────────┘
         MSB                                                       LSB

  IRO       0 = vibrational-energy term, 1 = higher-order term
  IQ1       localized-state index of rotor 1   (0..7),  stored as 8 + IQ1
  IQ2       localized-state index of rotor 2  (-7..7),  stored as 8 + IQ2
  MEG/IMG   omega (+1 even / -1 odd order of angular momenta) and the
            real/imaginary-part selector, combined as MEG + 2 - IMG
  KAP       |kappa| = distance of the matrix element from the diagonal
  JP        exponent of P^2  (total angular momentum)
  KP        exponent of Pz   (projection on the internal-rotor z axis)

  encode:   INPAR = (((((IRO*16 + 8+IQ1)*16 + 8+IQ2)*4 + MEG+2-IMG)
                       *16 + KAP)*16 + JP)*16 + KP
  decode:   IQ = INPAR // 256                 # drop JP, KP
            KAP = IQ % 16;  MEG_raw = (IQ // 16) % 4
            IQ2 = (IQ // 64) % 16 - 8;  IQ1 = (IQ // 1024) % 16 - 8
```

Because the codes sort numerically, the reader sorts parameters by `INPAR` so
that the vibrational-energy terms (`IRO = 0`) come first; routines exploit this
to process them as contiguous groups.

### Symmetry: ISCD, NC, MQ and JSYM

Two control integers from the input — `ISCD` (symmetry parameter) and `NC`
(direction-cosine parameter) — determine the molecular symmetry. From them the
reader derives `MQ` (rotor equivalence), and per symmetry block the code derives
`JSYM`, which selects how the Hamiltonian is symmetrised and diagonalised.

```text
 ISCD  ──►  MQ = 3 - min(max(|ISCD|, 1), 2)
            ├── |ISCD| ≤ 1  (ISCD ∈ {-1, 0, 1})  ──► MQ = 2  non-equivalent rotors
            └── |ISCD| ≥ 2                        ──► MQ = 1  equivalent rotors

 per block (IS1, IS2):  ISZ  ──► JSYM  (via the MSYM table + NC adjustment)

   IS1 == IS2 == 0 ───────────────► ISZ 1
   IS1 == IS2 != 0 ───────────────► ISZ 2        JSYM ──► diagonalisation
   IS1 != IS2, (IS1+IS2) % N1 == 0 ► ISZ 3        ───────────────────────────
   IS1 != IS2, otherwise ─────────► ISZ 4         0  full complex-Hermitian
                                                  1  Wang fold → eigh(e)+eigh(o)
                                                  2  Wang fold (alt. signs)
                                                  3  even/odd-Ka split (EVENODD)
                                                  4  Wang fold → EVENODD(e,o)
```

`JSYM` reduces the complex Hermitian block to smaller real/symmetric sub-blocks
using the **Wang transformation** (symmetric/antisymmetric `|K> ± |−K>`
combinations) and, where applicable, an even/odd-`Ka` split. Levels in the
antisymmetric sub-block are tagged with a `-` in the output. The mapping lives in
[`compute_jsym`](../erham/hamiltonian.py).

### Numerical substitutions

The refactor's core change: two large hand-coded numerical kernels are replaced
by their NumPy/LAPACK equivalents. This is exact for the quantities that matter.

| Fortran routine(s) | Replaced by | Why it is safe |
|--------------------|-------------|----------------|
| `SEIGCX` + `HOUSCX`/`BISECT`/`INVITR`/`REVCX` (complex-Hermitian eigensolver) | `numpy.linalg.eigh` | Identical eigenvalues; line strengths are phase-invariant; the `-` labels come from block structure, not eigenvector phase |
| `LEASQU1` + `LSVDF`/`LSVDB`/`LSVG2`/`VHS12`/`DROTG` (IMSL SVD) | `numpy.linalg.svd` | Identical singular values; the parameter changes, standard errors and covariance are recomputed from the SVD with the original formulas |

The expensive O(J³) "sandwich" transforms (`Σ_K1 Σ_K2 D1·B·EV·D2`) in the
Hamiltonian and its derivatives are also recast as NumPy matrix products, which
is the main reason the port is tractable in pure Python.

---

## Module reference

| Module | Fortran origin | Key public functions | Responsibility |
|--------|----------------|----------------------|----------------|
| [`workspace.py`](../erham/workspace.py) | `COMMON` blocks, dimensions | `Workspace`, `Model`, `vec`, `mat` | Shared scratch state, parsed model, physical constants |
| [`fortran_format.py`](../erham/fortran_format.py) | edit descriptors | `fI`, `fF`, `fD`, `fA` | Reproduce Fortran `I/F/D/A` number formatting |
| [`matrices.py`](../erham/matrices.py) | `ASYMRO`, `TRIG`, `DMAT`, `BMAT`, `INDMAT` | `asymro`, `trig`, `dmat`, `bmat`, `indmat` | Matrix-element building blocks |
| [`hamiltonian.py`](../erham/hamiltonian.py) | `HAMILT`, `EVENODD` | `hamilt`, `evenodd`, `compute_jsym`, `vib_sandwich` | Assemble and diagonalise the effective Hamiltonian |
| [`deriv.py`](../erham/deriv.py) | `DERIV` | `deriv` | Derivatives of energy levels w.r.t. each parameter |
| [`leastsq.py`](../erham/leastsq.py) | `LEASQU1` | `leasqu1`, `LsqResult` | SVD least-squares solve + statistics |
| [`reader.py`](../erham/reader.py) | `INPUT` | `read_input`, `FortranReader` | Parse and validate the input file, echo it |
| [`iterate.py`](../erham/iterate.py) | `ITER` | `iter_fit` | Non-linear least-squares refinement loop |
| [`derived.py`](../erham/derived.py) | `FTEST`, `DERPAR` | `ftest` | Derived molecular constants and their errors |
| [`predict.py`](../erham/predict.py) | `PREDIC`, `ORDER` | `predic`, `order` | Predict levels/lines; sort; write catalog |
| [`driver.py`](../erham/driver.py) | `PROGRAM ERHAM` | `run`, `main` | Entry point; wires the pipeline |

### Key signatures

```python
# read input, echo it, return the parsed model
read_input(lines: list[str], out) -> Model

# refine parameters in place; populate ws.VCM (variance-covariance)
iter_fit(model: Model, ws: Workspace, out) -> None

# assemble + diagonalise one block; fills ws.E (energies), ws.UC (vectors)
hamilt(model, iv, j, jsym, ws) -> (j1, j21, rjj1, lbl)

# dE_i/dp for parameter `ip`, returned in ws.E
deriv(model, iv, j, j1, j21, rjj1, ip, ws) -> None

# predict levels/lines for all states & blocks; emit ordered list + catalog
predic(model, ws, out, catalog_inputs=None) -> None

# full run from file to file
run(filein, fileout, catalog_inputs=None) -> None
```

---

## The effective Hamiltonian

`hamilt()` is the heart of the program. For one `(state, J, symmetry block)` it
builds the complex Hermitian matrix `HC` (stored as a lower triangle) from four
contributions, then symmetrises and diagonalises it.

```text
hamilt(model, iv, J, jsym, ws)
  │
  ├─ EJ[k] = sqrt((J-k)(J+k+1))                       ladder factors
  │
  ├─ ASYMRO ........... asymmetric rigid rotor (A-reduction)
  │                     → HC diagonal + 2nd sub-diagonal
  │
  ├─ DMAT  ............ Wigner d-matrices D1, D2  (recursion in J,
  │                     cached: only J = JDM..J-1 are (re)built)
  ├─ BMAT  ............ combine D1, D2 → B matrix (BC)
  ├─ vib_sandwich(EV)   tunneling-energy contribution, as NumPy matmuls
  │                     → HC (Hermitian accumulate)
  ├─ tunneling groups   higher-order / S-reduction distortion terms
  │                     → HC (off-diagonals at distance KAP)
  │
  └─ symmetrise + diagonalise, selected by JSYM
        ┌─────────────────────────────────────────────────────────┐
        │ JSYM 0   eigh(full  J21 × J21)                           │
        │ JSYM 1,2 Wang fold → eigh(even block) + eigh(odd block)  │
        │ JSYM 4   Wang fold → EVENODD(even) + EVENODD(odd)        │
        │ JSYM 3   EVENODD(full)                                   │
        └─────────────────────────────────────────────────────────┘
        → sort eigenvalues ascending, carry the '-' antisymmetry labels

  result:  ws.E[1..j21] = energies,  ws.UC[·,1..j21] = eigenvectors,
           returns (j1, j21, rjj1, lbl)
```

`vib_sandwich` collapses the original triple loop into
`G = D1ᵀ · (B ⊙ EVp) · D2`, applies the torsional phase factors, then adds the
Hermitian result to `HC`. `EVENODD` and the Wang-folding code are translated
literally (scalar, in-place) because their basis bookkeeping must be preserved
exactly; only the inner eigensolves are delegated to `numpy.linalg.eigh`.

---

## The least-squares fit

`iter_fit()` performs the Fortran `ITER` loop: each cycle walks the sorted list
of energy levels, builds each block once, accumulates calculated frequencies and
(scaled) derivatives into a Jacobian, solves the normal equations by SVD, prints
the parameter table, and applies the corrections.

```text
 for cycle in 1..NIT:
   │
   ├─ for each level group (IS1, IS2, IV, J)  [levels are pre-sorted]:
   │     INDMAT  (once per symmetry block)
   │     HAMILT  → energies E         → accumulate CALC[transition] += ±E
   │     for each variable parameter p:
   │        DERIV → dE/dp             → accumulate DER[transition, p] += ±dE/dp · scale
   │
   ├─ form weighted residuals  res = (obs - calc) · weight       (handle blends)
   ├─ leasqu1(DER, res)  ─ SVD ─►  Δp, std errors, precision, correlation, covariance
   ├─ print parameter table (old value, std error, change, precision, scale)
   └─ apply  p ← p + Δp   (and re-slave equivalent-rotor parameters)

 after the last cycle:  ws.VCM = scaled covariance  (used by FTEST and PREDIC)
```

> [!IMPORTANT]
> A transition can connect levels from **two different** `J` blocks (upper and
> lower state), so a calculated frequency is the difference `E_upper − E_lower`.
> The level-encoding (`ILEV`) packs a signed transition index into each level so
> the accumulation `CALC[transition] += E · sign` yields the frequency once both
> blocks have been processed.

Parameter scaling (`SCP`) keeps the normal matrix well-conditioned: each Jacobian
column is multiplied by a per-parameter scale, and the resulting changes/errors
are scaled back for display. Octic and other tiny constants therefore do not
trigger the SVD "ill-conditioned" failure.

---

## Predictions and the catalog

`predic()` loops over vibrational states and symmetry blocks, and within each
over `J = JMIN..JMAX`. It prints energy levels, then **R-transitions**
(`J ↔ J−1`) and **Q-transitions** (within `J`), computing line strengths for the
`a`, `b`, `c` dipole components and the relative intensity. `order()` then sorts
all predicted lines by frequency and writes them, plus an optional JPL catalog.

```text
 for each (state IV, symmetry block IS1,IS2):
   INDMAT
   for J = JMIN .. JMAX:
     HAMILT  → energies, eigenvectors
     print energy levels (with '-' labels)
     if NUNC: DERIV for every variable parameter  → per-level derivatives
     │
     ├─ R-transitions (J ↔ J−1):  use this J's vectors + saved (J−1) vectors
     │     line strengths S_a, S_b, S_c  →  intensity  →  σ uncertainty
     ├─ partition-function accumulation (Boltzmann × spin weight × degeneracy)
     └─ Q-transitions (within J):  line strengths, intensity, σ
     save (E, labels, vectors, derivatives) for the next J's R-transitions
   if JMIN == 0:  write the SUM OF STATES table (partition function)

 order():  collect all lines → sort by frequency → write ordered list
                                                  → write JPL catalog (IFPR=4)
```

Uncertainties come from propagating the post-fit covariance `ws.VCM`:
`σ = sqrt(dνᵀ · VCM · dν)`, where `dν` is the derivative of the transition
frequency with respect to the fitted parameters. The partition function is the
Boltzmann-weighted sum of states, printed only when `JMIN = 0`.

---

## Input file format

All input is free-format (whitespace/comma separated) except the title. Inputs 4
through 7 are **repeated for each vibrational state**. Terminators end the
variable-length lists. See `erham.txt` for the authoritative description.

```text
input file
│
├─ 1.  title line                                          (up to 80 chars)
├─ 2.  ISCD NC NIV NIT IFPR NUNC  DIP(a) DIP(b) DIP(c) TEMP   control
├─ 3a. N1 RHO1 ivRHO1 BETA1 ivBET1 ALPHA1 ivALP1             rotor 1
├─ 3b. N2 RHO2 ...                       (only if MQ = 2, non-equivalent)
│
│   for each vibrational state IV = 1 .. NIV:
├─ 4.  JMIN JMAX FMIN FMAX THRES  A(7..21)  IVR(7..21)       rot / CD consts
├─ 5.  tunneling parameter lines           (terminated by MEG = 0)
├─ 6.  symmetry-block lines                 (terminated by IS1 < 0)
└─ 7.  transition lines                     (terminated by IS1 < 0)
```

| Input | Fields | Notes |
|-------|--------|-------|
| 1 | title | Read as fixed `1X,10A8`; echoed back |
| 2 | `ISCD NC NIV NIT IFPR NUNC DIP(1:3) TEMP` | Symmetry, #states, #iterations, print/uncertainty options, dipole, temperature |
| 3a/3b | `N RHO ivRHO BETA ivBET ALPHA ivALP` | Periodicity and rho-axis parameters per rotor; `iv*` flag a variable (`1`), constant (`0`) or slaved (`2`) |
| 4 | `JMIN JMAX FMIN FMAX THRES` + 15 constants + 15 flags | `A(7..9)` rotational, `A(10..14)` quartic (kHz), `A(15..21)` sextic (Hz); `IVR<0` slaves to another state |
| 5 | `IQ1 IQ2 MEG KAP JP KP PAR IVAR SCPP` | One tunneling/distortion term; `MEG=0` ends the list; `SCPP` is the scaling factor |
| 6 | `IS1 IS2 ISP1 ISP2` | Symmetry block + spin multiplicities; `IS1<0` ends |
| 7 | `IS1 IS2 JQ NQ J NN FREQ BLE ER` | Transition (upper `JQ,NQ`; lower `J,NN`); `BLE` blends; `ER` is the uncertainty (`0` ⇒ zero weight) |

> [!TIP]
> Free-format reads consume exactly the fields they need and discard the rest of
> the line, which is why the example input can carry trailing comments such as
> `old 20075.6760` after a transition.

---

## Output sections

The output file is produced in this fixed order (matching the Fortran):

```text
1. timestamp banner
2. input echo            (control, rotor, per-state constants, tunneling, blocks)
3. transition list       (every transition as read)
4. fit cycles            (residual table, singular values, parameter table,
                          correlation matrix) × NIT
5. DERIVED PARAMETERS    (moments of inertia, angles, F-numbers, torsional
                          energy differences)  + standard errors
6. PREDICTIONS           (per block: energy levels, R/Q transitions)
7. SUM OF STATES         (partition function, when JMIN = 0)
8. CATALOG ENTRY INFO    (when IFPR = 4)
9. TRANSITIONS ORDERED BY FREQUENCY
10. timestamp banner
```

When `IFPR = 4`, a separate JPL-format catalog file is also written, prompting
interactively (or via `catalog_inputs`) for the catalog name, ID, partition
function and two log-intensity cutoffs.

---

## Validation

The port is validated against the bundled example. Running `ac10x-r3.in`
reproduces `ac10x-r3.out` and `ac10x-r3-catalog.txt` to the printed precision:

| Section | Agreement |
|---------|-----------|
| Fit parameters, std errors, changes, precision | ~12 significant figures |
| Standard deviation per cycle | 15.032383 → 2.4172771 (exact) |
| Derived parameters + error propagation | exact (e.g. 3.2337118 ± 0.0005069) |
| Torsional energy differences | exact |
| Predicted levels, line strengths, intensities, σ | exact |
| Partition function (sum of states) | exact (260395) |
| Ordered transitions + JPL catalog | exact, character-for-character |

> [!NOTE]
> The reference `.out` and catalog files shipped with the project are
> **abbreviated** (middle rows replaced with `etc...`), so validation compares
> anchor sections and the head/tail rather than a naive full-file diff.

A finite-difference derivative checker (`fd_check.py`) is included; it was the
tool that located the one substantive porting bug (a missing higher-order
tunneling term in the rho derivative).

> [!WARNING]
> `DERIV` must reproduce the **Fortran's** analytic derivative, not the
> mathematically exact one. The Fortran's rho-derivative of the odd-`KAP`
> higher-order terms carries a small built-in approximation; matching it is what
> reproduces the reference fit. Do not "correct" a derivative to satisfy the
> finite-difference check unless the discrepancy is a genuine missing term.

---

## Performance and extension notes

The full example (with `NUNC = 1` uncertainties and `JMAX = 60`) runs in roughly
**12 minutes** in pure Python; the Fortran takes seconds. The hot spots are:

- the scalar R/Q transition-moment loops in `predict.py`, and
- the `rho` derivative's `EV`-build trig loops in `deriv.py`.

Both are vectorisable with NumPy the same way the Hamiltonian sandwich already
is. The Hamiltonian assembly and the SVD are not bottlenecks.

```text
 cost (full example, NUNC=1, JMAX=60)
   PREDIC  R/Q moment loops      ████████████████████  (largest)
   DERIV   rho EV-build loops    ████████
   HAMILT  assembly + eigh       ██
   ITER    fit (6 cycles)        █
```

When extending the code:

- Keep the **1-based** array convention and Fortran-style loop bounds in the
  physics modules — it is what makes the port auditable against the original.
- New tunneling-parameter types flow through the `INPAR` encode/decode; add them
  in `reader.py` and handle them in both `hamilt` and `deriv` (value *and*
  derivative).
- Output formatting goes through `fortran_format.py`; reuse `fI/fF/fD` so columns
  stay aligned with the reference.

---

## Glossary

| Term | Meaning |
|------|---------|
| Internal rotor | A molecular subgroup (e.g. a methyl top) that rotates relative to the frame |
| Tunneling parameter | Constant describing the energy/coupling from tunneling between equivalent torsional minima |
| `J` | Total rotational angular-momentum quantum number |
| `Ka, Kc` | Asymmetric-rotor projection labels (printed for each level) |
| A-reduction | Watson's asymmetric centrifugal-distortion Hamiltonian reduction (the default) |
| S-reduction | Alternative distortion reduction, available via tunneling-parameter input |
| Wang transformation | Symmetric/antisymmetric `|K> ± |−K>` basis that block-diagonalises the Hamiltonian |
| R / Q transition | `ΔJ = ±1` (R) and `ΔJ = 0` (Q) rotational transitions |
| Spin weight | Nuclear-spin statistical weight of a torsional sub-level |
| Partition function | Boltzmann-weighted sum of states, used to scale absolute intensities |
| JPL catalog | Standard spectral line-list format from the JPL molecular spectroscopy catalog |
