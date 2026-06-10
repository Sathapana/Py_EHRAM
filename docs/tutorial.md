# Tutorial: Fit and predict the acetone spectrum with ERHAM

**Time:** ~20 minutes | **Level:** Beginner (no spectroscopy background needed)

In this tutorial you will run ERHAM end-to-end on a real molecule — acetone — and
learn to read every part of the output. You will refine a set of molecular
constants against ~600 measured lines, predict the spectrum, and produce a
catalog of predicted transitions. Terms are defined as they appear.

---

## What you'll build

By the end you will have:

- a **fitted parameter set** for acetone's lowest torsionally-excited state,
- a list of **predicted transitions** with frequencies, intensities and
  uncertainties,
- a set of **derived molecular constants** (moments of inertia, internal-rotation
  constants), and
- a **JPL-format catalog** file.

```text
   ac10x-r3.in                ERHAM                 results you'll read
 ┌──────────────┐         ┌───────────┐        ┌────────────────────────┐
 │ constants    │         │   fit     │  ───►  │ refined parameters     │
 │ + ~600 lines │  ───►   │   then    │  ───►  │ predicted spectrum     │
 │ + settings   │         │  predict  │  ───►  │ derived constants      │
 └──────────────┘         └───────────┘  ───►  │ JPL catalog            │
                                                └────────────────────────┘
```

> [!NOTE]
> ERHAM models molecules with **two internal rotors** — here, acetone's two
> methyl (CH₃) groups, which rock back and forth (*torsion*) and split each
> rotational level into closely-spaced sub-levels. You don't need to understand
> the physics to follow along; you'll just run the program and interpret numbers.

---

## Prerequisites

- **Python 3.9+** with **numpy** installed (`python -c "import numpy"` should
  succeed).
- The `erham` package — this repository.
- The example input `ac10x-r3.in` (ships in the project root, one level above
  `python/`).
- A terminal. Commands below use Windows PowerShell; on macOS/Linux swap `copy`
  for `cp` and `\` for `/`.

---

## Step 1 — Set up and verify

Open a terminal in the `python/` directory and confirm the package imports.

```powershell
cd C:\dev\ERHAM\python
python -c "import erham; print('ERHAM', erham.__version__)"
```

You should see `ERHAM 16g-R3-py`. If you get a `ModuleNotFoundError`, make sure
you're in the `python/` folder (the one that contains the `erham/` package).

---

## Step 2 — Look at what you're fitting

The input file describes the molecule, the calculation settings, and the measured
lines. Open `..\ac10x-r3.in` and look at the first few lines:

```text
 ACETONE  LOWEST EXCITED STATE (demonstration new features)        ← 1. title
 -2  -1   1   6   4   1   0. 2.93 0.   100.                         ← 2. control
 3   .0625926982487  1    25.8860386635  1    0.   0                ← 3. rotor 1
 0   60   5000.   50000.   .02                                      ← 4. ranges
 10177.2050877   8502.84897957   4910.25075253                     ←    A, B, C
```

Here's what those control numbers mean — the ones you'll touch in this tutorial:

| Value | Field | Meaning |
|-------|-------|---------|
| `-2` | `ISCD` | Symmetry: C₂ᵥ with two **equivalent** rotors (acetone) |
| `6` | `NIT` | Number of least-squares **iterations** (fit cycles) |
| `4` | `IFPR` | Print option `4` = also write a **JPL catalog** (and prompt for it) |
| `1` | `NUNC` | Compute **uncertainties** of predicted frequencies |
| `2.93` | `DIP(b)` | The b-axis **dipole moment** (Debye) — drives line intensities |
| `100.` | `TEMP` | Temperature (K) for the Boltzmann intensities |
| `0 60` | `JMIN JMAX` | Predict rotational levels for **J = 0 to 60** |

The `10177.2…`, `8502.8…`, `4910.2…` line holds the three **rotational
constants** A, B, C (in MHz) — the headline numbers that define the molecule's
shape. The fit will adjust these (and ~37 others) to best match the measured
lines further down the file.

---

## Step 3 — Make a fast version and run it

The shipped input predicts up to J = 60 and writes a catalog, which takes several
minutes. For a snappy first run, make a copy and change two things: lower the
prediction range, and turn off the catalog prompt.

```powershell
copy ..\ac10x-r3.in acetone-quick.in
```

Open `acetone-quick.in` and edit **two lines**:

```text
line 2:  -2  -1   1   6   4   1 ...   →   -2  -1   1   6   0   1 ...
                          ^IFPR=4                       ^IFPR=0  (no catalog)

line 4:   0   60   5000. ...          →    0   10   5000. ...
              ^JMAX=60                          ^JMAX=10  (predict J ≤ 10)
```

> [!TIP]
> The fit itself doesn't depend on `JMAX` — only the predictions do. So your
> quick run produces the **exact same fitted parameters** as the full run, just a
> shorter spectrum. This is a handy way to iterate quickly while developing a fit.

Now run it:

```powershell
python run_erham.py acetone-quick.in acetone-quick.out
```

You'll see the fit cycles print to the screen:

```text
ERHAM V16g-R3 ...
CYCLE 1
STANDARD DEVIATION 15.032382703978413
CYCLE 2
STANDARD DEVIATION 2.4173476247273444
...
CYCLE 6
STANDARD DEVIATION 2.417277121959847
PREDICTIONS
 IS1 0  IS2 0  IV 1
 ...
```

This takes about a minute. When it finishes, open `acetone-quick.out` — that's
where all the results live. The next steps walk through it section by section.

---

## Step 4 — Read the fit

Scroll (or search) to a block that begins with `NORMAL EQUATIONS`. There is one
per cycle; the **last** one holds the final fitted values. Above each you'll find
the line:

```text
 40 SINGULAR VALUES USED OUT OF 40, STANDARD DEVIATION =  0.15032383D+02
```

The **standard deviation** is the headline quality metric: how well the model
reproduces the measured lines (in MHz). Watch it drop across the cycles —
`15.0 → 2.4` — and level off. That flattening means the fit has **converged**.

Now read a parameter row from the final table:

```text
 STATE    OLD PARAMETER             STANDARD ERROR     CHANGE       PREC     SCALE FAC

    1  RHO1    0.625926982487D-01    0.48480D-04   -0.33036D-06  0.10116D-05  0.100D-02
 1  3  A       0.101772050877D+05    0.26223D-01   -0.24232D-02  0.31589D-02  0.100D+01
```

Reading the `A` (rotational constant) row left to right:

| Column | Value | Meaning |
|--------|-------|---------|
| State / index | `1 3` | Vibrational state 1, fit-variable #3 |
| Name | `A` | The parameter |
| Old parameter | `0.101772050877D+05` | Its value = **10177.2 MHz** (`D+05` ⇒ ×10⁵) |
| Standard error | `0.26223D-01` | 1σ uncertainty ≈ 0.026 MHz |
| Change | `-0.24232D-02` | How much this cycle adjusted it |
| Scale factor | `0.100D+01` | Internal scaling (1.0 here) |

> [!NOTE]
> `D+05` is Fortran scientific notation for `e+05`. So `0.101772050877D+05` reads
> as `0.101772050877 × 10⁵ = 10177.2050877`.

A small **change** relative to the **standard error** is another sign the fit has
settled. By cycle 6, the changes are tiny — acetone is well-determined.

---

## Step 5 — Read the predictions

Search for `PREDICTIONS`, then the first block header
`VIBRATIONAL STATE  1      IS1 0    IS2 1`. (`IS1`/`IS2` label a *symmetry block* —
a group of levels that don't mix.) Right below it are the **energy levels** for
each J:

```text
 J =  0
      5400.283
 J =  1
     18743.236      20114.892      24433.986
```

Each number is a rotational energy level (MHz). A trailing `-` on some levels (you
will see them in other blocks) marks a level that is *antisymmetric* under the
molecule's two-fold rotation — a label, not a minus sign on the energy.

After the levels come the **transitions**. Here is one Q-transition (a `ΔJ = 0`
transition — upper and lower state have the same J):

```text
   1   3   1   1      5690.7499  16   0.00018   1.31550   0.11239   0.11569     0.1356D-01
```

Read it with this map:

```text
   1     3      1     1     5690.7499  16   0.00018  1.31550  0.11239  0.11569   0.1356D-01
  J'    N'    J"    N"      frequency  g     S_a      S_b      S_c     intensity  uncertainty
  └ upper ┘   └ lower ┘     (MHz)     spin   └─ line strengths ─┘     (relative) (1σ, MHz)
              level         transition weight  for the a/b/c dipole
              labels                            components
```

- **frequency** — predicted line position, the difference of two energy levels.
- **g (spin weight)** — nuclear-spin statistical weight of the level.
- **S_a, S_b, S_c** — *line strengths* for the three dipole directions. Acetone's
  dipole lies along the b axis (`DIP(b) = 2.93`), so **S_b** dominates the
  intensity; S_a and S_c are near zero.
- **intensity** — the relative intensity you'd actually observe (line strength ×
  dipole² × spin weight × Boltzmann factor × frequency).
- **uncertainty** — the 1σ error on the predicted frequency, propagated from the
  fit. (This appears because you set `NUNC = 1`.)

`R-TRANSITIONS` (`ΔJ = ±1`, between J and J−1) are listed the same way.

---

## Step 6 — Read the derived parameters

Search for `DERIVED PARAMETERS`. These aren't fitted directly — ERHAM *computes*
them from the fitted constants and propagates their uncertainties:

```text
INTERNAL MOMENTS OF INERTIA (u*A**2)
                                      3.2337118   0.0005069
ANGLES  (A,1)                        30.1371185   0.0146787
F-NUMBERS (F1,F2,F') (MHz)
                                      166921.57       25.06
```

| Quantity | Example | Physical meaning |
|----------|---------|------------------|
| Internal moment of inertia | `3.2337118 ± 0.0005069 u·Å²` | How "heavy" each methyl rotor is, rotationally |
| Angles `(A,1)`, `(B,1)`, `(C,1)` | `30.14°` | Orientation of a rotor's axis vs. the principal axes |
| F-numbers `F1, F2, F'` | `166921.57 ± 25.06 MHz` | Internal-rotation constants (how fast the tops spin) |
| Torsional energy differences | `-17173.8262 MHz` | Tunneling splittings between torsional sub-levels |

Each value carries a **standard error** in the second column, computed by
propagating the fit's covariance matrix — so you know how trustworthy each derived
number is.

---

## Step 7 — Generate the full JPL catalog

Now run the real thing: the original input, which predicts to J = 60 and writes a
catalog. The catalog needs five answers; `run_erham.py` will prompt for them
interactively when it reaches the prediction stage.

```powershell
python run_erham.py ..\ac10x-r3.in acetone-full.out
```

When prompted, enter:

```text
Enter catalog filename !            →  acetone.cat
Enter catalog ID !                  →  58999
Enter Partition function !          →  0.260395e6
Enter first LOG Intensity Cutoff... →  -10
Enter second LOG Intensity Cutoff...→  -10
```

> [!IMPORTANT]
> The full run takes **several minutes** (it predicts thousands of lines with
> uncertainties). The screen will sit on the prediction blocks for a while — this
> is normal. Watch for `IS1 1  IS2 2  IV 1` (the last block) to know it's nearly
> done.

When it finishes, open `acetone.cat`. Each line is one predicted transition in the
standard JPL catalog format (frequency, error, log-intensity, quantum numbers):

```text
    5105.5348  0.0056 -6.5972 3    1.1992 30  589991404 1 1 0 0     1 0 1 0
    5690.7499  0.0136 -6.3523 3    0.6252 48  589991404 1 1 0 0     1 0 1 1
```

This file can be loaded into spectral-analysis tools that read JPL catalogs.

> [!TIP]
> To skip the interactive prompts (for scripting or batch jobs), drive ERHAM from
> Python instead and pass the catalog answers directly:
> ```python
> from erham.driver import run
> run("../ac10x-r3.in", "acetone-full.out",
>     catalog_inputs=("acetone.cat", 58999, 0.260395e6, -10.0, -10.0))
> ```

---

## Summary

You have completed a full ERHAM workflow:

- **Step 1** — verified the package imports.
- **Step 2** — read the input file's control settings and rotational constants.
- **Step 3** — made a fast variant (lower `JMAX`, no catalog) and ran the fit +
  prediction in about a minute.
- **Step 4** — read the fit: the standard deviation converged from 15.0 to 2.4 MHz,
  and you decoded a parameter row.
- **Step 5** — read predicted energy levels and decoded a transition line
  (frequency, spin weight, line strengths, intensity, uncertainty).
- **Step 6** — read derived constants (moments of inertia, F-numbers) with their
  propagated errors.
- **Step 7** — produced the full J = 60 spectrum and a JPL catalog.

---

## Next steps

- **Understand the internals** — the [architecture & reference](architecture.md)
  explains the pipeline, the Hamiltonian, the symmetry logic and every module.
- **The full input specification** — the original user manual `erham.txt` (project
  root) documents every input field and option.
- **Try your own changes** — edit `acetone-quick.in` and re-run to see the effect:
  raise `TEMP` to 200 K and watch the intensities shift; set `NIT` to `0` to skip
  the fit and predict from the input constants directly; or narrow `FMIN`/`FMAX`
  to focus on one frequency band.
