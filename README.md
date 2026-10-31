# Symbolic Regression of Dark-Matter-Only Auriga Halo Distribution Functions

This repository measures the isotropic distribution function f(E) of the 30 dark-matter-only
Auriga haloes, scales the curves to a common dimensionless form, and searches for analytic
expressions with symbolic regression (PySR) under cross-validation. It contains the code, the
processed results, and one notebook for every figure and table in the paper.

## Getting the simulations

The haloes are the 30 level-4 dark-matter-only runs (`Original_DMO/4`) of the Auriga public
release ([Grand et al. 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.532.1814G)). The data
is distributed through [Globus](https://globus.org): the collection is "The Auriga Simulations"
on the endpoint "MPCDF DataHub Virgotng", and the steps are described at
<https://wwwmpa.mpa-garching.mpg.de/auriga/data.html>.

For each halo you need the snapshots and group catalogues from snapshot 77 to 127 (the bound
particles are selected from their orbits over that range) and, for Fig. 8 only, the merger
trees `trees_sf1_127.*.hdf5`. Point `AURIGA_DATA` at the folder that holds the haloes:

```bash
export AURIGA_DATA=/path/to/Auriga
```

The code looks for `Halo N/` (or `Halo_NN/`) folders containing `Snapshots/` and `Groups/`,
either directly or one level down in a `Z*/` folder. If the release unpacks differently, a few
symlinks into this shape are enough.

You do not need the simulations to reproduce most of the paper: the processed results in
`AurigaDFs/Halo_Results/` (DFs, density fits, cross-validation fronts and the global fit) are
part of the repository. Only Figs. 1 and 8, and the particle profiles of Fig. 7, read particles.

## What is where

```
AurigaDFs/
  Empirical_DF_Construction/
    io/               reading snapshots and group catalogues; all paths (results_paths.py)
    bound_selection/  bound particles, from pericentre/apocentre sign flips
    core/             DF, density fits, scaling, Eddington inversion, SR complexity, physicality
    pipelines/        one script per stage, run in the order below
    paper_figures/    one notebook per figure or table
  Halo_Results/       outputs (the small final products are tracked; particle IDs are not)
  tests/              tests of core/physicality.py
```

No path is hard-coded. Everything resolves through `io/results_paths.py` or these variables:

| Variable | What | Default |
|---|---|---|
| `AURIGA_DATA` | the simulations | needed only to rebuild from particles |
| `HALO_RESULTS` | where results are read and written | `AurigaDFs/Halo_Results` |
| `CV_DIR` | the cross-validation run used by the figures | `$HALO_RESULTS/_sr/cv_with_hubble_flow_nfw_ref02` |
| `AURIGA_TREES` | merger trees, if kept outside `AURIGA_DATA` | – |
| `GNUPLOT` | gnuplot for rendering figures (with pdflatex on `PATH`) | `gnuplot` on `PATH` |

## Rebuilding the results

Run from `AurigaDFs/` in a Python environment with numpy, scipy, sympy, pandas, h5py and PySR.
Each stage reads what the previous one wrote to `Halo_Results`, skips work already done, and
redoes it with `--force`.

```bash
P="python -m Empirical_DF_Construction.pipelines"
H=$(seq 1 30)          # all haloes (in zsh, write --halos $(seq 1 30))

# 1. bound particles, density profiles and NFW/Einasto fits, raw DFs   (needs AURIGA_DATA)
$P.build_ids              --halos $H
$P.build_density_profiles --halos $H --method with_hubble_flow
$P.build_df_raw           --halos $H

# 2. dimensionless DFs, and every halo in Halo 2's units
$P.build_df_dimensionless --halos $H --ref-halo 2

# 3. cross-validation folds, then PySR on every fold and change of variable
$P.build_cv_folds
CV=Halo_Results/_sr/cv_with_hubble_flow_nfw_ref02
Empirical_DF_Construction/pipelines/run_pysr_cv.sh --cv-dir $CV

# 4. back to eps_hat, score on held-out haloes, physicality, families, global refit
$P.compose_pysr_fronts  --cv-dir $CV
$P.evaluate_pysr_cv     --cv-dir $CV
$P.check_physicality_cv --cv-dir $CV
$P.merge_pysr_families  --cv-dir $CV
$P.fit_families_global  --cv-dir $CV
```

The other particle selections compared in Figs. 1-2 use the same stages with
`--method without_hubble_flow`, plus `naive_radial_selection`, `build_density_naive_radial` and
`build_df_raw_fixed_point`. `check_against_catalogue` compares the bound sets with the
SUBFIND/FoF catalogue (masses, V_max, velocity dispersion, virial ratio).

## Figures and tables

Run the notebooks in `paper_figures/` from that folder. They write figure data to
`paper_figures/output/`; `table3_fig6_generalised_forms` goes first, since Figs. 7 and 8 read
its output.

| Notebook | Makes | Needs particles |
|---|---|---|
| `fig1_density` | Fig. 1 | yes |
| `fig2_selection` | Fig. 2 | |
| `fig3_4_self_similarity` | Figs. 3, 4 | |
| `table2_winners` | Table 2 | |
| `table3_fig6_generalised_forms` | Table 3, Fig. 6 | |
| `fig7_table4_physical_consequences` | Fig. 7, Table 4 | for the halo profiles |
| `fig8_radial_motion` | Fig. 8 | yes |

## Tests

```bash
python tests/test_physicality.py      # or: pytest tests
```
