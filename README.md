# SHIELD-W — course-project replication and Overleaf package

This package extends the supplied SHIELD project with a finite root-zone water
budget, seven-day dry-down, irrigation timing/budgets, independent trait ablation,
36 paired sensitivity settings, and explicit numerical/conservation checks.
The results are controlled synthetic experiments, not validated city forecasts.
The contribution is the original comparative analysis, not a claim that plant
hydraulics, urban ecohydrology or irrigation modelling was invented here.

## Read or edit the reports

- `report_short.tex`: course-length main report (two-page compiled version).
- `report_detailed.tex`: full mathematical model, results interpretation,
  assumptions, citations, verification and limitations.
- `references.bib`: verified paper/book/repository references.
- `preamble.tex`, `results/*.tex`: shared formatting and generated report values.
- `figures/`: all original simulation figures as editable-use PDF and PNG assets.

Upload this entire folder to Overleaf, preserving subdirectories. Select
`report_short.tex` or `report_detailed.tex` as the main document and use pdfLaTeX
with BibTeX (automatic with Overleaf's normal build). The Python code does not
run on Overleaf; the completed results and figure assets are already included.
Both reports end with an equal-contribution statement for D, H, M and S.
An AI-assistance disclosure is included in both reports.

## Reproduce locally (Python 3.12, tested)

```bash
python -m pip install -r requirements.txt
python -m unittest test_model.py -v
python run_experiments.py
python build_report_assets.py
python verify_outputs.py
```

The experiment script already calls the report-asset builder at the end.
Rerun `build_report_assets.py` after editing CSVs/figure presentation;
`verify_outputs.py` checks the specific experiment matrix supplied here. No
internet or external datasets are required after installing dependencies.
To compile PDFs locally, install a TeX distribution and run:

```bash
latexmk -pdf report_short.tex
latexmk -pdf report_detailed.tex
```

`shield.py` contains the model; `run_experiments.py` contains the experiment
matrix and figures; `build_report_assets.py` converts output tables into LaTeX
and renders the architecture. No random draws are used, so no seed is needed.
The exact execution environment and source hashes are in `results/metadata.json`.
Runtime is machine dependent (see the recorded elapsed seconds there).

## Result files and units

- `static_factorial.csv`: 12 prescribed-potential trait × LAI × drought cases;
  cooling is the 14–16 h mean at exactly 2 m, K (same numerical difference as °C).
- `resilience.csv`: dry/wet total and transpiration-increment cooling ratios.
- `static_timeseries.csv`: step-level static baseline/control traces.
- `dynamic_summary.csv`: finite-water cases, seven-day cooling service (K h),
  total transpiration and delivered irrigation (mm per ground area), final-day
  14–16 h cooling, final potential, incremental irrigation benefit (K h/mm).
- `dynamic_timeseries.csv`: full dynamic traces at 600 s intervals. `transp_mm`
  is the interval amount, NOT a rate; `theta_end` is end-of-interval content;
  `hour`/`time_h` label forcing midpoint. Temperature is the post-step solution.
- `independent_traits.csv`: 2×2 Kp/P50 dynamic ablation at LAI 3, budget 12 mm.
- `sensitivity.csv`: 36 prespecified configurations, each with paired traits.
  These are scenario responses, not independent field replicates or a posterior.
- `cutoff_audit.csv`: soil-storage cutoff dependence, including −4/−6/−8 MPa.
- `convergence.csv`: time/vertical/search resolution; both traits are checked.
- `splitting_audit.csv`: old versus revised linear transport integration with
  revised physics held constant; not a rerun of the complete original model.
- `run_audit.csv`: every model call and all configuration/diagnostic fields.
- `legacy/`: unchanged supplied numerical CSVs for provenance; not solver inputs.

All main simulations use one-sided leaf LAI. Transpiration is reported per ground
area. 1 mm = 1 kg/m² = 1 L/m². Leaf rates inside the model use mol H2O/(m² leaf s).
Cooling service integrates signed bare-minus-full temperature from 10 to 18 h
for seven days, with no nighttime contribution or clipping of negative cooling.
The model does not compute thermal comfort, mortality, carbon storage, pumping
energy or citywide avoided energy use.

## Important changes from the supplied prototype

1. Prescribed potential is retained for the static control, but dynamic cases
   deplete a van Genuchten root-zone bucket and receive budgeted irrigation.
2. Linear diffusion and ventilation use a combined matrix-exponential update.
   Nonlinear leaf feedback remains explicitly refreshed and is resolution-tested.
3. Every water integral uses actual interval amounts; the previous 540/600 s
   record/integration mismatch is removed.
4. Shortwave interception is conservative across finite layers, and one shared
   attenuation coefficient is used for transmission and ground-shading forcing.
5. A vectorized feasible-grid profit search includes the no-transpiration option;
   it replaces invalid-plateau local optimization. It is a discrete approximation.
6. A common 6 MPa drawdown domain replaces the former trait-dependent potential
   guard. The model does not claim that this domain is a measured safety margin.
7. One-day atmospheric spin-up, exact 2 m interpolation and interval-window means
   replace cold-start/nearest-cell/nearest-time comparisons.
8. Ground evaporation is disabled in all counterfactuals (mulched-surface
   assumption), isolating counted plant water demand. Ground heat and longwave
   are still prescribed; global urban energy conservation is not claimed.

As several aspects changed, revised absolute values should not be mixed with
old report numbers. Absolute cooling and the rankings depend on the selected
closures. The underlying P-Hydro and UT&C repositories were consulted as
references, not executed, imported, or bundled.


## Verification pass (25 October 2026)

The full experiment matrix was rerun, and the test suite was expanded from
6 to 11 checks. Independent scalar calculations verify the vectorized leaf
search; a separate ODE solver checks atmospheric transport. Diagnostic and
reproducibility fixes remove the hard-coded metric interval, record complete
plant parameters and unique run IDs, and include non-transpiring leaf-energy
residuals. The numerical results and conclusions are compared with the prior
package in `results/verification.json`; see `VERIFICATION.md` for scope/limits.
