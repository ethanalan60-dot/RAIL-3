# Saved-summary presentation

This local release candidate contains existing nonphotographic Figures 2–6
in `original_figures/`. Their bytes match the manuscript assets. They remain
local, with redistribution licensing pending. Figure 1 and the Q1–Q6 plates
are excluded because third-party image rights have not been confirmed.
Their existing numerical annotations and scientific limits remain in
`../evidence/summaries/qualitative_cases_source.csv` and the case notes.

The portable script below plots only six digest-bound saved input files. It
re-renders the statistical content of Figures 3–6 with a compact new layout;
it does not promise byte-identical rendering of the historical figures.
Figure 2 is an unchanged illustrative schematic supplied as a static vector,
not a measured mask or an executable experimental result.

```bash
python presentation/plot_saved_summaries.py --check-inputs
python presentation/plot_saved_summaries.py --output build/saved-summary-displays
```

The output directory must be new and separate from the saved sources. Python
and Matplotlib are sufficient; no model, SAM, raw target, prediction array,
network access, external archive, bootstrap routine or scientific evaluator
is imported. Font files are not distributed. `requirements.txt` records the
presentation dependency, not the historical SAM runtime environment.

The release source map is `../evidence/source_map.json`. The original 19
scientific table fragments are in `../evidence/tables/`, with their original
number macros and exact saved summaries. They are evidence fragments, not a
standalone manuscript: original equation/section cross-reference labels are
retained, and the complete manuscript and publisher template are not included.

## Display rules

- Fidelity intervals are paired image-group contrast intervals. They are
  not marginal error bars on individual model means. The local-control band
  is the frozen ±5% margin; the script reads already-saved relative display
  values and does not estimate the margin or reclassify practical ties.
- All six COCO models remain in the objective and sparse displays. MAE,
  DRRE and normalized regret have separate units. Points use the original
  printed precision without new ranking or significance tests.
- Sparse curves retain negative capture and fixed budgets. Budget/capture
  percentages are display conversions only. V6 and COCO have distinct
  populations and criteria; their side-by-side display is descriptive.
- The only utility interval plotted is the predeclared R3 interval for
  `1 - normalized_regret`. It is not an interval for raw gain or IoU.
  POSITIVE_GAIN_CAP remains available separately in the evidence summaries;
  it does not determine COCO E4 or E5.
- FIT composition and alternative-utility results are post-hoc. The two
  RC-001 sources remain separate; no averaging or explanation of their
  discrepancy is introduced.

This directory's existence is not a publication or a validation receipt.
See the release's actual check record for commands that were executed.
