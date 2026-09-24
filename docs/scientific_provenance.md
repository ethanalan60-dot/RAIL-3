# Scientific evidence and provenance

This is a local release candidate drawn from the existing research archive, not a
new experiment or a publication. Redistribution licensing remains pending. The
archive and its historical records remain separate from this export; a future
release commit cannot stand in for an experimental freeze date.

The source-linked evidence concerns a fixed prompt-repair decision with frozen
SAM 3.1, prospective features, registered VOC development and frozen COCO
transfer. VOC V6 remains **MIXED**. The existing COCO synthesis remains
**GAP_REPLICATES**, with **1 recorded GT read** and
**48 saved performance rows**. No scientific metric,
interval, model, prediction, action, criterion or sample was recomputed or selected
for this export. These historical totals are distinct from zero new scientific
runs during release preparation.

## Reader entry points

- [Table/figure source map](../evidence/source_map.json): all 19 tables and 12
  manuscript figure identities, experiment IDs, original digests, bound original
  display positions, public-candidate paths, transformations and exclusions.
- [Claim scope](../evidence/claim_scope.json): current permitted claims and limits.
- [Saved summaries](../evidence/summaries/): full saved paired values, both COCO
  sparse semantics, all six external models, all six R3 zero-cost VOC cells,
  F0/F1 precision, and FIT-only sensitivity. Values retain their saved precision.
- [Scientific method excerpts](../evidence/methods/): definitions, five propositions
  and proofs, architecture/input/weighting details, fold/seed/scaler rules,
  historical diagnostic limits and source-specific implementation identities.
  Original cross-reference labels remain in these excerpts; this is not the
  complete manuscript or a publisher-ready TeX package.
- [Historical authorities](../evidence/authorities/): the saved 48-row external
  core CSV, E1–E5 registry, missingness result, prediction lock/checkpoint identity
  record, and original E4/E5/sparse contracts. The copies retain historical
  pre-GT counters and null result fields where they occurred.
- [Presentation instructions](../presentation/README.md): separate saved-summary
  rendering with no scientific recomputation. Originals and re-renderings are
  distinguished; no marginal model CI or new ranking test is supplied.

Every direct copy has an original and destination SHA-256 in the source map.
A `historical_source_path` identifies a record cited by an existing summary;
**HISTORICAL_IDENTITY_ONLY** means that original is not distributed here, not
that it was read during this export or is publicly downloadable. Accessible
package paths point to the supplied field copies or summaries. A copied summary
can preserve a scientific value and its identity without supplying the original
prediction arrays required to regenerate it independently.

## Method identity and information boundary

The canonical IDs remain R0_SMALL_P, R0_CM_P, R1_P, RECT_P, UNION_P and R3_P
for the external panel; their display aliases are G-small, G-match, Atom, Rect,
Union and Relative. Aliases are not extra models. The model table and source
bindings retain the additional VOC-only decision and observation families.
The historical LL4TTA_P task adaptation adds no independent training jobs.

The prospective inputs contain 41 state, 16 action and 27 region coordinates.
Local predictors share regional encoding, unweighted mean/max context pooling
and a full-raster area-weighted residual sum. This prediction weighting differs
from target-valid accounting weights. First-batch R1/RECT/UNION use state-only
supervision; R2/R3 are complete training-recipe comparisons. Regional output is
not a demonstrated local-calibration guarantee. RECT is region-count-matched
guillotine partitioning, not per-region area matching; descriptor contents
also differ, preventing a boundary-only causal interpretation.

A0 and A1 have identical canonical/normalized text in the fixed vocabulary;
their registered call identities remain distinct, without assuming equal masks.
Atoms use every retained candidate in the observed calls, including connected
components of the zero signature, not three bits from three action-union masks.
STOP retains the union of all retained A0 candidates. The maximum-score source
candidate is used to plan applicable repairs; it does not replace that union.
The exact A0–A6 prompts, crop/box/point/flip/restoration rules and distinctions
between a valid empty outcome, structural infeasibility and technical failure
are retained in the method excerpts and original code/config identities.

Five outer image-group folds keep all states of an image together. Seeds are
13, 37 and 71. Inner stopping and prospective scaling use only training-side
groups; continuous seed outputs are combined before decisions. The frozen V6
recipe uses AdamW, learning rate 0.0005, weight decay 0.0001, at most 120 full
optimization-partition steps, patience 15 and a strict 1e-8 improvement rule.
It does not imply convergence. External full fits use the precommitted family
median epochs (120 for the first five external families, 108 for R3), with
all-FIT scalers and no COCO scaler fitting. The prediction lock retains all
checkpoint identities and seed ordering without distributing or reading weights.
The known predictor-refit environment is not evidence of the historical SAM
trajectory-generation environment.

## Metrics, denominators and uncertainty

The exact accounting identity concerns target-valid mismatch counts on a
partition. It is not a learning or intervention-performance guarantee. The
current MAE and DRRE pool defined action errors and STOP-relative pair errors
by their respective record counts; they are not averages of state means.
The exact legacy PDF printed a state-balanced DRRE formula, now verified in
its own version. Its executed historical reduction remains UNKNOWN. The
[definition/model crosswalk](../evidence/methods/definition_and_model_crosswalk.md)
keeps these estimands and R1/R1_P identities distinct. The legacy PDF itself is
not included in this release candidate.

Normalized regret is the ratio of summed selected regret to summed STOP
regret on the relevant complete-oracle population. A zero oracle denominator
is undefined. COCO utility improvement and its interval concern **1 − normalized
regret**, not raw signed gain, pixel counts or IoU improvement. Sparse gain
capture divides signed selected gain by the sum of the largest K nonnegative
oracle gains on the same eligible population. That top-K denominator differs
from full-population oracle headroom. Negative realized gains remain included.
FORCED_K takes the exact frozen K; POSITIVE_GAIN_CAP may use fewer states but
retains the same budget's oracle denominator.

The saved image-group paired intervals use 2000 resamples,
seed 13, PCG64 and linear 0.025/0.975 percentiles. Sampled image groups retain
all associated records and multiplicities. These are pointwise contrast
intervals conditional on fixed trained models, not intervals on individual
model means, independent folds/seeds, or simultaneous familywise guarantees.
Practical tie requires complete interval containment within the frozen
comparator-relative ±5% band; nonzero effects and intervals excluding zero
can still satisfy it. E3 comprises four comparisons, not a new overall gate.

## Stage-specific rules and outcomes

V6's utility translation rule requires its normalized-regret condition and
**all six** 1%, 2%, 5% cells under both sparse semantics to satisfy its positive
headroom and magnitude conditions. R3's existing signed gains are positive
in all six cells, but the 1% gain captures miss the V6 magnitude threshold.
The saved V6 MIXED terminal and original interpretation are retained without
reclassification.

COCO E4 keeps R3 as its predetermined model. A–D jointly require normalized
regret below one, a strictly positive lower bound for 1 − normalized regret,
a predicted non-STOP decision, and positive realized gain/headroom/capture
at **at least one** primary FORCED_K budget. The recorded booleans are
A=true, B=false, C=true, D=true, hence E4_NOT_SUPPORTED. E1/E2 support and
this E4 boolean yield E5 GAP_REPLICATES. E5 is a deterministic synthesis,
not independent statistical evidence. Positive-cap results cannot change
E4/E5. E4_NOT_SUPPORTED is not zero effect or necessary harm. Some other
models and budgets have positive capture; no all-model/all-budget failure
claim is supported. VOC-to-COCO displays are descriptive, not a new test of
a domain difference.

## Prediction freezing and first target access

| Event | Existing recorded UTC | Recorded phase |
| --- | --- | --- |
| Prediction lock | UNKNOWN | Before E4 operationalization |
| E4 operationalization | 2026-09-05T14:29:36Z | Post-prediction, pre-GT amendment |
| E4 FORCED_K binding | 2026-09-05T16:56:21.834821Z | Post-prediction, pre-GT binding |
| E5 synthesis contract | UNKNOWN | After prediction lock, before GT |
| First COCO GT read | 2026-09-05T17:29:07.825063Z | External target access |

The executable E4 operationalization and FORCED_K primary binding were
post-prediction, pre-GT additions. “Registered” identifies repository-recorded
contracts at their stated stages, not public preregistration or a complete
pre-prediction protocol. Unknown prediction-lock/E5 UTC fields are not inferred
from file timestamps or new release commits. Historical zero counters in the
contracts are not current COCO totals.

The official runner completed evaluation and scientific QA before exiting with
code 1 because its working copy lacked the ledger. The ledger was later restored
from the historical prefix and unchanged result append, without reevaluation.
The original exit code remains 1. The exact manuscript account is retained in
[external contract and history](../evidence/methods/external_contract_and_history.tex).

## Technical missingness and label blindness

The designed COCO panel contains 20000 states. One trajectory-level
failure and three A4 technical failures affect 4 states. Fidelity
retains 44414 defined action cells and 24415 STOP-relative
pairs across 19999 states. Complete-oracle and sparse utility use
19996 states; all 1000 image groups remain in the
resampling universe. Missing values are not zero and technical failure does
not make a repair structurally infeasible.

The three A4-failure states have offline availability-adjusted choices after
failure was observed but before GT access. This is label-blind, not pre-action
observable. Those states remain in the defined-prediction diagnostic population,
where only non-STOP choices count; they do not enter complete-oracle or sparse
utility and support no deployment-utility claim for failure-aware selection.
The technical-missingness authority retains the original state identities and
exact exclusion rules; no masks or raw labels are supplied.

## Historical limits retained

The seven diagnostic dimensions—scale, action composition, image-group folds,
classes, cost/budget, observation and mechanism—remain scientifically distinct.
Historical visual-feature EXP-011–013 were executed but are not the current
external panel. Historical retrospective runtime/geometry/descriptor boundaries
prevent treating V6 as a pure runtime single-factor intervention. Validation40
had earlier baseline/validation uses; the later locked transaction does not
make its images untouched across the project. V6 is registered FIT development,
not pristine held-out confirmation. EXP-032 remains execution-unverified and
adds no assumed training batch.

The finite-action propositions retain common feasible sets, identical cost
terms, fixed ties, strict STOP conditions, the unique-optimum margin assumption,
the singleton case and the expectation-bound conditions. Small mean MAE/DRRE is
not a statewise sup-error guarantee. The favorable FIT margin association did
not replicate in validation40 VH3; this contrary result remains disclosed.
Track B is NO_GO / NOT_EVALUATED / NOT_SCIENTIFIC_FAILURE and contributes no E5
input. No unavailable result is treated as a scientific failure.

RC-001 preserves separate primary decision/target-rich and utility-rescoring
sources. Its exact values and different roles remain in numeric_sources.json,
[the RC-001 paragraph](../evidence/methods/rc001_scope.tex) and the corresponding
saved summaries. They are not averaged, replaced or assigned an invented cause.
FIT foreground/alternative-utility analyses remain post-hoc and do not become
COCO multi-utility validation.

Six known limitations remain: exact F0/F1 evaluable/pair denominators; the
Error change action pair; the SAM trajectory environment; exact prediction-lock
and E5 UTC; historical executed DRRE reduction; and the remaining historical
nonzero-cost grid. None is marked repaired merely because this package exists.
These gaps restrict their corresponding claims without requiring a new full
project audit, sample selection or scientific rerun for this local export.

## Release transformations and reproducibility limits

The two exact host-location metadata replacements affect the historical R3
source-path field and historical interpreter-location field only. The complete
original file digests, selectors and replacement explanations remain in the
source map; original strings remain in private local export records. All other
bytes in those two copied files are retained, including numeric literal
precision. No identifier used by a state, prediction, checkpoint or scientific
hash namespace is renamed.

L1 is saved-summary presentation: the evidence permits inspecting table values
and rebuilding the supplied numerical displays. Photo-based displays cannot be
fully reconstructed from this candidate. L2 (evaluation from frozen predictions)
and L3 (complete retraining/trajectory generation from original inputs) require
additional licensed artifacts and are not validated by summary rendering or
synthetic tests. No full end-to-end reproduction is claimed. The current release
check record must distinguish actual checks from NOT_RUN steps.
