# RAIL-3

RAIL-3 is a research code and evidence snapshot for studying the gap between
residual prediction fidelity and sparse prompt-repair intervention utility in
promptable segmentation.

The scientific sequence is **exact accounting → residual fidelity → STOP-relative
fidelity → action choice → sparse allocation → intervention utility**. Experiments
use frozen SAM 3.1, VOC development and COCO external evaluation. Six COCO models
have different descriptive orderings for MAE, DRRE and intervention utility;
better residual prediction does not automatically establish sparse utility.
The saved V6 terminal is **MIXED** and COCO classification **GAP_REPLICATES**.
These labels do not imply zero effect or failure of every model at every budget.
Positive and negative cells and their interpretation are retained in the
[scientific evidence guide](docs/scientific_provenance.md).

Snapshot version: **v0.1.0**. Target repository:
[ethanalan60-dot/RAIL-3](https://github.com/ethanalan60-dot/RAIL-3).
The [MIT licence](LICENSE) covers author-owned original code only;
[third-party terms and exclusions](THIRD_PARTY_NOTICES.md) remain separate.

## What can be reproduced here?

**This snapshot does NOT provide a turnkey reproduction of all historical
experiments.**

| Level | Scope | Status |
| --- | --- | --- |
| L1 | Saved evidence and nonphoto figure reconstruction | PARTIAL |
| L2 | Evaluation of frozen predictions | NOT RUN / INPUT AND AUTHORITY INCOMPLETE |
| L3 | Full experimental execution | NOT RUN / INPUT, RUNTIME AND AUTHORITY INCOMPLETE |

The source map binds 19 tables and 12 figure identities. The full six-model
external panel, both sparse semantics and the six R3 zero-cost VOC cells remain.
A historical path/hash identifies a source; it does not make absent predictions,
scalers, checkpoints or panel manifests obtainable. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact inputs and source limitations.

Figure 2 is supplied as a nonphotographic vector schematic. The saved-summary
renderer rebuilds statistical displays for Figures 3–6 without estimating new
metrics or intervals. Figure 1 and Q1–Q6 photographic plates are excluded;
their numerical annotations, original case identities and repeat-state limits
remain documented. No replacement cases are selected.

## Start with engineering checks

Use Python >=3.12. The minimal runtime requirement is `numpy>=1.26,<3`;
the build backend requires `setuptools>=68`. No command below downloads data,
PyTorch, CUDA, SAM or weights.

```bash
python scripts/release.py verify
python scripts/release_entrypoints.py --help
python scripts/release_entrypoints.py --list
python -m pip install -e .
python scripts/release_smoke.py
python -m unittest tests.test_release_gateway tests.test_release_entrypoints -v
```

**INSTALLATION_TEST_STATUS = BLOCKED_BY_BUILD_DEPENDENCY_ACQUISITION** in the
recorded local Python 3.12.3 environment. Official dependency downloads timed out;
wheel/editable installation was not certified. This is a dependency-acquisition
limitation, not evidence of software functional failure. Existing-environment
source tests passed: 22 unique artificial tests. Fresh standard-library tests
passed 11 fixtures, while the fresh NumPy suite remained unavailable. See
[INSTALLATION.md](INSTALLATION.md) for actual scope and separate build/install
recipes. A source test is not an installed-wheel import or a scientific rerun.

The standard-library adapter can inspect configuration and diagnose missing inputs:

```bash
python scripts/release_entrypoints.py --entrypoint v6_classify --validate-only --config release_engineering_inputs.example.json --run-id inspection-001 --output release-runs/inspection-001
```

The supplied null-location example returns exit **3**. Identity/configuration
errors return **2**. Even exit **0** establishes engineering structure only:
input payloads and historical identities are not verified, scientific readiness
is NOT_ESTABLISHED, and no experiment or output directory is created.
`scripts/release.py` remains verify/plan only. The
[entry-point table](docs/entry_points.md) distinguishes this adapter from original
scientific drivers whose --check or preflight may read scientific inputs and run
analysis. Seven recovered historical files retain exact source identities in
[SOURCE_TO_RELEASE.json](SOURCE_TO_RELEASE.json); their presence does not complete
the historical runtime/authority closure.

## Saved nonphoto displays

```bash
python -m pip install -e '.[presentation]'
python presentation/plot_saved_summaries.py --check-inputs
python presentation/plot_saved_summaries.py --output build/saved-summary-displays
```

The output directory must be new. This reads six bound saved summaries and
reformats their existing values/intervals; it does not reconstruct results from
prediction arrays. See [presentation details](presentation/README.md).

## Scientific boundaries, citation and responsibility

A0–A6, candidate-level atoms, count-matched RECT/UNION controls, state-only
first-batch supervision, image-group splits/seeds/scalers, optimization budgets,
missingness, oracle denominators, paired intervals, pre-GT amendments and E1–E5
are preserved. RC-001's two sources, VH3 nonreplication, historical panel use,
GT-access timing, six source limitations and Track B NOT_EVALUATED remain
explicit. Public Git history begins at release preparation and does not replace
historical experiment identities; see [provenance](RELEASE_PROVENANCE.md).

[CITATION.cff](CITATION.cff) preserves the confirmed author order and cites one
unified unpublished manuscript, without inventing a DOI or journal acceptance.
Raw VOC/COCO images/labels, checkpoints, prediction arrays, third-party reference
PDFs, unauthorized photographs and the full photographic manuscript are excluded.

AI-assisted tools, including ChatGPT and Codex (OpenAI), were used to support code development and debugging, preparation of reproducible visualization scripts, and manuscript drafting, organization, and language editing. These tools provided suggestions and implementation support under the authors' direction. The authors determined and approved the experimental design, made the final methodological and analytical decisions, and approved the interpretation and conclusions. The authors retain full responsibility for the correctness, integrity, and reproducibility of the work.

The same disclosure is preserved in [AI_USE.md](AI_USE.md). The authors declare
no competing interests relevant to this work. Code publication is not a claim
of complete experimental reproduction.
