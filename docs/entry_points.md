# Entry-point capabilities

The seven exact omitted resources are now restored. This closes the named
source omissions, not the complete scientific runtime or original launch-lock
transaction. See [capabilities](../ENTRYPOINT_CAPABILITIES.json) and
[per-file identities](../SOURCE_TO_RELEASE.json). Original scientific
scripts were not changed to make help pass.

## Engineering and presentation routes

| Entry | Help / validate-only capability | Actual scope |
| --- | --- | --- |
| `scripts/release.py` | Standard-library help, verify, plan | Supplied checksum/syntax checks and a location plan. It retains its PLAN_ONLY/BLOCKED role and has no scientific launcher. |
| `scripts/release_entrypoints.py` | Standard-library help/list/validate-only | New adapter: source SHA/AST, configuration fields, distribution-presence metadata and lexical input/output checks. Missing resources exit 3; identity/config errors exit 2. Even engineering exit 0 leaves scientific readiness NOT_ESTABLISHED and historical verification NOT_PERFORMED. |
| `scripts/release_smoke.py` | Help before NumPy import; 11 artificial tests | NumPy is required for serialization, tiny atomic partitions and artificial E4 boolean tests. No scientific result is evaluated. |
| `python -m unittest tests.test_release_gateway tests.test_release_entrypoints -v` | 4 prior location tests plus 7 new adapter tests | Standard-library fixtures only; no discovery of historical research tests. |
| `presentation/plot_saved_summaries.py` | Standard-library help and saved-input hash check | Rendering needs optional Matplotlib and the six supplied summaries. Earlier rendering PASS remains historical; this round does not re-render or recompute results. |

Fresh Python 3.12 environments failed to acquire setuptools/NumPy because the
normal official endpoints timed out. Standard-library checks and missing-input
diagnostics still work. Source tests in the existing NumPy environment cannot
establish a successful fresh wheel/editable installation. The actual build/install and artificial-test scope is documented in
[INSTALLATION.md](../INSTALLATION.md); earlier NOT_RUN checks are not relabelled
as successful installation.

```bash
python scripts/release_entrypoints.py --help
python scripts/release_entrypoints.py --list
python scripts/release_entrypoints.py --entrypoint v6_classify --validate-only --config release_engineering_inputs.example.json --run-id inspect-001 --output release-runs/inspect-001
```

The example has null locations: exit 3 is the correct diagnostic. It opens and
probes no input payload, creates no output directory, and imports no historical
scientific driver. User-supplied paths/hashes are not independently verified
contents or access permission. Distribution metadata is not version, CUDA or
import compatibility. Listed code/configuration checks do not certify every
transitive import or original historical integrity file. These limits are also
returned in JSON. The verified planner is compiled from its exact hash-checked
source bytes, without accepting a separate bytecode cache.

## Historical source routes

| Source entry | Source / native-help status | Resources and execution boundary |
| --- | --- | --- |
| `scripts/run_tmlr_v6_first_batch_crossfit.py` | Exact source restored; native help NOT_RUN | Imports the training stack and checks CUDA before parsing help. Requires original FIT/fold/bundle/implementation authority and run layout. It does not consume action-cost tables. |
| `scripts/run_tmlr_v6_phase2_second_batch_crossfit.py` | Exact source restored; native help NOT_RUN | Training stack, fixed GPU/run layout, phase-2 launch lock and registered inputs/objectives/source guards. |
| `scripts/run_tmlr_v6_phase2_f0_f1_crossfit.py` | Exact source restored; native help NOT_RUN | CUDA check before parsing, phase-2 views/bundles/cost/fold/launch identities and original output paths. |
| `scripts/evaluate_tmlr_v6_phase2_f0_f1_fit.py` | Exact source restored; native help NOT_RUN | OOF predictions, targets/bundles and scale/fold/cost/view/launch identities. Its --check recomputes scientific outputs/bootstrap. |
| `scripts/evaluate_tmlr_v6_phase2_sensitivity_fit.py` | Exact source restored; native help NOT_RUN | Original OOF/bundle/confusion/cost/launch records. Its --check recomputes sensitivity results. |
| `scripts/build_tmlr_v6_phase2_qualitative_atlas.py` | Exact source restored; native help NOT_RUN | Torch/Pillow and original FIT images/labels/candidates. Even --check performs original case selection and raw image/label reads. It is not the saved-summary renderer. |
| `scripts/classify_tmlr_v6_phase2.py` | Three missing import files resolved; native help NOT_RUN | Torch/Pillow imports, complete reports/launch authority and historical tests; can recompute comparisons/bootstrap. The release adapter executes none of this and hardcodes no classification. |
| `configs/experiments/voc_m06_pilot_b_training_v1.json` | Exact constant-bound resource restored | Historical support resource, not a CLI or additional current experiment. |
| `scripts/evaluate_tmlr_v6_first_batch_fit.py`, `evaluate_tmlr_v6_phase2_second_batch_fit.py` | Existing source snapshots | Frozen OOF/target/fold/cost identities and statistics. Their --check is not engineering validation. |
| `scripts/train_tmlr_v9b1_fullfit.py` | Existing source; only --help is checked | Standard-library parameter parsing precedes I/O. --check/--worker/--finalize have real input/checkpoint/training effects and are not run. |
| `scripts/build_tmlr_v9b1_label_free_features.py` | Existing source; only --help with NumPy is checked | --check reads real candidate-cache/feature payloads. |
| Other feature/trajectory/prediction/COCO evaluation sources | Existing source snapshots; native execution NOT_RUN | Exact panel/FIT/fold manifests, caches, checkpoints/scalers, predictions/targets and runtime/source authority are required. V9C preflight reads scientific payloads and is not a safe release validation command. |

The original scientific functions exist. Future use requires their complete,
authorized inputs, environment and historical authority, or a separately
engineered new-run adapter that preserves semantics and uses fresh outputs.
This engineering adapter is not an unconditional future prohibition. Fixed
historical output/GPU/source identities prevent a truthful portable launcher
from being obtained by path substitution alone. No original guard was bypassed,
no release commit substituted for a historical one, and no exception hidden to
manufacture scientific success. Additional audit/materialization code and
historical-test identities remain explicitly outside the verified closure.

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for L1/L2/L3 and exact input roles.
Summaries cannot recreate unavailable prediction arrays, scalers or checkpoints.
Full scientific reproduction is neither attempted nor claimed.
