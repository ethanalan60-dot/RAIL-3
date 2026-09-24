# Reproducibility and input availability

This local release candidate provides the implemented methods, frozen contracts
and saved summary evidence for one unified study. It supports a bounded synthetic
CPU check and partial reconstruction of saved-summary displays. It does not supply
the original datasets, masks, learned checkpoints, trajectories or prediction
arrays needed for a complete scientific rerun.

The historical outcomes remain **VOC V6 = MIXED** and **COCO = GAP_REPLICATES**.
The recorded COCO totals are one GT read and 48 performance rows. Release
preparation adds no training, inference, SAM call, GT read, scientific metric
calculation or bootstrap. A release check is distinct from a new experiment.

## Three reproduction levels

| Level | Available here | Additional requirements | Status |
| --- | --- | --- | --- |
| L1: inspect saved results and rebuild displays | The [source map](evidence/source_map.json), 19 original table fragments and saved summaries; static nonphotographic Figure 2; saved-summary rendering of the statistical content of Figures 3–6 | Python and the presentation dependencies for rendering; image rights and the excluded original plates for the photo-based figures | **PARTIAL**. Rendering may change layout and PDF bytes. It does not reconstruct estimates or intervals from predictions. |
| L2: evaluate frozen predictions | Metric/decision implementation, E4/E5 contracts, prediction-lock identities and saved results | Exact prediction arrays, state/group alignment, target arrays and their inventories, feasibility/availability masks, trajectory and selection authorities, and the required historical source identities | **NOT_RUN; NOT_COMPLETE_IN_THIS_PACKAGE**. A summary CSV cannot independently regenerate paired intervals or per-state metrics. |
| L3: rerun the full experiment | Actual feature, model, training, SAM adapter, trajectory, prediction and evaluation implementations; selected configurations and historical environment records | Licensed datasets and SAM assets, the exact panel/FIT/fold manifests, original input and checkpoint authorities, additional historical authority dependencies, a compatible environment and separately authorized compute | **NOT_RUN; NOT_COMPLETE_IN_THIS_PACKAGE**. Preserved historical drivers are not turnkey release commands. |

Figure 1 and the Q1–Q6 photo-based plates are excluded from this candidate because
third-party image redistribution has not been confirmed. Their saved numerical
annotations and selection limitations remain available. No new examples replace
them. The method and table TeX excerpts are evidence fragments, not the complete
manuscript or a standalone publisher-template build.

The source map distinguishes a supplied `public_path` from
`HISTORICAL_IDENTITY_ONLY`. A historical relative path and digest identify an
original artifact; they do not establish that the artifact is downloadable.
Synthetic or rendering checks must be reported by their actual command results,
not inferred from the presence of a script.

## Safe local entry points

From the release directory, with the relevant dependencies installed:

```bash
python scripts/release_smoke.py --help
python scripts/release_smoke.py
python presentation/plot_saved_summaries.py --check-inputs
python presentation/plot_saved_summaries.py --output build/saved-summary-displays
```

The smoke runs only the explicit synthetic suite in
`tests/test_release_synthetic.py`. It checks canonical identities, tiny
hand-constructed atomic partitions and E4 boolean rules on artificial values.
It does not import a model, SAM backend, scientific metric evaluator or bootstrap
routine. The presentation script reads only its bound saved summaries. Its output
directory must be new; it does not overwrite source figures or evidence.

The separate configuration gateway produces a plan only:

```bash
python scripts/release.py plan --config release_inputs.example.json --run-id inspection-001 --output release-runs/inspection-001
```

The example deliberately leaves inputs unbound. A valid plan can exit zero while
reporting `status: PLAN_ONLY` and `scientific_execution: BLOCKED`. The gateway checks the release's
own content checksums and path structure; it neither opens input payloads nor
probes raw-label availability. It prints JSON without creating the run directory
or launching a historical driver. Input overlap checks are lexical; they do not verify symlink resolution or physical input identity. A fresh run ID and a new output location are
required even for planning. Local checksum agreement establishes content
consistency with the supplied manifest, not independent source authentication.
See [entry points](docs/entry_points.md).

## Inputs, identities and access

| Input | Bound version or identity source | Included / access |
| --- | --- | --- |
| VOC development data | VOC 2012; registered S1364 FIT selection, five image-group folds and seeds 13/37/71. The expected FIT-manifest digest is in `configs/experiments/tmlr_v9_coco_external_v2.json` under `voc_fit_authority`; full V6 rules are in `configs/experiments/tmlr_v6_p0_prospective_capacity.json`. | Raw data and the FIT/fold payloads are not supplied. The canonical [VOC 2012 page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) is the existing acquisition pointer. The earlier bounded acquisition-pointer check could not reach that page; live reachability was not newly verified. Public access to the exact research FIT/fold artifacts is **UNKNOWN**. |
| COCO external data | COCO 2017 validation images and `annotations/instances_val2017.json`. `configs/experiments/tmlr_v9_coco_external_v2.json` records official archive names, expected bytes and SHA-256. | Obtain authorized inputs through the [official COCO download page](https://cocodataset.org/#download). Archives and labels are not included or downloaded here. The original selection is a fixed subset; downloading the full dataset does not recreate its panel identity. |
| External panel and state alignment | The supplied [prediction lock](evidence/authorities/track_a_coco_prediction_lock.json) records the panel lock ID/digest, 20,000 designed states and prediction population. | Lock metadata is supplied. Public access to the exact panel-manifest payload and all alignment dependencies is **UNKNOWN**. No replacement panel is selected. |
| Upstream SAM source | Commit `96914d2425f90a64f45ca977c2b5165418099543`; adapter `sam31-real-text-v1`. | Use the [bound upstream source revision](https://github.com/facebookresearch/sam3/tree/96914d2425f90a64f45ca977c2b5165418099543) under its applicable terms. Upstream source is not vendored. |
| SAM checkpoint | `sam3.1_multiplex.pt`, 3,502,755,717 bytes; SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`, as recorded in `configs/models/sam/sam31_checkpoint_identity.yaml`. | Not included. The bound record identifies official gated Hugging Face access through `facebook/sam3.1`; follow upstream access and licence requirements. No automatic download or access-control bypass is provided. |
| Learned predictors and scalers | Six external canonical families and seeds 13/37/71. `checkpoint_identities`, `model_sha256` and `scaler_identity` in the prediction lock preserve the individual identities. | The 18 predictor checkpoints, fitted scalers and their original lock dependencies are not distributed. An approved public access route is **UNKNOWN**. |
| Frozen predictions | The prediction lock records 24 artifacts, comprising individual-seed and continuous-ensemble outputs, plus two agreeing prediction manifests and their digests. | Arrays and the full prediction-manifest payloads are absent. Access is **UNKNOWN**. Saved aggregate rows are not substitutes. |
| Trajectories and candidate caches | The prediction lock and `scripts/run_tmlr_v9c_track_a_pre_gt.py` bind the trajectory inventory/lock, action ordering and technical-missingness authority. | Candidate masks and cache/trajectory payloads are absent. Access is **UNKNOWN**. Existing failures cannot be repaired by regenerating cases and treating them as the frozen run. |
| Frozen evaluation targets | `src/rail3/evaluation/tmlr_v9c_execution.py` defines the target root and target inventory; the supplied missingness result describes the retained denominators. | Target arrays and complete target-input bindings are absent. Access is **UNKNOWN**. L2 needs these aligned targets; otherwise target construction would require a separately authorized label-reading stage. |

These are acquisition pointers and recorded expected identities, not new
download, permission or file-verification receipts. No referenced raw/source
payload was opened to prepare this document. A digest known from an authority
must not be presented as a fresh verification of an unavailable file.

### Exact input locators

Paths below are release-relative for supplied files and original repository-relative for omitted research payloads. Selectors identify fields in supplied metadata; they do not resolve or read the referenced files. `V6` denotes `configs/experiments/tmlr_v6_p0_prospective_capacity.json`; `V6-2` denotes `configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json`; `PRED` denotes `evidence/authorities/track_a_coco_prediction_lock.json`; `FULLFIT` denotes `configs/experiments/tmlr_v9b1_fullfit_v1.json`. A wildcard selector means every recorded member, not a replacement sample.

| Level / input | Exact location or binding selector | Supplied scope / missing identity |
| --- | --- | --- |
| L1 / saved display input | `presentation/inputs/coco_r3_interval.csv` | Supplied; 231 bytes; SHA-256 `1b9680c7a940bda624d3b5cedae4d8eaa6a399a13e06a9ce14c1d67e39f8a9c0` from `presentation/input_identities.json["coco_r3_interval.csv"]`. |
| L1 / saved display input | `presentation/inputs/fidelity_contrasts_v2.json` | Supplied; 21734 bytes; SHA-256 `459e0e443fbc5a9e394edff4edd7382d24b89de69199abb281588366c9142899` from `presentation/input_identities.json["fidelity_contrasts_v2.json"]`. |
| L1 / saved display input | `presentation/inputs/fit_sensitivity_v2.json` | Supplied; 21199 bytes; SHA-256 `38045a35ddab61c86b8c447a0ef8e287bb83d661ca2abe469b14fc23f0676c49` from `presentation/input_identities.json["fit_sensitivity_v2.json"]`. |
| L1 / saved display input | `presentation/inputs/six_model_three_objectives.csv` | Supplied; 566 bytes; SHA-256 `48c92ccadf1068cb9522b47a44f3a47ec366b8a960392a0df9262f5ccc635d36` from `presentation/input_identities.json["six_model_three_objectives.csv"]`. |
| L1 / saved display input | `presentation/inputs/sparse_utility_coco.csv` | Supplied; 2141 bytes; SHA-256 `cb1fa0be75f74de206fc3dc4d2a90ac2d078f9194fc3e93f5473975da50923dc` from `presentation/input_identities.json["sparse_utility_coco.csv"]`. |
| L1 / saved display input | `presentation/inputs/sparse_utility_voc_r3.csv` | Supplied; 2837 bytes; SHA-256 `82a00ef03c9e2743e8f29227408954fe5b617ef141013cef6684c8fb1249dc3a` from `presentation/input_identities.json["sparse_utility_voc_r3.csv"]`. |
| L2/L3 / VOC FIT and folds | `V6:/frozen_fit_inputs/s1364_manifest` gives `data/manifests/voc2012-m06e-fit1364.json`; `/frozen_fit_inputs/folds/{S250,S500,S1000,S1364}` gives each exact fold path and SHA. S1364 uses `data/manifests/voc2012-m06e-s_max-folds.json`. | Config supplied; FIT/fold payloads omitted. FIT and S1364 fold bytes/SHA are recorded; other fold bytes are UNKNOWN in this config. |
| L2/L3 / prospective feature/target bundles and phase-2 views | `V6:/prospective_bundle_identities/roles/*/partitions/*/{array,manifest}` records each exact path, bytes and SHA. External full-fit inputs are enumerated separately by `FULLFIT:/bundle_input_allowlist/*`. Phase-2 view registry: `artifacts/audits/tmlr_v6/phase2_f0_f1_view_registry.json` (`phase2_io.py:PHASE2_VIEW_REGISTRY_PATH`). | Code/config supplied; NPZ bundles and manifests/view-registry payloads omitted. View-registry bytes/SHA are UNKNOWN in the supplied locator configuration. Features and targets remain separate inputs. |
| L2 / VOC OOF arrays | Original roots: `artifacts/voc2012/tmlr-v6-p0/first-batch-crossfit`, `artifacts/voc2012/tmlr-v6-p0/phase2-second-batch-crossfit`, and `artifacts/voc2012/tmlr-v6-p0/phase2-f0-f1-crossfit`. Training code defines model/scale/fold/seed subdirectories; each run has `run-manifest.json` and `predictions.npz`, with exact prediction identities in the run manifest. | Original CLI/code supplied; run manifests and OOF arrays omitted. Individual prediction bytes/SHA are UNKNOWN here; registered run layouts or aggregate metrics cannot supply them. |
| L2/L3 / VOC cost profiles | `V6:/cost_profile/artifacts/{csv,json}` gives `artifacts/source_data/tmlr_v6/action_type_cost_profiles.{csv,json}` and bytes/SHA. `V6-2:/cost_profiles/required_future_registry/{csv_path,json_path}` gives `artifacts/source_data/tmlr_v6/phase2_scale_action_type_cost_profiles.{csv,json}`. | Metadata supplied; cost-profile payloads omitted. Phase-2 scale-profile identities were intentionally deferred to the launch lock, whose payload is omitted; those bytes/SHA are UNKNOWN here. These costs are not first-batch trainer inputs. |
| L2/L3 / VOC gates and launch authority | `V6-2:/first_batch_gate_and_phase2_launch_lock` names `artifacts/audits/tmlr_v6/phase2_launch_lock.json`, `first_batch_completion_gate.json`, `first_batch_run_inventory.json`, and the separate `artifacts/paper/source_data/tmlr_v6/first_batch/first_batch_evaluation_report.json`. Its launch-lock requirements also name historical OOF inventory, cost, fold and implementation identities. | Required schemas and paths supplied; complete original authority transactions omitted. The recorded launch-lock SHA `1446c5ff368705de59cbfec340f162142a57ba9b17ead9867e99f81702f27863` is retained in `SOURCE_TO_RELEASE.json:/recorded_authority_expectations`; it was not freshly verified against a payload. Other unexposed bytes/SHA remain UNKNOWN here. Historical Git objects and the additional source/test closure are separate requirements. |
| L2/L3 / COCO panel and alignment | `data/manifests/tmlr_v9a_r_coco_external_panel_v1.json` and `artifacts/paper/source_data/tmlr_v9a_r/coco_external_panel_identity.csv`; exact hashes are `coco_v9b1_trajectory.py:PANEL_MANIFEST_SHA256` and `PANEL_IDENTITY_SHA256`. `PRED:/panel` retains the panel lock ID/hash. | Code and lock metadata supplied; panel/identity CSV payloads omitted. `coco_v9b1_trajectory.py:TAXONOMY_PATH` separately names `artifacts/paper/source_data/tmlr_v9b0_r2/coco_voc20_taxonomy_verification.csv`; original alignment metadata is also required. |
| L2 / COCO frozen predictions and features | `PRED:/prediction_manifests/{build1,build2}` gives `artifacts/paper/source_data/tmlr_v9c/track_a_coco_prediction_manifest_build{1,2}.json`, each 36,533 bytes and SHA `c4b95dc752a21f529a4016fead16e6f5eae7a2447699c94fa62c95fe2db8f061`. `/prediction_inventory` and `/feature_inventory` give the exact inventory paths, bytes and hashes. The omitted prediction manifests enumerate individual array identities. | Lock metadata supplied; prediction manifests, array payloads and feature inventory/banks omitted. The 24 individual array locations/digests cannot be recovered from saved summary CSVs. |
| L2 / saved COCO targets | `artifacts/targets/tmlr-v9c/track-a/residual_targets.npz`; original inventory `artifacts/paper/source_data/tmlr_v9c/track_a_coco_target_inventory.json` binds `/target_arrays`, `/reference_union_artifacts` and `/first_read_receipt`. Reference union pattern: `artifacts/targets/tmlr-v9c/track-a/unions/{canonical_image_id}.npz`. These are literal paths/selectors in `tmlr_v9c_execution.py`. | Target-reader code supplied; target arrays, union masks and inventory payload omitted. Their bytes/SHA are UNKNOWN from this locator; no raw labels are opened to fill the gap. |
| L2/L3 / learned checkpoints and scalers | `PRED:/checkpoint_identities/*/checkpoint` gives every exact checkpoint path/bytes/SHA under `artifacts/voc2012/tmlr-v9b1-fullfit/S1364/{family}/seed_{seed}/checkpoint.pt`; `/checkpoint_lock` identifies `artifacts/paper/source_data/tmlr_v9b1/fullfit_checkpoint_lock.json`. Scalers are embedded under `scalers` in checkpoint payloads (`fullfit.py:checkpoint_payload`); `PRED:/scaler_identity/by_family/*/scaler_sha256_by_seed/*` binds their identities. | Metadata supplied; checkpoints/scalers and original checkpoint-lock payload omitted. A separate standalone scaler-file path is not asserted. The historical evaluator also verifies checkpoint bytes; a pure saved-summary rebuild does not. |
| L2/L3 / trajectories, availability and missingness | `PRED:/trajectory_inventory` and `/trajectory_lock` give exact paths/bytes/SHA for `artifacts/paper/source_data/tmlr_v9b1/coco_external_trajectory_inventory.json` and `artifacts/paper/source_data/tmlr_v9c/final_trajectory_lock.json`. The omitted inventory specifies cache roots; evaluator code additionally requires `artifacts/paper/source_data/tmlr_v9c/technical_missingness_report.json` and immutable plan/cache identities. | Lock metadata and saved `evidence/authorities/track_a_technical_missingness_evaluation.json` supplied; original cache/plan/availability transactions omitted. Saved missingness results are not replacement execution masks. |
| L3 / raw images, labels and SAM assets | VOC source locations are bound through `V6:/frozen_fit_inputs` and `/prospective_bundle_identities/fit_only_input_inventory`. COCO code names `data/public/coco2017/raw/val2017` and `data/public/coco2017/raw/annotations/instances_val2017.json`; `tmlr_v9_coco_external_v2.json:/official_archives` binds archive identities. SAM source/checkpoint identities are in `configs/models/sam/sam31_checkpoint_identity.yaml:/checkpoint`. | Config/code supplied; raw data, SAM source/weights and photographs omitted. Their access and redistribution terms remain separate from these recorded identities; no acquisition or payload verification occurred in this engineering task. |

For the omitted project-specific payloads above, a public retrieval endpoint and redistribution authorization remain **UNKNOWN**. The official third-party acquisition pointers in the preceding table retain their separately stated terms; they do not supply the project-specific panel, folds or predictions. These selectors are precise starting points for an authorized future input arrangement, not a claim that every historical execution dependency is supplied. No payload path was followed to produce this table.

## Model and environment scope

The six external identities remain R0_SMALL_P, R0_CM_P, R1_P, RECT_P, UNION_P
and R3_P, with aliases G-small, G-match, Atom, Rect, Union and Relative.
The V6 configurations retain the additional VOC-only objective and observation
families. Alias names do not add models or training jobs.

The prospective feature dimensions are region 27, state 41 and action 16.
Candidate-level signatures and connected cells, count-matched guillotine
rectangles, union/complement supports, A0-union STOP and full-raster prediction
weights remain implemented in the exported source. Target-valid accounting
weights are distinct. First-batch R1/RECT/UNION supervision is state-only;
R2/R3 compare complete training recipes. No release path adjustment changes
those definitions.

The recorded full-fit environment is Python 3.12.3, NumPy 1.26.4,
PyTorch 2.10.0+cu128 and CUDA runtime 12.8. Its configuration and
`evidence/methods/backend_and_rule_bindings.json` identify the stage.
`constraints/sam31-wsl-cu128.txt` and `constraints/sam31-supplemental.txt` are
historical setup records, including their original wheel/cache assumptions.
They do not certify the exact environment of every SAM trajectory run or serve
as a mandatory installation recipe for the small CPU example. The latter needs
NumPy; saved-summary plotting uses Matplotlib. Actual model/SAM execution has
additional dependencies, including PyTorch, Pillow, pycocotools and the pinned
upstream SAM package. No such scientific environment was installed or exercised
as part of these reproduction levels.

The previous local display/check environment was Python 3.12.3 with NumPy 2.5.2 and Matplotlib 3.11.1; its PDF inspection used PyMuPDF 1.28.2. That earlier installation check was NOT_RUN; the later actual attempts are distinguished in INSTALLATION.md. This engineering round created three isolated Python 3.12.3 environments with pip 24.0, without system/user-site packages. Official dependency downloads timed out, so wheel and editable backend attempts failed for missing setuptools; the missing NumPy also prevents the 11-test core suite in those fresh environments. Existing-environment source regressions are reported separately and cannot establish a fresh installation. No historical scientific environment was changed.

## Seven resource dispositions: recovered, with stage identities

The previous candidate's seven omissions were sparse-worktree omissions. Each
exact file was present in the reachable original research Git tree examined
during recovery and was recovered using
an already-bound stage identity, rather than its filename or modification time.
[Per-file provenance](SOURCE_TO_RELEASE.json) gives every source
commit, blob, SHA-256 and authority selector. No outside repository was read.

| File | Identity and disposition |
| --- | --- |
| `scripts/run_tmlr_v6_first_batch_crossfit.py` | Restored exactly from the inventory-bound `cac967264d4ec5cfc7443683a815625aedce6fce` source; core CLI source now present. |
| `scripts/run_tmlr_v6_phase2_second_batch_crossfit.py` | Restored exactly from the certified formal-end `1a9f6bd24e89e83f7b8fb84c054322a4919569e4` source. |
| `scripts/run_tmlr_v6_phase2_f0_f1_crossfit.py` | Same bound phase-2 source; no new wrapper substituted at this old path. |
| `scripts/evaluate_tmlr_v6_phase2_f0_f1_fit.py` | Same bound phase-2 source; classifier import file now present. |
| `scripts/evaluate_tmlr_v6_phase2_sensitivity_fit.py` | Same bound phase-2 source; classifier import file now present. |
| `scripts/build_tmlr_v6_phase2_qualitative_atlas.py` | Same bound phase-2 source; classifier import file now present, but its original selection/raw-input logic remains unexecuted. |
| `configs/experiments/voc_m06_pilot_b_training_v1.json` | Restored exactly; SHA-256 matches the original M06-B protocol constant. This is a historical resource, not an extra current experiment or a CLI. |

All seven recovered versions matched the original research HEAD blobs examined
during recovery (`da2780cdcd724f7cff4abac39257427ba53a40eb`). These
original Git objects are not part of the new public repository history.
This establishes those files' identities, not an independent reconstruction of
the entire historical launch-lock aggregate. The phase-2 launch-lock payload,
additional source/audit helpers and historical integrity-test identities remain
unsupplied. They are enumerated as direct guard references in
[ENTRYPOINT_CAPABILITIES.json](ENTRYPOINT_CAPABILITIES.json), not silently bypassed.
The earlier bound bundle-builder recovery also remains available.

No seven-path source-access request remains. Other original run inputs remain
unavailable as listed above; a recorded hash does not make them obtainable.
The classifier's old three-import omission is resolved. Top-level Torch/Pillow
imports, historical runtime checks, report validation, bootstrap and its
scientific integrity-test sequence remain separate. The release adapter does
not execute any of them or hardcode their outcome.

## Six existing scientific source limitations

| Limit | Consequence for interpretation or reproduction |
| --- | --- |
| Exact F0/F1 evaluable-state and pair denominators | The saved accuracy practical ties remain visible, but these summaries do not permit a full denominator audit or establish conditional utility equivalence. |
| The action pair in the old “Error change” panels | The original plates cannot establish an action-specific pixel-change claim from those panels; no pair is inferred from a photograph or rounded number. |
| Historical SAM trajectory-generation environment | Predictor-refit environment metadata cannot certify exact SAM trajectory reproducibility. |
| Exact prediction-lock and E5 UTC timestamps | The recorded event ordering remains stated; missing timestamps are not inferred from file modification times or the new release snapshot. |
| Historical executed DRRE reduction | The old printed state-balanced formula is verified in its version, but the executed historical reduction remains UNKNOWN. It is not pooled into the current pooled-record estimand. |
| Remaining historical nonzero-cost grid | The recovered R3 six zero-cost cells and current external sparse tables are retained; a complete historical cost-response surface cannot be reconstructed or used to support an all-cost claim. |

The current methods, denominators and limited conclusions are explained in
[scientific provenance](docs/scientific_provenance.md) and
[claim scope](evidence/claim_scope.json). The two RC-001 sources remain separate.
VH3's failure to replicate and Track B's non-evaluation remain disclosed.

## Preserved authorities and safe failure

Historical scientific drivers retain source/Git/input guards. Four exported
files have explicit machine-location substitutions recorded in
[path transformations](docs/path_transformations.json); the non-path scientific
content remains unchanged. Those substitutions do not make a new release commit
equivalent to a historical freeze or automatically satisfy an old source hash.

The plan gateway does not resolve these authority dependencies, fill missing
inputs, regenerate predictions, tune a controller or launch training/evaluation.
It rejects unsafe output structure and existing output directories. Missing or
unverified identities remain a block on scientific execution. Neither summary
rendering nor a successful synthetic check constitutes end-to-end reproduction
or permission to overwrite the archived results.

## Historical verification versus a new scientific run

The new `scripts/release_entrypoints.py --validate-only` checks shipped source
hashes/AST, configuration structure, distribution-presence metadata and lexical
input-location fields. It does not open input payloads, certify their existence,
check CUDA/version compatibility or verify an original historical transaction.
A complete-looking location plan still reports scientific readiness as
NOT_ESTABLISHED. First-batch training does not consume action-cost tables; those
resources belong to later evaluation and relevant phase-2 objectives.

The original scientific functions and their honest future execution routes are
retained. Their fixed output layouts, GPU assignment and launch/Git identities
prevent a safe generic new-run launcher here without more engineering and
authentic inputs. No identity resolver was changed to substitute a release
commit for a historical one. A future user may run the original route with the
complete authorized archive/environment, or develop a separately identified
new-run adapter that preserves the scientific semantics and uses fresh outputs.
The current engineering-only adapter is not a permanent ban on that work.
Missing full-rerun inputs constrain L2/L3 claims, not every possible bounded
code release once licence, target and rights conditions are resolved.
