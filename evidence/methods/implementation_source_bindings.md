# Implementation source identities

The source paths below identify historical scientific authorities. They are not promises that every original artifact is included. Consult ../source_map.json for package paths; no model or target file is distributed.

| Display | Canonical identity | Input | Pooling | Output | Parameters | Scope |
| --- | --- | --- | --- | --- | ---: | --- |
| G-small | R0_SMALL_P | state41 + action16 = 57 | none | sigmoid residual | 7809 | VOC_FIT_OOF, COCO_EXTERNAL |
| G-match | R0_CM_P | state41 + action16 = 57 | none | sigmoid residual | 103369 | VOC_FIT_OOF, COCO_EXTERNAL |
| Atom | R1_P | region27, state41, action16 | unweighted region-embedding mean and max; final full-raster area-weighted sum | sigmoid regional residual -> state residual | 103521 | VOC_FIT_OOF, COCO_EXTERNAL |
| Rect | RECT_P | region27, representation-specific state41, action16 | same local pooling; guillotine count-matched rectangles | sigmoid regional residual -> state residual | 103521 | VOC_FIT_OOF, COCO_EXTERNAL |
| Union | UNION_P | region27, representation-specific state41, action16 | same local pooling; union and complement connected components | sigmoid regional residual -> state residual | 103521 | VOC_FIT_OOF, COCO_EXTERNAL |
| Regret | R2_P | region27, state41, action16 | same local pooling | sigmoid regional residual -> state residual | 103521 | VOC_FIT_OOF |
| Relative | R3_P | region27, state41, action16 | same local pooling | sigmoid regional residual -> state residual | 103521 | VOC_FIT_OOF, COCO_EXTERNAL |
| Gain | Q2_P | state41 + action16 = 57 | none | unbounded gain; scientific STOP fixed zero | 24321 | VOC_FIT_OOF |
| Defer | L2D_P | state41 + flattened five action16 = 121 | none | five unbounded logits | 16517 | VOC_FIT_OOF |
| SPO+ | SPO_PLUS_P | state41 + flattened five action16 = 121 | none | five unbounded predicted cost-adjusted values | 16517 | VOC_FIT_OOF |
| F0 | F0_P | region27, state41, action16; physical A3 excluded | same local pooling | remaining-action residuals | 103521 | VOC_FIT_OOF |
| F1 | F1_P | region27 from A0--A3 candidates, state41, action16; physical A3 excluded | same local pooling; acquired-A3 status in incidence context | remaining-action residuals | 103521 | VOC_FIT_OOF |

## Method-to-implementation bindings

### I01 — Exact A0--A6 text/spatial prompts and maximum-score source choice

Experiments: EXP-016, EXP-034, EXP-035, EXP-036, EXP-037, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `configs/experiments/voc_m04_m05_action_protocol_v1.json` — `/actions and /normalization_mapping`; source SHA-256 `c558595b6be151307f04ae189ff0b8c74f1de2adbf7e0c046db57048fd12f985`. Readable path: `configs/experiments/voc_m04_m05_action_protocol_v1.json`.
- `src/rail3/data/voc_taxonomy.py` — `VOC_FOREGROUND_CLASSES`; source SHA-256 `d82b2ddf9004c955c680ead57e9136e465f9c551f791d00dff7a8f3bd364e75a`. Readable path: `src/rail3/data/voc_taxonomy.py`.
- `src/rail3/sam/voc_actions.py` — `select_source_candidate; expanded_crop_box; deepest_interior_point; build_action_plans`; source SHA-256 `12811c1c8fdfcfa17bc210376dc43b823e748ef638d4ee93cc7de9df3cd33706`. Readable path: `src/rail3/sam/voc_actions.py`.
### I02 — Outcome union over all retained candidates; STOP is A0 union; empty candidate retention

Experiments: EXP-016, EXP-034, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/data/m06e_sources.py` — `union_canonical; union_action`; source SHA-256 `81b519f56a58817c9c3d34633a06444b4dc993e6516b46b20e3371afe15e4f9e`. Readable path: `src/rail3/data/m06e_sources.py`.
- `scripts/build_tmlr_v6_prospective_bundles.py` — `_all_action_predictions`; source SHA-256 `89a161a80db9e1752459b0aaa2e34d8cb48cf24b1c7f839a91fc1acb74ba127b`. Readable path: `paper/unified_ijcv/source_data/method_sources/scripts/build_tmlr_v6_prospective_bundles.py`.
- `src/rail3/sam/sam31_backend.py` — `_convert_output retained_indices and no-result branches`; source SHA-256 `15072351b46b2d2160ae283b233fec90594eeb9d5df3ddeff72069d33607345e`. Readable path: `src/rail3/sam/sam31_backend.py`.
- `docs/experiments/TMLR_V9B1_INFERENCE_OUTCOME_POLICY.md` — `Q1--Q7`; source SHA-256 `4b9585272c5f88d454ca3797477c97cc79456c14c4ecca681f7dd3273a9083d5`. Readable path: `docs/experiments/TMLR_V9B1_INFERENCE_OUTCOME_POLICY.md`.
### I03 — Plan infeasibility vs legal empty vs technical missingness and saved selection availability

Experiments: EXP-052, EXP-053, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `configs/experiments/tmlr_v9b1_e4_missingness_disposition_v1.json` — `/semantic_layer /execution_layer /evaluation_denominator`; source SHA-256 `e3118fc03b03b81487705516450156eb75f5921c9f77f91850f596fa9df35db1`. Readable path: `configs/experiments/tmlr_v9b1_e4_missingness_disposition_v1.json`.
- `configs/experiments/tmlr_v9c_utility_aligned_addendum.json` — `technical-missingness action disposition`; source SHA-256 `e1c3bbdcdc16c2348cc861d85919fb12f591bcfd37b62b5527a356adb6fc53f8`. Readable path: `configs/experiments/tmlr_v9c_utility_aligned_addendum.json`.
- `scripts/run_tmlr_v9c_track_a_pre_gt.py` — `_mask_action_technical_missingness`; source SHA-256 `2d4dcc80d3428cc309192480458fda4fe7349282626098de08b3339aa94caaae`. Readable path: `scripts/run_tmlr_v9c_track_a_pre_gt.py`.
- `src/rail3/evaluation/tmlr_v9c_metrics.py` — `_validate; evaluate_metrics defined/complete/oracle_mask`; source SHA-256 `79d60062b3ffb2836b63f131557cd1f9b9b2c799e54c473092eb34a865a511e5`. Readable path: `src/rail3/evaluation/tmlr_v9c_metrics.py`.
### I04 — Atom candidate-level signature, candidate order, zero-signature background and stable index binding

Experiments: EXP-016, EXP-035, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/data/m06e_sources.py` — `canonical_observation; causal_base_inputs`; source SHA-256 `81b519f56a58817c9c3d34633a06444b4dc993e6516b46b20e3371afe15e4f9e`. Readable path: `src/rail3/data/m06e_sources.py`.
- `src/rail3/regions/m06e_causal.py` — `assert_m06e_causal_partition_integrity; build_m06e_causal_partition`; source SHA-256 `a9b7bc3ac76156f12231515115e776764ae937c18b7e627d56a0be7e2f3145b0`. Readable path: `src/rail3/regions/m06e_causal.py`.
- `src/rail3/regions/voc_state_atomic.py` — `build_voc_state_atomic_partition`; source SHA-256 `2e49ce1116e1415ff0309a9b0b4f87f60f528a3d9d289ca0b523c0c61c5ca91d`. Readable path: `src/rail3/regions/voc_state_atomic.py`.
- `src/rail3/regions/atomic.py` — `CandidateMaskInput; build_atomic_partition`; source SHA-256 `446dd3138514881347d5de69039e2944e2b5251bf85ca03ddd67e5bac9559141`. Readable path: `src/rail3/regions/atomic.py`.
- `src/rail3/models/tmlr_v9b1/inference.py` — `_candidate_inputs; build_state_role_features`; source SHA-256 `bd14045f40fcb86efb52f49fa86c80ab8ea461dfba437f5f2065eeb21c7e3cf5`. Readable path: `src/rail3/models/tmlr_v9b1/inference.py`.
### I05 — RECT is full-raster region-count matching, not per-atom area matching; UNION includes complement

Experiments: EXP-035, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/diagnostic/representations.py` — `guillotine_partition; union_component_partition; regional_feature_matrices`; source SHA-256 `adedf537f1ca78b9c583c29abc05ead92c6ee7e0c003e8dc8d6bcdbd3bf55bd4`. Readable path: `src/rail3/diagnostic/representations.py`.
- `scripts/build_tmlr_v6_prospective_bundles.py` — `_full_raster_rectangle; _partitions; build_features`; source SHA-256 `89a161a80db9e1752459b0aaa2e34d8cb48cf24b1c7f839a91fc1acb74ba127b`. Readable path: `paper/unified_ijcv/source_data/method_sources/scripts/build_tmlr_v6_prospective_bundles.py`.
- `src/rail3/models/tmlr_v9b1/inference.py` — `build_state_role_features`; source SHA-256 `bd14045f40fcb86efb52f49fa86c80ab8ea461dfba437f5f2065eeb21c7e3cf5`. Readable path: `src/rail3/models/tmlr_v9b1/inference.py`.
### I06 — 27/41/16 numeric schemas, region-dependent descriptor content and observed-A3 context

Experiments: EXP-034, EXP-035, EXP-036, EXP-037, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/models/m06e/features.py` — `atom_vector; state_vector; _hash_vector`; source SHA-256 `41357a229399a466527a6bc0cc9ec35bfdec18cee820d03e3a29faf82f98eec4`. Readable path: `src/rail3/models/m06e/features.py`.
- `src/rail3/models/tmlr_v6/schema.py` — `ACTION_FEATURE_NAMES; dimensions`; source SHA-256 `a407a6d37eb2ae47db58eb6523802d3496515117edb0cdb259d99507c8784763`. Readable path: `src/rail3/models/tmlr_v6/schema.py`.
- `src/rail3/models/tmlr_v6/descriptors.py` — `descriptor_from_plan; stop_descriptor; action_vector`; source SHA-256 `b88365e94d95784b71043d924ac3812967a2cc93cc7cede70eb05c1fe9412380`. Readable path: `src/rail3/models/tmlr_v6/descriptors.py`.
- `src/rail3/diagnostic/representations.py` — `regional_feature_matrices; observed_partition_feature_matrices`; source SHA-256 `adedf537f1ca78b9c583c29abc05ead92c6ee7e0c003e8dc8d6bcdbd3bf55bd4`. Readable path: `src/rail3/diagnostic/representations.py`.
- `src/rail3/models/m07a/features.py` — `build_feature_matrices; observation_context`; source SHA-256 `d56be909740bdc5b8a77f3e8051f46ccc1b9f081c13aae041ee0363b39a85a0e`. Readable path: `src/rail3/models/m07a/features.py`.
### I07 — Global graph widths, parameter-only capacity matching, counts and activations

Experiments: EXP-034, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/models/m06e/models.py` — `GlobalResidual`; source SHA-256 `8a2908a7db315c3d1c8a4950cbf35d1e2056955b7ed617c0088e74cb291b6484`. Readable path: `src/rail3/models/m06e/models.py`.
- `src/rail3/models/tmlr_v6/models.py` — `CapacityMatchedGlobalResidual; solve_capacity_matched_width`; source SHA-256 `6f5da8e38652f832938bad001d4b7eddc5ec3a604de296faa6cc78b2b5181099`. Readable path: `src/rail3/models/tmlr_v6/models.py`.
- `configs/experiments/tmlr_v6_p0_prospective_capacity.json` — `/models/first_batch; /capacity_match`; source SHA-256 `37ea7f814416fc21b4a48af876b7752b6c76ca44379559d75739d9ae3e826904`. Readable path: `configs/experiments/tmlr_v6_p0_prospective_capacity.json`.
### I08 — Shared local embedding/context/action/head graph and separate pooling reductions

Experiments: EXP-035, EXP-036, EXP-037, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/models/m06e/models.py` — `SharedAtomicResidual; aggregate_atomic`; source SHA-256 `8a2908a7db315c3d1c8a4950cbf35d1e2056955b7ed617c0088e74cb291b6484`. Readable path: `src/rail3/models/m06e/models.py`.
- `src/rail3/models/tmlr_v6/models.py` — `ProspectiveSharedAtomicResidual`; source SHA-256 `6f5da8e38652f832938bad001d4b7eddc5ec3a604de296faa6cc78b2b5181099`. Readable path: `src/rail3/models/tmlr_v6/models.py`.
- `src/rail3/models/tmlr_v6/training.py` — `inference_weights_from_features`; source SHA-256 `39fe1f084e53c9d8378a03dd15afca79f1a4fa7cf3dd0f3158843565a7e32970`. Readable path: `src/rail3/models/tmlr_v6/training.py`.
### I09 — Gain and direct multi-action architectures, no marginal residual interpretation and alias identity

Experiments: EXP-036. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/models/m06e/models.py` — `DirectGainBaseline`; source SHA-256 `8a2908a7db315c3d1c8a4950cbf35d1e2056955b7ed617c0088e74cb291b6484`. Readable path: `src/rail3/models/m06e/models.py`.
- `src/rail3/diagnostic/baselines.py` — `MultiActionNetwork`; source SHA-256 `7016c294f9adbc48f65521cc20c0ced70253baf95b67980c86df053a5dcab419`. Readable path: `src/rail3/diagnostic/baselines.py`.
- `src/rail3/models/tmlr_v6/phase2_models.py` — `MODEL_REGISTRY; LOGICAL_ALIAS_REGISTRY`; source SHA-256 `feed84c5c2414517f249bd89f2ea3353e7daa6936a1c13ce33e5b5c6cc1324b1`. Readable path: `src/rail3/models/tmlr_v6/phase2_models.py`.
- `configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json` — `/objectives/Q2_P /objectives/L2D_P /objectives/SPO_PLUS_P /objectives/LL4TTA_P`; source SHA-256 `c5356e4141ef21dda8956109a855a9b7f89ae51095b4233ee709ed4110c24131`. Readable path: `configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json`.
### I10 — Five fixed group folds, seed-specific inner stop split and scaler fit scope

Experiments: EXP-034, EXP-035, EXP-036, EXP-037. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `src/rail3/folds/m06e.py` — `build_m06e_scale_folds`; source SHA-256 `1ba6a9bb34936b3a60c291f95343268a2ffd46cf0688903ec5b4414540499a1a`. Readable path: `src/rail3/folds/m06e.py`.
- `src/rail3/models/tmlr_v6/training.py` — `_inner_split; outer_train_np = train_np | stop_np`; source SHA-256 `39fe1f084e53c9d8378a03dd15afca79f1a4fa7cf3dd0f3158843565a7e32970`. Readable path: `src/rail3/models/tmlr_v6/training.py`.
- `src/rail3/models/tmlr_v6/schema.py` — `scale_predictor_features`; source SHA-256 `a407a6d37eb2ae47db58eb6523802d3496515117edb0cdb259d99507c8784763`. Readable path: `src/rail3/models/tmlr_v6/schema.py`.
- `src/rail3/models/m06e/features.py` — `fit_scaler; FrozenScaler.transform`; source SHA-256 `41357a229399a466527a6bc0cc9ec35bfdec18cee820d03e3a29faf82f98eec4`. Readable path: `src/rail3/models/m06e/features.py`.
- `src/rail3/models/tmlr_v6/phase2_scaling.py` — `scale_remaining_predictor_features`; source SHA-256 `b2f38f4bebf009923309e81e278b3b47d3f5c3f37ddf833e4194fced772d3f26`. Readable path: `src/rail3/models/tmlr_v6/phase2_scaling.py`.
### I11 — One full-partition step per epoch, AdamW recipe and early-stop criterion

Experiments: EXP-034, EXP-035, EXP-036, EXP-037. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `configs/experiments/tmlr_v6_p0_prospective_capacity.json` — `/training`; source SHA-256 `37ea7f814416fc21b4a48af876b7752b6c76ca44379559d75739d9ae3e826904`. Readable path: `configs/experiments/tmlr_v6_p0_prospective_capacity.json`.
- `configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json` — `/training_recipe`; source SHA-256 `c5356e4141ef21dda8956109a855a9b7f89ae51095b4233ee709ed4110c24131`. Readable path: `configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json`.
- `src/rail3/models/tmlr_v6/training.py` — `_run_first_batch_with_gpu_lease optimization loop; _seed`; source SHA-256 `39fe1f084e53c9d8378a03dd15afca79f1a4fa7cf3dd0f3158843565a7e32970`. Readable path: `src/rail3/models/tmlr_v6/training.py`.
- `src/rail3/models/tmlr_v6/phase2_training.py` — `objective and single-step epoch loop`; source SHA-256 `8f82ef62a44a1ee557e7b31d1eb7596e3dc9aa7f362675b68a08aa4f6357e552`. Readable path: `src/rail3/models/tmlr_v6/phase2_training.py`.
- `src/rail3/models/tmlr_v6/phase2_f0_f1.py` — `objective and single-step epoch loop`; source SHA-256 `44703e52d1de7fa40870c7d09fd769fa1de36a5698458044a0972f5dba2aa7e5`. Readable path: `src/rail3/models/tmlr_v6/phase2_f0_f1.py`.
### I12 — Fixed family median epochs, full-population refit/scalers and continuous seed averaging

Experiments: EXP-050, EXP-053, EXP-056. Role: `METHOD_PROVENANCE_FOR_DEVELOPMENT_AND_EXTERNAL_CONFIRMATION`.

- `configs/experiments/tmlr_v9b1_fullfit_v1.json` — `/epoch_rule /training_contract /ensemble_contract`; source SHA-256 `b381a933c8d8cb6c302ed377708b8e938d3588e697e9cd09baf15e3465569d18`. Readable path: `configs/experiments/tmlr_v9b1_fullfit_v1.json`.
- `docs/experiments/TMLR_V9B0_RESULT_BLIND_RECOVERY_ADDENDUM.md` — `Full-fit epoch rule`; source SHA-256 `b0ef699ee22bbae6595ef01e2960c4a531746a2b472a8232ea093e0c28378142`. Readable path: `docs/experiments/TMLR_V9B0_RESULT_BLIND_RECOVERY_ADDENDUM.md`.
- `src/rail3/models/tmlr_v9b1/fullfit.py` — `scale_fullfit_features; run_fixed_epoch_loop`; source SHA-256 `22fce4c1e24a9b12af634767e45ea8de858e9315e022db480b5f7900bb74787b`. Readable path: `src/rail3/models/tmlr_v9b1/fullfit.py`.
- `src/rail3/models/tmlr_v9b1/inference.py` — `apply_frozen_voc_scalers; predict_frozen_seed; ensemble functions`; source SHA-256 `bd14045f40fcb86efb52f49fa86c80ab8ea461dfba437f5f2065eeb21c7e3cf5`. Readable path: `src/rail3/models/tmlr_v9b1/inference.py`.
- `scripts/run_tmlr_v9c_track_a_pre_gt.py` — `_predict; continuous ensemble then frozen selection availability`; source SHA-256 `2d4dcc80d3428cc309192480458fda4fe7349282626098de08b3339aa94caaae`. Readable path: `scripts/run_tmlr_v9c_track_a_pre_gt.py`.

## Training recipe

The V6 recipe uses AdamW, learning rate 0.0005, weight decay 0.0001, maximum 120 epochs, patience 15 and strict improvement greater than 1e-8. One epoch is one full optimization-partition step, not an inferred minibatch setting. Outer folds remain fixed; seeds are 13, 37 and 71. Inner-stop is the first floor(n_outer_train/5) hash-ranked image groups. Scalers fit inner-train plus inner-stop after prospective projection, never outer-held-out groups.

The external full fits use the same optimizer and fixed family epochs derived from the median of 15 registered V6 best epochs: 120 for R0_SMALL_P, R0_CM_P, R1_P, RECT_P and UNION_P; 108 for R3_P. Full-fit scalers use all authorized FIT rows; no COCO scaler fit occurs. Exact source selectors are I10–I12 and the machine-readable training_recipe record.

