#!/usr/bin/env python3
"""Build the deterministic, non-cherry-picked TMLR V6 FIT-OOF atlas.

Figure contract
---------------
Core conclusion: hash-selected FIT examples expose where accurate residual
prediction does or does not translate to correct action choice, plus the two
frozen model-disagreement controls.  The archetype is image plate + quant:
full uncropped image evidence leads to selected/oracle masks and the signed
error-change map, while the complete prediction/target vectors provide the
quantitative audit trail.  No brightness, contrast, crop, or local image
adjustment is applied.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.evaluate_tmlr_v6_first_batch_fit import (
    load_first_batch_surfaces,
    load_historical_fit_surfaces,
)
from rail3.cache import CandidateCache
from rail3.contracts import canonical_json_bytes
from rail3.data.m06e_sources import (
    TrajectorySource,
    action_observations,
    canonical_observation,
    load_trajectory_sources,
    state_by_class,
    union_action,
    union_canonical,
)
from rail3.analysis.tmlr_v6_phase2 import (
    EXTERNAL_SPLITS,
    Phase2Surface,
    assert_zero_external_access,
)
from rail3.analysis.tmlr_v6_qualitative import (
    NO_ELIGIBLE_STATE,
    QUALITATIVE_CATEGORIES,
    build_selection_manifest,
    error_change_map,
    validate_selection_manifest,
)
from rail3.models.tmlr_v6.phase2_contract import (
    safe_phase2_repository_path,
    validate_phase2_launch_lock,
    verify_exact_rebuild_directory,
)
from rail3.models.tmlr_v6.training import (
    load_protocol,
    sha256_file,
    validate_fold_manifest,
)


SOURCE_MIRROR = Path("data/v6_fit_only/source_mirror")
FIT_MANIFEST = Path("data/manifests/voc2012-m06e-fit1364.json")
STATE_INDEX = SOURCE_MIRROR / "artifacts/voc2012/m06e/causal-v2/state-index.jsonl"
BASELINE_CACHE = SOURCE_MIRROR / "artifacts/candidates/voc2012/sam31-text-v1"
DATASET_ROOT = SOURCE_MIRROR / "data/raw/voc2012/extracted/VOCdevkit/VOC2012"
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/paper/figures/tmlr_v6/qualitative_cases"
)
QUALITATIVE_REPORT_CHECKS = frozenset({
    "launch_lock_exact",
    "fit_oof_only",
    "deterministic_hash_selection",
    "no_cross_category_deduplication",
    "no_predicate_relaxation",
    "selected_cache_inputs_inventory_registered",
    "full_image_uncropped",
    "void_transparent_in_error_map",
    "prediction_and_target_vectors_complete",
    "q5_q6_both_selected_outputs_and_vectors",
    "q6_multi_component_label",
    "png_pdf_svg_source_csv_and_selection_manifest",
    "protected_access_zero",
    "sam_inference_false",
})
TRAJECTORY_SOURCES = (
    (
        "fit100", 100,
        Path("artifacts/voc2012/fit100-trajectory-merged.json"),
        Path("artifacts/trajectories/voc2012/m04-m05-v1/fit100-merged"),
    ),
    (
        "m06c_additional", 400,
        Path("artifacts/voc2012/m06c/trajectory-new400-merged.json"),
        Path("artifacts/trajectories/voc2012/m06c/new400"),
    ),
    (
        "m06d_additional", 500,
        Path("artifacts/voc2012/m06d/trajectory-new500-merged.json"),
        Path("artifacts/voc2012/m06d/trajectory-cache-merged"),
    ),
    (
        "m06e_additional", 364,
        Path("artifacts/voc2012/m06e/trajectory-new364-merged.json"),
        Path("artifacts/voc2012/m06e/trajectory-cache-merged"),
    ),
)
_CACHE_KEY = re.compile(r"(?:candidate|trajectory)_cache_([0-9a-f]{64})")
SOURCE_FIELDS = (
    "category",
    "state_id",
    "image_group_id",
    "class_id",
    "action",
    "plan_feasible",
    "r1_predicted_residual",
    "true_residual",
    "r1_selected",
    "display_oracle",
    "comparison_model",
    "comparison_prediction",
    "comparison_selected",
)


def _repository() -> Path:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("TMLR V6 qualitative atlas must run at worktree root")
    return root


def _inside(repository: Path, path: Path) -> Path:
    return safe_phase2_repository_path(repository, path)


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class FitInventoryGuard:
    """Verify every selected raw/cache read against the frozen FIT inventory."""

    def __init__(self, repository: Path, protocol: Mapping[str, Any]) -> None:
        self.repository = repository
        identity = protocol["prospective_bundle_identities"][
            "fit_only_input_inventory"
        ]
        path = _inside(repository, Path(str(identity["path"])))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(identity["bytes"])
            or sha256_file(path) != str(identity["sha256"])
        ):
            raise RuntimeError("TMLR V6 qualitative FIT inventory identity drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "TMLR_V6_FIT_ONLY_INPUTS_COMPLETE"
            or payload.get("heldout_access")
            != {name: 0 for name in EXTERNAL_SPLITS}
            or any(value is not True for value in payload.get("checks", {}).values())
        ):
            raise RuntimeError("TMLR V6 qualitative FIT inventory is not safe")
        self.identity = {
            "path": str(identity["path"]),
            "bytes": int(identity["bytes"]),
            "sha256": str(identity["sha256"]),
        }
        self.rows = {str(row["path"]): row for row in payload["files"]}
        self.opened: dict[str, dict[str, Any]] = {}

    def verify(self, logical: Path) -> Path:
        key = logical.as_posix()
        if key not in self.rows:
            raise RuntimeError(f"TMLR V6 qualitative input is not FIT-registered: {key}")
        identity = self.rows[key]
        path = _inside(self.repository, logical)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(identity["bytes"])
            or sha256_file(path) != str(identity["sha256"])
        ):
            raise RuntimeError("TMLR V6 qualitative selected input identity drift")
        self.opened[key] = {
            "path": key,
            "bytes": int(identity["bytes"]),
            "sha256": str(identity["sha256"]),
        }
        return path


def _cache_logical(root: Path, key: str) -> Path:
    match = _CACHE_KEY.fullmatch(key)
    if match is None:
        raise RuntimeError("TMLR V6 qualitative cache key is malformed")
    return root / match.group(1)[:2] / f"{key}.json"


def _load_surfaces(
    repository: Path,
) -> tuple[dict[str, Phase2Surface], Mapping[str, Any]]:
    protocol = load_protocol(repository)
    folds = validate_fold_manifest(repository, protocol)
    first, _ = load_first_batch_surfaces(
        repository=repository, protocol=protocol, fold_manifest=folds,
    )
    historical, _, historical_identity = load_historical_fit_surfaces(
        repository=repository,
        protocol=protocol,
        reference=first["R1_P"],
    )

    def convert(model: str, source: Any) -> Phase2Surface:
        return Phase2Surface(
            model_id=model,
            prediction_kind="residual",
            prediction=np.asarray(source.prediction),
            target=np.asarray(source.target),
            feasible=np.asarray(source.feasible),
            groups=np.asarray(source.groups),
            state_ids=source.state_ids,
        ).validated()

    return {
        "R1_P": convert("R1_P", first["R1_P"]),
        "R0_CM_P": convert("R0_CM_P", first["R0_CM_P"]),
        "RETRO_R1_FIT_OOF": convert(
            "RETRO_R1_FIT_OOF", historical["RETRO_R1_FIT_OOF"],
        ),
    }, historical_identity


def _selected_state_records(
    *, state_index_path: Path, entries: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    requested = {
        int(entry["selected_state_index"])
        for entry in entries if entry["status"] == "SELECTED"
    }
    result = {}
    for index, record in enumerate(_json_lines(state_index_path)):
        if index in requested:
            result[index] = record
        if len(result) == len(requested):
            break
    if set(result) != requested:
        raise RuntimeError("TMLR V6 selected state index lookup is incomplete")
    for entry in entries:
        if entry["status"] == "SELECTED" and str(
            result[int(entry["selected_state_index"])]["state_id"]
        ) != str(entry["selected_state_id"]):
            raise RuntimeError("TMLR V6 selected state ID/index drift")
    return result


def _trajectory_index(
    *, repository: Path, guard: FitInventoryGuard, manifest: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], Any, str]]:
    sources = []
    for origin, count, report, cache in TRAJECTORY_SOURCES:
        mirrored_report = SOURCE_MIRROR / report
        guard.verify(mirrored_report)
        sources.append(TrajectorySource(
            origin=origin,
            expected_images=count,
            report_path=_inside(repository, mirrored_report),
            cache_root=_inside(repository, SOURCE_MIRROR / cache),
        ))
    index, _ = load_trajectory_sources(manifest=manifest, sources=sources)
    return index


def _case_assets(
    *,
    repository: Path,
    guard: FitInventoryGuard,
    fit_manifest: Mapping[str, Any],
    trajectory_index: Mapping[str, tuple[dict[str, Any], Any, str]],
    state_reference: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_index = int(state_reference["manifest_index"])
    item = fit_manifest["records"][manifest_index]
    asset_id = str(item["asset_id"])
    if asset_id != str(state_reference["asset_id"]):
        raise RuntimeError("TMLR V6 qualitative state/manifest asset drift")
    image_record, trajectory_cache, origin = trajectory_index[asset_id]
    if origin != str(state_reference["origin"]):
        raise RuntimeError("TMLR V6 qualitative trajectory origin drift")
    class_id = int(state_reference["class_id"])
    class_state = state_by_class(image_record)[class_id]
    canonical_key = str(class_state["canonical_cache_key"])
    guard.verify(_cache_logical(BASELINE_CACHE, canonical_key))
    for action in class_state["actions"]:
        guard.verify(_cache_logical(
            Path(trajectory_cache.root),
            str(action["cache_key"]),
        ))
    baseline_cache = CandidateCache(_inside(repository, BASELINE_CACHE))
    canonical_result, _, _, _ = canonical_observation(
        item=item,
        class_state=class_state,
        baseline_cache=baseline_cache,
        class_id=class_id,
    )
    _, outcomes = action_observations(
        class_state=class_state, cache=trajectory_cache, class_id=class_id,
    )
    sample_id = str(item["sample_id"])
    image_logical = DATASET_ROOT / "JPEGImages" / f"{sample_id}.jpg"
    label_logical = DATASET_ROOT / "SegmentationClass" / f"{sample_id}.png"
    image_path = guard.verify(image_logical)
    label_path = guard.verify(label_logical)
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    with Image.open(label_path) as handle:
        label = np.asarray(handle, dtype=np.uint8)
    if image.shape[:2] != label.shape:
        raise RuntimeError("TMLR V6 qualitative image/label geometry drift")
    valid = label != 255
    truth = label == class_id
    shape = label.shape
    masks: dict[str, np.ndarray | None] = {
        "STOP": union_canonical(canonical_result, shape),
    }
    for outcome in outcomes[2:]:
        masks[outcome.action_code] = (
            union_action(outcome, shape)
            if outcome.outcome in {"result", "no_result"} else None
        )
    if tuple(masks) != ("STOP", "A3", "A4", "A5", "A6"):
        raise RuntimeError("TMLR V6 qualitative action-mask order drift")
    selected_action = str(entry["r1_selected_action"])
    oracle_action = str(entry["display_oracle_action"])
    if masks[selected_action] is None or masks[oracle_action] is None:
        raise RuntimeError("TMLR V6 selected/oracle qualitative mask undefined")
    comparison_action = entry.get("comparison_selected_action")
    if comparison_action is not None and masks[str(comparison_action)] is None:
        raise RuntimeError("TMLR V6 comparison qualitative mask undefined")
    partition_logical = SOURCE_MIRROR / Path(str(state_reference["partition_path"]))
    guard.verify(partition_logical)
    return {
        "image": image,
        "truth": truth,
        "valid": valid,
        "a0": masks["STOP"],
        "selected": masks[selected_action],
        "oracle": masks[oracle_action],
        "comparison": (
            masks[str(comparison_action)] if comparison_action is not None else None
        ),
        "error_change": error_change_map(
            truth=truth,
            stop_prediction=np.asarray(masks["STOP"], dtype=bool),
            selected_prediction=np.asarray(masks[selected_action], dtype=bool),
            valid=valid,
        ),
        "sample_id": sample_id,
        "asset_id": asset_id,
        "class_id": class_id,
        "image_identity": guard.opened[image_logical.as_posix()],
        "label_identity": guard.opened[label_logical.as_posix()],
        "partition_identity": guard.opened[partition_logical.as_posix()],
    }


def _binary_plate(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 4), dtype=np.float32)
    result[..., :3] = (0.93, 0.93, 0.93)
    result[..., 3] = valid.astype(np.float32)
    result[mask & valid, :3] = (0.12, 0.55, 0.62)
    return result


def _error_plate(values: np.ndarray) -> np.ndarray:
    """Encode signed error change with genuinely transparent void pixels."""

    value = np.asarray(values, dtype=np.float64)
    if not np.all(np.isin(value[np.isfinite(value)], (-1.0, 0.0, 1.0))):
        raise RuntimeError("TMLR V6 error-change plate has an invalid value")
    plate = np.zeros((*value.shape, 4), dtype=np.uint8)
    plate[value == -1.0] = (47, 107, 154, 255)
    plate[value == 0.0] = (232, 232, 232, 255)
    plate[value == 1.0] = (217, 119, 47, 255)
    # NaN is the registered FIT-void encoding.  Its alpha deliberately stays
    # zero in PNG and in the PNG embedded by SVG.  PDF is explicitly rendered
    # as a white-background composite because PDF has no portable page alpha.
    return plate


def _render_case(
    *, output_stem: Path, category: str, entry: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> None:
    width, height = 2160, 1590  # exactly 7.2 x 5.3 inches at 300 dpi
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.is_file():
        raise RuntimeError("TMLR V6 deterministic DejaVu font is unavailable")
    title_font = ImageFont.truetype(str(font_path), 33)
    # 30 px at the frozen 300 dpi is exactly 7.2 pt, safely above the
    # registered 7 pt floor after export.
    panel_font = ImageFont.truetype(str(font_path), 30)
    text_font = ImageFont.truetype(str(font_path), 30)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    def mask_plate(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
        plate = np.full((*mask.shape, 3), 255, dtype=np.uint8)
        plate[valid] = (237, 237, 237)
        plate[np.asarray(mask, dtype=bool) & valid] = (31, 140, 158)
        plate[~valid] = (248, 248, 248)
        return plate

    panels: list[tuple[str, np.ndarray]] = [
        ("Full FIT image", np.asarray(assets["image"], dtype=np.uint8)),
        ("Fit ground truth", mask_plate(assets["truth"], assets["valid"])),
        ("A0 / STOP output", mask_plate(assets["a0"], assets["valid"])),
        (
            f"R1-P selected: {entry['r1_selected_action']}",
            mask_plate(assets["selected"], assets["valid"]),
        ),
    ]
    if assets["comparison"] is not None:
        panels.append((
            f"{entry['comparison_model']} selected: "
            f"{entry['comparison_selected_action']}",
            mask_plate(assets["comparison"], assets["valid"]),
        ))
    panels.extend([
        (
            f"Oracle: {entry['display_oracle_action']}",
            mask_plate(assets["oracle"], assets["valid"]),
        ),
        ("Error change", _error_plate(assets["error_change"])),
    ])
    if len(panels) > 8:
        raise RuntimeError("TMLR V6 qualitative panel count exceeds layout")

    outer_x, gap_x = 54, 24
    cell_width = (width - 2 * outer_x - 3 * gap_x) // 4
    image_height = 400
    row_tops = (105, 580)
    for index, (panel_title, array) in enumerate(panels):
        row, column = divmod(index, 4)
        x0 = outer_x + column * (cell_width + gap_x)
        y0 = row_tops[row]
        title_box = draw.textbbox((0, 0), panel_title, font=panel_font)
        title_width = title_box[2] - title_box[0]
        draw.text(
            (x0 + (cell_width - title_width) / 2, y0),
            panel_title, fill="black", font=panel_font,
        )
        source_mode = "RGBA" if array.ndim == 3 and array.shape[-1] == 4 else "RGB"
        source = Image.fromarray(array, mode=source_mode)
        resampling = (
            Image.Resampling.NEAREST
            if source_mode == "RGBA" else Image.Resampling.LANCZOS
        )
        source.thumbnail((cell_width, image_height), resampling)
        panel_x = x0 + (cell_width - source.width) // 2
        panel_y = y0 + 43 + (image_height - source.height) // 2
        # Copy RGBA values directly: using an alpha mask here would composite
        # transparent FIT-void pixels onto white and silently destroy alpha.
        canvas.paste(source, (panel_x, panel_y))
        draw.rectangle(
            (panel_x, panel_y, panel_x + source.width - 1, panel_y + source.height - 1),
            outline=(170, 170, 170), width=2,
        )
    prediction = ", ".join(
        f"{action}={value:.4f}"
        for action, value in zip(("STOP", "A3", "A4", "A5", "A6"), entry["r1_prediction"])
    )
    target = ", ".join(
        f"{action}={'NA' if value is None else f'{value:.4f}'}"
        for action, value in zip(("STOP", "A3", "A4", "A5", "A6"), entry["true_residual"])
    )
    vector_lines = [
        f"R1-P predicted residuals: {prediction}",
        f"True residuals: {target}",
    ]
    if entry.get("comparison_prediction") is not None:
        comparison = ", ".join(
            f"{action}={value:.4f}"
            for action, value in zip(
                ("STOP", "A3", "A4", "A5", "A6"),
                entry["comparison_prediction"],
            )
        )
        vector_lines.append(
            f"{entry['comparison_model']} predicted residuals: {comparison}"
        )
    vector_lines.append(
        "Error change: blue = repaired (-1); grey = unchanged (0); "
        "orange = newly wrong (+1); void = transparent"
    )
    headline = (
        f"{category} | state {entry['selected_state_id']} | "
        f"class {assets['class_id']}"
    )
    headline_width = draw.textbbox((0, 0), headline, font=title_font)[2]
    draw.text(
        ((width - headline_width) / 2, 24), headline,
        fill="black", font=title_font,
    )
    text_y = 1085
    for line in vector_lines:
        for wrapped in textwrap.wrap(line, width=126, break_long_words=False):
            draw.text((60, text_y), wrapped, fill="black", font=text_font)
            text_y += 38
    if text_y > height - 20:
        raise RuntimeError("TMLR V6 qualitative annotation overflow")

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    canvas.save(
        png_path, format="PNG", dpi=(300, 300), compress_level=9,
    )
    pdf_canvas = Image.new("RGB", canvas.size, "white")
    pdf_canvas.paste(canvas, mask=canvas.getchannel("A"))
    pdf_canvas.save(
        pdf_path,
        format="PDF",
        resolution=300,
        title=f"TMLR V6 qualitative {category}",
        creator="RAIL-3 TMLR V6 deterministic Pillow renderer",
        producer="Pillow",
        creationDate=time.gmtime(0),
        modDate=time.gmtime(0),
    )
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="7.2in" '
        f'height="5.3in" viewBox="0 0 {width} {height}">\n'
        f'  <title>TMLR V6 qualitative {category}</title>\n'
        f'  <image width="{width}" height="{height}" '
        f'href="data:image/png;base64,{encoded}"/>\n'
        '</svg>\n'
    ).encode("utf-8")
    with svg_path.open("xb") as handle:
        handle.write(svg)
        handle.flush()
        os.fsync(handle.fileno())


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = list(SOURCE_FIELDS)
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError("TMLR V6 qualitative source CSV field-set drift")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in rows)
    return output.getvalue().encode("utf-8")


def _write(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _identity(logical: Path, physical: Path) -> dict[str, Any]:
    return {
        "path": logical.as_posix(),
        "bytes": physical.stat().st_size,
        "sha256": sha256_file(physical),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repository = _repository()
    launch = validate_phase2_launch_lock(repository)
    surfaces, historical_identity = _load_surfaces(repository)
    selection = build_selection_manifest(surfaces)
    validate_selection_manifest(selection)
    protocol = load_protocol(repository)
    guard = FitInventoryGuard(repository, protocol)
    fit_path = _inside(repository, FIT_MANIFEST)
    fit_identity = protocol["frozen_fit_inputs"]["s1364_manifest"]
    if (
        fit_path.is_symlink()
        or not fit_path.is_file()
        or sha256_file(fit_path) != str(fit_identity["sha256"])
    ):
        raise RuntimeError("TMLR V6 qualitative fit manifest identity drift")
    fit_manifest = json.loads(fit_path.read_text(encoding="utf-8"))
    state_index_path = guard.verify(STATE_INDEX)
    entries = selection["categories"]
    state_records = _selected_state_records(
        state_index_path=state_index_path, entries=entries,
    )
    trajectory_index = _trajectory_index(
        repository=repository, guard=guard, manifest=fit_manifest,
    )
    if args.output_root != DEFAULT_OUTPUT_ROOT:
        raise RuntimeError("TMLR V6 qualitative output root is not registered")
    output_root = _inside(repository, args.output_root)
    if args.check and not output_root.exists():
        raise FileNotFoundError(output_root)
    if not args.check and output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.tmp-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        source_rows: list[dict[str, Any]] = []
        case_assets: dict[str, Any] = {}
        for entry in entries:
            category = str(entry["category"])
            if entry["status"] == NO_ELIGIBLE_STATE:
                continue
            assets = _case_assets(
                repository=repository,
                guard=guard,
                fit_manifest=fit_manifest,
                trajectory_index=trajectory_index,
                state_reference=state_records[int(entry["selected_state_index"])],
                entry=entry,
            )
            _render_case(
                output_stem=temporary / category,
                category=category,
                entry=entry,
                assets=assets,
            )
            case_assets[category] = {
                "sample_id": assets["sample_id"],
                "asset_id": assets["asset_id"],
                "class_id": assets["class_id"],
                "image": assets["image_identity"],
                "fit_label": assets["label_identity"],
                "causal_partition": assets["partition_identity"],
                "crop": "none_full_image",
                "brightness_contrast_gamma": "none",
                "pseudo_color": "binary_masks_and_signed_error_map_only",
                "stitching": "none",
            }
            for action_index, action in enumerate(("STOP", "A3", "A4", "A5", "A6")):
                source_rows.append({
                    "category": category,
                    "state_id": entry["selected_state_id"],
                    "image_group_id": entry["image_group_id"],
                    "class_id": assets["class_id"],
                    "action": action,
                    "plan_feasible": entry["plan_feasible"][action_index],
                    "r1_predicted_residual": entry["r1_prediction"][action_index],
                    "true_residual": entry["true_residual"][action_index],
                    "r1_selected": action == entry["r1_selected_action"],
                    "display_oracle": action == entry["display_oracle_action"],
                    "comparison_model": entry.get("comparison_model"),
                    "comparison_prediction": (
                        entry.get("comparison_prediction", [None] * 5)[action_index]
                    ),
                    "comparison_selected": action == entry.get("comparison_selected_action"),
                })
        source_path = temporary / "qualitative_cases_source.csv"
        _write(source_path, _csv_bytes(source_rows))
        selection_path = temporary / "qualitative_selection_manifest.json"
        _write(selection_path, canonical_json_bytes(selection) + b"\n")
        outputs: dict[str, Any] = {
            source_path.name: _identity(args.output_root / source_path.name, source_path),
            selection_path.name: _identity(
                args.output_root / selection_path.name, selection_path,
            ),
        }
        for entry in entries:
            if entry["status"] != "SELECTED":
                continue
            for suffix in (".png", ".pdf", ".svg"):
                path = temporary / f"{entry['category']}{suffix}"
                outputs[path.name] = _identity(args.output_root / path.name, path)
        report = {
            **selection,
            "schema_version": "rail3.tmlr-v6.phase2-qualitative-atlas.v1",
            "status": "TMLR_V6_PHASE2_QUALITATIVE_ATLAS_PASS",
            "selection_manifest_schema": selection["schema_version"],
            "figure_contract": {
                "core_conclusion": (
                    "Deterministically selected FIT examples expose success, "
                    "decision failures, and registered model disagreements."
                ),
                "archetype": "image_plate_plus_quant",
                "backend": "python_pillow_deterministic_raster",
                "width_inches": 7.2,
                "height_inches": 5.3,
                "font_floor_points": 7,
                "png_dpi": 300,
                "void_rendering": {
                    "png": "alpha_zero",
                    "svg": "embedded_png_alpha_zero",
                    "pdf": "white_background_composite",
                },
            },
            "input_bindings": {
                "phase2_launch_lock_sha256": launch.sha256,
                "first_batch_completion_gate_sha256": launch.payload[
                    "first_batch_completion_gate"
                ]["sha256"],
                "first_batch_run_inventory_sha256": launch.payload[
                    "first_batch_run_inventory"
                ]["sha256"],
                "first_batch_evaluation_sha256": launch.payload[
                    "first_batch_evaluation"
                ]["sha256"],
                "fit_only_input_inventory": guard.identity,
                "historical_fit_oof": historical_identity,
            },
            "case_assets": case_assets,
            "selected_fit_inputs": {
                "count": len(guard.opened),
                "identities": [
                    guard.opened[key] for key in sorted(guard.opened)
                ],
                "identities_sha256": hashlib.sha256(canonical_json_bytes(
                    [guard.opened[key] for key in sorted(guard.opened)]
                )).hexdigest(),
            },
            "outputs": outputs,
            "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
            "sam_inference": False,
            "prompt_or_trajectory_generation": False,
            "manual_selection": False,
            "predicate_relaxation": False,
            "checks": {
                "launch_lock_exact": True,
                "fit_oof_only": True,
                "deterministic_hash_selection": True,
                "no_cross_category_deduplication": True,
                "no_predicate_relaxation": True,
                "selected_cache_inputs_inventory_registered": True,
                "full_image_uncropped": True,
                "void_transparent_in_error_map": True,
                "prediction_and_target_vectors_complete": True,
                "q5_q6_both_selected_outputs_and_vectors": all(
                    entry["status"] == NO_ELIGIBLE_STATE
                    or (
                        entry.get("selected_state_id") is not None
                        and len(entry.get("r1_prediction", ())) == 5
                        and len(entry.get("true_residual", ())) == 5
                    )
                    for entry in entries
                    if entry["category"] in {"Q5", "Q6"}
                ),
                "q6_multi_component_label": True,
                "png_pdf_svg_source_csv_and_selection_manifest": True,
                "protected_access_zero": True,
                "sam_inference_false": True,
            },
        }
        manifest_path = temporary / "qualitative_atlas_manifest.json"
        if (
            set(report["checks"]) != set(QUALITATIVE_REPORT_CHECKS)
            or any(value is not True for value in report["checks"].values())
        ):
            raise RuntimeError("TMLR V6 qualitative report checks are incomplete")
        assert_zero_external_access(report["heldout_access"])
        _write(manifest_path, canonical_json_bytes(report) + b"\n")
        if args.check:
            verify_exact_rebuild_directory(output_root, temporary)
        else:
            os.replace(temporary, output_root)
    print(json.dumps({
        "status": "TMLR_V6_PHASE2_QUALITATIVE_ATLAS_PASS",
        "selected_categories": sum(
            entry["status"] == "SELECTED" for entry in entries
        ),
        "no_eligible_categories": sum(
            entry["status"] == NO_ELIGIBLE_STATE for entry in entries
        ),
        "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
        "check": args.check,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
