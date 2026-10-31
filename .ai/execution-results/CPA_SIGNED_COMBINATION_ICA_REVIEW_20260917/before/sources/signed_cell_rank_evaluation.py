"""Label-free offline comparison for completed signed naive/cell-rank arms.

This evaluator is intentionally separate from the attack runner.  It performs
all public/final-artifact provenance checks before loading private evaluation
truth, then evaluates both final candidate sets with one pixel-MSE Hungarian
matching per arm.  Labels are never read.  The command is documented for a
future explicit research run and is not invoked by this implementation task.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch.nn import functional as F

from . import signed_cell_rank as core
from . import upstream_naive
from .signed_baseline import ATTACK_LAYER, CPA_CONFIG, FIA_CONFIG, SEED
from .signed_baseline_observation import CHECKPOINT_SHA, FC_SHAPES, TRUTH_SHA, digest, read_bundle
from .full_update_evaluation import evaluate_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PUBLIC_ROOT = PROJECT_ROOT / "results" / "signed_int4_baseline_B16_T10"
CELL_PUBLIC_ROOT = PROJECT_ROOT / "results" / "signed_cell_rank_B16_T10"
BASELINE_PRIVATE_ROOT = PROJECT_ROOT / "results_private" / "signed_int4_baseline_B16_T10"
CELL_PRIVATE_ROOT = PROJECT_ROOT / "results_private" / "signed_cell_rank_B16_T10"
EVALUATOR_FORMAT = "signed_cell_rank_label_free_offline_comparison_v1"
RUNTIME_FIELDS = (
    "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms",
    "deterministic_warn_only", "matmul_tf32", "cudnn_tf32",
    "cublas_workspace_config", "threads",
)
RUNTIME_CONTRACT = {
    "cudnn_benchmark": True,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
    "deterministic_warn_only": True,
    "matmul_tf32": False,
    "cudnn_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "threads": 1,
}
INITIALIZATION_SHAPE = (16, 3, 224, 224)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_DELTA_CONVENTION = "theta_after10_minus_theta_start"

CPA_EXECUTED_SETTINGS = {
    "n_components": 16,
    "seed": SEED,
    "device": "cuda",
    "n_iter": CPA_CONFIG["cpa_ica_max_iter"],
    "lr": CPA_CONFIG["cpa_ica_lr"],
    "opt": CPA_CONFIG["cpa_ica_optimizer"],
    "sch": CPA_CONFIG["cpa_ica_scheduler"],
    "decor": CPA_CONFIG["cpa_ica_decor"],
    "T": CPA_CONFIG["cpa_ica_T"],
    "ne": CPA_CONFIG["cpa_ica_ne"],
    "nv": CPA_CONFIG["cpa_ica_nv"],
    "l1": CPA_CONFIG["cpa_ica_l1"],
    "tv": CPA_CONFIG["cpa_ica_tv"],
    "log_every": CPA_CONFIG["cpa_ica_log_every"],
    "whitening": "upstream cov/(d-1), torch.linalg.eig, top abs eigenvalues",
    "backend": "official_cpa/src/gradient_inversion.py:CocktailPartyAttack",
    "debias": False,
    "mp_truncation": False,
    "mixing_matrix_computed": False,
}
FIA_EXECUTED_SETTINGS = {
    "seed": SEED,
    "device": "cuda",
    "n_iter": FIA_CONFIG["n_iter"],
    "lr": FIA_CONFIG["lr"],
    "tv": FIA_CONFIG["tv"],
    "fi": FIA_CONFIG["fi"],
    "optimizer": FIA_CONFIG["optimizer"],
    "scheduler": FIA_CONFIG["scheduler"],
    "log_every": FIA_CONFIG["log_every"],
    "backend": "official_cpa/src/feature_inversion.py:Direct",
    "feature_adapter": "VGG16 IMAGENET1K_V1 features/avgpool; unused classifier skipped",
    "evaluation_inputs": False,
    "cudnn_benchmark": True,
    "cudnn_deterministic": True,
}


class ProvenanceError(ValueError):
    """The supplied public arms do not form the registered comparison."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProvenanceError(f"expected JSON object: {path}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    """Require a canonical lowercase SHA256 string before comparing it."""
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ProvenanceError(f"{label} must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_hash_map(value: Any, label: str) -> dict[str, str]:
    if type(value) is not dict:
        raise ProvenanceError(f"{label} must be an object")
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ProvenanceError(f"{label} contains an invalid path key")
        _require_sha256(item, f"{label}[{key}]")
    return value


def _require_typed_fields(value: Any, expected: dict[str, Any], label: str) -> dict[str, Any]:
    """Require fields and exact Python types before equality checks.

    ``type(...) is ...`` is intentional: JSON ``true`` must not satisfy an
    integer field because Python considers ``bool`` a subclass of ``int``.
    """
    if type(value) is not dict:
        raise ProvenanceError(f"{label} must be an object")
    for key, expected_value in expected.items():
        if key not in value:
            raise ProvenanceError(f"{label}.{key} is missing")
        if type(value[key]) is not type(expected_value):
            raise ProvenanceError(
                f"{label}.{key} has type {type(value[key]).__name__}; "
                f"expected {type(expected_value).__name__}")
    return value


def _validate_initialization(initial: Any, arm: str) -> dict[str, Any]:
    """Validate the complete FIA initialization evidence before comparing arms."""
    if type(initial) is not dict:
        raise ProvenanceError(f"{arm} FIA initialization must be an object")
    required = {"tensor_sha256": "", "shape": [], "seed": SEED, "runtime": {}}
    for key in required:
        if key not in initial:
            raise ProvenanceError(f"{arm} FIA initialization missing {key}")
    _require_sha256(initial["tensor_sha256"], f"{arm} FIA initialization tensor_sha256")
    shape = initial["shape"]
    if type(shape) is not list or any(type(item) is not int for item in shape):
        raise ProvenanceError(f"{arm} FIA initialization shape must be an integer list")
    if tuple(shape) != INITIALIZATION_SHAPE:
        raise ProvenanceError(f"{arm} FIA initialization shape mismatch")
    if type(initial["seed"]) is not int or type(initial["seed"]) is bool or initial["seed"] != SEED:
        raise ProvenanceError(f"{arm} FIA initialization seed mismatch")
    runtime_value = _require_typed_fields(initial["runtime"], RUNTIME_CONTRACT, f"{arm} FIA runtime")
    if any(runtime_value[field] != expected for field, expected in RUNTIME_CONTRACT.items()):
        raise ProvenanceError(f"{arm} FIA runtime differs from the fixed baseline contract")
    return initial


def _validate_initialization_pair(left: Any, right: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    left_value = _validate_initialization(left, "naive")
    right_value = _validate_initialization(right, "cell_rank")
    if left_value["tensor_sha256"] != right_value["tensor_sha256"]:
        raise ProvenanceError("FIA pixel initializations differ")
    if left_value["shape"] != right_value["shape"] or left_value["seed"] != right_value["seed"]:
        raise ProvenanceError("FIA initialization metadata differs")
    for field in RUNTIME_FIELDS:
        if left_value["runtime"][field] != right_value["runtime"][field]:
            raise ProvenanceError(f"FIA numerical runtime differs: {field}")
    return left_value, right_value


def _within(path: Path, root: Path, *, allow_root: bool = False) -> bool:
    return (allow_root and path == root) or (path != root and path.is_relative_to(root))


def _validate_paths(*, naive: Path, cell_rank: Path, public_observation: Path,
                    private_observation: Path, output: Path) -> dict[str, Path]:
    paths = {"naive": naive.resolve(), "cell_rank": cell_rank.resolve(),
             "public_observation": public_observation.resolve(),
             "private_observation": private_observation.resolve(), "output": output.resolve()}
    if not _within(paths["naive"], BASELINE_PUBLIC_ROOT.resolve()) or not _within(
            paths["public_observation"], BASELINE_PUBLIC_ROOT.resolve()):
        raise ProvenanceError("naive/public observation must remain in the baseline public tree")
    if not _within(paths["cell_rank"], CELL_PUBLIC_ROOT.resolve()):
        raise ProvenanceError("cell-rank arm must remain in the new public tree")
    if not _within(paths["private_observation"], BASELINE_PRIVATE_ROOT.resolve()):
        raise ProvenanceError("private observation must remain in the baseline private tree")
    if not _within(paths["output"], CELL_PRIVATE_ROOT.resolve()):
        raise ProvenanceError("comparison output must remain in the new private tree")
    if paths["output"].exists():
        raise FileExistsError(paths["output"])
    values = list(paths.items())
    for index, (name, path) in enumerate(values):
        for other, other_path in values[index + 1:]:
            if path == other_path or path in other_path.parents or other_path in path.parents:
                raise ProvenanceError(f"overlapping evaluation paths: {name}, {other}")
    return paths


def _completed(folder: Path, expected_arm: str, *, legacy_naive: bool = False) -> dict[str, Any]:
    result = _read_json(folder / "result.json")
    if (result.get("status") != "COMPLETED" or result.get("cpa_iterations") != 25000
            or result.get("fia_iterations") != 25000
            or (not legacy_naive and result.get("arm") != expected_arm)
            or (legacy_naive and result.get("arm") not in (None, expected_arm))):
        raise ProvenanceError(f"{expected_arm} is not a completed CPA25k/FIA25k arm")
    if not legacy_naive and result.get("attack_status") != "COMPLETED":
        raise ProvenanceError(f"{expected_arm} attack stage is not completed")
    if not legacy_naive and result.get("evaluation_status") not in ("NOT_RUN", "PENDING"):
        raise ProvenanceError(f"{expected_arm} evaluation stage is not pending/complete for offline comparison")
    for filename, key in (("candidate.pt", "candidate_sha256"), ("sources.pt", "sources_sha256")):
        path = folder / filename
        expected_hash = _require_sha256(result.get(key), f"{expected_arm} result {key}")
        if not path.is_file() or core.file_sha256(path) != expected_hash:
            raise ProvenanceError(f"{expected_arm} final artifact hash mismatch: {filename}")
    return result


def _require_executed_settings(kind: str, settings: dict[str, Any], arm: str) -> None:
    expected = CPA_EXECUTED_SETTINGS if kind == "cpa" else FIA_EXECUTED_SETTINGS
    _require_typed_fields(settings, expected, f"{arm} {kind} settings")
    mismatches = {key: (settings.get(key), value) for key, value in expected.items()
                  if settings.get(key) != value}
    if mismatches:
        raise ProvenanceError(f"{arm} {kind} executed settings mismatch: {mismatches}")


def _executed_settings(folder: Path, result: dict[str, Any], source_hashes: dict[str, str],
                       input_sha: str) -> None:
    input_sha = _require_sha256(input_sha, f"{result.get('arm')} canonical CPA input hash")
    _require_hash_map(source_hashes, f"{result.get('arm')} source hashes")
    for kind, expected_iterations in (("cpa", 25000), ("fia", 25000)):
        executed = _read_json(folder / f"{kind}_result.json")
        executed_hashes = _require_hash_map(executed.get("source_hashes"),
                                            f"{result.get('arm')} {kind} source hashes")
        if executed_hashes != source_hashes:
            raise ProvenanceError(f"{result.get('arm')} {kind} source hashes differ from result")
        settings = executed.get("settings", {})
        _require_executed_settings(kind, settings, result.get("arm"))
        if type(settings.get("n_iter")) is not int or settings.get("n_iter") != expected_iterations:
            raise ProvenanceError(f"{result.get('arm')} {kind} iteration count mismatch")
        if executed_hashes != source_hashes:
            raise ProvenanceError(f"{result.get('arm')} {kind} source hashes differ from result")
    manifest = _read_json(folder / "cpa" / "cpa_manifest.json")
    cpa_fingerprint = manifest.get("fingerprint", {})
    if type(cpa_fingerprint) is not dict:
        raise ProvenanceError(f"{result.get('arm')} CPA fingerprint is missing")
    fingerprint_input_sha = _require_sha256(cpa_fingerprint.get("input_sha256"),
                                            f"{result.get('arm')} CPA fingerprint input_sha256")
    if fingerprint_input_sha != input_sha:
        raise ProvenanceError(f"{result.get('arm')} CPA input fingerprint mismatch")
    _require_typed_fields(cpa_fingerprint.get("settings"), CPA_EXECUTED_SETTINGS,
                          f"{result.get('arm')} CPA manifest settings")
    if cpa_fingerprint.get("settings") != CPA_EXECUTED_SETTINGS:
        raise ProvenanceError(f"{result.get('arm')} CPA manifest settings mismatch")
    _require_hash_map(cpa_fingerprint.get("source_hashes"),
                      f"{result.get('arm')} CPA manifest source hashes")
    if cpa_fingerprint.get("source_hashes") != source_hashes:
        raise ProvenanceError(f"{result.get('arm')} CPA manifest source hashes mismatch")
    if type(manifest.get("cache_root")) is not str or not manifest.get("cache_root"):
        raise ProvenanceError(f"{result.get('arm')} CPA cache root is missing")
    cache_root = Path(manifest["cache_root"]).resolve()
    if not cache_root.is_relative_to(folder.resolve()):
        raise ProvenanceError(f"{result.get('arm')} CPA cache escaped its arm directory")
    fia_manifest_path = folder / "fia" / "fia_manifest.json"
    if not fia_manifest_path.is_file():
        raise ProvenanceError(f"{result.get('arm')} FIA manifest is missing")
    fia_manifest = _read_json(fia_manifest_path)
    fia_fingerprint = fia_manifest.get("fingerprint", {})
    if type(fia_fingerprint) is not dict:
        raise ProvenanceError(f"{result.get('arm')} FIA fingerprint is missing")
    _require_typed_fields(fia_fingerprint.get("settings"), FIA_EXECUTED_SETTINGS,
                          f"{result.get('arm')} FIA manifest settings")
    if fia_fingerprint.get("settings") != FIA_EXECUTED_SETTINGS:
        raise ProvenanceError(f"{result.get('arm')} FIA manifest settings mismatch")
    _require_hash_map(fia_fingerprint.get("source_hashes"),
                      f"{result.get('arm')} FIA manifest source hashes")
    if fia_fingerprint.get("source_hashes") != source_hashes:
        raise ProvenanceError(f"{result.get('arm')} FIA manifest source hashes mismatch")
    if type(fia_manifest.get("cache_root")) is not str or not fia_manifest.get("cache_root"):
        raise ProvenanceError(f"{result.get('arm')} FIA cache root is missing")
    fia_cache_root = Path(fia_manifest["cache_root"]).resolve()
    if not fia_cache_root.is_relative_to((folder / "fia").resolve()):
        raise ProvenanceError(f"{result.get('arm')} FIA cache escaped its FIA directory")


def _check_arm(folder: Path, expected_arm: str, *, public_sha: str, checkpoint_sha: str,
               source_hashes: dict[str, str], naive: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = _completed(folder, expected_arm, legacy_naive=naive)
    registration = _read_json(folder / "registered_arm.json")
    registered_checkpoint = _require_sha256(registration.get("checkpoint_sha256"),
                                            f"{expected_arm} registered checkpoint_sha256")
    if "checkpoint_sha256" in result:
        result_checkpoint = _require_sha256(result.get("checkpoint_sha256"),
                                            f"{expected_arm} result checkpoint_sha256")
    elif naive:
        # The completed legacy baseline predates a duplicated result-level
        # checkpoint field; its registration is the explicit compatibility
        # source, while new cell-rank arms must write the field.
        result_checkpoint = registered_checkpoint
    else:
        raise ProvenanceError(f"{expected_arm} result checkpoint_sha256 is missing")
    if registered_checkpoint != checkpoint_sha or result_checkpoint != checkpoint_sha:
        raise ProvenanceError(f"{expected_arm} registered checkpoint/seed mismatch")
    if type(registration.get("seed")) is not int or type(registration.get("seed")) is bool:
        raise ProvenanceError(f"{expected_arm} registered seed has the wrong type")
    if registration.get("seed") != SEED:
        raise ProvenanceError(f"{expected_arm} registered checkpoint/seed mismatch")
    registered_cpa = _require_typed_fields(registration.get("cpa"), CPA_CONFIG,
                                           f"{expected_arm} registered CPA settings")
    registered_fia = _require_typed_fields(registration.get("fia"), FIA_CONFIG,
                                           f"{expected_arm} registered FIA settings")
    if registered_cpa != CPA_CONFIG or registered_fia != FIA_CONFIG:
        raise ProvenanceError(f"{expected_arm} registered CPA/FIA settings differ")
    if registration.get("source_adapter") != "abs(S_hat); estimated ordering; no oracle correction":
        raise ProvenanceError(f"{expected_arm} source adapter changed")
    registered_hashes = _require_hash_map(registration.get("source_hashes"),
                                          f"{expected_arm} registered source hashes")
    result_hashes = result.get("source_hashes")
    if result_hashes is not None:
        result_hashes = _require_hash_map(result_hashes, f"{expected_arm} result source hashes")
    if registered_hashes != source_hashes or (result_hashes is not None and result_hashes != source_hashes):
        raise ProvenanceError(f"{expected_arm} original source hashes changed")
    input_info = registration.get("input")
    if type(input_info) is not dict:
        raise ProvenanceError(f"{expected_arm} input provenance is missing")
    input_public_sha = _require_sha256(input_info.get("public_manifest_sha256"),
                                       f"{expected_arm} input public_manifest_sha256")
    if input_public_sha != public_sha:
        raise ProvenanceError(f"{expected_arm} public observation binding mismatch")
    registered_private = registration.get("private_inputs_used", input_info.get("private_inputs_used"))
    registered_labels = registration.get("labels_used", False)
    if registered_private is not False or registered_labels is not False:
        raise ProvenanceError(f"{expected_arm} private/label contract violated")
    result_private = result.get("private_inputs_used", result.get("input", {}).get("private_inputs_used"))
    if result_private is not False or result.get("labels_used", False) is not False:
        raise ProvenanceError(f"{expected_arm} result private/label contract violated")
    if naive:
        expected = {"arm": "INT4_naive", "private_inputs_used": False}
        if registration.get("input_conversion") != "identity_raw_Delta_or_raw_s_q; no sign/lr/T/B transformation":
            raise ProvenanceError("INT4_naive input conversion changed")
    else:
        expected = {"arm": "INT4_cell_rank", "private_inputs_used": False}
        if registration.get("input_conversion") != "raw_signed_dequantized_V_final_cell_projected_from_public_q_s":
            raise ProvenanceError("INT4_cell_rank input conversion is not truthful")
        preprocess = registration.get("preprocess")
        if type(preprocess) is not dict:
            raise ProvenanceError("INT4_cell_rank preprocessing metadata is missing")
        preprocess_types = {
            "format": str, "schema_version": int, "rank": int, "ap_iterations": int,
            "expected_ap_iterations": int, "attack_layer": str, "centered": bool,
            "rank_backend": str, "projection_dtype": str, "projection_device": str,
            "final_selection": str, "final_v_file_sha256": str,
            "final_v_tensor_sha256": str, "scale_is_metric_only": bool,
        }
        for key, expected_type in preprocess_types.items():
            if key not in preprocess or type(preprocess[key]) is not expected_type:
                raise ProvenanceError(f"INT4_cell_rank preprocessing field {key} is missing or mistyped")
        if (preprocess["format"] != "signed_external_uncentered_gram_eigh_cell_rank_v1"
                or preprocess["schema_version"] != 1 or preprocess["rank"] != 16
                or preprocess["ap_iterations"] not in (0, 5)
                or preprocess["expected_ap_iterations"] != preprocess["ap_iterations"]
                or preprocess["attack_layer"] != ATTACK_LAYER or preprocess["centered"] is not False
                or preprocess["rank_backend"] != core.RANK_BACKEND
                or preprocess["projection_dtype"] != "torch.float32"
                or preprocess["projection_device"] not in ("cpu", "cuda")
                or preprocess["final_selection"] != "final_iteration_only"
                or preprocess["scale_is_metric_only"] is not True):
            raise ProvenanceError("INT4_cell_rank preprocessing contract mismatch")
        _require_sha256(preprocess["final_v_file_sha256"],
                        "INT4_cell_rank preprocessing final V file SHA256")
        _require_sha256(preprocess["final_v_tensor_sha256"],
                        "INT4_cell_rank preprocessing final V tensor SHA256")
        input_refs = input_info
        for key in ("preprocessed_v_path", "preprocessed_result_path"):
            if type(input_refs.get(key)) is not str or not input_refs[key]:
                raise ProvenanceError(f"INT4_cell_rank {key} is missing")
            ref_path = Path(input_refs[key]).resolve()
            if not _within(ref_path, CELL_PUBLIC_ROOT.resolve()) or not ref_path.is_file():
                raise ProvenanceError(f"INT4_cell_rank {key} escaped or is missing")
        _require_sha256(input_refs.get("preprocessed_result_sha256"),
                        "INT4_cell_rank preprocessing result SHA256")
        _require_sha256(input_refs.get("preprocessed_v_file_sha256"),
                        "INT4_cell_rank preprocessed V file SHA256")
        _require_sha256(input_refs.get("preprocessed_v_tensor_sha256"),
                        "INT4_cell_rank preprocessed V tensor SHA256")
        if core.file_sha256(Path(input_refs["preprocessed_result_path"])) != input_info.get("preprocessed_result_sha256"):
            raise ProvenanceError("INT4_cell_rank preprocessing result binding mismatch")
        if input_refs["preprocessed_v_file_sha256"] != preprocess["final_v_file_sha256"]:
            raise ProvenanceError("INT4_cell_rank final V file hash differs from preprocessing metadata")
        if input_refs["preprocessed_v_tensor_sha256"] != preprocess["final_v_tensor_sha256"]:
            raise ProvenanceError("INT4_cell_rank final V tensor hash differs from preprocessing metadata")
        if "preprocess" in result:
            if type(result.get("preprocess")) is not dict or result["preprocess"] != preprocess:
                raise ProvenanceError("INT4_cell_rank result/preprocessing metadata mismatch")
            if result.get("preprocess_status") != "COMPLETED":
                raise ProvenanceError("INT4_cell_rank preprocessing stage is not completed")
    if any(input_info.get(key) != value for key, value in expected.items()):
        raise ProvenanceError(f"{expected_arm} arm identity/input provenance mismatch")
    if result.get("input") is not None and result.get("input") != input_info:
        raise ProvenanceError(f"{expected_arm} result input provenance differs from registration")
    _executed_settings(folder, result, source_hashes, registration.get("input_tensor_sha256"))
    # The protected legacy naive arm wrote telemetry at its arm root; the new
    # runner writes it beside its independent ``fia`` cache.  Accept only these
    # exact locations so the initialization hash remains auditable.
    initial_path = folder / "fia" / "fia_initial.json"
    if not initial_path.is_file() and naive:
        initial_path = folder / "fia_initial.json"
    initial = _read_json(initial_path)
    _validate_initialization(initial, expected_arm)
    return result, initial, registration


def validate_provenance(*, naive: str | Path, cell_rank: str | Path,
                        public_observation: str | Path,
                        private_observation: str | Path,
                        output: str | Path) -> dict[str, Any]:
    """Authenticate both finished public arms before private tensor loading."""
    paths = _validate_paths(
        naive=_resolve(naive), cell_rank=_resolve(cell_rank),
        public_observation=_resolve(public_observation),
        private_observation=_resolve(private_observation), output=_resolve(output))
    if paths["naive"] == paths["cell_rank"]:
        raise ProvenanceError("the same arm cannot be supplied twice")
    public_manifest = _read_json(paths["public_observation"] / "manifest.json")
    public_sha = digest(paths["public_observation"] / "manifest.json")
    _require_sha256(public_manifest.get("checkpoint_sha256"), "public manifest checkpoint_sha256")
    if public_manifest.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ProvenanceError("public observation checkpoint mismatch")
    sender = _read_json(paths["private_observation"] / "sender_validation.json")
    # Hash private files for provenance without deserializing their tensors.
    raw_path = paths["private_observation"] / "raw_updates.pt"
    truth_path = paths["private_observation"] / "evaluation_truth.pt"
    raw_sha = digest(raw_path)
    truth_sha = digest(truth_path)
    for field in ("checkpoint_sha256", "source_truth_sha256", "public_manifest_sha256",
                  "raw_updates_sha256", "evaluation_truth_sha256"):
        _require_sha256(sender.get(field), f"sender {field}")
    if "source_hashes" in sender:
        _require_hash_map(sender["source_hashes"], "sender source_hashes")
    if type(sender.get("status")) is not str or sender.get("status") != "COMPLETED":
        raise ProvenanceError("sender status is not COMPLETED")
    if (sender.get("checkpoint_sha256") != CHECKPOINT_SHA
            or sender.get("source_truth_sha256") != TRUTH_SHA
            or sender.get("public_manifest_sha256") != public_sha
            or sender.get("raw_updates_sha256") != raw_sha
            or sender.get("evaluation_truth_sha256") != truth_sha):
        raise ProvenanceError("sender/public/private provenance mismatch")
    source_hashes = upstream_naive.source_hashes()
    _require_hash_map(source_hashes, "original source hashes")
    naive_result, naive_initial, naive_registration = _check_arm(
        paths["naive"], "INT4_naive", public_sha=public_sha, checkpoint_sha=CHECKPOINT_SHA,
        source_hashes=source_hashes, naive=True)
    cell_result, cell_initial, cell_registration = _check_arm(
        paths["cell_rank"], "INT4_cell_rank", public_sha=public_sha, checkpoint_sha=CHECKPOINT_SHA,
        source_hashes=source_hashes, naive=False)
    _validate_initialization_pair(naive_initial, cell_initial)
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "public_manifest_sha256": public_sha,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "raw_updates_sha256": raw_sha,
        "evaluation_truth_sha256": truth_sha,
        "source_hashes": source_hashes,
        "distinct_arm_identities": True,
        "registered_and_executed_settings_valid": True,
        "private_loaded": False,
        "initialization_sha256": naive_initial.get("tensor_sha256"),
        "results": {"naive": naive_result, "cell_rank": cell_result},
        "registrations": {"naive": naive_registration, "cell_rank": cell_registration},
        "raw_updates_path": str(raw_path),
    }


def _validated_public_input(folder: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only the public signed packet after public provenance checks."""
    manifest, entries = read_bundle(folder)
    if type(manifest) is not dict or _require_sha256(manifest.get("checkpoint_sha256"),
                                                    "public manifest checkpoint_sha256") != CHECKPOINT_SHA:
        raise ProvenanceError("public observation checkpoint mismatch")
    if ATTACK_LAYER not in entries:
        raise ProvenanceError(f"public observation lacks {ATTACK_LAYER}")
    q, s = entries[ATTACK_LAYER]
    expected_shape = tuple(FC_SHAPES[ATTACK_LAYER])
    if (type(q) is not torch.Tensor or q.layout != torch.strided or q.dtype != torch.int8
            or q.device.type != "cpu" or tuple(q.shape) != expected_shape):
        raise ProvenanceError("public signed codes have the wrong shape or dtype")
    if (type(s) is not torch.Tensor or s.layout != torch.strided or s.dtype != torch.float32
            or s.device.type != "cpu" or s.ndim != 0 or not bool(torch.isfinite(s)) or not bool(s > 0)):
        raise ProvenanceError("public signed scale has the wrong type or value")
    return q, s


def _load_true_update(path: Path, *, expected_shape: tuple[int, ...]) -> tuple[torch.Tensor, str]:
    """Load and authenticate true G only inside the final offline evaluator."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise ProvenanceError("raw_updates.pt must contain an object")
    checkpoint = _require_sha256(payload.get("checkpoint_sha256"), "raw update checkpoint_sha256")
    if checkpoint != CHECKPOINT_SHA:
        raise ProvenanceError("raw update checkpoint mismatch")
    if type(payload.get("delta_convention")) is not str or payload["delta_convention"] != RAW_DELTA_CONVENTION:
        raise ProvenanceError("raw update delta convention mismatch")
    updates = payload.get("updates")
    if type(updates) is not dict or ATTACK_LAYER not in updates:
        raise ProvenanceError("raw update lacks the requested attack layer")
    true_g = updates[ATTACK_LAYER]
    if (type(true_g) is not torch.Tensor or true_g.layout != torch.strided
            or true_g.dtype != torch.float32 or true_g.device.type != "cpu"
            or tuple(true_g.shape) != tuple(expected_shape) or not bool(torch.isfinite(true_g).all())):
        raise ProvenanceError("true G has the wrong CPU FP32 shape or finite-value status")
    return true_g.contiguous(), payload["delta_convention"]


def _squared_l2(value: torch.Tensor, *, chunk_rows: int = 64) -> float:
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    flat = value.reshape(value.shape[0], -1)
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, flat.shape[0], chunk_rows):
        chunk = flat[start:start + chunk_rows].to(dtype=torch.float64)
        total += (chunk * chunk).sum(dtype=torch.float64)
    return float(torch.sqrt(total))


def update_error_metrics(estimate: torch.Tensor, true_g: torch.Tensor, *, arm: str,
                         estimate_input_sha256: str, delta_convention: str) -> dict[str, Any]:
    """Return evaluation-only true-G errors without affecting candidate selection."""
    if (type(estimate) is not torch.Tensor or estimate.layout != torch.strided
            or estimate.dtype != torch.float32 or estimate.device.type != "cpu"
            or type(true_g) is not torch.Tensor or true_g.layout != torch.strided
            or true_g.dtype != torch.float32 or true_g.device.type != "cpu"
            or tuple(estimate.shape) != tuple(true_g.shape)
            or not bool(torch.isfinite(estimate).all()) or not bool(torch.isfinite(true_g).all())):
        raise ProvenanceError(f"{arm} true-G metric received an invalid tensor")
    input_sha = _require_sha256(estimate_input_sha256, f"{arm} estimate input SHA256")
    if type(delta_convention) is not str or delta_convention != RAW_DELTA_CONVENTION:
        raise ProvenanceError("true-G metric delta convention mismatch")
    absolute = _squared_l2((estimate - true_g).contiguous())
    reference = _squared_l2(true_g)
    if reference == 0.0:
        relative = None
        relative_status = "UNDEFINED_ZERO_REFERENCE_NORM"
    else:
        relative = absolute / reference
        if not math.isfinite(relative):
            raise ProvenanceError(f"{arm} true-G relative error is nonfinite")
        relative_status = "COMPLETED"
    return {
        "status": relative_status,
        "arm": arm,
        "attack_layer": ATTACK_LAYER,
        "estimate_input_sha256": input_sha,
        "true_g_source": "raw_updates.pt",
        "delta_convention": delta_convention,
        "absolute_l2_error": absolute,
        "reference_l2_norm": reference,
        "relative_l2_error": relative,
        "relative_l2_error_status": relative_status,
        "selection_used": False,
    }


def _load_bound_cell_v(registration: dict[str, Any], *, expected_shape: tuple[int, ...]) -> torch.Tensor:
    """Load the exact finalized V referenced by the cell arm registration."""
    input_info = registration.get("input")
    if type(input_info) is not dict:
        raise ProvenanceError("INT4_cell_rank input provenance is missing")
    path_value = input_info.get("preprocessed_v_path")
    if type(path_value) is not str or not path_value:
        raise ProvenanceError("INT4_cell_rank finalized V path is missing")
    path = Path(path_value).resolve()
    if not _within(path, CELL_PUBLIC_ROOT.resolve()):
        raise ProvenanceError("INT4_cell_rank finalized V path escaped the cell public tree")
    file_sha = _require_sha256(input_info.get("preprocessed_v_file_sha256"),
                               "INT4_cell_rank finalized V file SHA256")
    tensor_sha = _require_sha256(input_info.get("preprocessed_v_tensor_sha256"),
                                 "INT4_cell_rank finalized V tensor SHA256")
    if not path.is_file() or core.file_sha256(path) != file_sha:
        raise ProvenanceError("INT4_cell_rank finalized V file binding mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (type(value) is not torch.Tensor or value.layout != torch.strided
            or value.dtype != torch.float32 or value.device.type != "cpu"
            or tuple(value.shape) != tuple(expected_shape) or not bool(torch.isfinite(value).all())):
        raise ProvenanceError("INT4_cell_rank finalized V has the wrong shape or dtype")
    if core.tensor_sha256(value) != tensor_sha:
        raise ProvenanceError("INT4_cell_rank finalized V tensor binding mismatch")
    if upstream_naive._tensor_hash(value) != _require_sha256(
            registration.get("input_tensor_sha256"), "INT4_cell_rank canonical input SHA256"):
        raise ProvenanceError("INT4_cell_rank finalized V canonical input binding mismatch")
    return value.contiguous()


def embedding_evaluation(target: torch.Tensor, truth_z: torch.Tensor) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment
    if (not isinstance(target, torch.Tensor) or not isinstance(truth_z, torch.Tensor)
            or target.shape != truth_z.shape or target.ndim != 2
            or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(truth_z).all())):
        raise ValueError("invalid source/embedding shape or values")
    cosine = F.normalize(target.double(), dim=1) @ F.normalize(truth_z.double(), dim=1).T
    rows, cols = linear_sum_assignment(-cosine.numpy())
    pairs = [{"source_index": int(i), "truth_index": int(j), "cosine": float(cosine[i, j])}
             for i, j in zip(rows, cols)]
    return {
        "matching": "Hungarian_max_embedding_cosine_offline_only",
        "per_source": pairs,
        "mean_cosine": sum(row["cosine"] for row in pairs) / len(pairs),
        "all_sources_included": True,
        "adapter": "abs(S_hat); no scaling/permutation before FIA",
        "labels_consumed": False,
    }


def _image_evaluation(folder: Path, truth_path: Path, originals: torch.Tensor):
    image = evaluate_candidate(folder / "candidate.pt", truth_path, include_labels=False)
    candidate = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)
    pixels = candidate["pixels"].float()
    from .report_full_update_iterations import _make_lpips_metric
    metric, lpips_provenance = _make_lpips_metric()
    values = []
    for pair in image["per_image"]:
        i, j = pair["candidate_index"], pair["truth_index"]
        metric.reset()
        score = metric((pixels[i:i + 1] - .5) * 2, (originals[j:j + 1] - .5) * 2)
        if score.numel() != 1 or not bool(torch.isfinite(score).all()):
            raise ProvenanceError("nonfinite LPIPS score")
        pair["lpips"] = float(score.detach().cpu())
        values.append(pair["lpips"])
    image["lpips_provenance"] = lpips_provenance
    image["mean_lpips"] = sum(values) / len(values)
    image["labels_consumed"] = False
    return image


def _write_figures(output: Path, originals: torch.Tensor, candidates: dict[str, torch.Tensor],
                   reports: dict[str, dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for compact in (False, True):
        # Four triplets per compact row require twelve columns (4 * original,
        # naive, cell-rank).  The full view remains one triplet per image.
        rows, cols = (4, 12) if compact else (16, 3)
        figure, axes = plt.subplots(rows, cols, figsize=(18, 24) if compact else (9, 42))
        axes = axes.reshape(rows, cols)
        try:
            for truth_index in range(16):
                row, offset = (truth_index // 4, (truth_index % 4) * 3) if compact else (truth_index, 0)
                axes[row, offset].imshow(originals[truth_index].permute(1, 2, 0))
                axes[row, offset].set_title(f"{truth_index + 1} Original")
                for col, arm in enumerate(("naive", "cell_rank"), 1):
                    pair = next(item for item in reports[arm]["per_image"] if item["truth_index"] == truth_index)
                    axes[row, offset + col].imshow(candidates[arm][pair["candidate_index"]].permute(1, 2, 0))
                    axes[row, offset + col].set_title("INT4 naive" if arm == "naive" else "INT4 cell-rank")
                for col in range(3):
                    axes[row, offset + col].axis("off")
            figure.tight_layout(pad=.5)
            figure.savefig(output / ("all16_compact.png" if compact else "all16_comparison.png"), dpi=110)
        finally:
            plt.close(figure)


def _atomic_text(path: Path, content: str) -> None:
    """Replace one result artifact atomically within its output directory."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _comparison_report(comparison: dict[str, Any]) -> str:
    deltas = comparison.get("cell_rank_minus_naive", {})
    lines = [
        "# Signed INT4 naive / external cell-rank comparison",
        "",
        f"[FACT] evaluation_status={comparison.get('evaluation_status')}; plot_status={comparison.get('plot_status')}.",
        "[FACT] Both completed arms use the original CPA25k, abs source adapter, and Direct FIA25k.",
        "[FACT] All 16 final candidates are evaluated label-free with one pixel-MSE Hungarian matching per arm.",
        "[FACT] Cell/code and scale checks describe the inferred update input; they do not describe a reconstructed local update.",
        "[FACT] Cell-rank minus naive mean deltas: " + json.dumps(deltas, allow_nan=False),
    ]
    if comparison.get("offline_update_error") is not None:
        lines.append("[FACT] Evaluation-only true-G errors: "
                     + json.dumps(comparison["offline_update_error"], allow_nan=False))
    if comparison.get("plot_status") == "FAILED":
        lines.append("[FACT] Numerical comparison JSON was committed before plotting and is retained despite the figure failure.")
        lines.append("[UNKNOWN] Plotting failure: " + str(comparison.get("plot_error")))
    else:
        lines.append("[UNKNOWN] No threshold, significance, or generalization is established by one batch/seed.")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def evaluate(*, naive: str | Path, cell_rank: str | Path,
             public_observation: str | Path, private_observation: str | Path,
             output: str | Path) -> dict[str, Any]:
    """Evaluate finalized candidates once; no attack or label path is called."""
    audit = validate_provenance(
        naive=naive, cell_rank=cell_rank, public_observation=public_observation,
        private_observation=private_observation, output=output)
    paths = {name: Path(value) for name, value in audit["paths"].items()}
    q, s = _validated_public_input(paths["public_observation"])
    naive_registration = audit["registrations"]["naive"]
    cell_registration = audit["registrations"]["cell_rank"]
    naive_estimate = (q.to(dtype=torch.float32) * s).contiguous()
    naive_input_sha = _require_sha256(naive_registration.get("input_tensor_sha256"),
                                      "INT4_naive canonical input SHA256")
    if upstream_naive._tensor_hash(naive_estimate) != naive_input_sha:
        raise ProvenanceError("INT4_naive finalized input binding mismatch")
    cell_estimate = _load_bound_cell_v(cell_registration, expected_shape=tuple(q.shape))
    # Only after all public/final-arm bindings have passed may the evaluator
    # deserialize the private true-G payload.
    true_g, delta_convention = _load_true_update(
        paths["private_observation"] / "raw_updates.pt", expected_shape=tuple(q.shape))
    update_errors = {
        "naive_sq": update_error_metrics(
            naive_estimate, true_g, arm="INT4_naive_sq", estimate_input_sha256=naive_input_sha,
            delta_convention=delta_convention),
        "cell_rank_V": update_error_metrics(
            cell_estimate, true_g, arm="INT4_cell_rank_V",
            estimate_input_sha256=_require_sha256(cell_registration.get("input_tensor_sha256"),
                                                  "INT4_cell_rank canonical input SHA256"),
            delta_convention=delta_convention),
    }
    truth_path = paths["private_observation"] / "evaluation_truth.pt"
    truth = torch.load(truth_path, map_location="cpu", weights_only=True)
    if "x" not in truth or "z" not in truth:
        raise ProvenanceError("evaluation truth must contain x and z")
    # A sender artifact may carry private labels for other approved analyses;
    # this evaluator never reads or indexes that field.
    images = truth["x"]
    if images.ndim != 4 or tuple(images.shape) != (16, 3, 224, 224):
        raise ProvenanceError("evaluation requires all sixteen 224x224 images")
    mean = images.new_tensor((.485, .456, .406)).reshape(1, 3, 1, 1)
    std = images.new_tensor((.229, .224, .225)).reshape(1, 3, 1, 1)
    originals = (images * std + mean).clamp(0, 1)
    output_path = paths["output"]
    output_path.mkdir(parents=True)
    reports, candidates, embeddings = {}, {}, {}
    for arm in ("naive", "cell_rank"):
        folder = paths[arm]
        reports[arm] = _image_evaluation(folder, truth_path, originals)
        candidate = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)
        candidates[arm] = candidate["pixels"].float()
        source = torch.load(folder / "sources.pt", map_location="cpu", weights_only=True)
        if set(source) != {"S_hat", "target_z"}:
            raise ProvenanceError(f"{arm} sources artifact has unexpected fields")
        embeddings[arm] = embedding_evaluation(source["target_z"], truth["z"])
    paired = []
    for truth_index in range(16):
        rows = {arm: next(item for item in reports[arm]["per_image"] if item["truth_index"] == truth_index)
                for arm in ("naive", "cell_rank")}
        paired.append({
            "truth_index": truth_index,
            "naive_candidate_index": rows["naive"]["candidate_index"],
            "cell_rank_candidate_index": rows["cell_rank"]["candidate_index"],
            **{key: rows["cell_rank"][key] - rows["naive"][key]
               for key in ("mse", "psnr_db", "ssim", "lpips")},
        })
    deltas = {f"cell_rank_minus_naive_{key}": reports["cell_rank"][f"mean_{key}"] - reports["naive"][f"mean_{key}"]
              for key in ("mse", "psnr_db", "ssim", "lpips")}
    comparison = {
        # Commit numerical results before plotting so figure errors cannot erase
        # an otherwise valid offline metric result.
        "status": "PENDING_PLOTS",
        "evaluation_status": "PENDING_PLOTS",
        "numeric_results_status": "COMPLETED",
        "plot_status": "PENDING",
        "full_experiment_status": "PENDING",
        "format": EVALUATOR_FORMAT,
        "arms": {arm: {"image_metrics": reports[arm], "embedding_oracle": embeddings[arm],
                        "final_result": audit["results"][arm]} for arm in ("naive", "cell_rank")},
        "cell_rank_minus_naive": deltas,
        "paired_cell_rank_minus_naive_by_truth": paired,
        "all_images_included": True,
        "matching": "one pixel-MSE Hungarian matching per arm; no rematching for LPIPS",
        "labels_consumed": False,
        "oracle_scope": "offline embedding analysis only; no source/image reorder before FIA",
        "provenance": audit,
        "truth_sha256": audit["evaluation_truth_sha256"],
        "offline_update_error": update_errors,
        "offline_private_loaded": True,
        "success_threshold": None,
        "generalization": "NOT_ESTABLISHED_SINGLE_BATCH_SINGLE_SEED",
    }
    _atomic_json(output_path / "comparison.json", comparison)
    _atomic_text(output_path / "report.md", _comparison_report(comparison))
    try:
        _write_figures(output_path, originals, candidates, reports)
        figure_paths = (output_path / "all16_comparison.png", output_path / "all16_compact.png")
        if not all(path.is_file() and path.stat().st_size > 0 for path in figure_paths):
            raise RuntimeError("required all16 figure artifact is missing or empty")
    except Exception as error:
        comparison.update({
            "status": "FAILED",
            "evaluation_status": "FAILED",
            "plot_status": "FAILED",
            "full_experiment_status": "FAILED",
            "plot_error_type": type(error).__name__,
            "plot_error": str(error),
        })
        _atomic_json(output_path / "comparison.json", comparison)
        _atomic_text(output_path / "report.md", _comparison_report(comparison))
        return comparison
    comparison.update({
        "status": "COMPLETED",
        "evaluation_status": "COMPLETED",
        "plot_status": "COMPLETED",
        "full_experiment_status": "COMPLETED",
        "figure_artifacts": ["all16_comparison.png", "all16_compact.png"],
    })
    _atomic_json(output_path / "comparison.json", comparison)
    _atomic_text(output_path / "report.md", _comparison_report(comparison))
    return comparison


def _resolve(value: str | Path) -> Path:
    return Path(value).resolve()


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naive", type=Path, required=True)
    parser.add_argument("--cell-rank", type=Path, required=True)
    parser.add_argument("--public-observation", type=Path, required=True)
    parser.add_argument("--private-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    result = evaluate(**vars(args))
    print(json.dumps({"status": result.get("status"),
                      "evaluation_status": result.get("evaluation_status"),
                      "plot_status": result.get("plot_status")}, allow_nan=False))
    return 0 if result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
