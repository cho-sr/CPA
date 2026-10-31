"""Authorized short INT4 CoRA: two public samples, restored NE CPA and Direct FIA."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "quantization_aware_cpa"), str(ROOT)]
import torch
from qacpa import upstream_naive as backend
from qacpa.deterministic_hybrid_setup import configure_runtime
from qacpa.signed_baseline_observation import CHECKPOINT_SHA, read_bundle
from qacpa.signed_bits_randomized import sample_matrix

TASK = "CORA_INT4_NE_SHORT_20261006"
HERE = Path(__file__).resolve().parent
PUBLIC = ROOT / "quantization_aware_cpa/results/cora_int4_ne_short_20261006"
PRIVATE = ROOT / "quantization_aware_cpa/results_private/cora_int4_ne_short_20261006"
OBSERVATION = ROOT / "quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation"
TRUTH = ROOT / "quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
MANIFEST_SHA = "9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69"
TRUTH_SHA = "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2"
CPA_STEPS = FIA_STEPS = 500
MEMBER_SEEDS = [1234, 1001237]
SEED = 42


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hashes():
    files = [Path(__file__), ROOT / "sparsified_update_cpa/cora_ica.py",
             ROOT / "quantization_aware_cpa/qacpa/signed_baseline_observation.py",
             ROOT / "quantization_aware_cpa/qacpa/signed_bits_randomized.py",
             ROOT / "quantization_aware_cpa/qacpa/signed_int8_screen3_fia.py",
             ROOT / "quantization_aware_cpa/qacpa/full_update_evaluation.py",
             ROOT / "quantization_aware_cpa/qacpa/report_full_update_iterations.py"]
    return {str(p): digest(p) for p in files} | backend.source_hashes()


def input_hashes():
    if digest(OBSERVATION / "manifest.json") != MANIFEST_SHA or digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Registered public observation/checkpoint identity changed")
    manifest = read(OBSERVATION / "manifest.json")
    if manifest["codec_metadata"]["bits"] != 4 or manifest["observation_source"] != "single_client_single_local_update":
        raise ValueError("Expected archived single-client INT4 observation")
    result = {str(OBSERVATION / "manifest.json"): MANIFEST_SHA, str(CHECKPOINT): CHECKPOINT_SHA}
    for layer in manifest["layers"].values():
        path = OBSERVATION / layer["file"]
        if path.resolve().parent != OBSERVATION.resolve() or path.is_symlink() or digest(path) != layer["sha256"]:
            raise ValueError("Public packet hash/path failure")
        result[str(path)] = layer["sha256"]
    return result


def handoff(status):
    path = ROOT / ".ai/EXECUTION_HANDOFF.md"
    text = (f"# Short INT4 CoRA with negentropy restored\n\nTASK_ID: {TASK}\nPLAN_VERSION: 1\nSTATUS: {status}\n"
            f"ASSIGNED_ROLE: CODEX_EXECUTOR\nCLAIMED_BY: CURRENT_CONVERSATION:{os.environ.get('CODEX_THREAD_ID', 'recorded_at_launch')}\n"
            "USER_EXECUTION_AUTHORIZATION: 짧게 돌려볼수있나 4비트로; preceding direct assignment: 걍 negentropy 켜줘 너가\n\n"
            "One short run only: archived single-client INT4 B16/T10 observation; frozen VGG16 Conv, all FC local SGD0.001, seed42. "
            "Two bounded-cell candidates with registered seeds1234/1001237; exact public codes and per-tensor scale preserved. "
            "Each uses restored upstream log-cosh NE (ne1), Adam0.001, decor5.3/T7.7/nv0.13/l1=5/TV0.1, CPA500 steps; "
            "then unchanged Direct FIA500 steps, Adam0.1, cosine scheduler, fi1/TV0.1. All16 sources/images kept. "
            "Rank final candidates by public feature-cosine distance+0.1 normalized TV, tie by member index. "
            "Only after both attacks and public selection finish, evaluate the winner on all16 images with frozen pixel-MSE Hungarian pairing "
            "and PSNR/SSIM/VGG LPIPS. No new victim training, private feedback, additional arm, restart or success cutoff.\n\n"
            f"Worker: {HERE / 'run_short.py'} run\nPublic: {PUBLIC}\nPrivate: {PRIVATE}\nRecord: {HERE / 'execution_record.json'}\n"
            "Use one hidden persistent worker and one verified OS-exit callback to this same conversation. End the model turn after launch; no progress polling.\n")
    if path.exists() and status == "READY":
        archive = HERE / "previous_execution_handoff.md"
        if archive.exists():
            raise FileExistsError("Previous handoff archive already exists")
        shutil.copyfile(path, archive)
    elif path.exists() and f"TASK_ID: {TASK}" not in path.read_text(encoding="utf-8"):
        raise ValueError("Another active handoff replaced this task")
    path.write_text(text, encoding="utf-8")


def prepare():
    configure_runtime(SEED)
    if PUBLIC.exists() or PRIVATE.exists():
        raise FileExistsError("Do not overwrite or restart an existing short run")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise RuntimeError("CUDA with 8GiB free is required")
    if digest(TRUTH) != TRUTH_SHA:
        raise ValueError("Registered evaluation bundle changed")
    from qacpa.report_full_update_iterations import _verified_lpips_weights
    registration = dict(task_id=TASK, status="READY", bits=4, source_count=16,
                        cpa_steps=CPA_STEPS, fia_steps=FIA_STEPS, member_seeds=MEMBER_SEEDS,
                        negentropy_enabled=True, input_hashes=input_hashes(), code_hashes=code_hashes(),
                        private_truth_sha256=TRUTH_SHA, private_loaded=False,
                        lpips_weights=_verified_lpips_weights(), selection="feature_cosine_distance_plus_0.1_normalized_TV",
                        authorized_thread=os.environ.get("CODEX_THREAD_ID"))
    PUBLIC.mkdir(parents=True)
    write(PUBLIC / "registration.json", registration)
    write(HERE / "execution_record.json", dict(task_id=TASK, status="READY", public=str(PUBLIC), private=str(PRIVATE)))
    handoff("READY")
    print("PREPARED " + TASK, flush=True)


def load_cora(upstream):
    previous = sys.modules.get("gradient_inversion")
    try:
        sys.modules["gradient_inversion"] = upstream.gradient_inversion
        spec = importlib.util.spec_from_file_location("task_cora_ica", ROOT / "sparsified_update_cpa/cora_ica.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CoRACocktailPartyAttack
    finally:
        if previous is None:
            sys.modules.pop("gradient_inversion", None)
        else:
            sys.modules["gradient_inversion"] = previous


def validate_member(folder):
    folder = Path(folder)
    record = read(folder / "result.json")
    for kind, expected in [("cpa", CPA_STEPS), ("fia", FIA_STEPS)]:
        checkpoint = torch.load(folder / kind / f"{kind}_checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["completed_iterations"] != expected or record[f"{kind}_completed"] != expected:
            raise ValueError("Incomplete short-run optimization")
    if not record["sampler"]["code_consistency"]["all_equal"] or not record["sampler"]["scale_reproduced"]:
        raise ValueError("Sample did not reproduce public codes/scale")
    if not math.isfinite(record["negentropy_last"]) or record["negentropy_last"] >= 0:
        raise ValueError("Negentropy was not active")
    if digest(folder / "candidate.pt") != record["candidate_sha256"]:
        raise ValueError("Candidate integrity failure")
    pixels = torch.load(folder / "candidate.pt", map_location="cpu", weights_only=True)["pixels"]
    if pixels.shape != (16, 3, 224, 224) or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all() or ((pixels < 0) | (pixels > 1)).any():
        raise ValueError("Invalid final pixels")
    return record, pixels


def run_member(index, q, scale, upstream, cpa_class):
    from qacpa.signed_int8_screen3_fia import score_attack
    folder = PUBLIC / f"member{index:02d}"
    folder.mkdir()
    X, sampler = sample_matrix(q, scale, bits=4, seed=MEMBER_SEEDS[index])
    torch.manual_seed(SEED)
    args = SimpleNamespace(n_iter=CPA_STEPS, lr=.001, opt="adam", sch="none", decor=5.3, T=7.7,
                           ne=1., nv=.13, l1=5., tv=.1, use_labels=False, batch_size=16,
                           sigma=0., C=0., ds="imagenet", model="vgg16", defense="observed",
                           exp_name="cora_short", attack="cp", fi_method="direct", fl_alg="fedavg")
    device = torch.device("cuda")
    cpa_dir = folder / "cpa"
    cpa_dir.mkdir()
    with backend._runtime_environment(upstream, device, cpa_dir):
        Path(upstream.utils.get_attack_exp_path(args)).mkdir(parents=True, exist_ok=True)
        attack = cpa_class(backend._ObservedMatrixModel(), [X.to(device)], None, args, 0)
    log, cpa_seconds = backend._optimize(attack, CPA_STEPS, 100, cpa_dir, "cpa", device)
    target = attack.get_rec().detach().cpu().abs().contiguous()
    gradient_norm = float(attack.W_hat.grad.norm())
    if not math.isfinite(gradient_norm) or not torch.isfinite(target).all():
        raise ValueError("Nonfinite restored-NE separation")
    torch.save({"target_z": target}, folder / "target.pt")
    ne_last = float(log[-1]["loss_ne"])
    del attack, X
    gc.collect()
    torch.cuda.empty_cache()

    fia_dir = folder / "fia"
    fia_dir.mkdir()
    fia_args = SimpleNamespace(tv=.1, opt_fi="adam", sch_fi="cosine", lr_fi=.1,
                              ds="imagenet", n_iter_fi=FIA_STEPS, use_labels=False, gm=0., fi=1.)
    with backend._runtime_environment(upstream, device, fia_dir):
        model = backend.VGG16FeatureAdapter().to(device)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        attack = upstream.feature_inversion.Direct(target.to(device), model, fia_args, None, None)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    _, fia_seconds = backend._optimize(attack, FIA_STEPS, 100, fia_dir, "fia", device)
    score = score_attack(attack, target)
    pixels = attack.get_rec().detach().cpu().contiguous().clamp(0, 1)
    torch.save({"pixels": pixels}, folder / "candidate.pt")
    result = dict(member=index, cpa_completed=CPA_STEPS, fia_completed=FIA_STEPS, sampler=sampler,
                  negentropy_last=ne_last, cpa_gradient_norm=gradient_norm, score=score,
                  cpa_seconds=cpa_seconds, fia_seconds=fia_seconds,
                  candidate_sha256=digest(folder / "candidate.pt"), private_loaded=False)
    write(folder / "result.json", result)
    del model, attack
    gc.collect()
    torch.cuda.empty_cache()
    validate_member(folder)
    return result


def evaluate(winner):
    from qacpa.full_update_evaluation import evaluate_candidate
    from qacpa.report_full_update_iterations import _make_lpips_metric
    folder = PUBLIC / f"member{winner:02d}"
    _, pixels = validate_member(folder)
    if digest(TRUTH) != TRUTH_SHA:
        raise ValueError("Evaluation truth identity changed")
    metrics = evaluate_candidate(folder / "candidate.pt", TRUTH, include_labels=False)
    payload = torch.load(TRUTH, map_location="cpu", weights_only=True)
    originals = (payload["x"] * pixels.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
                 + pixels.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)).clamp(0, 1)
    metrics["private_keys_materialized"] = sorted(payload)
    del payload
    metric, provenance = _make_lpips_metric()
    for pair in metrics["per_image"]:
        i, j = pair["candidate_index"], pair["truth_index"]
        metric.reset()
        pair["lpips"] = float(metric((pixels[i:i+1] - .5) * 2, (originals[j:j+1] - .5) * 2))
        if not math.isfinite(pair["lpips"]):
            raise ValueError("Nonfinite LPIPS")
    metrics.update(mean_lpips=sum(p["lpips"] for p in metrics["per_image"]) / 16,
                   lpips="computed_offline_vgg", lpips_provenance=provenance,
                   private_fields_consumed=["x"], selected_member=winner)
    PRIVATE.mkdir(parents=True)
    write(PRIVATE / "metrics.json", metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    for slot, pair in enumerate(sorted(metrics["per_image"], key=lambda p: p["truth_index"])):
        row, col = divmod(slot, 4)
        i, j = pair["candidate_index"], pair["truth_index"]
        for offset, tensor, label in [(0, originals[j], f"Original {j}"), (1, pixels[i], f"CoRA {i}")]:
            ax = axes[row, 2*col+offset]
            ax.imshow(tensor.permute(1, 2, 0).numpy())
            ax.set_title(label, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(PRIVATE / "comparison.png", dpi=140)
    plt.close(fig)
    return metrics


def run():
    started = time.perf_counter()
    configure_runtime(SEED)
    registration = read(PUBLIC / "registration.json")
    if registration["status"] != "READY" or registration["input_hashes"] != input_hashes() or registration["code_hashes"] != code_hashes():
        raise ValueError("Frozen short-run inputs/code changed, or already started")
    registration["status"] = "RUNNING"
    write(PUBLIC / "registration.json", registration)
    handoff("RUNNING")
    write(HERE / "execution_record.json", dict(task_id=TASK, status="RUNNING", pid=os.getpid()))
    torch.cuda.reset_peak_memory_stats()
    _, packets = read_bundle(OBSERVATION)
    q, scale = packets["classifier.0.weight"]
    del packets
    upstream = backend.load_upstream_modules()
    cpa_class = load_cora(upstream)
    members = [run_member(i, q, scale, upstream, cpa_class) for i in range(2)]
    winner = min(range(2), key=lambda i: (members[i]["score"]["score"], i))
    selection = dict(selected_member=winner, rule="feature_cosine_distance_plus_0.1_normalized_TV",
                     private_metrics_used=False, members=members)
    write(PUBLIC / "selection.json", selection)
    for i in range(2):
        validate_member(PUBLIC / f"member{i:02d}")
    if registration["input_hashes"] != input_hashes() or registration["code_hashes"] != code_hashes():
        raise ValueError("Provenance changed during attack")
    metrics = evaluate(winner)
    seconds = time.perf_counter() - started
    report = (f"# {TASK}\n\nCompleted short INT4 CoRA with restored negentropy. "
              "Two randomized bounded-cell members, CPA500 + Direct FIA500 each; same16/B16/T10; seed42. "
              "Public codes/scales reproduced exactly. Selection uses feature cosine + 0.1 normalized TV, with no private feedback.\n\n"
              f"Selected member: {winner}; PSNR {metrics['mean_psnr_db']:.6f}dB, SSIM {metrics['mean_ssim']:.6f}, "
              f"VGG LPIPS {metrics['mean_lpips']:.6f}. Runtime {seconds:.1f}s.\n\n"
              "All16 final images evaluated once with shared pixel-MSE Hungarian pairing. "
              "This short run checks behavior; no restoration-success threshold or comparison to full-budget results is claimed.\n\n"
              "![Original and CoRA reconstruction](comparison.png)\n")
    (PRIVATE / "report.md").write_text(report, encoding="utf-8")
    write(HERE / "execution_record.json", dict(task_id=TASK, status="COMPLETED", pid=os.getpid(),
          total_seconds=seconds, selected_member=winner, members=members,
          mean_psnr_db=metrics["mean_psnr_db"], mean_ssim=metrics["mean_ssim"], mean_lpips=metrics["mean_lpips"],
          peak_cuda_allocated=torch.cuda.max_memory_allocated(), report=str(PRIVATE / "report.md"),
          comparison=str(PRIVATE / "comparison.png"), candidate_sha256=members[winner]["candidate_sha256"]))
    handoff("COMPLETED")
    print("COMPLETED " + str(PRIVATE / "report.md"), flush=True)


if __name__ == "__main__":
    try:
        {"prepare": prepare, "run": run}[sys.argv[1]]()
    except Exception as error:
        traceback.print_exc()
        write(HERE / "execution_record.json", dict(task_id=TASK, status="FAILED", error=str(error), pid=os.getpid()))
        if sys.argv[1] == "run":
            handoff("FAILED")
        sys.exit(1)
