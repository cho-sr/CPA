"""Reporting-only audit of the stopped CoRA run; no optimization or private loading."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PUBLIC = ROOT / "quantization_aware_cpa/results/cora_int4_ne_short_20261006"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


registration = json.loads((PUBLIC / "registration.json").read_text(encoding="utf-8"))
for group in ("input_hashes", "code_hashes"):
    for name, expected in registration[group].items():
        if digest(name) != expected:
            raise ValueError(f"Provenance changed: {name}")
checkpoint_path = PUBLIC / "member00/cpa/cpa_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
target_path = PUBLIC / "member00/target.pt"
target = torch.load(target_path, map_location="cpu", weights_only=True)["target_z"]
assert checkpoint["completed_iterations"] == 500
assert checkpoint["log"][-1]["iter"] == 499
assert all(entry["loss_ne"] < 0 for entry in checkpoint["log"])
assert target.shape == (16, 25088) and torch.isfinite(target).all()
assert not (PUBLIC / "member00/fia/fia_checkpoint.pt").exists()
assert not (PUBLIC / "member01").exists()
assert not (PUBLIC / "selection.json").exists()
record_path = HERE / "execution_record.json"
original = record_path.read_bytes()
failure = json.loads(original.decode("utf-8"))
assert failure["status"] == "FAILED" and "adaptive_avg_pool2d_backward_cuda" in failure["error"]
archive = HERE / "worker_failure_original.json"
with archive.open("xb") as stream:
    stream.write(original)
audit = {
    "task_id": registration["task_id"], "status": "FAILED_VERIFIED",
    "exit_code": 1, "public_input_and_code_hashes_verified": True,
    "members": [
        {"member": 0, "cpa_completed": 500, "fia_completed": 0,
         "negentropy_active_in_all_saved_logs": True,
         "first_loss_ne": checkpoint["log"][0]["loss_ne"],
         "last_loss_ne": checkpoint["log"][-1]["loss_ne"],
         "cpa_optimizer_seconds": checkpoint["wall_time_sec"],
         "target_shape": list(target.shape), "target_finite": True,
         "cpa_checkpoint_sha256": digest(checkpoint_path), "target_sha256": digest(target_path)},
        {"member": 1, "cpa_completed": 0, "fia_completed": 0, "status": "NOT_STARTED"},
    ],
    "failure_stage": "member00 first FIA backward, before its first optimizer step",
    "failure_reason": failure["error"],
    "implementation_issue": "The short-run adapter retained strict deterministic algorithms from configure_runtime instead of the established FIA seed_all warn_only mode.",
    "public_selection_completed": False, "private_evaluation_run": False,
    "reconstructed_image_candidate_saved": False, "quality_metrics": None,
    "optimization_restarted": False, "new_experiments_run": False,
}
save(HERE / "completion_verification.json", audit)
report = (
    "# CORA_INT4_NE_SHORT_20261006\n\nStatus: FAILED, partial separation verified. Worker exited1.\n\n"
    "Member0: CPA500/500 completed; FIA0/500 completed. Member1 was not started. "
    "The saved target has shape16x25088 and is finite. Public input and registered code hashes match.\n\n"
    f"Negentropy was active in all six saved loss logs: first {audit['members'][0]['first_loss_ne']:.9f}, "
    f"last {audit['members'][0]['last_loss_ne']:.9f}. The contrast was included in the total loss; this does not establish reconstruction quality.\n\n"
    "FIA stopped during its first backward call because adaptive_avg_pool2d_backward_cuda does not support "
    "the strict deterministic-algorithm mode retained by this task's short-run adapter. The established FIA helper "
    "uses seed_all(...), which selects warn_only=True; that runtime setup was omitted here. "
    "This is an execution-adapter error, not evidence that the enabled negentropy failed.\n\n"
    "No final pixels, public candidate selection, private evaluation, PSNR/SSIM/LPIPS or comparison image were produced. "
    "No optimization was restarted. Original worker failure is preserved in worker_failure_original.json.\n"
)
(HERE / "report.md").write_text(report, encoding="utf-8")
failure.update(completion_verification=str(HERE / "completion_verification.json"),
               report=str(HERE / "report.md"), completion="PARTIAL_SEPARATION_ONLY_VERIFIED", members=audit["members"],
               public_selection_completed=False, private_evaluation_run=False, reporting_exit_code=0)
save(record_path, failure)
with (ROOT / ".ai/EXECUTION_HANDOFF.md").open("a", encoding="utf-8") as stream:
    stream.write("\n## Verified failure (2026-10-06)\n\nMember0 completed CPA500 with active negentropy; FIA failed before its first optimizer step due to strict deterministic CUDA adaptive pooling. Member1 did not start. No candidate selection, reconstruction pixels or private quality metrics. No restart. Reporting audit: " + str(HERE / "completion_verification.json") + "\n")
print(json.dumps({"status": "FAILED_VERIFIED", "cpa_completed": 500, "fia_completed": 0,
                  "negentropy_last": audit["members"][0]["last_loss_ne"], "report": str(HERE / "report.md")}))
