"""User-authorized single completion notice to the existing bound Planner."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

import psutil

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def main():
    receipt_path=OUT/"planner_final_delivery.json"
    if receipt_path.exists():
        raise RuntimeError("Final handoff already attempted; do not automatically resend")
    result=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
    assert result["status"]=="COMPLETED"
    peer=json.loads((ROOT/".ai/relay-runs/visible/peers.json").read_text(encoding="utf-8"))["CODEX"]
    process=psutil.Process(peer["pid"])
    expected_ms=int(re.search(r"\d+",peer["started"]).group())
    assert process.name().lower()=="codex.exe" and abs(process.create_time()*1000-expected_ms)<2
    assert peer["id"]!=result["claim"]["session_id"]
    executable=process.exe()
    means=result["image_means"]
    reference,naive=means["reference"],means["naive"]
    message=(
        "[CPA Executor final result | user-authorized ONE handoff]\n"
        "TASK_ID: CPA_SIGNED_BASELINE_B16_T10_20260914; PLAN_VERSION: 1; STATUS: COMPLETED.\n"
        "Existing high executor finished the approved job. Collection once with all six FC ordinary-SGD/replay deltas exactly equal; "
        "INT4 naive and FP32 Reference each completed official CPA 25000 and Direct FIA 25000. "
        "All stage exit codes 0. Both final all16 candidates are hash-fixed; no arm was rerun.\n"
        f"FP32 Reference: PSNR {reference['mean_psnr_db']:.6f} dB, SSIM {reference['mean_ssim']:.6f}, VGG LPIPS {reference['mean_lpips']:.6f}.\n"
        f"INT4 naive: PSNR {naive['mean_psnr_db']:.6f} dB, SSIM {naive['mean_ssim']:.6f}, VGG LPIPS {naive['mean_lpips']:.6f}.\n"
        "Naive minus reference: "+json.dumps(result["naive_minus_reference"])+".\n"
        "Single batch/seed only: naive is worse on these three image means; no general success threshold/significance claim. "
        "FP32 Reference is a privileged precompression reference. Both arms share identical FIA initial hashes and numeric settings; "
        "upstream adaptive_avg_pool2d_backward_cuda nondeterminism warning remains recorded, no complete-determinism claim.\n"
        "Planner review 01 was applied before offline evaluation: distinct arm identities, actual sender/input hash binding, "
        "checkpoint/seed/config/source checks, and private-only output guards PASS. Initial 158 CPU tests plus 25 focused review tests PASS. "
        "No codec/math/budget/handoff-plan changes. Original official/codec/affine/STATE/RELAY sources preserved; "
        "existing image evaluator only adds a default-preserving labels-free option.\n"
        f"Report: {OUT/'report.md'}\nResults/hashes/commands: {OUT/'results.json'}\n"
        f"Evaluation metrics and paired all16 differences: {ROOT/'quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/evaluation/comparison.json'}\n"
        f"All16 figure: {ROOT/'quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/evaluation/all16_compact.png'}\n"
        "Show this result once to the user and END YOUR TURN. Do not poll, launch further work/experiments, "
        "restart an arm, or reply with another monitoring loop. No reply to high is required."
    )
    (OUT/"planner_final_notice.txt").write_text(message,encoding="utf-8")
    record={"state":"SENDING","peer":peer,"executable":executable,
            "attempted_at_utc":datetime.now(timezone.utc).isoformat(),"authorized_one_time":True}
    receipt_path.write_text(json.dumps(record,indent=2),encoding="utf-8")
    try:
        completed=subprocess.run([executable,"queue","--thread",peer["id"],"--message",message],
                                 cwd=ROOT,capture_output=True,timeout=30,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        (OUT/"planner_final_stdout.log").write_bytes(completed.stdout)
        (OUT/"planner_final_stderr.log").write_bytes(completed.stderr)
        record.update(state="QUEUED" if completed.returncode==0 else "FAILED",exit_code=completed.returncode,
                      receipt=completed.stdout.decode("utf-8",errors="replace").strip(),
                      stderr=completed.stderr.decode("utf-8",errors="replace").strip())
    except Exception as error:
        record.update(state="DELIVERY_UNCONFIRMED",error=str(error))
    receipt_path.write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2))
    return 0 if record["state"]=="QUEUED" else 1


if __name__=="__main__":
    raise SystemExit(main())
