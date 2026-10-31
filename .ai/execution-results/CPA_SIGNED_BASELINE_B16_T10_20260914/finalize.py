"""Read completed artifacts, verify scope/hashes, and produce the job handback."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PUBLIC = ROOT/"quantization_aware_cpa/results/signed_int4_baseline_B16_T10"
PRIVATE = ROOT/"quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    claim=read(OUT/"claim.json")
    executions={stage:read(OUT/f"{stage}_execution.json") for stage in ("tests","collect","naive","reference","evaluate")}
    assert all(v["status"]=="COMPLETED" and v["exit_status"]==0 for v in executions.values())
    comparison=read(PRIVATE/"evaluation/comparison.json")
    assert comparison["status"]=="COMPLETED" and comparison["fia_initializations_equal"] and comparison["candidate_hashes_unchanged"]
    arms={"naive":PUBLIC/"naive","reference":PRIVATE/"reference"}
    for name,folder in arms.items():
        result=read(folder/"result.json")
        assert result["candidate_sha256"]==digest(folder/"candidate.pt")
        assert result["sources_sha256"]==digest(folder/"sources.pt")
        assert result["cpa_iterations"]==25000 and result["fia_iterations"]==25000
    registered={arm:read(folder/"registered_arm.json") for arm,folder in arms.items()}
    for key in ("cpa","fia","seed","input_conversion","source_adapter","checkpoint_sha256","source_hashes"):
        assert registered["naive"][key]==registered["reference"][key],key
    for kind in ("cpa","fia"):
        for arm,folder in arms.items():
            manifest=read(folder/kind/f"{kind}_manifest.json")
            assert Path(manifest["cache_root"]).is_relative_to(folder)
    before=read(OUT/"sources_before.json")
    after={name:digest(ROOT/name) for name in before}
    changed=[name for name in before if before[name]!=after[name]]
    assert changed==["quantization_aware_cpa/qacpa/full_update_evaluation.py"],changed
    # Review 01 changes only the previously unused offline evaluator. Preserve
    # the earlier stage hashes instead of rewriting their execution records.
    revision=read(OUT/"planner_review_01_revision.json")
    for stage in ("collect","naive","reference","evaluate"):
        for name,sha in executions[stage]["source_hashes"].items():
            if Path(name).name=="signed_baseline_evaluation.py" and stage!="evaluate":
                assert sha==revision["old_evaluation_sha256"],(stage,name)
                assert digest(ROOT/name)==revision["new_evaluation_sha256"],(stage,name)
            else:
                assert digest(ROOT/name)==sha,(stage,name)
    assert digest(ROOT/"quantization_aware_cpa/qacpa/signed_baseline.py")==revision["core_sha256"]
    assert revision["focused_tests"]["exit_status"]==0
    preflight=read(OUT/"preflight.json")
    for name in ("truth","checkpoint"):
        assert digest(Path(preflight[name]["path"]))==preflight[name]["sha256"]
    sender=read(PRIVATE/"observation/sender_validation.json")
    assert sender["ordinary_runs"]==sender["replay_runs"]==1 and sender["all_raw_deltas_torch_equal"]
    suite=ET.parse(OUT/"pytest.xml").getroot().find("testsuite")
    tests={k:int(suite.attrib[k]) for k in ("tests","failures","errors","skipped")}
    artifacts={}
    for base in (PUBLIC,PRIVATE):
        for p in base.rglob("*"):
            if p.is_file():
                artifacts[str(p.relative_to(ROOT))]={"bytes":p.stat().st_size,"sha256":digest(p)}
    result={"task_id":claim["task_id"],"plan_version":1,"status":"COMPLETED","claim":claim,
            "completed_at_utc":datetime.now(timezone.utc).isoformat(),"tests":tests,
            "executions":executions,"communication":comparison["communication"],
            "offline_review_01":revision,"provenance_validation":comparison["provenance_validation"],
            "all_raw_deltas_torch_equal":True,"arm_settings_equal":True,"fia_initializations_equal":True,
            "image_means":{arm:{key:comparison["arms"][arm]["image_oracle"][key] for key in
                                 ("mean_psnr_db","mean_ssim","mean_lpips","mean_mse")} for arm in arms},
            "embedding_mean_cosine":{arm:comparison["arms"][arm]["embedding_oracle"]["mean_cosine"] for arm in arms},
            "naive_minus_reference":comparison["naive_minus_reference"],
            "paired_differences_path":str(PRIVATE/"evaluation/comparison.json"),
            "runtime":{arm:comparison["arms"][arm]["final_result"]["final_runtime"] for arm in arms},
            "protected_sources_before":before,"protected_sources_after":after,
            "authorized_shared_helper_change":changed,"all_other_protected_hashes_equal":True,
            "artifacts":artifacts,"limitations":["single batch/seed; no success threshold or generalization",
               "FP32 Reference has raw precompression information unavailable to INT4 attacker",
               "upstream negentropy overwritten to zero, unchanged",
               "adaptive_avg_pool2d_backward_cuda nondeterministic warning with warn_only=True; no full determinism claim",
               "code/bin/scale metrics describe observed/raw matrices, not reconstructed image-update replay"],
            "scope_exclusions_preserved":["quantization-aware CPA","ensemble","MP","FIA update-consistency loss",
                                            "oracle candidate selection","additional seed/batch/arm","GT-label attack"]}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# CPA_SIGNED_BASELINE_B16_T10_20260914 v1 — Executor report", "",
           "[FACT] COMPLETED. New signed observation, INT4 naive, FP32 Reference, and all16 offline evaluation finished.",
           "", "| Arm | PSNR dB | SSIM | VGG LPIPS | Embedding cosine | CUDA peak allocated B |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("reference","naive"):
        e=result["image_means"][arm]
        lines.append(f"| {arm} | {e['mean_psnr_db']:.6f} | {e['mean_ssim']:.6f} | {e['mean_lpips']:.6f} | {result['embedding_mean_cosine'][arm]:.6f} | {result['runtime'][arm]['cuda_peak_allocated']} |")
    lines += ["", "[FACT] Naive minus reference: "+json.dumps(result["naive_minus_reference"]),"",
        "[FACT] Same raw local update, frozen Conv/all FC, batch00 B16 T10 seed42; exact ordinary SGD/replay agreement on all six raw tensors. Signed CPU FP32 per-tensor scale, nearest-even; q+7 nibble payload with reserved 15. Original affine observations preserved.",
        "[FACT] Both arms use raw Delta/raw sq without -Delta/lr or T/B transforms; official eig whitening and CPA 25,000, abs(S_hat), Direct FIA 25,000. Separate input fingerprints/caches. Identical FIA initial tensor hashes and actual numeric runtime flags.",
        "[FACT] Public-only naive process has no private truth/raw-update API. Reference receives raw Delta only. Separate offline process loads private truth only after both final artifacts are complete and hashed. No oracle matching/scaling before FIA.",
        "[FACT] One pixel-MSE Hungarian per arm for PSNR/SSIM/VGG LPIPS on all16; distinct embedding cosine Hungarian. Matching used only for final evaluation; paired differences are aligned by truth image in comparison.json.",
        f"[FACT] CPU gates: {tests['tests']} passed; 0 failures/errors/skips. Exact execution commands, timestamps, exit statuses and stdout are in *_execution.json / *_attempt_1.log.",
        "[FACT] Planner review 01 added offline identity/input/source binding and private-output guards, validated by focused CPU rejection tests. No collection or optimized arm was rerun; earlier execution source hashes are preserved with this explicit offline-only revision.",
        "[FACT] New modules: signed_baseline_observation.py, signed_baseline.py, signed_baseline_evaluation.py; new tests: test_signed_baseline.py. Existing full_update_evaluation.py only adds include_labels=False while preserving the original default path; regression gates passed. Existing quantizer/cell_utils/official CPA/FIA/collector/STATE/RELAY unchanged.",
        "[FACT] Communication: "+json.dumps(comparison["communication"]),
        "[FACT] Raw source/objective diagnostics remain distinct from actual MI/negentropy. MI and actual NE are not computed. Quantization/scale fit is a diagnostic on supplied observation matrices, not an image-update constraint.",
        "[FACT] adaptive_avg_pool2d_backward_cuda emitted a nondeterminism warning under the upstream adapter's warn_only=True setting. Both arms retain identical settings; complete determinism is not claimed.",
        "[UNKNOWN] No success threshold, statistical significance, generalization or causal mechanism established by one batch/seed. This is a newly approved signed per-tensor experiment, not a reproduction of the old affine per-row or -Delta/lr baseline.",
        "[PROPOSAL] No additional run was scheduled. Planner can review the saved evidence.","",
        "Evaluation outputs: quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/evaluation/report.md, comparison.json, all16_comparison.png, all16_compact.png.",
        "Public bundle and naive candidates: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/.",
        "Private raw/reference/truth and offline artifacts: quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    handoff=ROOT/".ai/EXECUTION_HANDOFF.md"
    text=handoff.read_text(encoding="utf-8")
    assert f"TASK_ID: {claim['task_id']}\nPLAN_VERSION: 1\nSTATUS: RUNNING" in text
    text=text.replace("STATUS: RUNNING","STATUS: COMPLETED",1)
    text=text.replace(" / RUNNING"," / "+result["completed_at_utc"])
    text="\n".join("수집/naive/reference/evaluation 상태: COMPLETED / COMPLETED / COMPLETED / COMPLETED" if line.startswith("수집/naive/reference/evaluation 상태:") else line for line in text.split("\n"))
    text=text.replace("결과: NOT_STARTED","결과: COMPLETED; .ai/execution-results/CPA_SIGNED_BASELINE_B16_T10_20260914/report.md 및 results.json; all16 offline 평가 완료")
    handoff.write_text(text,encoding="utf-8")
    print(json.dumps({"status":"COMPLETED","means":result["image_means"],"delta":result["naive_minus_reference"]},indent=2))


if __name__=="__main__": main()
