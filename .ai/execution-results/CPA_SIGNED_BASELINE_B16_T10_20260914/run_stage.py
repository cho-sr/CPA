"""One explicitly selected authorized stage; records command, log and exit status."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PUBLIC = ROOT/"quantization_aware_cpa/results/signed_int4_baseline_B16_T10"
PRIVATE = ROOT/"quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10"
CHECKPOINT = Path.home()/".cache/torch/hub/checkpoints/vgg16-397923af.pth"
TRUTH = ROOT/"quantization_aware_cpa/results_private/deterministic_fc_hybrid_pilot_B16_T10/evaluation_truth.pt"


def main():
    stage = sys.argv[1]
    args = {
        "collect": ["--truth-source",TRUTH,"--checkpoint",CHECKPOINT,
                    "--public-output",PUBLIC/"observation","--private-output",PRIVATE/"observation"],
        "naive": ["--public-observation",PUBLIC/"observation","--checkpoint",CHECKPOINT,"--output",PUBLIC/"naive"],
        "reference": ["--raw-updates",PRIVATE/"observation/raw_updates.pt","--checkpoint",CHECKPOINT,"--output",PRIVATE/"reference"],
        "evaluate": ["--naive",PUBLIC/"naive","--reference",PRIVATE/"reference",
                     "--private-observation",PRIVATE/"observation","--public-observation",PUBLIC/"observation",
                     "--output",PRIVATE/"evaluation"],
    }
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT),str(ROOT/"quantization_aware_cpa")])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = "42"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if stage=="tests":
        command = [sys.executable,"-B","-m","pytest","-q",
                   "quantization_aware_cpa/tests/test_signed_baseline.py",
                   "quantization_aware_cpa/tests/test_full_update_evaluation.py",
                   "quantized_update_cpa/tests/test_cell_utils.py",
                   "-o",f"cache_dir={OUT/'pytest-cache'}",f"--basetemp={OUT/'pytest-tmp'}",f"--junitxml={OUT/'pytest.xml'}"]
    else:
        command = [sys.executable,"-B","-m","qacpa.signed_baseline",stage,*map(str,args[stage])]
    previous = OUT/f"{stage}_execution.json"
    if previous.exists() and json.loads(previous.read_text())["status"]=="COMPLETED":
        raise RuntimeError("refusing to repeat completed stage")
    attempt = 1+len(list(OUT.glob(f"{stage}_attempt_*.log")))
    log_path = OUT/f"{stage}_attempt_{attempt}.log"
    source_paths = list((ROOT/"quantization_aware_cpa/qacpa").glob("signed_baseline*.py"))
    source_paths += [ROOT/"quantization_aware_cpa/qacpa/full_update_evaluation.py"]
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    record = {"stage":stage,"status":"RUNNING","command_argv":command,"cwd":str(ROOT),
              "source_hashes":source_hashes,"log":str(log_path),"attempt":attempt,
              "started_at_utc":datetime.now(timezone.utc).isoformat()}
    previous.write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record),flush=True)
    tick=time.perf_counter()
    with log_path.open("w",encoding="utf-8") as log:
        log.write("Command argv: "+json.dumps(command)+"\n")
        log.flush()
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record["pid"]=process.pid
        previous.write_text(json.dumps(record,indent=2),encoding="utf-8")
        exit_code=process.wait()
        elapsed=time.perf_counter()-tick
        log.write(f"\nEXIT_STATUS: {exit_code}\nWALL_SECONDS: {elapsed}\n")
    record.update(status="COMPLETED" if exit_code==0 else "NEEDS_REVIEW",exit_status=exit_code,
                  wall_seconds=elapsed,ended_at_utc=datetime.now(timezone.utc).isoformat())
    previous.write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record),flush=True)
    if exit_code:
        print(log_path.read_text(encoding="utf-8")[-12000:],flush=True)
    return exit_code


if __name__=="__main__":
    raise SystemExit(main())
