"""One-time real-packet/GPU plumbing validation, not an additional attack arm."""
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from qacpa.random_fill import random_fill_packet, candidate_seed, consistency_score
from qacpa.topk_ratio_codec import encode_topk
from qacpa.run_random_fill_same80 import read_json, write_json, file_hash, utc_now, free_memory


def main():
    torch.set_num_threads(4)
    baseline = PROJECT / "results/cpa_fia_topk10_same80_B16"
    output = PROJECT / "results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/setup_checks"
    output.mkdir(parents=True, exist_ok=True)
    audits = []
    for batch in range(5):
        path = baseline / f"batch{batch:02d}/observation.bin"
        packet = path.read_bytes()
        assert file_hash(path) == read_json(path.parent / "communication.json")["sha256"]
        filled, audit = random_fill_packet(packet, candidate_seed(batch, 0))
        assert audit["tau"] > 0
        assert encode_topk(filled, 10) == packet, "Imputed candidate is not packet-consistent"
        audits.append({"batch_index": batch, **audit, "reencoded_packet_identical": True})
        print(f"REAL PACKET batch={batch} tau={audit['tau']:.9g} retained bits exact; recompressed bytes identical", flush=True)
        del packet, filled
        free_memory()
    device = torch.device("cuda")
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    # Already available reconstruction used only to validate replay/ranking IO;
    # this score is NOT a candidate and is NOT used in randomized selection.
    rec = torch.load(baseline / "batch00/source_order_rec.pt", map_location="cpu", weights_only=True)
    packet = (baseline / "batch00/observation.bin").read_bytes()
    started = time.perf_counter()
    score = consistency_score(rec=rec, packet=packet, keep_percent=10, device=device)
    report = {"created_at_utc": utc_now(), "packet_checks": audits, "gpu": torch.cuda.get_device_name(),
              "gpu_capability": list(torch.cuda.get_device_capability()),
              "ranking_path_check": score, "ranking_path_check_sec": time.perf_counter()-started,
              "ranking_input": "existing zero-fill source-order reconstruction; no original images/labels/features",
              "scope": "validation only; no CPA/FIA attack arm; score never enters randomized selection"}
    write_json(output / "real_packet_and_ranking_verification.json", report)
    print(f"GPU REPLAY VERIFIED relative_l2={score['relative_l2']:.6g} time={report['ranking_path_check_sec']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
