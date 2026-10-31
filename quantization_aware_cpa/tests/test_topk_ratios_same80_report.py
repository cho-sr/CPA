"""CPU coverage/provenance checks for the original-five-batch Top-k 50% report."""
from copy import deepcopy
import json

import pytest
import torch

from qacpa import report_topk_ratios_same80 as report


@pytest.fixture(params=(30, 20, 10))
def completed_run(tmp_path, monkeypatch, request):
    monkeypatch.setattr(report, "PROJECT", tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    from qacpa import run_topk_ratios_same80 as runner
    cfg = runner.read_json(runner.PROJECT / f"configs/cpa_fia_topk{request.param}_same80_B16.json")
    monkeypatch.setattr(report, "_communication", lambda folder, batch, keep_percent: {"batch_index": batch, **report.read_json(folder / "communication.json")})
    identity = {"config_sha256": "a" * 64}
    manifest = {"run_tag": cfg["run_tag"], "identity": identity, "git_sha": "1" * 40,
                "inputs_sha256": {}, "metadata": []}
    output = tmp_path / "results/synthetic_same80"
    for batch in range(5):
        source = tmp_path / report.SOURCE_OUTPUT / f"batch{batch:02d}"
        source.mkdir(parents=True)
        metadata = [{"global_index": batch * 16 + i, "batch_index": batch, "source_index": i,
                     "dataset_index": 1000 + batch * 16 + i, "label": i,
                     "image_path": f"/historical/{batch * 16 + i:02d}.JPEG"} for i in range(16)]
        torch.save({"metadata": metadata, "x": torch.tensor(.2).expand(16, 3, 224, 224)}, source / "inputs.pt")
        manifest["metadata"].extend(deepcopy(metadata))
        manifest["inputs_sha256"][(source / "inputs.pt").relative_to(tmp_path).as_posix()] = report.file_hash(source / "inputs.pt")
        folder = output / f"batch{batch:02d}"
        folder.mkdir(parents=True)
        source_recs = (torch.arange(16, dtype=torch.float32) / 16)[:, None, None, None].expand(16, 3, 224, 224)
        torch.save(source_recs, folder / "source_order_rec.pt")
        torch.save(source_recs.flip(0), folder / "rec.pt")
        torch.save(torch.tensor(.2).expand(16, 25088), folder / "sources.pt")
        rows = [{**meta, "run_tag": cfg["run_tag"], "n": 16, "T": 10, "estimated_source_index": 15 - i,
                 "oracle_fill": False, "ls_A": False, "lpips": .1 + batch / 10 + i / 1000,
                 "psnr": 20 + i / 100, "ssim": .3, "source_abs_cosine": .95 if i < batch else .5}
                for i, meta in enumerate(metadata)]
        report.write_csv(folder / "per_image_metrics.csv", rows)
        for name in ("cpa_log.json", "fia_log.json", "sparsification_diagnostics.json", "packet_manifest.json"):
            report.write_json(folder / name, {})
        (folder / "observation.bin").write_bytes(bytes(48))
        report.write_json(folder / "communication.json", {"fp32_payload_bytes": 128, "fp32_packet_bytes": 160,
                          "value_bytes": 16, "bitmap_bytes": 0, "header_bytes": 32,
                          "packet_bytes": 48, "n_values": 32})
        report.write_json(folder / "summary.json", {"identity": identity, "batch_index": batch,
                          "n_images": 16, "n": 16, "T": 10, "oracle_fills": 0, "ls_A": False,
                          **report._metrics(rows), "runtime_sec": {"cpa": 1., "fia": 2., "invocation": 3.},
                          "output_hashes": {name: report.file_hash(folder / name) for name in report.REQUIRED_OUTPUTS}})
    rendered = []

    def grid(originals, recs, rows, path, **kwargs):
        assert originals.shape == recs.shape == (len(rows), 3, 224, 224)
        ids = [int(row["global_index"]) for row in rows]
        rendered.append(ids)
        path.write_bytes(b"synthetic grid")
        return {"file": path.name, "n_images": len(rows), "global_indices": ids}

    monkeypatch.setattr(report, "render_grid", grid)
    monkeypatch.setattr(report, "render_distributions", lambda rows, path, **kwargs: path.write_bytes(b"synthetic distribution"))
    yield cfg, manifest, output, rendered
    torch.set_num_threads(previous_threads)


def _rehash(folder, name):
    done = report.read_json(folder / "summary.json")
    done["output_hashes"][name] = report.file_hash(folder / name)
    report.write_json(folder / "summary.json", done)


def test_all80_report_keeps_original_order_per_batch_metrics_and_one_ledger_row(completed_run):
    cfg, manifest, output, rendered = completed_run
    result = report.summarize(cfg, manifest, output)
    assert len(result["per_image_metrics"]) == 80
    assert [int(row["global_index"]) for row in result["per_image_metrics"]] == list(range(80))
    assert result["metrics"]["embedding_cos09"] == 10
    assert len(result["metrics"]["source_cos_per_source"]) == 80
    assert [batch["embedding_cos09"] for batch in result["per_victim_batch"]] == [0, 1, 2, 3, 4]
    assert all(len(batch["source_cos_per_source"]) == 16 for batch in result["per_victim_batch"])
    assert result["communication"]["packet_bytes"] == 5 * 48
    assert result["runtime_sec"]["invocation"] == 15.
    assert rendered == [list(range(80)), *[list(range(batch * 16, (batch + 1) * 16)) for batch in range(5)]]
    assert torch.load(output / "all80_rec.pt", weights_only=True).shape == (80, 3, 224, 224)
    text = (output / "report.md").read_text(encoding="utf-8")
    assert "five 16-image batches" in text and "not an FP32 attack result" in text
    assert "best10" not in text
    report.summarize(cfg, manifest, output)
    ledger = [json.loads(line) for line in (output.parent / "runs.jsonl").read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]["config"] == cfg


@pytest.mark.parametrize("fault", ["missing_batch", "changed_artifact", "manifest_metadata", "row_metadata", "permutation", "missing_input_hash"])
def test_report_rejects_incomplete_or_changed_historical_provenance(completed_run, fault):
    cfg, manifest, output, _ = completed_run
    folder = output / "batch04"
    if fault == "missing_batch":
        (folder / "summary.json").unlink()
    elif fault == "changed_artifact":
        (folder / "cpa_log.json").write_text("changed", encoding="utf-8")
    elif fault == "manifest_metadata":
        manifest["metadata"][79]["dataset_index"] = 99999
    elif fault == "row_metadata":
        rows = report.read_csv(folder / "per_image_metrics.csv")
        rows[-1]["image_path"] = "/different.JPEG"
        report.write_csv(folder / "per_image_metrics.csv", rows)
        _rehash(folder, "per_image_metrics.csv")
    elif fault == "permutation":
        torch.save(torch.load(folder / "source_order_rec.pt", weights_only=True), folder / "rec.pt")
        _rehash(folder, "rec.pt")
    else:
        del manifest["inputs_sha256"][f"{report.SOURCE_OUTPUT}/batch04/inputs.pt"]
    with pytest.raises((RuntimeError, FileNotFoundError)):
        report.summarize(cfg, manifest, output)
    assert not (output / "summary.json").exists()
    assert not (output.parent / "runs.jsonl").exists()
