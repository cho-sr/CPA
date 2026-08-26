#!/usr/bin/env python3
"""Recompute image metrics from saved reconstructions without rerunning CPA/FIA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from author_runner import IMAGE_METRIC_FIELDS, SUMMARY_FIELDS, ImageMetricComputer
from common import get_dataloaders, read_json, read_pickle, write_json
from metrics import summarize_metric_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Completed CPA/FIA run directory containing reconstructions.pkl.",
    )
    parser.add_argument(
        "--ds",
        default=None,
        help="Dataset name. Defaults to manifest['ds'] when manifest.json exists.",
    )
    parser.add_argument(
        "--source",
        choices=["dataset", "update_file"],
        default="dataset",
        help=(
            "Where to get originals. dataset reloads samples by sample_indices; "
            "update_file loads manifest['update_file'] and matches the original runner exactly."
        ),
    )
    parser.add_argument(
        "--matching",
        choices=["direct", "oracle_embedding"],
        default="direct",
        help=(
            "direct compares reconstruction row i to original row i. "
            "oracle_embedding uses oracle_evaluation.json embedding Hungarian assignment "
            "to pair reconstruction rows with original rows for evaluation/visualization only."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["test"],
        default="test",
        help="Dataset split used by sparse CPA collection. Currently only test is supported.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Metric device, e.g. cuda or cpu. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--success_lpips_threshold",
        type=float,
        default=None,
        help="LPIPS success threshold. Defaults to manifest value or 0.5.",
    )
    parser.add_argument(
        "--output_suffix",
        default="recomputed",
        help="Suffix for generated metric files.",
    )
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return read_json(path)


def infer_dataset_from_run_dir(run_dir: Path) -> str | None:
    parts = list(run_dir.parts)
    for marker in ("author_runs", "shared_updates"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return None


def normalize_suffix(value: str) -> str:
    suffix = value.strip()
    if not suffix:
        raise ValueError("output_suffix must not be empty")
    return suffix.lstrip("_")


def build_dataset_original_loader(ds: str) -> Callable[[Mapping[str, Any], list[int]], tuple[torch.Tensor, list[int]]]:
    _, dl_test = get_dataloaders(ds, batch_size=1, shuffle_test=False, num_workers=0)
    dataset = dl_test.dataset

    def load(entry: Mapping[str, Any], positions: list[int]) -> tuple[torch.Tensor, list[int]]:
        sample_indices = [int(value) for value in entry.get("sample_indices", [])]
        if not sample_indices:
            raise ValueError("reconstruction entry does not contain sample_indices; use --source update_file")
        selected = [sample_indices[position] for position in positions]
        xs = []
        for sample_index in selected:
            item = dataset[sample_index]
            x = item[0] if isinstance(item, (tuple, list)) else item
            xs.append(torch.as_tensor(x, dtype=torch.float32))
        return torch.stack(xs, dim=0), selected

    return load


def build_update_file_original_loader(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> Callable[[Mapping[str, Any], list[int]], tuple[torch.Tensor, list[int]]]:
    update_file_value = manifest.get("update_file")
    if not update_file_value:
        raise ValueError(f"{run_dir / 'manifest.json'} does not contain update_file")
    update_data = read_pickle(Path(update_file_value))

    def load(entry: Mapping[str, Any], positions: list[int]) -> tuple[torch.Tensor, list[int]]:
        batch = int(entry["batch"])
        truth_all = torch.as_tensor(update_data["x"][batch], dtype=torch.float32)
        truth = truth_all[positions]
        sample_indices = update_data.get("sample_indices", [])
        if len(sample_indices) > batch:
            selected = [int(sample_indices[batch][position]) for position in positions]
        else:
            entry_indices = entry.get("sample_indices", [])
            selected = [int(entry_indices[position]) for position in positions]
        return truth, selected

    return load


def load_oracle_embedding_matches(run_dir: Path) -> dict[int, dict[int, int]]:
    path = run_dir / "oracle_evaluation.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is required for --matching oracle_embedding")
    data = read_json(path)
    result: dict[int, dict[int, int]] = {}
    for batch_entry in data.get("batches", []):
        batch = int(batch_entry["batch"])
        embedding = batch_entry.get("embedding")
        if not isinstance(embedding, Mapping):
            raise ValueError(f"Batch {batch} has no embedding oracle assignment")
        truth_rows = embedding.get("truth_rows")
        recovered_cols = embedding.get("recovered_cols")
        if not isinstance(truth_rows, list) or not isinstance(recovered_cols, list):
            raise ValueError(f"Batch {batch} embedding assignment is missing truth_rows/recovered_cols")
        result[batch] = {int(rec_col): int(truth_row) for truth_row, rec_col in zip(truth_rows, recovered_cols)}
    return result


def original_positions_for_entry(
    entry: Mapping[str, Any],
    *,
    n: int,
    matching: str,
    oracle_matches: Mapping[int, Mapping[int, int]],
) -> list[int]:
    if matching == "direct":
        positions = list(range(n))
    elif matching == "oracle_embedding":
        batch = int(entry["batch"])
        batch_matches = oracle_matches.get(batch)
        if batch_matches is None:
            raise ValueError(f"No oracle embedding assignment for batch {batch}")
        positions = []
        for rec_sample in range(n):
            if rec_sample not in batch_matches:
                raise ValueError(f"No oracle match for batch {batch} recovered row {rec_sample}")
            positions.append(int(batch_matches[rec_sample]))
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(f"Unknown matching mode {matching}")
    n_candidates = len(entry.get("sample_indices", []))
    if n_candidates and any(position < 0 or position >= n_candidates for position in positions):
        raise ValueError(f"Matched original position outside batch candidate range: {positions}")
    return positions


def output_paths(run_dir: Path, suffix: str) -> dict[str, Path]:
    return {
        "image": run_dir / f"image_metrics_by_sample_{suffix}.csv",
        "summary": run_dir / f"metric_summary_{suffix}.csv",
        "oracle": run_dir / f"oracle_evaluation_{suffix}.json",
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    recon_path = run_dir / "reconstructions.pkl"
    if not recon_path.exists():
        raise FileNotFoundError(recon_path)

    manifest = load_manifest(run_dir)
    ds = args.ds or manifest.get("ds") or manifest.get("dataset") or infer_dataset_from_run_dir(run_dir)
    if not ds:
        raise ValueError("--ds is required when run_dir/manifest.json is missing or has no ds")
    success_threshold = (
        float(args.success_lpips_threshold)
        if args.success_lpips_threshold is not None
        else float(manifest.get("success_lpips_threshold", 0.5))
    )
    device = choose_device(args.device)
    suffix = normalize_suffix(args.output_suffix)

    if args.source == "dataset":
        load_originals = build_dataset_original_loader(str(ds))
    else:
        load_originals = build_update_file_original_loader(run_dir, manifest)
    oracle_matches = (
        load_oracle_embedding_matches(run_dir) if args.matching == "oracle_embedding" else {}
    )

    recon_entries = read_pickle(recon_path)
    metric_computer = ImageMetricComputer(device, success_lpips_threshold=success_threshold)
    image_rows: list[dict[str, Any]] = []
    eval_entries: list[dict[str, Any]] = []

    for entry in recon_entries:
        batch = int(entry["batch"])
        rec = torch.as_tensor(entry["rec_fi_attack_order"], dtype=torch.float32, device=device)
        n_rec = int(rec.shape[0])
        original_positions = original_positions_for_entry(
            entry,
            n=n_rec,
            matching=args.matching,
            oracle_matches=oracle_matches,
        )
        direct_indices = [int(value) for value in entry.get("sample_indices", [])[:n_rec]]
        truth_cpu, sample_indices = load_originals(entry, original_positions)
        truth = truth_cpu.to(device=device, dtype=torch.float32)
        rows, summary = metric_computer.compute_batch(
            batch=batch,
            rec=rec,
            truth_normalized=truth,
            ds=str(ds),
            sample_indices=sample_indices,
        )
        for row in rows:
            rec_sample = int(row["sample"])
            row["matching"] = args.matching
            row["rec_sample"] = rec_sample
            row["original_sample"] = original_positions[rec_sample]
            row["direct_sample_index"] = direct_indices[rec_sample] if rec_sample < len(direct_indices) else None
            row["matched_sample_index"] = row["sample_index"]
        image_rows.extend(rows)
        eval_entries.append(
            {
                "batch": batch,
                "metric_note": (
                    "Recomputed from saved reconstructions only; CPA/FIA was not rerun. "
                    f"matching={args.matching}."
                ),
                "image": summary,
            }
        )

    summary_rows = summarize_metric_rows(
        image_rows=image_rows,
        eval_entries=eval_entries,
        success_lpips_threshold=success_threshold,
    )
    paths = output_paths(run_dir, suffix)
    image_fieldnames = (
        IMAGE_METRIC_FIELDS
        + ("matching", "rec_sample", "original_sample", "direct_sample_index", "matched_sample_index")
    )
    write_csv(paths["image"], image_rows, fieldnames=image_fieldnames)
    write_csv(paths["summary"], summary_rows, fieldnames=SUMMARY_FIELDS)
    write_json(
        paths["oracle"],
        {
            "source": args.source,
            "matching": args.matching,
            "dataset": ds,
            "run_dir": str(run_dir),
            "reconstruction_file": str(recon_path),
            "metric_note": "Evaluation-only recomputation from reconstructions.pkl.",
            "image_metric_dependencies": metric_computer.dependency_status(),
            "metric_summary": summary_rows,
        },
    )

    print(f"wrote {paths['image']}")
    print(f"wrote {paths['summary']}")
    for row in summary_rows:
        if row["metric"] in {"lpips", "psnr", "ssim", "reconstruction_success_rate"}:
            print(
                f"{row['metric']}: mean={row['mean']} std={row['std']} "
                f"min={row['min']} max={row['max']} n={row['n']}"
            )


if __name__ == "__main__":
    main()
