"""M7 controls evaluated on the exact stored float FIA images.

Public helpers use RGB tensors in [0, 1].  ``load_historical`` also provides
the original ImageNet-normalized targets under ``x``.  This module performs
no ICA, unmixing, FIA optimization, or reconstruction rematching.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torchmetrics.functional.image.lpips import _normalize_tensor, _spatial_average
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .fia import denormalize_imagenet, normalize_lpips
from .victim import load_imagenet_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_CSV = PROJECT_ROOT / "results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv"
METHOD_LABELS = {
    "fp32": "fp32-CPA (36 CPA + 4 exact-source)",
    "naive": "4bit-CPA / CPA (미수정) + naive dequant",
}
METHOD_KEYS = {"fp32": "fp32_CPA", "naive": "four_bit_CPA"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_historical(aggregate_csv: Path = HISTORICAL_CSV) -> dict[str, dict[str, Any]]:
    """Load the historical 40 pairs by result directory and global-index metadata.

    Returned keys are the *historical storage* names ``fp32`` and ``naive``;
    user-facing labels and new method identifiers are in ``label`` and ``key``.
    No PNG is used, and the four fp32 exact-source replacements are retained.
    """
    aggregate_csv = Path(aggregate_csv).resolve()
    with aggregate_csv.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, Any]] = {}
    loaded: dict[Path, tuple[torch.Tensor, dict[str, Any]]] = {}
    for method in METHOD_LABELS:
        rows = sorted((dict(r) for r in all_rows if r["method"] == method), key=lambda r: int(r["global_index"]))
        if [int(r["global_index"]) for r in rows] != list(range(40)):
            raise ValueError(f"{method} must contain each global index 0..39 exactly once")
        rec_list, x_list, metadata, provenance = [], [], [], []
        for row in rows:
            result_dir = Path(row["result_dir"].replace("\\", "/"))
            if not result_dir.is_absolute():
                result_dir = PROJECT_ROOT / result_dir
            method_dir = result_dir / method
            if method_dir not in loaded:
                loaded[method_dir] = (
                    torch.load(method_dir / "rec.pt", map_location="cpu", weights_only=False),
                    torch.load(method_dir / "targets.pt", map_location="cpu", weights_only=False),
                )
            rec, targets = loaded[method_dir]
            indices = [i for i, m in enumerate(targets["metadata"]) if int(m["global_index"]) == int(row["global_index"])]
            if len(indices) != 1:
                raise ValueError(f"Expected one target for gid={row['global_index']} in {method_dir}, got {indices}")
            local = indices[0]
            if len(rec) != len(targets["metadata"]) or len(targets["x"]) != len(rec):
                raise ValueError(f"Unaligned tensor lengths in {method_dir}")
            meta = dict(targets["metadata"][local])
            for key in ("batch_index", "source_index", "label"):
                if int(meta[key]) != int(row[key]):
                    raise ValueError(f"CSV / target mismatch for {key}: {row} versus {meta}")
            rec_list.append(rec[local].float())
            x_list.append(targets["x"][local].float())
            metadata.append(meta)
            provenance.append({"global_index": int(row["global_index"]), "method_dir": str(method_dir), "local_index": local,
                               "source_kind": "exact_source" if "fp32_truth" in str(result_dir) else "CPA"})
        x = torch.stack(x_list).contiguous()
        output[method] = {"rec": torch.stack(rec_list).contiguous(), "x": x, "truth": denormalize_imagenet(x),
                          "rows": rows, "metadata": metadata, "provenance": provenance,
                          "label": METHOD_LABELS[method], "key": METHOD_KEYS[method]}
    if not torch.equal(output["fp32"]["x"], output["naive"]["x"]):
        raise ValueError("The historical fp32 and 4-bit original-image tensors differ")
    return output


class LPIPSEvaluator:
    """The FIA torchmetrics VGG LPIPS, retaining individual pair distances.

    Feature caching repeats the installed LPIPS forward operations exactly:
    scaling, VGG, channel normalization, squared difference, learned 1x1
    convolution and spatial average. It avoids 3,200 redundant VGG forwards.
    The run additionally checks all matrix diagonals against stored metrics.
    """

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device).eval()

    @torch.inference_mode()
    def paired(self, rec: torch.Tensor, truth: torch.Tensor, batch_size: int = 4) -> np.ndarray:
        if rec.shape != truth.shape:
            raise ValueError(f"Paired shapes differ: {rec.shape}, {truth.shape}")
        values = []
        for start in range(0, len(rec), batch_size):
            pred = normalize_lpips(rec[start:start + batch_size].to(self.device, dtype=torch.float32).clamp(0, 1))
            target = normalize_lpips(truth[start:start + batch_size].to(self.device, dtype=torch.float32).clamp(0, 1))
            values.append(self.metric.net(pred, target).reshape(-1).cpu())
        return torch.cat(values).numpy()

    @torch.inference_mode()
    def features(self, images: torch.Tensor, batch_size: int = 4) -> list[torch.Tensor]:
        net = self.metric.net
        if net.resize is not None or net.spatial:
            raise ValueError("Cached LPIPS requires the FIA default resize=None, spatial=False")
        layers: list[list[torch.Tensor]] = [[] for _ in range(net.L)]
        for start in range(0, len(images), batch_size):
            rgb = images[start:start + batch_size].to(self.device, dtype=torch.float32).clamp(0, 1)
            outputs = net.net(net.scaling_layer(normalize_lpips(rgb)))
            for layer, features in zip(layers, outputs):
                layer.append(_normalize_tensor(features).cpu())
        return [torch.cat(layer) for layer in layers]

    @torch.inference_mode()
    def pairwise(self, rec: torch.Tensor, truth: torch.Tensor, batch_size: int = 4,
                 truth_features: list[torch.Tensor] | None = None) -> np.ndarray:
        """Return D[i,j] = LPIPS(reconstruction_i, original_j), without matching."""
        rec_features = self.features(rec, batch_size=batch_size)
        truth_features = truth_features if truth_features is not None else self.features(truth, batch_size=batch_size)
        matrix = torch.zeros((len(rec), len(truth)), dtype=torch.float32)
        # Layer-outer order transfers each feature set only once and preserves
        # the original left-to-right sum over LPIPS layers.
        for layer, (pred_layer, truth_layer) in enumerate(zip(rec_features, truth_features)):
            pred_layer, truth_layer = pred_layer.to(self.device), truth_layer.to(self.device)
            for i in range(len(rec)):
                for start in range(0, len(truth), batch_size):
                    differences = (pred_layer[i:i + 1] - truth_layer[start:start + batch_size]) ** 2
                    values = _spatial_average(self.metric.net.lins[layer](differences), keep_dim=True).reshape(-1)
                    matrix[i, start:start + len(values)] += values.cpu()
            print(f"pairwise LPIPS layer={layer + 1}/{len(rec_features)} completed", flush=True)
        return matrix.numpy()


def dataset_mean_image(*, data_root: Path, split: str, output_dir: Path, workers: int = 4,
                       batch_size: int = 64, expected_count: int = 50000) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average every transformed image in the full requested ImageNet split."""
    mean_path, meta_path = output_dir / "dataset_mean_image.pt", output_dir / "dataset_mean_metadata.json"
    if mean_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (Path(meta["data_root"]).resolve() != data_root.resolve() or meta["split"] != split
                or meta["count"] != expected_count):
            raise ValueError("Existing dataset mean metadata does not match the requested full dataset")
        if sha256_file(mean_path) != meta["tensor_file_sha256"]:
            raise ValueError("Existing dataset mean tensor does not match its recorded SHA256")
        return torch.load(mean_path, map_location="cpu", weights_only=True), meta
    started = time.time()
    dataset = load_imagenet_dataset(data_root, split)
    if len(dataset) != expected_count:
        raise ValueError(f"Full ImageNet {split} requires {expected_count} images; found {len(dataset)}. No 40-image substitute is allowed.")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                         persistent_workers=workers > 0)
    total = torch.zeros((3, 224, 224), dtype=torch.float64)
    count = 0
    last_print = 0.0
    for images, _labels in loader:
        total += denormalize_imagenet(images).to(torch.float64).sum(dim=0)
        count += len(images)
        if time.time() - last_print > 30:
            print(f"dataset mean images={count}/{len(dataset)} elapsed={time.time() - started:.1f}s", flush=True)
            last_print = time.time()
    if count != len(dataset):
        raise RuntimeError(f"Incomplete dataset mean: {count}/{len(dataset)}")
    mean_image = (total / count).float()
    sample_manifest = "\n".join(str(Path(p).relative_to(data_root)) for p, _ in dataset.samples)
    meta = {"data_root": str(data_root.resolve()), "split": split, "count": count,
            "definition": "Per-pixel arithmetic mean of all transformed RGB images in the full ImageNet validation split",
            "transform": "qacpa.victim.load_imagenet_dataset: VGG16_Weights.IMAGENET1K_V1.transforms(); qacpa.fia.denormalize_imagenet",
            "accumulator_dtype": "float64", "saved_dtype": "float32", "shape": list(mean_image.shape),
            "sample_path_manifest_sha256": hashlib.sha256(sample_manifest.encode("utf-8")).hexdigest(),
            "wall_time_sec": time.time() - started}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mean_image, mean_path)
    meta["tensor_file_sha256"] = sha256_file(mean_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    from torchvision.utils import save_image
    save_image(mean_image, output_dir / "dataset_mean_image.png")
    return mean_image, meta


def shuffled_rows(matrix: np.ndarray, *, method: str, seeds: range = range(20)) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Permute target partners once per seed; fixed points are retained/reported."""
    if matrix.shape != (40, 40) or not np.isfinite(matrix).all():
        raise ValueError("M7 shuffled controls require a finite, complete 40 x 40 LPIPS matrix")
    n = len(matrix)
    rows, summaries = [], []
    for seed in seeds:
        permutation = np.random.default_rng(seed).permutation(n)
        values = matrix[np.arange(n), permutation]
        fixed = int(np.count_nonzero(permutation == np.arange(n)))
        rows.extend({"control": "shuffled", "method": METHOD_KEYS[method], "label": METHOD_LABELS[method],
                     "seed": seed, "global_index": i, "reconstruction_global_index": i,
                     "target_global_index": int(j), "fixed_point": int(i == j), "lpips": float(value)}
                    for i, (j, value) in enumerate(zip(permutation, values)))
        summaries.append({"method": METHOD_KEYS[method], "seed": seed, "n_images": n, "fixed_points": fixed,
                          "lpips_mean": float(np.mean(values)), "lpips_std": float(np.std(values, ddof=1)),
                          "permutation": " ".join(map(str, permutation.tolist()))})
    return rows, summaries


def per_victim_batch_summary(rows: list[dict[str, Any]], *, per_seed: bool = False) -> list[dict[str, Any]]:
    """Summarize against the original's victim batch, including shuffled pairs."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        target_index = int(row["target_global_index"])
        if not 0 <= target_index < 40 or not np.isfinite(float(row["lpips"])):
            raise ValueError(f"Invalid M7 target or LPIPS: {row}")
        if per_seed and row["control"] != "shuffled":
            continue
        key = (row["control"], row["method"], target_index // 8)
        if per_seed:
            key += (int(row["seed"]),)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, selected in sorted(groups.items()):
        values = np.array([float(row["lpips"]) for row in selected], dtype=np.float64)
        targets = {int(row["target_global_index"]) for row in selected}
        seeds = {int(row["seed"]) for row in selected if str(row.get("seed", "")) != ""}
        control, method, batch_index = key[:3]
        expected_n = 8 if per_seed or control != "shuffled" else 160
        if len(targets) != 8 or len(values) != expected_n:
            raise ValueError(f"Incomplete victim-batch group {key}: n={len(values)}, targets={sorted(targets)}")
        summaries.append({"control": control, "method": method, "batch_index": batch_index,
                          "seed": key[3] if per_seed else "all" if control == "shuffled" else selected[0].get("seed", ""),
                          "n_values": len(values), "n_target_images": len(targets), "n_seeds": len(seeds),
                          "lpips_mean": float(values.mean()), "lpips_std": float(values.std(ddof=1)),
                          "lpips_min": float(values.min()), "lpips_max": float(values.max()),
                          "batch_definition": "target_global_index // 8"})
    return summaries


def derive_batch_summaries(output_dir: Path, *, append_supplement: bool = True) -> dict[str, Any]:
    """Add batch summaries to an existing completed run without evaluating LPIPS."""
    source_path = output_dir / "per_image_metrics.csv"
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    per_batch = per_victim_batch_summary(rows)
    per_seed_batch = per_victim_batch_summary(rows, per_seed=True)
    write_csv(output_dir / "per_batch_summary.csv", per_batch)
    write_csv(output_dir / "shuffled_per_seed_batch_summary.csv", per_seed_batch)
    summary["per_victim_batch"] = per_batch
    summary["per_victim_batch_source"] = {
        "source_csv": str(source_path.resolve()), "source_csv_sha256": sha256_file(source_path),
        "batch_definition": "target_global_index // 8; original-image victim batch even when shuffled",
        "shuffled_per_seed_batch_csv": str((output_dir / "shuffled_per_seed_batch_summary.csv").resolve()),
        "metric_evaluations_performed": 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if append_supplement:
        from .run import append_jsonl, git_sha
        append_jsonl(PROJECT_ROOT / "results/runs.jsonl", {
            "schema_version": 1, "record_type": "supplemental_derived_batch_summary", "supplemental_only": True,
            "run_tag": summary["run_tag"], "parent_run_tag": summary["run_tag"], "git_sha": git_sha(),
            "spec_sha256": summary["spec_sha256"], "provenance": summary["per_victim_batch_source"],
            "per_victim_batch": per_batch,
        })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-csv", type=Path, default=HISTORICAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/m7_recompute_controls")
    parser.add_argument("--run-tag", default="m7_recompute_controls")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-mean-only", action="store_true")
    parser.add_argument("--derive-batch-summaries-only", action="store_true",
                        help="Derive batch summaries from the completed per-image CSV without evaluating LPIPS")
    parser.add_argument("--spec", type=Path, help="Previously locked preregistration file")
    parser.add_argument("--spec-sha256", help="Locked preregistration SHA256; required before computing metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.derive_batch_summaries_only:
        derive_batch_summaries(output_dir)
        print(f"Derived victim-batch summaries from existing LPIPS CSV: {output_dir}", flush=True)
        return
    if args.data_root is None:
        historical_summary = json.loads((PROJECT_ROOT / "results/v5_m3_fia_4bit/summary.json").read_text(encoding="utf-8"))
        args.data_root = Path(historical_summary["config"]["data"]["root"])
    if not args.prepare_mean_only:
        if args.spec is None or args.spec_sha256 is None:
            raise ValueError("Metric evaluation requires --spec and its previously locked --spec-sha256")
        if sha256_file(args.spec).lower() != args.spec_sha256.lower():
            raise ValueError("Preregistration SHA256 mismatch; no metrics were evaluated")
    if args.prepare_mean_only:
        _mean_image, mean_metadata = dataset_mean_image(data_root=args.data_root, split=args.split, output_dir=output_dir,
                                                        workers=args.workers)
        print(json.dumps(mean_metadata, indent=2), flush=True)
        return
    started = time.time()
    historical = load_historical(args.aggregate_csv)
    evaluator = LPIPSEvaluator(args.device)
    truth = historical["fp32"]["truth"]
    truth_features = evaluator.features(truth, batch_size=args.batch_size)
    control_rows, seed_rows, historical_rows, matrices = [], [], [], {}
    validation: dict[str, Any] = {}
    for method, data in historical.items():
        matrix = evaluator.pairwise(data["rec"], truth, batch_size=args.batch_size, truth_features=truth_features)
        expected = np.array([float(r["lpips"]) for r in data["rows"]])
        diagonal = np.diag(matrix)
        original_api = evaluator.paired(data["rec"], truth, batch_size=1)
        old_max_error = float(np.max(np.abs(diagonal - expected)))
        direct_max_error = float(np.max(np.abs(diagonal - original_api)))
        if not np.isfinite(matrix).all() or old_max_error > 1e-5 or direct_max_error > 1e-5:
            raise ValueError(f"Historical LPIPS diagonal validation failed for {method}: old={old_max_error}, direct={direct_max_error}")
        validation[METHOD_KEYS[method]] = {"max_abs_error_vs_historical": old_max_error,
                                         "max_abs_error_vs_direct_single_pair": direct_max_error,
                                         "historical_mean": float(expected.mean()), "recomputed_diagonal_mean": float(diagonal.mean()),
                                         "tolerance": 1e-5}
        np.save(output_dir / f"{METHOD_KEYS[method]}_pairwise_lpips.npy", matrix)
        matrices[method] = matrix
        shuffled, seeds = shuffled_rows(matrix, method=method)
        control_rows.extend(shuffled)
        seed_rows.extend(seeds)
        historical_rows.extend({"method": data["key"], "label": data["label"], "global_index": int(row["global_index"]),
                                "batch_index": int(row["batch_index"]), "source_index": int(row["source_index"]),
                                "lpips": float(row["lpips"]), "recomputed_lpips": float(value), **provenance}
                               for row, value, provenance in zip(data["rows"], diagonal, data["provenance"]))
        print(f"{data['key']} pairwise verified: {validation[data['key']]}", flush=True)
    generator = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.rand(truth.shape, generator=generator, dtype=torch.float32)
    torch.save(noise, output_dir / "uniform_noise_seed42.pt")
    constants = {"gray_0.5": torch.full_like(truth, 0.5), "uniform_noise": noise}
    for control, images in constants.items():
        values = evaluator.paired(images, truth, batch_size=args.batch_size)
        control_rows.extend({"control": control, "method": "constant", "label": control, "seed": 42 if control == "uniform_noise" else "",
                             "global_index": i, "target_global_index": i, "lpips": float(value)} for i, value in enumerate(values))
        print(f"{control}: n={len(values)} LPIPS={float(values.mean()):.6f}", flush=True)
    del evaluator, truth_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU LPIPS complete; remaining dataset mean and its LPIPS use CPU only", flush=True)
    mean_image, mean_metadata = dataset_mean_image(data_root=args.data_root, split=args.split, output_dir=output_dir,
                                                   workers=args.workers)
    mean_evaluator = LPIPSEvaluator("cpu")
    values = mean_evaluator.paired(mean_image.unsqueeze(0).expand_as(truth), truth, batch_size=args.batch_size)
    control_rows.extend({"control": "dataset_mean", "method": "constant", "label": "dataset_mean", "seed": "",
                         "global_index": i, "target_global_index": i, "lpips": float(value)} for i, value in enumerate(values))
    print(f"dataset_mean: n={len(values)} LPIPS={float(values.mean()):.6f}", flush=True)
    write_csv(output_dir / "per_image_metrics.csv", control_rows)
    write_csv(output_dir / "shuffled_seed_summary.csv", seed_rows)
    write_csv(output_dir / "historical_per_image.csv", historical_rows)
    summaries = []
    for control, method in dict.fromkeys((r["control"], r["method"]) for r in control_rows):
        values = np.array([r["lpips"] for r in control_rows if (r["control"], r["method"]) == (control, method)])
        summaries.append({"control": control, "method": method, "n_values": len(values), "lpips_mean": float(values.mean()),
                          "lpips_std": float(values.std(ddof=1)), "lpips_min": float(values.min()), "lpips_max": float(values.max())})
    summary = {"run_tag": args.run_tag, "spec": str(args.spec.resolve()), "spec_sha256": args.spec_sha256,
               "historical_csv": str(args.aggregate_csv.resolve()), "historical_csv_sha256": sha256_file(args.aggregate_csv),
               "metric": "torchmetrics LearnedPerceptualImagePatchSimilarity(net_type='vgg'); normalize_lpips as in qacpa.fia",
               "pairwise_axes": "rows=reconstruction global_index; columns=original global_index; both 0..39",
               "shuffled": {"seeds": list(range(20)), "rng": "numpy.random.default_rng(seed).permutation(40)",
                            "rematching": False, "fixed_points": "retained and reported", "same_permutations_both_precisions": True},
               "noise": {"distribution": "Uniform[0,1)", "seed": 42, "generator": "torch CPU", "shape": list(noise.shape),
                         "tensor_file_sha256": sha256_file(output_dir / "uniform_noise_seed42.pt")},
               "dataset_mean": mean_metadata, "historical_validation": validation,
               "summaries": summaries, "shuffled_seed_summaries": seed_rows, "wall_time_sec": time.time() - started}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = derive_batch_summaries(output_dir, append_supplement=False)
    from .run import append_jsonl, git_sha
    run_record = {**summary, "schema_version": 1, "record_type": "full_run", "milestone": "m7_recompute_controls", "git_sha": git_sha()}
    append_jsonl(PROJECT_ROOT / "results/runs.jsonl", run_record)
    print(f"Completed {args.run_tag}: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
