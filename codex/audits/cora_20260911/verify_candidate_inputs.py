"""Regenerate CPU desparsification inputs only; never run training, CPA or FIA."""
import ast
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_observations import BASE, ROOT, OUT, long, read_json, read_pickle

source_code = (ROOT / "cora_fedavg_single_client_cpa/CoRA_sparsification.py").read_text(encoding="utf-8")
tree = ast.parse(source_code)
pure = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
        node.name in ("sample_missing_values", "_hash_arrays")]
namespace = {"np": np, "hashlib": hashlib}
exec("from __future__ import annotations\n" + "\n\n".join(ast.unparse(node) for node in pure), namespace)

author = ROOT / "sparsified_update_cpa/outputs_v1/author_runs/imagenet/vgg16/attack/cp_direct/nodef"
inventory = read_json(OUT / "run_inventory.json")
results = []
for item in inventory:
    if not item["path"].startswith("imagenet/"):
        continue
    is_cora = "/R8_" in item["path"]
    if not is_cora and item["sampler"] != "zero_fill":
        continue
    ratio = item["keep"]
    if is_cora:
        choice = read_json((BASE / item["path"]).parent / "manifest.json")
        member = choice["selected_member"]
        run = Path(choice["selected_member_run_path"])
    else:
        member = 0
        tag = "zf_keep10_e920" if ratio == .1 else "cora_obs_zero_fill_keep80_50_30"
        ratio_tag = str(ratio).replace(".", "p")
        run = author / f"cora_fedavg_single_client_cpa_{tag}_keep{ratio_tag}_n24_steps1_trials1_R1_seed1234_m000_uia"
    manifest = read_json(run / "manifest.json")
    index_path = Path(read_json(BASE / item["path"])["observation_file"])
    observation = read_pickle(index_path)
    row = observation["observations"][0][0]
    with np.load(long(index_path.parent / row["shard_file"]), allow_pickle=False) as archive:
        values = archive["values"]
        shape = archive["mask_shape"]
        mask = np.unpackbits(archive["mask_packed_bits"], count=int(np.prod(shape))).astype(bool).reshape(tuple(shape))
    seed = manifest["desparsification_seed"]
    candidate = namespace["sample_missing_values"](
        shape=shape, mask=mask, values=values, tau=row["tau"],
        rng=np.random.default_rng(seed), sampler=item["sampler"])
    retained_exact = np.array_equal(candidate[mask].view(np.uint32), values.view(np.uint32))
    dropped_bound = float(np.max(np.abs(candidate[~mask]), initial=0))
    attack_hash = namespace["_hash_arrays"]([candidate])
    del candidate, mask, values
    gc.collect()
    base = read_pickle(item["source_file"])
    expected = hashlib.sha256(json.dumps({
        "base_update_tensor_hash": base["metadata"]["update_tensor_hash"],
        "attack_layer_hash": attack_hash, "member": member, "seed": seed,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    del base
    gc.collect()
    result = {"keep_ratio": ratio, "method": "CoRA" if is_cora else "zero_fill",
              "selected_member": member, "run_path": str(run), "seed": seed,
              "retained_observed_values_bit_exact": retained_exact,
              "maximum_generated_dropped_abs": dropped_bound, "tau": row["tau"],
              "candidate_matches_recorded_update_tensor_hash": expected == manifest["update_tensor_hash"],
              "computed_update_tensor_hash": expected, "recorded_update_tensor_hash": manifest["update_tensor_hash"],
              "reconstructed_count": manifest.get("total_reconstructed_samples"),
              "checkpoint": manifest["checkpoint"], "sample_indices_hash": manifest.get("sample_indices_hash"),
              "author_source_hash": manifest.get("author_source_hash")}
    results.append(result)
    print(json.dumps({k:result[k] for k in ("keep_ratio", "method", "selected_member", "retained_observed_values_bit_exact", "candidate_matches_recorded_update_tensor_hash")}), flush=True)
with open(long(OUT / "candidate_input_verification.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
