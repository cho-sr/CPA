"""Read-only numerical audit of saved CoRA observations; writes only audit files."""
from pathlib import Path
import gc
import hashlib
import json
import math
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "codex/audits/cora_20260911"
BASE = ROOT / "cora_fedavg_single_client_cpa/outputs_v1"


def long(path):
    path = str(Path(path).absolute())
    return path if path.startswith("\\\\?\\") else "\\\\?\\" + path


def read_json(path):
    with open(long(path), encoding="utf-8") as f:
        return json.load(f)


def read_pickle(path):
    with open(long(path), "rb") as f:
        return pickle.load(f)


def sha(path):
    h = hashlib.sha256()
    with open(long(path), "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(arrays):
    h = hashlib.sha256()
    def update(x):
        if isinstance(x, (list, tuple)):
            h.update(f"sequence:{len(x)}".encode("ascii") + b"\0")
            for item in x:
                update(item)
            return
        a = np.ascontiguousarray(x)
        h.update(str(tuple(a.shape)).encode("ascii"))
        h.update(("torch." + str(a.dtype)).encode("ascii"))
        h.update(memoryview(a).cast("B"))
        h.update(b"\0")
    update(arrays)
    return h.hexdigest()


def compare(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Observation shapes differ")
    sq = ref = maximum = 0.
    different = 0
    for i in range(0, a.size, 1_048_576):
        x, y = a[i:i+1_048_576], b[i:i+1_048_576]
        d = x.astype(np.float64) - y.astype(np.float64)
        sq += float(d @ d)
        z = y.astype(np.float64)
        ref += float(z @ z)
        maximum = max(maximum, float(np.abs(d).max(initial=0)))
        different += int(np.count_nonzero(x != y))
    return {"numerically_equal": different == 0, "different_elements": different,
            "max_abs_error": maximum, "relative_l2": math.sqrt(sq / ref) if ref else None}


def audit_source(source_file, observation_file):
    data, obs = read_pickle(source_file), read_pickle(observation_file)
    metadata = data["metadata"]
    idx = int(data["attack_layer"]["index"])
    actual = np.asarray(data["grad"][0][idx], dtype=np.float32)
    row = obs["observations"][0][0]
    shard_path = Path(observation_file).parent / row["shard_file"]
    with np.load(long(shard_path), allow_pickle=False) as archive:
        shape = archive["mask_shape"]
        bits = archive["mask_packed_bits"]
        values = archive["values"]
    assert values.dtype == np.float32 and bits.dtype == np.uint8
    mask = np.unpackbits(bits, count=actual.size).astype(bool).reshape(tuple(shape))
    assert int(mask.sum()) == row["observed_elements"] == values.size
    # Stored shape is Python int; normalize NumPy 2 repr before comparison.
    h = hashlib.sha256()
    h.update(str(tuple(int(v) for v in shape)).encode("ascii"))
    h.update(memoryview(bits).cast("B"))
    h.update(str(tuple(values.shape)).encode("ascii"))
    h.update(memoryview(values).cast("B"))
    observation_hash_ok = h.hexdigest() == row["observation_hash"]
    decoded = np.zeros(tuple(shape), dtype=np.float32)
    decoded[mask] = values
    equality = compare(decoded, actual)
    assert observation_hash_ok and equality["numerically_equal"]
    source_hash = sha(source_file)
    source_link_ok = obs["metadata"].get("source_sparse_update_hash") == source_hash
    recorded_update_hash = metadata.get("update_tensor_hash")
    recomputed_update_hash = tensor_hash(data["grad"])
    assert recorded_update_hash == recomputed_update_hash
    checkpoint = metadata["checkpoint"]
    checkpoint_hash = sha(checkpoint["path"])
    checkpoint_ok = checkpoint_hash == checkpoint["sha256"]
    assert checkpoint_ok
    size = Path(long(shard_path)).stat().st_size
    dense_bytes = actual.nbytes
    result = {
        "source_file": str(source_file), "source_file_sha256": source_hash,
        "source_file_bytes": Path(long(source_file)).stat().st_size,
        "observation_file": str(observation_file), "observation_index_sha256": sha(observation_file),
        "observation_index_bytes": Path(long(observation_file)).stat().st_size,
        "source_index_link_verified": source_link_ok, "shard_hash_verified": observation_hash_ok,
        "update_tensor_hash_verified": True, "checkpoint_verified": checkpoint_ok,
        "metadata": {k:metadata.get(k) for k in ("keep_ratio", "n_samples", "local_steps", "local_epochs", "local_batch_size", "micro_batch_size", "lr", "seed", "observed_round", "observed_client_id", "sample_indices_hash", "label_hash", "source_fp32_update_hash", "sparse_relative_l2", "sparse_nonzero_fraction", "sparse_cosine_similarity_to_fp32", "shared_fp32_control", "shared_fp32_control_id")},
        "checkpoint": checkpoint, "attack_layer": data["attack_layer"],
        "npz_path": str(shard_path), "npz_bytes": size,
        "npz_plus_index_bytes": size + Path(long(observation_file)).stat().st_size,
        "dense_fc1_bytes": dense_bytes, "values_bytes": values.nbytes,
        "bitmap_bytes": bits.nbytes, "npz_container_and_shape_bytes": size-values.nbytes-bits.nbytes,
        "fc1_kept_elements": values.size, "fc1_elements": actual.size,
        "fc1_retained_fraction": values.size / actual.size,
        "fc1_nonzero_elements": int(np.count_nonzero(actual)),
        "fc1_savings_fraction_npz_only": 1-size/dense_bytes,
        "fc1_savings_fraction_npz_plus_index": 1-(size+Path(long(observation_file)).stat().st_size)/dense_bytes,
        "tau": row["tau"], "global_total": row["global_total_elements"],
        "global_kept": row["global_kept_elements"], "decode_vs_saved_sparse_update": equality,
        "per_tensor_stats": data["sparse_stats"][0]["per_tensor"],
        "sample_indices": data["sample_indices"],
    }
    print(json.dumps({k:result[k] for k in ("source_file", "source_index_link_verified", "npz_bytes", "fc1_retained_fraction", "fc1_savings_fraction_npz_plus_index", "tau", "decode_vs_saved_sparse_update")}), flush=True)
    del data, obs, decoded, actual, mask, values, bits
    gc.collect()
    return result


def main():
    inventory = read_json(OUT/"run_inventory.json")
    chosen = [r for r in inventory if r["path"].startswith("imagenet/") and
              ("/R8_" in r["path"] or r["sampler"] == "zero_fill")]
    source_keys = sorted({(r["source_file"], read_json(BASE/r["path"])["observation_file"]) for r in chosen})
    sources = [audit_source(Path(a), Path(b)) for a,b in source_keys]
    matches = []
    for r in chosen:
        source = next(s for s in sources if s["source_file"] == r["source_file"])
        matches.append({**r, "current_source_hash_matches_run": source["source_file_sha256"] == r["source_hash"],
                        "current_observation_hash_matches_run": source["observation_index_sha256"] == r["observation_hash"]})
    result = {"scope": "recent zero-fill baselines and historical CoRA R8, keep80/50/30/10", "sources": sources, "run_links": matches}
    with open(long(OUT/"observation_audit.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("AUDIT COMPLETE", len(sources), "source packets", len(matches), "run links", flush=True)


if __name__ == "__main__":
    main()
