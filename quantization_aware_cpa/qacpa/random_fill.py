"""Packet-only random imputation and public-model consistency selection.

No original image, true label, true feature, or uncompressed client update is
accepted here. Random filling is attacker postprocessing, not extra traffic.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from .topk_ratio_codec import CHUNK, _parse, decode_packet, encode_topk
from .upstream_naive import VGG16FeatureAdapter, _tensor_hash
from .uia import load_vgg16_classifier


def candidate_seed(batch: int, candidate: int) -> int:
    if not 0 <= batch < 5 or not 0 <= candidate < 20:
        raise ValueError("Expected batch 0..4 and candidate 0..19")
    # Reuse the seed plan across retained fractions; CPA/FIA seeds stay fixed.
    return 100000 + 100 * batch + candidate


def random_fill_packet(packet: bytes, seed: int) -> tuple[torch.Tensor, dict]:
    info, mask, values = _parse(packet)
    tau = float(np.abs(values).min())
    rng = np.random.Generator(np.random.PCG64(seed))
    flat = np.empty(info["elements"], dtype=np.float32)
    offset = 0
    minimum, maximum = math.inf, -math.inf
    for start in range(0, len(mask), CHUNK):
        m = mask[start:start + CHUNK]
        block = flat[start:start + CHUNK]
        count = int(m.sum())
        block[m] = values[offset:offset + count]
        offset += count
        sampled = rng.uniform(-tau, tau, size=int((~m).sum())).astype(np.float32)
        # FP32 rounding can hit +/-tau although the continuous draw has zero
        # boundary mass. Stay strictly inside so Top-k's index tie rule cannot
        # replace an observed cutoff coordinate on recompression.
        if tau > 0:
            interior = np.nextafter(np.float32(tau), np.float32(0))
            np.clip(sampled, -interior, interior, out=sampled)
        block[~m] = sampled
        if sampled.size:
            minimum = min(minimum, float(sampled.min()))
            maximum = max(maximum, float(sampled.max()))
    if offset != len(values) or not np.array_equal(flat[mask].view(np.uint32), values.view(np.uint32)):
        raise RuntimeError("Random filling altered transmitted FP32 bits")
    if minimum < -tau or maximum > tau:
        raise RuntimeError("Random filling exceeded the observed Top-k cutoff")
    result = torch.from_numpy(flat.reshape(info["shape"]))
    return result, {"sampler": "uniform_tau_pcg64_v1", "seed": int(seed),
                    "tau": tau, "tau_source": "min(abs(transmitted FP32 values))",
                    "random_min": minimum, "random_max": maximum,
                    "retained_fp32_bits_exact": True, "dropped_count": len(mask) - len(values),
                    "fp32_boundary_rule": "clip rounded endpoints to nextafter(tau,0)",
                    "rng_chunk_elements": CHUNK, "packet_sha256": hashlib.sha256(packet).hexdigest(),
                    "filled_tensor_sha256": _tensor_hash(result), "extra_transmitted_bytes": 0}


def relative_l2(predicted: torch.Tensor, observed: torch.Tensor) -> float:
    if predicted.shape != observed.shape:
        raise ValueError("Consistency shapes differ")
    a = predicted.detach().cpu().numpy().reshape(-1)
    b = observed.detach().cpu().numpy().reshape(-1)
    error = energy = 0.
    for start in range(0, len(a), CHUNK):
        p, o = a[start:start + CHUNK].astype(np.float64), b[start:start + CHUNK].astype(np.float64)
        d = p - o
        error += float(d @ d)
        energy += float(o @ o)
    score = math.sqrt(error) / max(math.sqrt(energy), 1e-12)
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite consistency score")
    return score


def rank_candidates(rows: list[dict], expected_ids: list[int]) -> list[dict]:
    if sorted(int(row["candidate"]) for row in rows) != sorted(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Candidate coverage is incomplete or duplicated")
    if any(not math.isfinite(float(row["relative_l2"])) or float(row["relative_l2"]) < 0 for row in rows):
        raise ValueError("Invalid observable ranking score")
    return sorted(rows, key=lambda row: (float(row["relative_l2"]), int(row["candidate"])))


def replay_classifier_update(classifier, features: torch.Tensor, labels: torch.Tensor,
                             *, lr: float = .001, steps: int = 10) -> torch.Tensor:
    """The same FC-only SGD recurrence as the victim, using reconstructed data."""
    classifier.eval()  # Disable classifier dropout exactly as in the victim.
    classifier.requires_grad_(True)
    initial = classifier[0].weight.detach().clone()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0., weight_decay=0.)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            hidden = classifier[0](features)
            loss = F.cross_entropy(classifier[1:](hidden), labels, reduction="none").mean()
            loss.backward()
            optimizer.step()
    return ((initial - classifier[0].weight.detach()) / lr).cpu().contiguous()


def consistency_score(*, rec: torch.Tensor, packet: bytes, keep_percent: int,
                      device: torch.device) -> dict:
    """Replay 10 local SGD steps on all16 reconstructions with predicted labels.

    Recompress that FC1 update with the SAME global Top-k bitmap codec and
    compare its zero-filled decoding to the received packet's decoding.
    This is an observable proxy, not an image-quality or true-label oracle.
    """
    if rec.shape != (16, 3, 224, 224) or not torch.isfinite(rec).all():
        raise ValueError("Ranking requires all16 finite source-order images")
    mean = rec.new_tensor((.485, .456, .406))[None, :, None, None]
    std = rec.new_tensor((.229, .224, .225))[None, :, None, None]
    x = ((rec.clamp(0, 1) - mean) / std).to(device)
    feature_model = VGG16FeatureAdapter().to(device).eval()
    with torch.no_grad():
        _, z = feature_model(x, return_z=True)
    del feature_model, x
    classifier = load_vgg16_classifier(device)
    classifier.eval()
    with torch.no_grad():
        labels = classifier(z).argmax(dim=1)
    predicted_labels = labels.cpu().tolist()
    simulated = replay_classifier_update(classifier, z, labels)
    del classifier, z, labels
    predicted_packet = encode_topk(simulated, keep_percent)
    del simulated
    predicted = decode_packet(predicted_packet)
    del predicted_packet
    observed = decode_packet(packet)
    value = relative_l2(predicted, observed)
    return {"relative_l2": value, "predicted_labels": predicted_labels,
            "label_mode": "public round0 model argmax on reconstructed images",
            "score": "norm(TopK(replayed_Y)-decoded_received_Y)/norm(decoded_received_Y)",
            "replay": "all16; round0 frozen conv; all classifier SGD lr0.001 T10; dropout off",
            "true_labels_used": False, "true_images_used": False, "true_features_used": False}
