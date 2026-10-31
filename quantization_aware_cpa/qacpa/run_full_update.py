"""Full VGG16 Conv+FC quantized single-client update inversion CLI."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .full_update_pipeline import (
    ContractError, attack_observation, collect_observation, file_digest,
    validate_contract, validate_design, vgg16_shapes, write_json,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--threads", type=int, default=2)
    commands = result.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read registered design; no training or attack")
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--checkpoint", type=Path)
    collect = commands.add_parser("collect", help="Sender: generate NEW full-model quantized updates")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--inputs", type=Path, required=True, help="Sender-only normalized images and labels")
    collect.add_argument("--output", type=Path, required=True, help="Public observation directory")
    collect.add_argument("--private-output", type=Path, required=True, help="Separate sender/evaluation directory")
    collect.add_argument("--codec-seed", type=int, required=True, help="Private sender RNG seed")
    collect.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for name in ("probe", "attack"):
        command = commands.add_parser(name, help="Receiver: one outer gradient" if name == "probe" else "Receiver: optimize images")
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--observation", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--mode", choices=("cell", "center"), default="cell")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate = commands.add_parser("evaluate", help="Offline only: compare final candidate with isolated truth")
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--truth", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_preexisted = getattr(args, "output", None) is not None and args.output.exists()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    try:
        if args.command == "evaluate":
            from .full_update_evaluation import evaluate_candidate
            if args.output.exists():
                raise FileExistsError(f"Evaluation output already exists: {args.output}")
            report = evaluate_candidate(args.candidate, args.truth)
            write_json(args.output, report)
        else:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or set(config) != {"contract", "attack"}:
                raise ContractError("Config must contain exactly contract and attack")
            validate_design(config["attack"])
            if args.command == "inspect":
                shapes = vgg16_shapes()
                elements = sum(math.prod(shape) for shape in shapes.values())
                code_bytes = sum((math.prod(shape) + 1) // 2 for shape in shapes.values())
                radius_bytes = sum((shape[0] if len(shape) >= 2 else 1) * 4 for shape in shapes.values())
                pending = None
                try:
                    validate_contract(config["contract"])
                except ContractError as error:
                    pending = str(error)
                report = {"status": "pending_contract" if pending else "contract_valid",
                          "pending_reason": pending, "model": "vgg16", "parameter_tensors": len(shapes),
                          "conv_tensors": sum(name.startswith("features.") for name in shapes),
                          "fc_tensors": sum(name.startswith("classifier.") for name in shapes),
                          "elements": elements, "code_bytes": code_bytes, "radius_bytes": radius_bytes,
                          "packet_header_bytes": len(shapes) * 32,
                          "layer_packet_bytes": code_bytes + radius_bytes + len(shapes) * 32,
                          "manifest_bytes": "additional; count serialized manifest at collection",
                          "checkpoint_sha256": file_digest(args.checkpoint) if args.checkpoint else None,
                          "training_or_attack_executed": False}
            else:
                validate_contract(config["contract"])
                device = torch.device(args.device)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable; no automatic device/model substitution")
                if args.command == "collect":
                    report = collect_observation(inputs=args.inputs, checkpoint=args.checkpoint,
                        contract=config["contract"], output=args.output, private_output=args.private_output,
                        codec_seed=args.codec_seed, device=device)
                else:
                    if output_preexisted:
                        raise FileExistsError(f"Refusing to overwrite attack output: {args.output}")
                    # The supplied design must name the same observation/training contract.
                    public_manifest = json.loads((args.observation / "manifest.json").read_text(encoding="utf-8"))
                    if public_manifest.get("contract") != config["contract"]:
                        raise ContractError("Config training/public contract differs from the observed packet")
                    report = attack_observation(observation=args.observation, checkpoint=args.checkpoint,
                        design=config["attack"], output=args.output, device=device, mode=args.mode,
                        probe=args.command == "probe")
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (ContractError, ValueError, OSError, RuntimeError) as error:
        report = {"status": "failed", "error_type": type(error).__name__, "message": str(error),
                  "conditions_automatically_reduced": False, "reconstruction_success": "not_established"}
        if isinstance(error, torch.OutOfMemoryError):
            report["status"] = "out_of_memory"
        # Keep a failure artifact in a newly registered receiver run, without
        # replacing any completed run or sender observation.
        output = getattr(args, "output", None)
        if (not output_preexisted and args.command in {"probe", "attack"} and output is not None and
                (output / "registered_design.json").exists() and not (output / "result.json").exists()):
            write_json(output / "failure.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
