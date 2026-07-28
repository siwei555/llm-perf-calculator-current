"""Validate a saved Qwen3.6 config with Transformers' PretrainedConfig.

This script does not download model weights. It accepts a local config.json,
loads it through Transformers, validates the text-layer schedule, and prints a
stable JSON summary that can be compared with the model registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import PretrainedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to the saved config.json")
    parser.add_argument("--expect-layers", type=int)
    parser.add_argument("--expect-full-layers", type=int)
    parser.add_argument("--expect-linear-layers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.config.read_text(encoding="utf-8"))
    config = PretrainedConfig.from_json_file(str(args.config))
    # PretrainedConfig verifies that Transformers can consume the local JSON.
    # Keep the source object for model-specific nested fields that the generic
    # base class does not promote to typed attributes.
    raw = config.to_dict()
    raw.update(source)
    text = raw["text_config"]
    schedule = text["layer_types"]
    full_layers = schedule.count("full_attention")
    linear_layers = schedule.count("linear_attention")

    checks = {
        "schedule_matches_num_hidden_layers":
            len(schedule) == text["num_hidden_layers"],
        "expected_layers":
            args.expect_layers is None or len(schedule) == args.expect_layers,
        "expected_full_layers":
            args.expect_full_layers is None
            or full_layers == args.expect_full_layers,
        "expected_linear_layers":
            args.expect_linear_layers is None
            or linear_layers == args.expect_linear_layers,
    }

    summary = {
        "model_type": raw["model_type"],
        "text_model_type": text["model_type"],
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "num_experts": text.get("num_experts", 0),
        "num_experts_per_tok": text.get("num_experts_per_tok", 0),
        "max_position_embeddings": text["max_position_embeddings"],
        "checks": checks,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not all(checks.values()):
        raise SystemExit("config validation failed")


if __name__ == "__main__":
    main()
