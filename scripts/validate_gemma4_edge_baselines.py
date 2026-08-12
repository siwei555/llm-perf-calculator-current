"""Validate Gemma 4 E2B/E4B text-path engineering baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "docs" / "gemma_4" / "config"
S = 131072
BYTES = 2

EXPECTED = {
    "gemma-4-E2B-it-config.json": {
        "checkpoint_b": 5.1,
        "text_b": 4.628569344,
        "lookup_b": 2.751463424,
        "independent_sliding": 12,
        "independent_full": 3,
        "prefill_tflops": 1492.552674967552,
        "decode_gflops": 18.903810048,
        "persistent_cache_gb": 0.811597824,
        "total_memory_gb": 17.159097856,
    },
    "gemma-4-E4B-it-config.json": {
        "checkpoint_b": 8.0,
        "text_b": 7.463013376,
        "lookup_b": 3.489660928,
        "independent_sliding": 20,
        "independent_full": 4,
        "prefill_tflops": 2045.847541907456,
        "decode_gflops": 23.125172224,
        "persistent_cache_gb": 2.168455168,
        "total_memory_gb": 24.3159552,
    },
}


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{actual:.12f} != {expected:.12f}")


def validate(filename: str, expected: dict[str, float]) -> None:
    config = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))["text_config"]
    d = config["hidden_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    c_sliding = config["head_dim"]
    c_full = config["global_head_dim"]
    window = config["sliding_window"]
    intermediate = config["intermediate_size"]
    ple_width = config["hidden_size_per_layer_input"]
    shared_layers = config["num_kv_shared_layers"]
    independent_layers = layers - shared_layers
    layer_types = config["layer_types"]
    independent_sliding = layer_types[:independent_layers].count("sliding_attention")
    independent_full = layer_types[:independent_layers].count("full_attention")
    sliding_layers = layer_types.count("sliding_attention")
    full_layers = layer_types.count("full_attention")
    shared_sliding = sliding_layers - independent_sliding
    shared_full = full_layers - independent_full
    close(independent_sliding, expected["independent_sliding"])
    close(independent_full, expected["independent_full"])

    def attention(seq: int, visible: int, c: int, independent: int, shared: int, core_factor: int = 2) -> int:
        q = 2 * seq * d * heads * c
        kv = 4 * seq * d * kv_heads * c
        core = core_factor * seq * visible * heads * c
        output = 2 * seq * heads * c * d
        return independent * (q + kv + core + output) + shared * (q + core + output)

    shared_width = 2 if config["use_double_wide_mlp"] else 1
    mlp_layer_units = independent_layers + shared_layers * shared_width
    mlp_prefill = 6 * S * d * intermediate * mlp_layer_units
    ple_prefill = 2 * S * d * layers * ple_width + layers * 4 * S * d * ple_width
    prefill = (
        attention(S, window, c_sliding, independent_sliding, shared_sliding, 4)
        + attention(S, S, c_full, independent_full, shared_full)
        + mlp_prefill
        + ple_prefill
    )

    mlp_decode = 6 * d * intermediate * mlp_layer_units
    ple_decode = 2 * d * layers * ple_width + layers * 4 * d * ple_width
    decode = (
        attention(1, window + 1, c_sliding, independent_sliding, shared_sliding, 4)
        + attention(1, S + 1, c_full, independent_full, shared_full, 4)
        + mlp_decode
        + ple_decode
    )
    persistent_cache = (
        independent_sliding * 2 * window * kv_heads * c_sliding
        + independent_full * 2 * S * kv_heads * c_full
    ) * BYTES / 1e9
    temporary = 2 * heads * (S + 1) * max(c_sliding, c_full) * BYTES / 1e9
    total_memory = expected["checkpoint_b"] * BYTES + persistent_cache + temporary + 4

    close(prefill / 1e12, expected["prefill_tflops"])
    close(decode / 1e9, expected["decode_gflops"])
    close(persistent_cache, expected["persistent_cache_gb"])
    close(total_memory, expected["total_memory_gb"])
    print(f"{filename}: prefill={prefill / 1e12:.3f} TFLOPs, decode={decode / 1e9:.3f} GFLOPs/token, memory={total_memory:.3f} GB")


def main() -> None:
    for filename, expected in EXPECTED.items():
        validate(filename, expected)
    print("Gemma 4 E2B/E4B baseline validation passed")


if __name__ == "__main__":
    main()
