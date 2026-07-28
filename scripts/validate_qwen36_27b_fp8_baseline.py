"""Validate Qwen/Qwen3.6-27B-FP8 dense-hybrid engineering baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA = ROOT / "data/models/Qwen_3.6/Qwen3.6-27B-FP8.json"


def close(actual: float, expected: float, tolerance: float = 0.002) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual:.6f} != {expected:.6f}")


def main() -> None:
    model = json.loads(MODEL_DATA.read_text(encoding="utf-8"))
    a = model["architecture"]
    p = model["precision"]
    expected = model["acceptedBaseline"]

    s = expected["promptTokens"]
    b = expected["batchSize"]
    d = a["hiddenSize"]
    i = a["intermediateSize"]
    nh = a["attentionHeads"]
    nkv = a["kvHeads"]
    c = a["headDim"]
    full_layers = a["layerSchedule"]["fullAttention"]
    linear_layers = a["layerSchedule"]["gatedDeltaNet"]
    linear = a["linearAttention"]
    nkh = linear["keyHeads"]
    ck = linear["keyHeadDim"]
    nvh = linear["valueHeads"]
    cv = linear["valueHeadDim"]
    kernel = linear["convKernelSize"]
    key_dim = nkh * ck
    value_dim = nvh * cv
    conv_dim = 2 * key_dim + value_dim
    e = p["bytesPerActivation"]

    ffn = 6 * s * d * i
    full = (
        2 * s * d * (2 * nh * c)
        + 2 * s * d * nkv * c * 2
        + 2 * s * s * nh * c
        + 2 * s * nh * c * d
        + ffn
    )
    linear_layer = (
        2 * s * d * conv_dim
        + 2 * s * d * value_dim
        + 2 * s * d * (2 * nvh)
        + 2 * kernel * s * conv_dim
        + 2 * s * nvh * ck * cv
        + 2 * s * value_dim * d
        + ffn
    )
    prefill = full_layers * full + linear_layers * linear_layer
    decode_full = (
        2 * d * (2 * nh * c)
        + 2 * d * nkv * c * 2
        + 4 * s * nh * c
        + 2 * nh * c * d
        + 6 * d * i
    )
    decode_linear = (
        2 * d * conv_dim
        + 2 * d * value_dim
        + 2 * d * (2 * nvh)
        + 2 * kernel * conv_dim
        + 2 * nvh * ck * cv
        + 2 * value_dim * d
        + 6 * d * i
    )
    decode = full_layers * decode_full + linear_layers * decode_linear

    weights = model["parameterEstimate"]["totalParamsB"] * p["bytesPerWeight"]
    full_kv = full_layers * b * 2 * nkv * s * c * e / 1e9
    linear_state = (
        linear_layers
        * b
        * (conv_dim * kernel * e + nvh * ck * cv * 2 * e)
        / 1e9
    )
    cache_traffic = (
        full_layers * b * 2 * nkv * s * c * e
        + linear_layers * b * (conv_dim * kernel + nvh * ck * cv) * e
    ) / 1e9
    temporary = b * 2 * e * nh * (s + 1) * c / 1e9
    total_memory = weights + full_kv + linear_state + temporary + expected["runtimeOverheadGb"]

    close(full / 1e12, expected["fullLayerPrefillTflops"])
    close(linear_layer / 1e12, expected["linearLayerPrefillTflops"])
    close(prefill / 1e12, expected["totalPrefillTflops"])
    close(decode / 1e9, expected["decodeGflopsPerToken"])
    close(weights, expected["weightTrafficGbPerToken"])
    close(cache_traffic, expected["cacheTrafficGbPerToken"])
    close(full_kv, expected["persistentFullKvCacheGb"])
    close(linear_state, expected["persistentLinearStateGb"])
    close(temporary, expected["peakTemporaryMemoryGb"])
    close(total_memory, expected["estimatedTotalMemoryGb"])

    print("Qwen/Qwen3.6-27B-FP8 baseline validation passed")
    print(f"prefill={prefill / 1e12:.3f} TFLOPs")
    print(f"decode={decode / 1e9:.3f} GFLOPs/token")
    print(f"memory={total_memory:.3f} GB")


if __name__ == "__main__":
    main()
