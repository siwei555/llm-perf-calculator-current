"""Validate the confirmed Qwen3.6-35B-A3B engineering baseline."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATA = ROOT / "data/models/Qwen_3.6/Qwen3.6-35B-A3B.json"


def close(actual: float, expected: float, tolerance: float = 0.002) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual:.6f} != {expected:.6f}")


def main() -> None:
    model_data = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_MODEL_DATA
    model = json.loads(model_data.read_text(encoding="utf-8"))
    a = model["architecture"]
    p = model["precision"]
    params = model["parameterEstimate"]
    expected = model["acceptedBaseline"]

    s = expected["promptTokens"]
    batch = expected["batchSize"]
    d = a["hiddenSize"]
    nh = a["attentionHeads"]
    nkv = a["kvHeads"]
    c = a["headDim"]
    experts = a["moe"]["routedExperts"]
    active = a["moe"]["activeExpertsPerToken"]
    intermediate = a["moe"]["intermediateSize"]
    full_layers = a["layerSchedule"]["fullAttention"]
    linear_layers = a["layerSchedule"]["gatedDeltaNet"]
    key_heads = a["linearAttention"]["keyHeads"]
    key_dim = a["linearAttention"]["keyHeadDim"]
    value_heads = a["linearAttention"]["valueHeads"]
    value_dim = a["linearAttention"]["valueHeadDim"]
    conv_kernel = a["linearAttention"]["convKernelSize"]
    e = p["bytesPerActivation"]

    moe = 6 * s * d * intermediate * (active + 1)
    full = (
        2 * s * d * (2 * nh * c)
        + 2 * s * d * nkv * c * 2
        + 2 * s * s * nh * c
        + 2 * s * nh * c * d
        + moe
    )
    linear = (
        2 * s * d * (key_heads * key_dim + key_heads * key_dim + value_heads * value_dim)
        + 2 * s * d * (value_heads * value_dim)
        + 2 * s * d * (value_heads * 2)
        + 2 * s * conv_kernel * (key_heads * key_dim * 2 + value_heads * value_dim)
        + 2 * s * value_heads * key_dim * value_dim
        + 2 * s * value_heads * value_dim * d
        + moe
    )
    prefill = full * full_layers + linear * linear_layers
    decode_full = (
        2 * d * (2 * nh * c)
        + 2 * d * nkv * c * 2
        + 2 * s * nh * c * 2
        + 2 * nh * c * d
        + 6 * d * intermediate * (active + 1)
    )
    decode_linear = (
        2 * d * (key_heads * key_dim + key_heads * key_dim + value_heads * value_dim)
        + 2 * d * (value_heads * value_dim)
        + 2 * d * (2 * value_heads)
        + 2 * conv_kernel * (key_heads * key_dim * 2 + value_heads * value_dim)
        + 2 * value_heads * key_dim * value_dim
        + 2 * value_heads * value_dim * d
        + 6 * d * intermediate * (active + 1)
    )
    decode = full_layers * decode_full + linear_layers * decode_linear

    active_params_b = (
        params["totalParamsB"]
        - params["totalExpertParamsB"]
        + params["totalExpertParamsB"] * active / experts
    )
    active_weight_traffic = active_params_b * p["bytesPerWeight"]
    full_kv = batch * full_layers * 2 * s * nkv * c * e / 1e9
    linear_state = (
        batch
        * linear_layers
        * (
            conv_kernel * (key_heads * key_dim * 2 + value_heads * value_dim) * e
            + value_heads * key_dim * value_dim * 2 * e
        )
        / 1e9
    )
    decode_linear_traffic = (
        batch
        * linear_layers
        * (
            conv_kernel * (key_heads * key_dim * 2 + value_heads * value_dim)
            + value_heads * key_dim * value_dim
        )
        * e
        / 1e9
    )
    cache_traffic = full_kv + decode_linear_traffic
    weights = params["totalParamsB"] * p["bytesPerWeight"]
    temporary = batch * 2 * e * nh * s * c / 1e9
    total_memory = weights + full_kv + linear_state + temporary + expected["runtimeOverheadGb"]

    actual = {
        "fullLayerPrefillTflops": full / 1e12,
        "linearLayerPrefillTflops": linear / 1e12,
        "totalPrefillTflops": prefill / 1e12,
        "decodeGflopsPerToken": decode / 1e9,
        "activeWeightTrafficGbPerToken": active_weight_traffic,
        "cacheTrafficGbPerToken": cache_traffic,
        "weightMemoryGb": weights,
        "persistentFullKvCacheGb": full_kv,
        "persistentLinearStateGb": linear_state,
        "peakTemporaryMemoryGb": temporary,
        "estimatedTotalMemoryGb": total_memory,
    }
    print(json.dumps(actual, indent=2))

    close(actual["fullLayerPrefillTflops"], expected["fullLayerPrefillTflops"])
    close(linear / 1e12, expected["linearLayerPrefillTflops"])
    close(prefill / 1e12, expected["totalPrefillTflops"])
    close(decode / 1e9, expected["decodeGflopsPerToken"])
    close(active_weight_traffic, expected["activeWeightTrafficGbPerToken"])
    close(cache_traffic, expected["cacheTrafficGbPerToken"])
    close(weights, expected["weightMemoryGb"])
    close(full_kv, expected["persistentFullKvCacheGb"])
    close(linear_state, expected["persistentLinearStateGb"])
    close(temporary, expected["peakTemporaryMemoryGb"])
    close(total_memory, expected["estimatedTotalMemoryGb"])

    print(f"{model['displayName']} baseline validation passed")
    print(f"prefill={prefill / 1e12:.3f} TFLOPs")
    print(f"decode={decode / 1e9:.3f} GFLOPs/token")
    print(f"memory={total_memory:.3f} GB")


if __name__ == "__main__":
    main()
