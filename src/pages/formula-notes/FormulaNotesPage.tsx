import { useEffect, useRef, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import type { FormulaStrategyId, ModelDefinition, ModelId } from "../../domain/model/types";
import type { FormulaTraceSection } from "../../domain/performance/types";
import { getFormulaTraceRowTarget } from "../../features/performance-calculator/utils/formulaTraceTargets";
import { useCalculatorContext } from "../../features/performance-calculator/state/CalculatorProvider";

type PrefillFormulaGuide = {
  expression: string;
  notes: string[];
  variables: FormulaVariable[];
};

type FormulaVariable = {
  symbol: string;
  meaning: string;
};

type TraceVariableDefinition = FormulaVariable & {
  source: string;
};

const traceVariableCatalog: Record<string, TraceVariableDefinition> = {
  S: { symbol: "S", meaning: "参与当前阶段计算的 token 数。", source: "工作负载输入" },
  S_ctx: { symbol: "S_ctx", meaning: "Decode 开始时已经存在的上下文 token 数。", source: "工作负载输入" },
  D: { symbol: "D", meaning: "模型隐藏层维度。", source: "模型 config.json" },
  I: { symbol: "I", meaning: "FFN 或单个专家的中间层维度。", source: "模型 config.json" },
  I_layer: { symbol: "I_layer", meaning: "当前 decoder 层实际使用的 MLP intermediate size。", source: "由基础宽度和分层 MLP schedule 派生" },
  P: { symbol: "P", meaning: "每个 decoder 层的 PLE 输入维度。", source: "模型 config.json：hidden_size_per_layer_input" },
  I_moe: { symbol: "I_moe", meaning: "单个 MoE 专家的中间层维度。", source: "模型 config.json" },
  k: { symbol: "k", meaning: "每个 token 激活的 routed expert 数量。", source: "模型 config.json" },
  E: { symbol: "E", meaning: "每层 routed expert 总数。", source: "模型 config.json" },
  n_h: { symbol: "n_h", meaning: "Attention query head 数量。", source: "模型 config.json" },
  n_h_I: { symbol: "n_h_I", meaning: "CSA indexer 的 attention head 数量。", source: "模型 config.json" },
  n_kv: { symbol: "n_kv", meaning: "Key/Value head 数量。", source: "模型 config.json" },
  n_kv_s: { symbol: "n_kv_s", meaning: "Sliding Attention 的 KV head 数量。", source: "模型 config.json" },
  n_kv_f: { symbol: "n_kv_f", meaning: "Full Attention 的 KV head 数量。", source: "模型 config.json" },
  n_v_heads: { symbol: "n_v_heads", meaning: "Linear Attention 的 value head 数量。", source: "模型 config.json" },
  c: { symbol: "c", meaning: "Attention 单个 head 的维度。", source: "模型 config.json" },
  c_s: { symbol: "c_s", meaning: "Sliding Attention 单个 head 的维度。", source: "模型 config.json" },
  c_f: { symbol: "c_f", meaning: "Full Attention 单个 head 的维度。", source: "模型 config.json" },
  c_I: { symbol: "c_I", meaning: "CSA indexer 单个 head 的维度。", source: "模型 config.json" },
  c_kL: { symbol: "c_kL", meaning: "Linear Attention 的 key head 维度。", source: "模型 config.json" },
  c_vL: { symbol: "c_vL", meaning: "Linear Attention 的 value head 维度。", source: "模型 config.json" },
  r_q: { symbol: "r_q", meaning: "Query 低秩投影维度。", source: "模型 config.json" },
  r_o: { symbol: "r_o", meaning: "Attention 输出低秩投影维度。", source: "模型 config.json" },
  o_groups: { symbol: "o_groups", meaning: "Attention 输出投影的分组数。", source: "模型 config.json" },
  sliding_window: { symbol: "sliding_window", meaning: "滑动注意力可见的局部窗口长度。", source: "模型 config.json" },
  index_topk: { symbol: "index_topk", meaning: "CSA indexer 选取的历史 token 数量。", source: "模型 config.json" },
  m_hca: { symbol: "m_hca", meaning: "HCA 历史 token 压缩倍率。", source: "模型 config.json" },
  m_csa: { symbol: "m_csa", meaning: "CSA 历史 token 压缩倍率。", source: "模型 config.json" },
  n_win: { symbol: "n_win", meaning: "滑动窗口长度的公式记号。", source: "模型 config.json" },
  key_dim: { symbol: "key_dim", meaning: "Linear Attention 的总 key 投影维度。", source: "模型 config.json" },
  value_dim: { symbol: "value_dim", meaning: "Linear Attention 的总 value 投影维度。", source: "模型 config.json" },
  kernel: { symbol: "kernel", meaning: "Linear Attention Conv1D 的卷积核宽度。", source: "模型 config.json" },
  conv_dim: { symbol: "conv_dim", meaning: "Linear Attention 卷积路径的通道维度。", source: "模型结构派生" },
  causal_factor: { symbol: "causal_factor", meaning: "因果 Attention 对有效 QK 配对数采用的系数。", source: "公式假设" },
  has_v_proj: { symbol: "has_v_proj", meaning: "是否存在独立 Value 投影的指示量。", source: "模型结构派生" },
  B: { symbol: "B", meaning: "推理批大小。", source: "工作负载输入" },
  e: { symbol: "e", meaning: "每个 cache/activation 元素占用的字节数。", source: "平台精度输入" },
  bytes_per_elem: { symbol: "bytes_per_elem", meaning: "单个缓存元素的字节数。", source: "平台精度输入" },
  bpw: { symbol: "bpw", meaning: "普通权重每个参数占用的字节数。", source: "平台精度输入" },
  bpe: { symbol: "bpe", meaning: "专家权重每个参数占用的字节数。", source: "平台精度输入" },
  N_text: { symbol: "N_text", meaning: "参与文本推理的骨干参数量，单位为十亿参数（B）；不等同于包含全部 checkpoint 组件的总参数量。", source: "模型注册数据：textBackboneParamsB" },
  N_lookup: { symbol: "N_lookup", meaning: "整张 token lookup 参数表的参数量，单位为十亿参数（B），包括词嵌入表及 PLE lookup 表；该项先从文本骨干权重中扣除。", source: "模型注册数据：tokenLookupParamsB" },
  N_non: { symbol: "N_non", meaning: "非专家权重参数量。", source: "模型注册数据/权重文件" },
  N_exp: { symbol: "N_exp", meaning: "专家权重参数量。", source: "模型注册数据/权重文件" },
  N_sliding: { symbol: "N_sliding", meaning: "Sliding Attention 层数。", source: "模型 config.json" },
  N_csa: { symbol: "N_csa", meaning: "CSA 层数。", source: "模型 config.json" },
  N_hca: { symbol: "N_hca", meaning: "HCA 层数。", source: "模型 config.json" },
  L: { symbol: "L", meaning: "参与当前公式的 decoder 层数。", source: "模型 config.json" },
  L_sliding: { symbol: "L_sliding", meaning: "Sliding Attention 层数。", source: "模型 config.json" },
  L_full: { symbol: "L_full", meaning: "Full Attention 层数。", source: "模型 config.json" },
  L_linear: { symbol: "L_linear", meaning: "Linear Attention 层数。", source: "模型 config.json" },
  L_kv: { symbol: "L_kv", meaning: "单步 Decode 实际读取的 KV 可见长度。", source: "由上下文与模型结构派生" },
  L_kv_decode: { symbol: "L_kv_decode", meaning: "Decode 阶段的有效 KV 可见长度。", source: "由上下文与模型结构派生" },
  F_Q: { symbol: "F_Q", meaning: "Query 路径的 FLOPs。", source: "公式派生" },
  F_KV: { symbol: "F_KV", meaning: "Key/Value 投影的 FLOPs。", source: "公式派生" },
  F_core: { symbol: "F_core", meaning: "Attention 核心计算的 FLOPs。", source: "公式派生" },
  F_core_sliding: { symbol: "F_core_sliding", meaning: "单个 Sliding 层处理当前 token 时，QKᵀ 与 AV 两趟 Attention core 的 FLOPs。", source: "Sliding Decode 可见长度与 Attention 维度派生" },
  F_core_csa: { symbol: "F_core_csa", meaning: "单个 CSA 层处理当前 token 时，QKᵀ 与 AV 两趟主 Attention core 的 FLOPs。", source: "CSA Decode 可见长度与 Attention 维度派生" },
  F_core_hca: { symbol: "F_core_hca", meaning: "单个 HCA 层处理当前 token 时，QKᵀ 与 AV 两趟 Attention core 的 FLOPs。", source: "HCA Decode 可见长度与 Attention 维度派生" },
  F_O: { symbol: "F_O", meaning: "Attention 输出投影的 FLOPs。", source: "公式派生" },
  F_MLP: { symbol: "F_MLP", meaning: "Dense MLP 的 FLOPs。", source: "公式派生" },
  F_PLE: { symbol: "F_PLE", meaning: "Per-Layer Embeddings 全局投影及所有逐层门控/投影的 FLOPs。", source: "PLE 公式派生" },
  F_decode: { symbol: "F_decode", meaning: "生成一个 token 时，当前 Gemma decoder 执行的总浮点运算量。", source: "由各类 decoder 层 FLOPs 与 PLE FLOPs 汇总" },
  F_si: { symbol: "F_si", meaning: "所有 Sliding Attention 独立 KV 层在单步 Decode 中的 FLOPs，包含 Q/K/V、Attention core、输出投影及 MLP/MoE。", source: "由模型层 schedule 与 Decode 公式派生" },
  F_ss: { symbol: "F_ss", meaning: "所有 Sliding Attention 共享 KV 层在单步 Decode 中的 FLOPs；这些层复用 KV，因此不重复计算 K/V 投影。", source: "由模型层 schedule 与 Decode 公式派生" },
  F_fi: { symbol: "F_fi", meaning: "所有 Full Attention 独立 KV 层在单步 Decode 中的 FLOPs，包含 Q/K/V、Attention core、输出投影及 MLP/MoE。", source: "由模型层 schedule 与 Decode 公式派生" },
  F_fs: { symbol: "F_fs", meaning: "所有 Full Attention 共享 KV 层在单步 Decode 中的 FLOPs；这些层复用 KV，因此不重复计算 K/V 投影。", source: "由模型层 schedule 与 Decode 公式派生" },
  F_MoE: { symbol: "F_MoE", meaning: "MoE 前馈路径的 FLOPs。", source: "公式派生" },
  F_FFN: { symbol: "F_FFN", meaning: "Dense FFN 路径的 FLOPs。", source: "公式派生" },
  F_compressor: { symbol: "F_compressor", meaning: "Token compressor 的 FLOPs。", source: "公式派生" },
  F_indexer: { symbol: "F_indexer", meaning: "CSA indexer 的 FLOPs。", source: "公式派生" },
  F_compressor_csa: { symbol: "F_compressor_csa", meaning: "单个 CSA 层对当前 token 执行 compressor 路径的 FLOPs。", source: "CSA compressor 公式派生" },
  F_compressor_hca: { symbol: "F_compressor_hca", meaning: "单个 HCA 层对当前 token 执行 compressor 路径的 FLOPs。", source: "HCA compressor 公式派生" },
  F_indexer_lin: { symbol: "F_indexer_lin", meaning: "单个 CSA 层为当前 token 生成索引 Query 与相关线性投影的 FLOPs。", source: "CSA indexer 线性投影公式派生" },
  F_indexer_attn: { symbol: "F_indexer_attn", meaning: "单个 CSA 层的当前 token 扫描压缩历史块所需的 Indexer attention FLOPs。", source: "CSA 压缩历史长度与 indexer 维度派生" },
  F_inproj: { symbol: "F_inproj", meaning: "Linear Attention 输入投影的 FLOPs。", source: "公式派生" },
  F_conv: { symbol: "F_conv", meaning: "Linear Attention Conv1D 的 FLOPs。", source: "公式派生" },
  F_scan: { symbol: "F_scan", meaning: "Linear Attention recurrent scan 的 FLOPs。", source: "公式派生" },
  F_sliding: { symbol: "F_sliding", meaning: "单个 Sliding 层的总 FLOPs。", source: "公式派生" },
  F_csa: { symbol: "F_csa", meaning: "单个 CSA 层的总 FLOPs。", source: "公式派生" },
  F_hca: { symbol: "F_hca", meaning: "单个 HCA 层的总 FLOPs。", source: "公式派生" },
  F_sliding_decode: { symbol: "F_sliding_decode", meaning: "全部 Sliding 层生成一个 token 的 FLOPs 总和。", source: "Sliding 层逐算子结果乘层数后汇总" },
  F_csa_decode: { symbol: "F_csa_decode", meaning: "全部 CSA 层生成一个 token 的 FLOPs 总和。", source: "CSA 层逐算子结果乘层数后汇总" },
  F_hca_decode: { symbol: "F_hca_decode", meaning: "全部 HCA 层生成一个 token 的 FLOPs 总和。", source: "HCA 层逐算子结果乘层数后汇总" },
  F_full: { symbol: "F_full", meaning: "单个 Full Attention 层的总 FLOPs。", source: "公式派生" },
  F_linear: { symbol: "F_linear", meaning: "单个 Linear Attention 层的总 FLOPs。", source: "公式派生" },
  F_tmp: { symbol: "F_tmp", meaning: "当前步骤的临时计算量。", source: "公式派生" },
  FLOPs_per_token: { symbol: "FLOPs_per_token", meaning: "生成一个 token 所需的估算 FLOPs。", source: "公式派生" },
  effective_compute: { symbol: "effective_compute", meaning: "计入效率系数后的有效算力。", source: "平台参数" },
  effective_bandwidth: { symbol: "effective_bandwidth", meaning: "计入效率系数后的有效带宽。", source: "平台参数" },
  compute_ceiling: { symbol: "compute_ceiling", meaning: "由有效算力决定的吞吐上限。", source: "公式派生" },
  bandwidth_ceiling: { symbol: "bandwidth_ceiling", meaning: "由有效带宽决定的吞吐上限。", source: "公式派生" },
  decode_compute_ceiling: { symbol: "decode_compute_ceiling", meaning: "Decode 的算力吞吐上限。", source: "公式派生" },
  decode_bandwidth_ceiling: { symbol: "decode_bandwidth_ceiling", meaning: "Decode 的带宽吞吐上限。", source: "公式派生" },
  bytes_per_token: { symbol: "bytes_per_token", meaning: "每生成一个 token 的内存访问量。", source: "公式派生" },
  B_sliding: { symbol: "B_sliding", meaning: "Sliding 层每 token 的缓存流量。", source: "公式派生" },
  B_csa: { symbol: "B_csa", meaning: "CSA 层每 token 的缓存流量。", source: "公式派生" },
  B_hca: { symbol: "B_hca", meaning: "HCA 层每 token 的缓存流量。", source: "公式派生" },
  B_full: { symbol: "B_full", meaning: "Full Attention 层每 token 的缓存流量。", source: "公式派生" },
  B_linear: { symbol: "B_linear", meaning: "Linear Attention 层每 token 的状态流量。", source: "公式派生" },
  B_weights: { symbol: "B_weights", meaning: "每 token 访问的权重字节数。", source: "公式派生" },
  B_cache: { symbol: "B_cache", meaning: "每 token 访问的缓存字节数。", source: "公式派生" },
  B_decode: { symbol: "B_decode", meaning: "每 token 的 Decode 总内存流量。", source: "公式派生" },
  M_weights: { symbol: "M_weights", meaning: "驻留模型权重占用。", source: "模型注册数据/权重文件" },
  M_cache: { symbol: "M_cache", meaning: "持久 Decode cache 或 recurrent state 占用。", source: "公式派生" },
  M_sliding: { symbol: "M_sliding", meaning: "Sliding Attention 持久缓存占用。", source: "公式派生" },
  M_full: { symbol: "M_full", meaning: "Full Attention 持久 KV 缓存占用。", source: "公式派生" },
  M_hca: { symbol: "M_hca", meaning: "HCA 持久缓存占用。", source: "公式派生" },
  M_csa: { symbol: "M_csa", meaning: "CSA 持久缓存占用。", source: "公式派生" },
  M_fullKV: { symbol: "M_fullKV", meaning: "Full Attention 持久 KV 缓存占用。", source: "公式派生" },
  M_linearState: { symbol: "M_linearState", meaning: "Linear Attention 持久 recurrent state 占用。", source: "公式派生" },
  M_decode_cache: { symbol: "M_decode_cache", meaning: "Decode 阶段持久缓存总占用。", source: "公式派生" },
  M_tmp: { symbol: "M_tmp", meaning: "单步 Decode 临时工作集。", source: "公式派生" },
  M_tmp_peak: { symbol: "M_tmp_peak", meaning: "单步 Decode 临时工作集峰值。", source: "公式派生" },
  M_overhead: { symbol: "M_overhead", meaning: "运行时额外显存开销。", source: "平台参数：Runtime Overhead" },
  M_runtime_overhead: { symbol: "M_runtime_overhead", meaning: "运行时额外显存开销。", source: "平台参数：Runtime Overhead" },
  conv_state: { symbol: "conv_state", meaning: "Linear Attention 的卷积状态大小。", source: "模型结构派生" },
  recurrent_state: { symbol: "recurrent_state", meaning: "Linear Attention 的递归状态大小。", source: "模型结构派生" }
};

function getTraceVariables(expression: string): TraceVariableDefinition[] {
  const normalizedExpression = expression
    .replace(/n_h\^I/g, "n_h_I")
    .replace(/c\^I/g, "c_I");
  const usedSymbols = new Set(
    normalizedExpression.match(/[A-Za-z][A-Za-z0-9_]*/g) ?? []
  );

  return Object.entries(traceVariableCatalog)
    .filter(([key]) => usedSymbols.has(key))
    .map(([, definition]) => definition);
}

const prefillFormulaGuides = {
  "deepseek-v4-compressed-moe": {
    expression: `F_prefill = N_sliding * F_sliding
          + N_csa * F_csa
          + N_hca * F_hca

F_indexer_attn = S^2 * n_h_I * c_I / m_csa
F_moe = 6 * S * D * I * (k + 1)`,
    notes: [
      "总 FLOPs 由 Sliding、CSA 与 HCA 三种层的单层 FLOPs 乘以各自层数后相加。",
      "CSA indexer attention 是随输入长度呈 O(S^2) 增长的独立项。",
      "MoE 按每个 token 激活的 routed experts 与 shared expert 计算。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "N_sliding / N_csa / N_hca", meaning: "Sliding、CSA、HCA 三类 decoder 层的数量。" },
      { symbol: "F_sliding / F_csa / F_hca", meaning: "对应类型单个 decoder 层的 Prefill FLOPs。" },
      { symbol: "F_indexer_attn", meaning: "CSA 索引器注意力的 FLOPs。" },
      { symbol: "S", meaning: "Prompt token length，即参与 Prefill 的 token 数。" },
      { symbol: "n_h_I", meaning: "CSA 索引器的 attention head 数。" },
      { symbol: "c_I", meaning: "CSA 索引器每个 head 的维度。" },
      { symbol: "m_csa", meaning: "CSA 的 token 压缩倍率。" },
      { symbol: "F_moe", meaning: "单层 MoE 前馈网络的 FLOPs。" },
      { symbol: "D", meaning: "模型 hidden size。" },
      { symbol: "I", meaning: "单个专家的中间层维度。" },
      { symbol: "k", meaning: "每个 token 激活的 routed expert 数；k + 1 包含 shared expert。" }
    ]
  },
  "dense-decoder-transformer": {
    expression: `F_prefill = L_sliding * F_sliding
          + L_full * F_full + F_PLE

F_layer = F_Q + F_KV + F_attention
        + F_O + F_MLP
F_attention_sliding = 4 * S * Lkv * n_h * c
F_attention_full = 2 * S^2 * n_h * c
F_KV(shared layer) = 0
F_MLP = 6 * S * D * I_layer
I_layer = I                    (base / independent-KV layer)
I_layer = 2 * I                (shared-KV layer when double-wide is enabled)
F_PLE = 2 * S * D * (L * P)
      + L * 4 * S * D * P`,
    notes: [
      "总 FLOPs 由 Sliding Attention 层与 Full Attention 层的计算量汇总。",
      "两种层分别展开 Q、KV、Attention、输出投影和 Dense MLP。",
      "Full Attention 核心项随输入长度呈 O(S^2) 增长。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_sliding / L_full", meaning: "Sliding Attention 层数与 Full Attention 层数。" },
      { symbol: "F_sliding / F_full", meaning: "单个 Sliding 层与单个 Full 层的 FLOPs。" },
      { symbol: "F_layer", meaning: "当前类型单个 decoder 层的总 FLOPs。" },
      { symbol: "F_Q / F_KV", meaning: "Query 投影与 Key/Value 投影的 FLOPs。" },
      { symbol: "F_attention", meaning: "Attention 核心计算的 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影的 FLOPs。" },
      { symbol: "F_MLP", meaning: "Dense GeGLU/SwiGLU 前馈网络的 FLOPs。" },
      { symbol: "F_PLE", meaning: "PLE 总 FLOPs，包括一次 D→L×P 全局投影，以及每层 D→P gate 和 P→D projection。" },
      { symbol: "S", meaning: "Prompt token length。" },
      { symbol: "D", meaning: "模型 hidden size。" },
      { symbol: "L", meaning: "Decoder 总层数；用于 PLE packed projection 和逐层 PLE 路径计数。" },
      { symbol: "P", meaning: "每层 PLE 输入维度，即 hidden_size_per_layer_input；Gemma-4-E2B 为 256。" },
      { symbol: "I", meaning: "Dense MLP 的基础 intermediate size。" },
      { symbol: "I_layer", meaning: "当前层实际使用的 intermediate size；普通层为 I，启用双宽的 KV 共享层为 2I。" }
    ]
  },
  "dense-decoder-moe": {
    expression: `F_prefill = L_sliding * F_sliding
          + L_full * F_full

F_layer = F_Q + F_KV + F_attention
        + F_O + F_moe
F_moe = 6 * S * D * I_moe * (k + 1)`,
    notes: [
      "总 FLOPs 由 Sliding Attention 层与 Full Attention 层的计算量汇总。",
      "Attention 路径按层类型分别计算，MoE 路径应用于全部 decoder layers。",
      "MoE 只计算每个 token 实际激活的 routed experts 与 shared expert。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_sliding / L_full", meaning: "Sliding Attention 层数与 Full Attention 层数。" },
      { symbol: "F_sliding / F_full", meaning: "单个 Sliding 层与单个 Full 层的 FLOPs。" },
      { symbol: "F_layer", meaning: "当前类型单个 decoder 层的总 FLOPs。" },
      { symbol: "F_Q / F_KV", meaning: "Query 投影与 Key/Value 投影的 FLOPs。" },
      { symbol: "F_attention", meaning: "Attention 核心计算的 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影的 FLOPs。" },
      { symbol: "F_moe", meaning: "单层 MoE 前馈网络的 FLOPs。" },
      { symbol: "S", meaning: "Prompt token length。" },
      { symbol: "D", meaning: "模型 hidden size。" },
      { symbol: "I_moe", meaning: "单个专家的 intermediate size。" },
      { symbol: "k", meaning: "每个 token 激活的 routed expert 数；k + 1 包含 shared expert。" }
    ]
  },
  "hybrid-linear-moe": {
    expression: `F_prefill = L_full * F_full
          + L_linear * F_linear

F_full = F_Q+gate + F_KV + F_attention
       + F_O + F_moe
F_linear = F_inproj + F_conv + F_scan
         + F_O + F_moe`,
    notes: [
      "总 FLOPs 由 Full Attention 层和 Linear Attention 层分别汇总。",
      "Full 层包含二次复杂度 Attention；Linear 层展开输入投影、Conv1D 与 gated delta scan。",
      "MoE 按每个 token 实际激活的 routed experts 与 shared expert 计算。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_full / L_linear", meaning: "Full Attention 层数与 Linear Attention 层数。" },
      { symbol: "F_full / F_linear", meaning: "单个 Full 层与单个 Linear 层的 FLOPs。" },
      { symbol: "F_Q+gate", meaning: "Full Attention 的 Query 与 gate 投影 FLOPs。" },
      { symbol: "F_KV", meaning: "Key/Value 投影 FLOPs。" },
      { symbol: "F_attention", meaning: "Full Attention 核心计算 FLOPs。" },
      { symbol: "F_inproj", meaning: "Linear Attention 输入投影 FLOPs。" },
      { symbol: "F_conv", meaning: "Linear Attention Conv1D FLOPs。" },
      { symbol: "F_scan", meaning: "Gated delta scan 递推计算 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影 FLOPs。" },
      { symbol: "F_moe", meaning: "MoE 前馈网络 FLOPs。" }
    ]
  },
  "hybrid-linear-dense": {
    expression: `F_prefill = L_full * F_full
          + L_linear * F_linear

F_full = F_Q+gate + F_KV + F_attention
       + F_O + F_FFN
F_linear = F_inproj + F_conv + F_scan
         + F_O + F_FFN`,
    notes: [
      "总 FLOPs 由 Full Attention 层和 Linear Attention 层分别汇总。",
      "Full 层包含二次复杂度 Attention；Linear 层展开输入投影、Conv1D 与 gated delta scan。",
      "该策略使用 Dense SwiGLU FFN，不使用 routed-expert MoE 项。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_full / L_linear", meaning: "Full Attention 层数与 Linear Attention 层数。" },
      { symbol: "F_full / F_linear", meaning: "单个 Full 层与单个 Linear 层的 FLOPs。" },
      { symbol: "F_Q+gate", meaning: "Full Attention 的 Query 与 gate 投影 FLOPs。" },
      { symbol: "F_KV", meaning: "Key/Value 投影 FLOPs。" },
      { symbol: "F_attention", meaning: "Full Attention 核心计算 FLOPs。" },
      { symbol: "F_inproj", meaning: "Linear Attention 输入投影 FLOPs。" },
      { symbol: "F_conv", meaning: "Linear Attention Conv1D FLOPs。" },
      { symbol: "F_scan", meaning: "Gated delta scan 递推计算 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影 FLOPs。" },
      { symbol: "F_FFN", meaning: "Dense SwiGLU 前馈网络 FLOPs。" }
    ]
  }
} satisfies Record<FormulaStrategyId, PrefillFormulaGuide>;

function getPrefillFormulaGuide(model: ModelDefinition): PrefillFormulaGuide {
  const guide = prefillFormulaGuides[model.formulaStrategyId];
  if (model.formulaStrategyId !== "dense-decoder-transformer") return guide;

  const sharedLayers = model.kvSharedLayerCount ?? 0;
  const baseLayers = model.decoderLayers - sharedLayers;
  const I = model.intermediateSize ?? 0;
  const sharedWidth = model.doubleWideMlpInKvSharedLayers ? 2 * I : I;

  return {
    ...guide,
    expression: `${guide.expression}

Current model:
L_base = ${baseLayers}, I_base = ${I}
L_shared_kv = ${sharedLayers}, I_shared_kv = ${sharedWidth}
F_MLP_total = 6 * S * D * (${baseLayers} * ${I} + ${sharedLayers} * ${sharedWidth})`,
    notes: [
      ...guide.notes,
      model.doubleWideMlpInKvSharedLayers
        ? `当前 ${model.displayName} 的 ${sharedLayers} 个 KV 共享层使用双宽 MLP（I_layer = 2I = ${sharedWidth}）。`
        : sharedLayers > 0
          ? `当前 ${model.displayName} 虽有 ${sharedLayers} 个 KV 共享层，但这些层仍使用基础 MLP 宽度 I = ${I}。`
          : `当前 ${model.displayName} 所有层均使用基础 MLP 宽度 I = ${I}。`
    ]
  };
}

const prefillTpsVariables: FormulaVariable[] = [
  { symbol: "TPS_prefill", meaning: "Prefill 阶段每秒处理的 token 数。" },
  { symbol: "S", meaning: "Prompt token length。" },
  { symbol: "effective_compute", meaning: "平台峰值算力乘以 compute efficiency 后的有效算力。" },
  { symbol: "F_prefill", meaning: "完成整个 Prompt Prefill 的总 FLOPs。" },
  { symbol: "effective_bandwidth", meaning: "内存带宽乘以 bandwidth efficiency 后的有效带宽。" },
  { symbol: "B_prefill", meaning: "完成整个 Prompt Prefill 预计产生的总内存流量。" }
];

const decodeTpsVariables: FormulaVariable[] = [
  { symbol: "TPS_decode", meaning: "Decode 阶段单序列每秒生成的 token 数。" },
  { symbol: "decode_compute_ceiling", meaning: "由单 token FLOPs 和有效算力决定的 Decode 吞吐上限。" },
  { symbol: "decode_bandwidth_ceiling", meaning: "由权重及缓存流量和有效带宽决定的 Decode 吞吐上限。" }
];

const decodeMemoryVariables: FormulaVariable[] = [
  { symbol: "M_decode_total", meaning: "Decode 运行期间预计需要的总显存。" },
  { symbol: "M_weights", meaning: "常驻模型权重占用。" },
  { symbol: "M_decode_cache", meaning: "跨生成 token 持久保存的 KV cache 或线性注意力状态。" },
  { symbol: "M_decode_tmp_peak", meaning: "单个 Decode step 的临时工作集峰值。" },
  { symbol: "M_runtime_overhead", meaning: "运行时框架、CUDA context、allocator 与 kernel workspace 的估算开销。" }
];

const variableSourceBySymbol: Record<string, string> = {
  S: "工作负载输入：Prompt Token Length",
  "N_sliding / N_csa / N_hca": "模型 config.json：各 Attention 层数",
  "L_sliding / L_full": "模型 config.json：Sliding / Full 层数",
  "L_full / L_linear": "模型 config.json：Full / Linear 层数",
  n_h_I: "模型 config.json：Indexer Attention Heads",
  c_I: "模型 config.json：Indexer Head Dimension",
  m_csa: "模型 config.json：CSA Compression Rate",
  D: "模型 config.json：Hidden Size",
  I: "模型 config.json：Intermediate Size",
  I_layer: "模型 config.json 与 MLP schedule：基础层为 I，双宽共享 KV 层为 2I",
  L: "模型 config.json：Num Hidden Layers",
  P: "模型 config.json：Hidden Size Per Layer Input",
  F_PLE: "由 PLE 全局投影和逐层 gate/projection 公式派生",
  I_moe: "模型 config.json：MoE Intermediate Size",
  k: "模型 config.json：Active Experts / Token",
  effective_compute: "平台参数：Compute Throughput × Compute Efficiency",
  effective_bandwidth: "平台参数：Memory Bandwidth × Bandwidth Efficiency",
  B_prefill: "计算器工程估算：Prefill Memory Traffic",
  M_weights: "模型参数量与 Bytes / Weight、Bytes / Expert",
  M_decode_cache: "模型 config.json、输入长度与缓存精度",
  M_decode_tmp_peak: "模型 config.json、输入长度与 Activation Precision",
  M_runtime_overhead: "平台参数：Runtime Overhead"
};

function getVariableSource(symbol: string): string {
  return variableSourceBySymbol[symbol] ?? "由当前公式及其上游变量派生";
}

function FormulaAccordionItem({
  id,
  title,
  open,
  onOpenChange,
  children
}: {
  id: string;
  title: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}) {
  return (
    <details
      id={id}
      className="formula-accordion"
      open={open}
      onToggle={(event) => onOpenChange(event.currentTarget.open)}
    >
      <summary>
        <span>{title}</span>
        <span className="formula-accordion__icon" aria-hidden="true" />
      </summary>
      <div className="formula-accordion__content">{children}</div>
    </details>
  );
}

function FormulaBlock({
  title,
  stage,
  expression,
  notes,
  variables,
  trace
}: {
  title: string;
  stage: string;
  expression: string;
  notes: string[];
  variables: FormulaVariable[];
  trace?: FormulaTraceSection;
}) {
  return (
    <article className="formula-block">
      <div className="formula-block__header">
        <div>
          <p className="eyebrow">{stage}</p>
          <h3>{title}</h3>
        </div>
        <span className="source-badge">calculator source</span>
      </div>
      <pre>{expression}</pre>
      <div className="formula-variable-list">
        <h4>变量含义</h4>
        <dl>
          <div className="formula-variable-list__header" aria-hidden="true">
            <span>变量</span>
            <span>含义</span>
            <span>数据来源</span>
          </div>
          {variables.map((variable) => (
            <div key={variable.symbol} className="formula-variable-list__row">
              <dt><code>{variable.symbol}</code></dt>
              <dd>{variable.meaning}</dd>
              <dd className="formula-variable-list__source">
                {getVariableSource(variable.symbol)}
              </dd>
            </div>
          ))}
        </dl>
      </div>
      <ul className="plain-list">
        {notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
      {trace ? <FormulaTracePreview trace={trace} /> : null}
    </article>
  );
}

function FormulaTracePreview({ trace }: { trace: FormulaTraceSection }) {
  return (
    <div className="trace-preview">
      <p className="eyebrow">formula trace</p>
      <div className="trace-preview__grid">
        {trace.rows.map((row, rowIndex) => {
          const variables = getTraceVariables(row.expression);

          return (
            <div
              key={row.label}
              id={getFormulaTraceRowTarget(trace.category, rowIndex)}
              className="trace-preview__row"
            >
              <span className="trace-preview__label">{row.label}</span>
              <code>{row.expression}</code>
              {row.explanation ? (
                <ul className="trace-preview__explanation">
                  {row.explanation.map((item) => <li key={item}>{item}</li>)}
                </ul>
              ) : null}
              {row.sourceUrl ? (
                <a
                  className="trace-preview__source"
                  href={row.sourceUrl}
                  target="_blank"
                  rel="noreferrer"
                >
                  {row.sourceLabel ?? "来源"} ↗
                </a>
              ) : null}
              {variables.length > 0 ? (
                <details className="trace-variable-table">
                  <summary>变量说明（{variables.length}）</summary>
                  <div className="table-scroll">
                    <table className="data-table data-table--compact">
                      <thead>
                        <tr>
                          <th>变量</th>
                          <th>含义</th>
                          <th>数据来源</th>
                        </tr>
                      </thead>
                      <tbody>
                        {variables.map((variable) => (
                          <tr key={variable.symbol}>
                            <td><code>{variable.symbol}</code></td>
                            <td>{variable.meaning}</td>
                            <td>{variable.source}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function FormulaNotesPage() {
  const [searchParams] = useSearchParams();
  const [openSections, setOpenSections] = useState<Set<string>>(() => new Set());
  const pageRef = useRef<HTMLElement>(null);
  const {
    availableFamilies,
    calculationRevision,
    formulaModelId,
    formulaSelectedFamily,
    formulaAvailableModels,
    formulaSelectedModel: model,
    formulaResult: result,
    updateFormulaModelFamily,
    updateFormulaModelId
  } = useCalculatorContext();
  const targetSection = searchParams.get("section");
  const targetFormula = searchParams.get("formula");

  useEffect(() => {
    setOpenSections(new Set());
    pageRef.current
      ?.querySelectorAll<HTMLDetailsElement>("details")
      .forEach((details) => {
        details.open = false;
      });
  }, [calculationRevision]);

  useEffect(() => {
    if (!targetSection) {
      return;
    }

    setOpenSections((current) => {
      if (current.has(targetSection)) {
        return current;
      }
      const next = new Set(current);
      next.add(targetSection);
      return next;
    });

  }, [targetSection]);

  useEffect(() => {
    if (!targetSection || !openSections.has(targetSection)) {
      return;
    }

    const targetId = targetFormula ?? targetSection;
    const frame = window.requestAnimationFrame(() => {
      document.getElementById(targetId)?.scrollIntoView({
        behavior: "smooth",
        block: targetFormula ? "center" : "start"
      });
    });

    return () => window.cancelAnimationFrame(frame);
  }, [openSections, targetFormula, targetSection]);

  const updateSectionOpen = (sectionId: string, open: boolean) => {
    setOpenSections((current) => {
      const next = new Set(current);
      if (open) {
        next.add(sectionId);
      } else {
        next.delete(sectionId);
      }
      return next;
    });
  };

  if (!result) {
    return null;
  }

  const prefillTrace = result.formulaTrace.find((trace) => trace.category === "prefill");
  const decodeTrace = result.formulaTrace.find((trace) => trace.category === "decode");
  const memoryTrace = result.formulaTrace.find((trace) => trace.category === "memory");
  const prefillFormulaGuide = getPrefillFormulaGuide(model);

  return (
    <section className="page-section" ref={pageRef}>
      <div className="page-heading">
        <div>
          <p className="eyebrow">Traceable Formulae</p>
          <h2>公式说明</h2>
        </div>
        <p className="page-description">
          该页面将整理 prefill、decode 与内存估算公式及其变量定义。
        </p>
      </div>

      <div className="formula-layout">
        <main className="formula-nav panel">
          <div id="formula-model-selection" className="formula-model-selectors page-section-anchor">
            <label className="field">
              <span>模型家族</span>
              <select
                value={formulaSelectedFamily}
                onChange={(event) => updateFormulaModelFamily(event.target.value)}
              >
                {availableFamilies.map((family) => (
                  <option key={family.id} value={family.id}>
                    {family.displayName}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>当前模型</span>
              <select
                value={formulaModelId}
                onChange={(event) => updateFormulaModelId(event.target.value as ModelId)}
              >
                {formulaAvailableModels.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.displayName}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="formula-accordion-list">
            <FormulaAccordionItem
              id="prefill-flops"
              title="Prefill FLOPs"
              open={openSections.has("prefill-flops")}
              onOpenChange={(open) => updateSectionOpen("prefill-flops", open)}
            >
              <FormulaBlock
                title="Prefill FLOPs"
                stage="prefill"
                expression={prefillFormulaGuide.expression}
                notes={prefillFormulaGuide.notes}
                variables={prefillFormulaGuide.variables}
                trace={prefillTrace}
              />
            </FormulaAccordionItem>

            <FormulaAccordionItem
              id="prefill-tps"
              title="Prefill TPS"
              open={openSections.has("prefill-tps")}
              onOpenChange={(open) => updateSectionOpen("prefill-tps", open)}
            >
              <FormulaBlock
                title="Prefill TPS"
                stage="prefill"
                expression={`TPS_prefill = min(
  S * effective_compute / F_prefill,
  S * effective_bandwidth / B_prefill
)`}
                notes={[
                  "effective_compute comes from platform TFLOPS and compute efficiency.",
                  "effective_bandwidth comes from memory bandwidth and bandwidth efficiency.",
                  "Current B_prefill is an engineering estimate and should be tightened later."
                ]}
                variables={prefillTpsVariables}
              />
            </FormulaAccordionItem>

            <FormulaAccordionItem
              id="decode-tps"
              title="Decode TPS"
              open={openSections.has("decode-tps")}
              onOpenChange={(open) => updateSectionOpen("decode-tps", open)}
            >
              <FormulaBlock
                title="Decode TPS"
                stage="decode"
                expression={`TPS_decode = min(
  decode_compute_ceiling,
  decode_bandwidth_ceiling
)`}
                notes={[
                  "Decode bandwidth is dominated by visible cache traffic and repeated per-token access.",
                  "The current compute ceiling is an approximation, not a per-kernel trace.",
                  "Weight reads are included in decode bandwidth traffic."
                ]}
                variables={decodeTpsVariables}
                trace={decodeTrace}
              />
            </FormulaAccordionItem>

            <FormulaAccordionItem
              id="decode-memory"
              title="Decode Memory"
              open={openSections.has("decode-memory")}
              onOpenChange={(open) => updateSectionOpen("decode-memory", open)}
            >
              <FormulaBlock
                title="Decode Memory"
                stage="memory"
                expression={`M_decode_total ~= M_weights
  + M_decode_cache
  + M_decode_tmp_peak
  + M_runtime_overhead`}
                notes={[
                  "M_weights is resident during decode and must be included.",
                  "M_decode_cache is persistent across generated tokens.",
                  "M_decode_tmp_peak captures the single-step attention working set."
                ]}
                variables={decodeMemoryVariables}
                trace={memoryTrace}
              />
            </FormulaAccordionItem>

          </div>
        </main>

      </div>
    </section>
  );
}
