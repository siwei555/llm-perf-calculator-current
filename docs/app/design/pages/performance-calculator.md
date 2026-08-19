# Performance Calculator Page Spec

> Quick token inputs provide `4K`, `8K`, `32K`, `64K`, `128K`, and `1M`. Clicking a supported target input switches the destination of these shortcuts; values above the selected model context limit remain disabled except when editing the sweep step.

> Dense Sliding Attention core uses `4 × S × Lkv × heads × head_dim`, covering both `QKᵀ` and `AV`. Dense Full Attention retains the causal-equivalent form `2 × S² × heads × head_dim`, which already equals `4 × (S²/2) × heads × head_dim` and must not be doubled again.

> `Decode Output Tokens` accepts zero. An explicit `0` means no Decode tokens are generated: Decode Time and Average Decode TPS are reported as zero, Final Context remains the prompt length, and peak memory remains the initial-context estimate. An empty field still defaults to Prompt Token Length.

## Gemma 4 E2B/E4B calculation and report requirements (2026-08-10)

- The Gemma 4 family selector includes `Gemma-4-E2B` and `Gemma-4-E4B`.
- Resident weight memory uses complete checkpoint parameters, while Decode weight traffic uses text-backbone parameters and row-level embedding lookup traffic.
- Persistent KV cache is allocated only for layers that own K/V. Shared-KV layers reread the reused states during attention and remain part of Decode cache traffic.
- Prefill and Decode compute include PLE global projection, per-layer PLE gate/projection, Attention and FFN.
- E2B/E4B HTML reports must contain separate colored blocks for Dense FFN, Sliding Attention, Full Attention and PLE, with independent/shared-KV layer counts visible.

> Source-link rule: the model summary's `模型参数来源` link opens the selected
> model's official `config.json` through `parameterSourceUrl`. The `Weight
> memory` row displays a separate `权重文件出处` link read from
> `weightSourceUrl`, rather than hardcoding either URL in a UI component.
> The same source link is rendered by both the calculator trace card and the
> Formula Notes page's `FormulaTracePreview`; these are separate render paths.
>
> The Formula Notes page selects its Prefill summary formula and explanatory
> notes from an exhaustive `formulaStrategyId` map. The summary must therefore
> match the selected model's dense, compressed-attention, or hybrid-linear
> trace instead of showing a model-independent static formula.
>
> Formula Notes is definition-only: it omits the right-side current-substitution
> panel, the summary substitution row, and evaluated numeric values under trace
> formula cards. Formula expressions, explanatory notes, and source links remain.
> Each summary formula renders an adjacent `变量含义` definition list. Prefill
> variable definitions come from the same exhaustive `formulaStrategyId` guide
> as the formula; TPS and memory formulas use their own complete definitions.
> The standalone Symbol Table accordion is intentionally omitted. Each formula's
> local variable list is the source of truth and contains three columns:
> variable, meaning, and data source. Data-source labels distinguish workload
> input, model config.json, platform/precision input, engineering estimates, and
> formula-derived values.
>
> Desktop shell navigation remains pinned to the viewport with a `100vh`
> independently scrollable sidebar. The right-side content is a separate
> `100vh` scroll container. Its offset is cached by route pathname, so
> Performance Calculator, Model Structure, Formula Notes, and History retain
> independent scroll positions when users switch with the sidebar. The active
> route's offset is recorded continuously while scrolling and restored only
> after navigation; it must not be sampled after the old page DOM has already
> been replaced. Route views remain mounted while inactive, preserving local
> UI state such as expanded Formula Notes accordions and History detail rows.
> At the mobile breakpoint the sidebar and content return to normal document
> flow so the sidebar cannot cover page content.
>
> On desktop, the main content area is also its own scroll container. The
> performance calculator, model structure, formula notes, and history routes
> retain separate scroll positions and restore them when revisited from the
> sidebar. Mobile keeps normal document scrolling.
>
> The primary result area is the third content row, after the page heading and
> input controls. It contains the calculate/reset/status toolbar followed by
> TTFT, Prefill TPS, Decode TPS, runtime memory, and memory breakdown. Detailed
> comparison/trend sections follow below it.
>
> The primary memory breakdown renders a small `权重文件出处` link directly
> beneath the Weights label. It receives the selected model's `weightSourceUrl`
> from the registry and does not duplicate or hardcode repository URLs.
>
> Every calculator Formula Trace card links to the matching Formula Notes
> section through `/formula-notes?section=...`. The destination accordion opens
> on navigation and scrolls into view. Prefill, Decode, and Memory traces map to
> `prefill-flops`, `decode-tps`, and `decode-memory`; nested source links remain
> independent anchors.
>
> Performance Calculator and Formula Notes own separate model selections.
> Editing either selector does not update the other. A successful performance
> calculation synchronizes the Formula Notes model to the model recorded in
> that calculation snapshot. Reset does not override the Formula Notes selector.
> Formula Notes traces are generated from its own model selection and the last
> successful calculation's platform/workload snapshot.
>
> Every individual Formula Notes trace card ends with its own collapsed
> `变量说明` table. Its rows are derived only from the symbols present in that
> card's expression and show symbol, meaning, and data source. Switching models
> or formula strategies therefore changes each local table without a
> page-level shared variable list.

## 1. 页面目的

`Performance Calculator` 是应用的主工作台页面，用于在给定模型、平台参数和 token 输入条件下，计算并展示：

- `TTFT`
- `Prefill TPS`
- `Decode TPS`
- `Total Runtime Memory`
- `Bottleneck Classification`
- `Formula Trace`
- `Token Sweep Trend`

页面面向内部研发和性能分析人员，优先保证高密度、可追溯和工程可解释性。

## 2. 页面范围

本页负责：

- 模型选择
- 平台参数输入
- token 输入与范围输入
- 单点性能计算
- token 趋势图计算与展示
- 显存估算摘要展示
- 结构摘要跳转
- 公式说明跳转

本页不负责：

- 完整结构详情展示
- 完整公式文档浏览
- 多模型并排对比
- 历史记录存档

## 3. 页面状态模型

页面共享一份工作台状态 `calculatorState`，至少包含：

### 3.1 Model State

```ts
type SelectedModelState = {
  modelFamily: "deepseek-v4";
  modelId: "deepseek-v4-flash" | "deepseek-v4-pro";
};
```

### 3.2 Platform Input State

```ts
type PlatformInputState = {
  computeThroughputTflops: number | null;
  memoryBandwidthGbps: number | null;
  memoryCapacityGb: number | null;
  computeEfficiency: number;
  bandwidthEfficiency: number;
  batchSize: number;
  precisionAssumptions: string;
};
```

默认值建议：

- `computeThroughputTflops = 248`（推荐精度标签包含 FP8 的模型）；其他模型默认 `124`
- `memoryBandwidthGbps = 273`
- `memoryCapacityGb = 256`
- `computeEfficiency = 0.4`（所有模型统一默认值）
- `bandwidthEfficiency = 0.6`
- `batchSize = 1`
- `precisionAssumptions = "FP8 weights + BF16 activations + FP4 experts"`

### 3.3 Workload Input State

```ts
type WorkloadInputState = {
  prefillTokenLength: number | null;
  decodeOutputTokens: number | null;
  tokenRangeStart: number | null;
  tokenRangeEnd: number | null;
  tokenRangeStep: number | null;
  tokenSweepMode: "fixed-step";
};
```

默认值建议：

- `prefillTokenLength = 131072`
- `decodeOutputTokens = null`，计算时回退为 `prefillTokenLength`
- `tokenRangeStart = 4096`
- `tokenRangeEnd = 131072`
- `tokenRangeStep = 4096`
- `tokenSweepMode = "fixed-step"`

### 3.4 View State

```ts
type PerformancePageViewState = {
  showIntermediateMetrics: boolean;
  showFormulaTrace: boolean;
  selectedTrendMetric: "prefillTps" | "decodeTps" | "ttft" | "totalRuntimeMemory";
  showBottleneckBackground: boolean;
  showTrendDataPoints: boolean;
  selectedBreakdownMode: "prefill" | "decode";
};
```

默认值建议：

- `showIntermediateMetrics = true`
- `showFormulaTrace = true`
- `selectedTrendMetric = "prefillTps"`
- `showBottleneckBackground = true`
- `showTrendDataPoints = true`
- `selectedBreakdownMode = "prefill"`

### 3.5 Computation State

```ts
type CalculationStatus =
  | "idle"
  | "invalid"
  | "ready"
  | "calculating"
  | "calculated"
  | "error";
```

```ts
type CalculationState = {
  status: CalculationStatus;
  validationErrors: Record<string, string>;
  lastCalculatedAt: string | null;
  results: PerformanceResultState | null;
};
```

## 4. 结果数据契约

```ts
type PerformanceResultState = {
  summary: {
    ttftMs: number;
    prefillTps: number;
    decodeTps: number;
    totalRuntimeMemoryGb: number;
    prefillBottleneck: BottleneckType;
    decodeBottleneck: BottleneckType;
    memoryFitsCapacity: boolean | null;
    memoryExcessGb: number | null;
  };
  comparison: {
    dominantCostPrefill: string;
    dominantCostDecode: string;
    computeDemandPrefill: number;
    computeDemandDecode: number;
    memoryTrafficPrefill: number;
    memoryTrafficDecode: number;
    effectiveThroughputPrefill: number;
    effectiveThroughputDecode: number;
  };
  breakdown: {
    prefill: BreakdownRow[];
    decode: BreakdownRow[];
  };
  intermediateMetrics: IntermediateMetric[];
  memoryBreakdown: MemoryBreakdownRow[];
  formulaTrace: FormulaTraceSection[];
  tokenSweepSeries: TokenSweepSeriesPoint[];
  structureSummary: StructureSummarySnapshot;
};
```

```ts
type BottleneckType = "compute-bound" | "bandwidth-bound";
```

```ts
type BreakdownRow = {
  module: "attention-core" | "compressor" | "indexer" | "moe" | "output-projection";
  value: number;
  share: number;
  boundBy: BottleneckType;
};
```

```ts
type IntermediateMetric = {
  key: string;
  symbol: string;
  value: number | string;
  unit: string;
  source: "config" | "derived" | "formula";
};
```

```ts
type MemoryBreakdownRow = {
  key: "weights" | "persistentDecodeCache" | "peakTempWorkingSet" | "runtimeOverhead" | "estimatedTotal";
  valueGb: number;
  share: number | null;
  note: string;
};
```

```ts
type FormulaTraceSection = {
  category: "prefill" | "decode" | "memory";
  rows: {
    label: string;
    expression: string;
    evaluated: string;
  }[];
};
```

```ts
type TokenSweepSeriesPoint = {
  tokenLength: number;
  prefillTps: number;
  decodeTps: number;
  ttftMs: number;
  totalRuntimeMemoryGb: number;
  prefillBottleneck: BottleneckType;
  decodeBottleneck: BottleneckType;
  intermediateSummary: {
    prefillFlops?: number;
    decodeCacheGb?: number;
    kvVisibleLength?: number;
  };
};
```

## 5. 布局规格

## 5.1 整体骨架

页面采用三层结构：

1. 左侧导航
2. 顶部应用栏
3. 主内容工作区

主内容工作区采用：

- `参数控制区`
- `结果分析区`

结果分析区采用全宽语义分区：

1. 第一行：六张指标卡采用两行三列；第一行依次为 `TTFT`、`Prefill TPS`、`Peak Runtime Memory`，第二行集中展示 `Initial Decode TPS`、`Average Decode TPS`、`Decode Time`。内存拆解与指标区并排时保持至少 320px
2. 第二行：`Prefill / Decode 对比` 独占全宽
3. 第三行：结构摘要与当前上下文摘要并排，宽度比例约为 58:42，为结构摘要中的长标签和数值预留足够空间。结构摘要沿用紧凑键值格式，显示 Decoder Layers、Hidden Size、Attention Heads、KV Heads、Experts、Active Experts / Token、MoE Intermediate Size 和 Context Limit，数值直接取自当前模型定义
4. 第四行：Token 趋势图独占全宽
5. 第五行：公式追溯独占全宽
6. 第六行：中间量结果表独占全宽

`公式追溯` 必须位于 `Token 趋势图` 正下方，并按 `Prefill`、`Decode`、`Memory` 小标题形成三行独立折叠区。三个折叠区首次进入均为收起状态，可分别展开；展开后该阶段的公式项按一行三列排列并顶部对齐。

每张追溯小公式卡片必须链接到公式说明页中与其对应的小公式位置。跳转时自动展开所属的 Prefill、Decode 或 Memory 折叠板块，并将对应小公式滚动到可视区域；不能只定位到阶段大板块。

`dense-decoder-transformer` 与 `dense-decoder-moe` 策略同样必须提供完整的 Decode 追溯，至少包含 Sliding/Full Attention 可见长度、分层 cache 流量、权重流量、Decode 总字节、每 token FLOPs、Compute Ceiling、Bandwidth Ceiling 和 Effective Decode TPS，不能只返回 Prefill 与 Memory。
Prefill/Decode 对比表宽度不足时允许容器内部横向滚动，禁止内容越界覆盖相邻卡片。
窄屏下第一行、摘要行和公式三列均改为单列堆叠。

## 5.2 参数控制区布局

参数控制区为四个卡片组。桌面默认使用两列两行，保证长字段名、模型推荐精度和说明文字具有足够宽度；窄屏改为单列。计算假设中的复选项占满卡片整行。

### Card 1: 模型选择

字段：

- `模型家族`
- `模型`

只读摘要：

- `Layers`
- `Hidden Size`
- `Attention Heads`
- `Experts`
- `Context Limit`

交互：

- 性能计算页切换模型时，仅更新本页待计算上下文，不实时修改模型结构页与公式说明页的手动选择
- 每次成功执行 `计算性能` 后，模型结构页与公式说明页同步到本次计算使用的模型
- 切换模型不会自动清空平台参数

### Card 2: 输入长度

字段：

- `Prompt Token Length`
- `Decode Output Tokens`

趋势图范围：

- `Token Sweep Start`
- `Token Sweep End`
- `Token Sweep Step`

快捷按钮：

- `4K`
- `8K`
- `32K`
- `128K`
- `1M`

交互：

- 点击快捷按钮时，应更新 `prefillTokenLength`，并使 decode 从该 prompt 的 KV cache 开始：
  - 若当前趋势范围为空，可同步初始化 `tokenRangeEnd`
- `Start <= End`
- `Step > 0`
- `Step` 不允许大于 `End - Start`，若大于则显示校验错误
- `Decode Output Tokens` 可留空；留空时按 `Prompt Token Length` 计算

### Card 3: 平台参数

字段：

- `Compute Throughput (TFLOPS)`
- `Memory Bandwidth (GB/s)`
- `HBM / VRAM Capacity (GB)`
- `Platform Template`

首版要求：

- `Platform Template` 至少支持 `Custom`
- 模板切换后会写入对应默认值，但用户仍可继续修改

### Card 4: 计算假设

字段：

- `Batch Size`
- `Compute Efficiency`
- `Bandwidth Efficiency`
- `Show Intermediate Metrics`
- `Show Formula Trace`

交互：

- `Show Intermediate Metrics` 控制中间量表的展示，不影响计算
- `Show Formula Trace` 控制公式追踪卡的展示，不影响计算
- 当 `Total Runtime Memory > HBM / VRAM Capacity` 时，页面以红字提示内存不足，不再对 `Prefill TPS` / `Decode TPS` 施加惩罚系数

### 参数区底部操作条

组件：

- `计算性能`
- `重置`
- `复制当前配置`
- `状态提示`

状态规则：

- 所有必填项合法时：`ready`
- 存在非法输入时：`invalid`
- 点击计算后进入：`calculating`
- 计算完成：`calculated`

## 5.3 左列布局

### Section A: 核心指标卡组

六张卡，桌面端固定为两行三列：

- 第一行：`TTFT`、`Prefill TPS`、`Peak Runtime Memory`
- 第二行：`Initial Decode TPS`、`Average Decode TPS`、`Decode Time`
- 所有 Decode 吞吐与耗时指标必须集中在第二行；窄屏可按响应式布局降为两列或一列。

每张卡必须显示：

- 主数值
- 单位
- 解释副标题
- 绑定标签

示例：

- `Prefill TPS`
  - main: `1234`
  - unit: `tokens/s`
  - sub: `Current platform estimate`
  - tag: `Compute-bound`

### Section B: Prefill / Decode 对比表

固定行：

- `Dominant Cost`
- `Compute Demand`
- `Memory Traffic`
- `Effective Throughput`
- `Latency`

展示规则：

- 统一左右列对比
- 单位明确
- 对差异大的行做视觉强调

### Section C: Token 趋势图区

这是页面的一等区域，默认展开。

#### 顶部控制

字段：

- `Metric`
  - `Prefill TPS`
  - `Decode TPS`
  - `TTFT`
  - `Total Runtime Memory`
- `Sweep Mode`
  - 首版固定 `Fixed Step`
- `Show Bottleneck Background`
- `Show Data Points`

#### 图表契约

X 轴：

- `Token Length`

Y 轴：

- 由 `selectedTrendMetric` 决定

展示规则：

- 若 `selectedTrendMetric` 为 `prefillTps`：
  - 主曲线显示 `prefillTps`
  - 次曲线可同时显示 `decodeTps`
- 若 `selectedTrendMetric` 为 `decodeTps`：
  - 主曲线显示 `decodeTps`
  - 次曲线可同时显示 `prefillTps`
- 若 `selectedTrendMetric` 为 `ttft` 或 `totalRuntimeMemory`：
  - 单曲线模式

Tooltip 必须显示：

- `Token Length`
- 当前曲线值
- `Prefill Bottleneck`
- `Decode Bottleneck`
- 至少一个中间量摘要

数据生成规则：

- 使用 `[tokenRangeStart, tokenRangeEnd]` 按 `tokenRangeStep` 生成离散点
- 每个点都调用同一套计算逻辑重新求值
- 禁止仅通过 UI 插值生成伪数据点

趋势图下方必须展示相同离散点的表格，至少包含 Token Length、Prefill TPS、Decode TPS、TTFT、Runtime Memory 和两阶段 Bottleneck。

每张趋势图的纵轴必须根据该指标当前采样点的实际最小值和最大值独立缩放并保留少量上下边距；不得强制最小跨度为 1。纵轴刻度的小数位根据当前数值跨度自动调整，避免 Decode TPS 等小范围数据的多个刻度被舍入成相同整数。

边界规则：

- 若点数超过 `500`，前端应提示范围过密，并阻止直接计算
- 若点数低于 `2`，不生成趋势图，显示校验提示

### Section D: 瓶颈拆解区

组件：

- `Breakdown Mode Toggle`
  - `Prefill`
  - `Decode`
- 堆叠条 / 条形图
- 明细表

模块固定为：

- `Attention Core`
- `Compressor`
- `Indexer`
- `MoE`
- `Output Projection`

### Section E: 中间量结果表

默认展示，受 `Show Intermediate Metrics` 控制。

列固定为：

- `Metric`
- `Symbol`
- `Value`
- `Unit`
- `Source`

排序规则：

- 先展示 summary 直接相关项
- 再展示 prefill
- 再展示 decode
- 再展示 memory

## 5.4 右列布局

### Section A: 模型结构摘要卡

内容：

- `Decoder Layers`
- `Attention Heads`
- `Head Dim`
- `KV Heads`
- `MoE Experts`
- `Hash-MoE Layers`
- `Compression Schedule`

底部按钮：

- `查看完整结构页`

### Section B: 显存需求分析卡

固定分段：

- `Weights`
- `Persistent Decode Cache`
- `Peak Temp Working Set`
- `Runtime Overhead`

`Runtime Overhead` 是运行框架、CUDA 上下文、内存池、碎片和工作缓冲区等未单独建模开销的估算假设。默认值为 `4 GB`，在内存拆解中明确标注“估算假设，可手动编辑”。用户修改后，重新计算的总显存、容量判断、趋势数据、公式追溯和历史记录统一使用该输入值。

模型摘要中的 `Context` 来自模型配置声明的最大上下文窗口（`max_position_embeddings` 或 `text_config.max_position_embeddings`），不得使用性能验收场景长度代替。`Prompt Token Length` 默认值 `131072`（128K）是验收计算长度，输入项下方应明确注明它不是模型最大上下文窗口。

输入长度快捷值必须受当前模型 `contextLimit` 约束：超过最大上下文的快捷按钮禁用，状态更新函数同时拒绝越界值；手动输入的 Prompt 长度和趋势扫描起点、终点也必须进行相同校验。`Token Sweep Start`、`Token Sweep End`、`Token Sweep Step` 三项归入独立的“Token趋势图扫描”小节。

快捷输入区位于输入长度板块最下方，标题为“快捷输入”。用户先聚焦 `Prompt Token Length`、`Decode Output Tokens` 或任一 Token Sweep 输入框，再点击 `4K / 8K / 32K / 128K / 1M`，快捷值写入当前聚焦的目标字段；界面同时显示当前输入目标。

桌面双栏布局中的“模型选择”与“输入长度”保持等宽。模型选择卡片内部的摘要使用左列 `0.75fr`、右列 `1.25fr`，将列分界明显向左移动，为 `Routed Experts / Layer` 和推荐精度一侧保留更多空间，避免专家总数等数字被拆行；不改变字体、字段结构或其他视觉样式。

模型选择摘要不得使用含义模糊的 `Experts: active / total` 合并展示。专家信息拆分为 `Active Experts / Token`（每个 token 被路由选中的专家数）和 `Routed Experts / Layer`（每个 MoE 层可供路由选择的专家总数）。

默认平台按 NVIDIA GB10 / DGX Spark 标称规格设置，`HBM / VRAM Capacity` 默认值为 `128 GB`。该值表示统一系统内存的标称容量；其余平台参数和模型精度参数保持各自现有口径。

模型注册表通过 `parameterSourceUrl` 保存官方模型页。模型选择卡片底部展示“模型参数来源”外部链接，并随当前模型自动切换；页面组件不得硬编码模型 URL。
- `Estimated Total`

每段显示：

- 绝对值
- 占比
- 注释

若填写了 `memoryCapacityGb`：

- 显示 `Fits Capacity` 或 `Exceeds Capacity`
- 若超出，显示超出量

### Section C: 公式追踪卡

Tab：

- `Prefill`
- `Decode`
- `Memory`

每行显示：

- `label`
- `expression`
- `evaluated`

底部按钮：

- `查看完整公式说明页`

### Section D: 假设与备注卡

内容：

- 当前平台假设
- 精度假设
- 模型来源
- 解析估算免责声明

## 6. 交互流程

## 6.1 首次进入

- 页面加载默认模型：`deepseek-v4-flash`
- 使用默认 token 与假设值填充表单
- 若关键平台参数为空，则结果区显示空态

## 6.2 输入与校验

校验规则：

- 所有数值字段必须为正数
- `tokenRangeStart <= tokenRangeEnd`
- `tokenRangeStep > 0`
- `tokenRangeStep <= tokenRangeEnd - tokenRangeStart`，除非 `Start == End`
- `batchSize >= 1`
- `computeEfficiency`、`bandwidthEfficiency` 应限制在 `(0, 1]`

校验失败时：

- 保持现有结果不清空
- 顶部状态改为 `invalid`
- 对应字段显示错误

## 6.3 计算触发

首版采用显式触发：

- 点击 `计算性能` 才执行计算

触发后必须一次性产出：

- 单点 summary 结果
- comparison
- breakdown
- memory breakdown
- formula trace
- token sweep series

## 6.4 页面跳转

- 点击 `查看完整结构页` 跳转 `模型结构`
- 点击 `查看完整公式说明页` 跳转 `公式说明`
- 跳转时保持当前模型和输入状态
- 每次成功执行 `计算性能` 后，`模型结构` 与 `公式说明` 页的滚动位置重置到页面顶端
- 每次成功执行 `计算性能` 后，`公式说明` 页的公式折叠项及小公式变量说明全部恢复为收起状态

## 6.5 HTML 报告与 JSON 导出

- 性能计算页标题区右上角提供 `打开 HTML 报告` 操作，不再提供 Excel 导出或自动下载 HTML 文件。
- 报告必须基于最近一次成功计算的快照，而不是尚未重新计算的表单草稿。
- HTML 为 UTF-8、自包含的新页面报告，不依赖外部 CSS、JavaScript 或在线资源；支持浏览器横向打印和按需另存网页。
- 主表采用 `Type / Calculation Layer / Item / Value / Notes (Description / Formula)` 的逻辑，其中 Calculation Layer 与 Item 在页面中合并为一列。
- `Type` 使用纵向合并和固定颜色表达模型的大运算量板块；板块内逐行列出投影、attention core、卷积、递归扫描等计算层。
- HTML 报告按 `formulaStrategyId` 选择专用分栏模板，不按模型名称硬编码：
  - `hybrid-linear-dense`：Model Config、Dense FFN、Full GQA、Gated DeltaNet、Prefill Total；
  - `hybrid-linear-moe`：Model Config、Active + Shared MoE FFN、Full GQA、Gated DeltaNet、Prefill Total；
  - `dense-decoder-transformer`：Model Config、Dense FFN、Sliding-window Attention、Full Attention、Prefill Total；
  - `dense-decoder-moe`：Model Config、Active + Shared MoE FFN、Sliding-window Attention、Full Attention、Prefill Total；
  - `deepseek-v4-compressed-moe`：Model Config、Sparse MoE FFN、Sliding Attention、CSA、HCA、Prefill Total；CSA 块内必须列出 Compressor 与 Indexer，HCA 块内必须列出 Compressor。
- Qwen3.6-27B 使用 Dense FFN 口径；Qwen3.6-35B-A3B 使用每 token 激活 routed experts 加 shared expert 的 MoE 口径。FP8 作为平台精度配置，不单独注册成模型；不得用 routed experts 总数冒充实际计算专家数。
- Projection 行展示单层、单 token 理论 FLOPs；块汇总按模型注册表中的实际层数累加；Full Attention core 使用当前成功计算快照的 Prompt Token Length。
- 报告显式声明其为理论工程估算而非实测数据，并展示模型、Prompt Length、Prefill TPS、TTFT、导出时间和公式策略。
- 算子分块表下方追加两张标准上下文明细表：
  - Prefill Detail：`Context / GFLOPs per Token / TPS @20% Compute Util / TPS @40% Compute Util / TTFT(sec) @40%`；情景只替换 Compute Efficiency，Bandwidth Efficiency 保持计算快照值。
  - Decode Detailed Data：`Context / Persistent Cache / Temp Peak / Total Memory / TPS @40% BW Util / TPS @60% BW Util / TPS @80% BW Util`；情景只替换 Bandwidth Efficiency，Compute Efficiency 保持计算快照值。
- 明细表标准 Context 使用 `1K / 2K / 4K / 8K / 16K / 32K / 64K / 128K` 且不超过 Token Range End；没有 MTP 参数和公式时不得添加 MTP 倍率列。
- 新增公式策略时必须同步新增 HTML 专用模板；未知策略应显式报错，不能静默套用不匹配的通用表格。
- 性能计算页同时保留 `导出 JSON`；JSON 直接由计算快照生成，并包含模型、平台、工作负载、核心结果、Prefill/Decode Projection、中间量、公式追溯和 Token Trend。数值必须保持 JSON number，单位在独立 `units` 字段或语义明确的字段名中表达。
- 输入校验失败、未实际完成计算时，不重置上述页面状态

## 6.6 Decode 区间与高级估算参数

- `Decode Output Tokens` 必须实际参与 Decode 估算，并满足 `Prompt Length + Decode Output Tokens <= Context Limit`。
- 结果区分别展示：
  - `Initial Decode TPS`：Prompt 上下文结束后生成第一个 token 的瞬时吞吐；
  - `Average Decode TPS`：完整输出区间内，以逐步延迟累加得到的平均吞吐；
  - `Total Decode Time`：完整生成区间的延迟总和；
  - `Peak Runtime Memory`：生成全部输出 token 后的最终上下文显存峰值。
- 超长输出区间最多采样 256 个均匀上下文点估计平均单步延迟；输出不超过 256 tokens 时逐 token 精确累加。
- `Prefill Cache Traffic Factor` 直接显示在“计算假设”区域，不折叠：
  - 默认值 `0.10`；
  - 允许范围 `0–1`；
  - 用户可手动编辑；
  - 代入 `B_prefill = B_weights + M_cache × prefill_cache_traffic_factor`。
- Dense 与 Hybrid 路径的 Attention 临时工作集使用 `Bytes / Activation`，不得将元素字节数写死为 `2`。
- 公式追溯、中间量、历史记录、HTML 报告和 JSON 导出必须包含与各自用途相关的上述参数与区间结果。

## 7. 组件清单

页面至少需要这些前端组件：

- `ModelSelectorCard`
- `WorkloadInputCard`
- `PlatformInputCard`
- `CalculationAssumptionsCard`
- `CalculationToolbar`
- `SummaryMetricCards`
- `PrefillDecodeComparisonTable`
- `TokenTrendChartPanel`
- `BottleneckBreakdownPanel`
- `IntermediateMetricsTable`
- `StructureSummaryCard`
- `MemoryBreakdownCard`
- `FormulaTraceCard`
- `AssumptionsNoteCard`

## 8. 图表实现约束

- 趋势图组件必须支持多序列和 tooltip 自定义
- breakdown 图必须支持模块占比展示
- 所有图表颜色语义必须稳定：
  - `Prefill` 一套固定色
  - `Decode` 一套固定色
  - `Memory` 一套固定色
  - `MoE / Attention / Indexer / Compressor / Output` 各自固定色

## 9. 空态与异常态

### 空态

- 结果区显示引导说明
- 说明需要先填写平台参数并点击计算

### 计算失败态

- 保留用户输入
- 在结果区顶部显示错误条
- 错误信息要区分：
  - 输入非法
  - 计算逻辑错误
  - 不支持的模型策略

### 超显存态

- 不阻止展示结果
- 显存卡和 summary 卡明确标出超出

## 10. 验收标准

- 页面可在不切页的情况下完成一次完整计算
- 单点结果和 token 趋势图使用同一套输入上下文
- token 范围可配置，且趋势图不是静态占位
- 显存分析中必须包含 `weights`
- 趋势图 tooltip 能看到 token 值、指标值和瓶颈分类
- 结构摘要和公式追踪都能跳转到对应页面

## 11. 后续扩展预留

- `Platform Template` 扩展为预置 GPU 列表
- `tokenSweepMode` 扩展为 `log-scale`
- 支持多模型横向趋势对比
- 支持导出当前图表与配置

## 12. Qwen3.6-35B-A3B 验收基线

`Qwen/Qwen3.6-35B-A3B` 在 128K prompt、Batch 1、BF16
weights / experts / activations 下，公式策略必须得到：

- Prefill：约 `2045.006 TFLOPs`
- Decode：约 `26.340 GFLOPs/token`
- 运行时显存：约 `78.217 GB`

公式追溯必须分别列出 Full GQA、Gated DeltaNet、MoE、Full KV cache
与线性 recurrent state，不能只显示聚合总量。

Qwen3.6-35B-A3B 在相同128K、Batch 1场景下切换为 FP8 平台精度配置时，
理论FLOPs保持不变；权重显存为 `34.660 GB`、active weight traffic 为
`3.311 GB/token`、运行时总显存约为 `43.557 GB`。FP8 不作为独立模型；
用户按平台参数区建议手动应用 1/2/1 bytes（weight/activation/expert）。

`Qwen3.6-27B`使用独立的`hybrid-linear-dense`策略。128K、
Batch 1验收值为Prefill `9771.463 TFLOPs`、Decode
`100.320 GFLOPs/token`、权重流量`27.000 GB/token`及运行时总显存
`42.966 GB`。切换模型后自动应用1/2 bytes（weight/activation）。
