# LLM Perf Calculator App Design Spec

## 1. 文档目的

本文档用于沉淀 `LLM Perf Calculator` 应用的产品设定、界面结构、交互骨架和实现约束，作为后续视觉稿、前端实现和桌面打包的统一事实源。

后续与本应用相关的讨论，应优先更新本文件，再按需要补充分页设计文档和效果图。

## 2. 文档与资产约定

- 设计文档目录：`docs/app/design/`
- 主 spec：`docs/app/design/app-design-spec.md`
- 架构设计：`docs/app/design/architecture-design.md`
- 设计效果图目录：`docs/app/design/images/`
- 效果图文件命名建议：
  - `performance-calculator-v1.png`
  - `model-structure-v1.png`
  - `formula-notes-v1.png`
- 当前已生成效果图：
  - `docs/app/design/images/performance-calculator-v1.png`
  - `docs/app/design/images/model-structure-v1.png`
  - `docs/app/design/images/formula-notes-v1.png`
- 当前已创建分页设计文档：
  - `docs/app/design/pages/performance-calculator.md`
  - `docs/app/design/pages/model-structure.md`
  - `docs/app/design/pages/formula-notes.md`
- 后续如继续拆分页设计文档，建议放在：
  - `docs/app/design/pages/performance-calculator.md`
  - `docs/app/design/pages/model-structure.md`
  - `docs/app/design/pages/formula-notes.md`

## 3. 当前已确认设定

### 3.1 产品目标

应用需要支持以下核心能力：

1. 选定模型后，结合平台参数（算力、内存带宽等），在给定 token 长度下计算模型的 `prefill` 和 `decode` 性能。
2. 结果至少包含：
   - `TTFT`
   - `Prefill TPS`
   - `Decode TPS`
3. 选定模型后，可以查看模型结构。
4. 应用既可以在浏览器打开，也可以打包成桌面应用。
5. 首版先支持 `DeepSeek V4` 家族：
   - `DeepSeek-V4-Flash`
   - `DeepSeek-V4-Pro`
6. 当前已扩展支持 `Gemma 4` family：
   - `Gemma-4-12B-it`
   - `google/gemma-4-26B-A4B-it`
7. 当前已扩展支持 `Qwen3.5` 与 `Qwen3.6` hybrid-linear-MoE family：
   - `Qwen3.5-35B-A3B`
   - `Qwen/Qwen3.6-35B-A3B`
   - `Qwen/Qwen3.6-35B-A3B-FP8`
   - `Qwen/Qwen3.6-27B-FP8`
   - `Qwen/Qwen3.6-35B-A3B` 的已确认工程事实源为
     `data/models/Qwen_3.6/Qwen3.6-35B-A3B.json`；其结构为 40 层
     hybrid decoder（10 层 Full GQA + 30 层 Gated DeltaNet）、
     256 routed experts、每 token 激活 8 个专家并带 1 个 shared expert。
     Base 版本默认使用 BF16 weights / experts / activations。
   - `Qwen/Qwen3.6-35B-A3B-FP8` 与 Base 共享相同结构和理论 FLOPs；
     推荐精度为 FP8 weights / FP8 experts / BF16 activations。默认权重
     显存采用文本 decoder 简化口径 `34.660 GB`，不把 KV cache 或
     recurrent state 错误地按 FP8 计算。
   - `Qwen/Qwen3.6-27B-FP8` 是 `hybrid-linear-dense`，包含48层
     Gated DeltaNet、16层Full GQA和64层dense SwiGLU FFN；它没有
     routed experts。推荐精度为FP8 weights / BF16 activations。
8. 架构必须为后续支持更多模型留出通用扩展空间。

### 3.2 技术与交付方向

- 前端技术栈：`React + Vite`
- 桌面打包：`Tauri`
- 计算口径：解析公式优先
- 首版不做在线推理，不接服务端，默认前端本地完成计算和展示

### 3.3 视觉与信息架构方向

- 定位：内部研发/性能分析工具
- 视觉风格：工程工具导向，而非营销展示页
- 导航结构：左侧导航多页
- 首版页面：
  - `性能计算`
  - `模型结构`
  - `公式说明`
- 首页信息密度：高密度分析视图

## 4. 设计原则

### 4.1 工程可追溯

所有关键结果都应能追溯到：

- 输入参数
- 模型配置
- 使用的公式
- 中间推导结果

不能只展示结论，不展示计算来源。

### 4.2 高密度但可扫描

页面服务对象是内部研发，因此允许较高信息密度，但必须保证：

- 视觉分区清晰
- 主指标优先突出
- 中间量和解释性内容可折叠或按块组织

### 4.3 模型扩展优先

页面和数据结构不能把 DeepSeek V4 写死在组件逻辑中。界面应围绕“模型定义 + 公式策略 + 平台输入 + 结果输出”设计。

### 4.4 Web / Desktop 统一体验

同一套 UI 同时面向：

- 浏览器内使用
- Tauri 桌面打包

因此应避免依赖浏览器特有的站点式信息架构，优先采用应用工作台式布局。

## 5. 页面地图

左侧导航包含以下项目：

1. `性能计算`
2. `模型结构`
3. `公式说明`
4. `历史记录`
   - 成功计算后保存日期、模型类型、平台/工作负载参数与结果摘要
   - 支持按时间升降序、按模型筛选、展开详情与清空全部记录
   - 使用 Web / Tauri WebView 的 localStorage 本地持久化

页面关系：

- `性能计算` 是主工作台
- `模型结构` 用于解释当前选中模型
- `公式说明` 用于解释当前结果的公式与符号
- 三页共享同一套“当前模型 + 当前平台参数 + 当前输入长度”状态

## 6. 共享数据模型

### 6.1 Model Definition

抽象层面需要支持：

- `modelFamily`
- `modelId`
- `displayName`
- `configSource`
- `architectureSummary`
- `structureMetrics`
- `formulaStrategyId`
- `supportedViews`

对于 DeepSeek V4 首版，至少包含：

- `deepseek-v4-flash`
- `deepseek-v4-pro`

### 6.2 Platform Input

平台参数至少包含：

- `computeThroughputTflops`
- `memoryBandwidthGbps`
- `memoryCapacityGb`
- `computeEfficiency`
- `bandwidthEfficiency`
- `batchSize`
- `precisionAssumptions`

### 6.3 Workload Input

负载输入至少包含：

- `prefillTokenLength`
- `decodeOutputTokens`
- `tokenRangeStart`
- `tokenRangeEnd`
- `tokenRangeStep`
- `tokenSweepMode`

### 6.4 Result Model

结果对象至少包含：

- `ttft`
- `prefillTps`
- `decodeTps`
- `totalRuntimeMemory`
- `bottleneckClassification`
- `memoryBreakdown`
- `formulaTrace`
- `intermediateMetrics`
- `tokenSweepSeries`

## 7. 详细文字线框

## 7.1 性能计算页

### 页面目标

让用户在一个高密度工作台内完成：

- 选择模型
- 输入平台参数
- 输入 token 长度
- 计算 prefill / decode 性能
- 查看性能随 token 数变化的趋势
- 查看结构摘要
- 查看显存估算
- 追溯关键公式

### 页面总体布局

页面采用：

- 左侧固定导航
- 顶部应用栏
- 主内容区

主内容区采用：

- 上方参数控制区
- 下方双列分析区

### 左侧导航

固定垂直导航，包含：

- `性能计算`
- `模型结构`
- `公式说明`
- `历史记录`

样式要求：

- 当前页面高亮
- 支持图标 + 文本
- `历史记录` 显示为已启用的本地计算日志入口

### 顶部应用栏

包含：

## 8. 项目目录结构

当前仓库建议采用以下整体结构，以兼顾：

- `React + Vite` 前端实现
- `Tauri` 桌面打包
- 模型定义与公式策略的可扩展性
- 页面层与计算引擎层解耦

```text
.
├── data/
│   ├── models/
│   │   └── deepseek-v4/
│   └── platform-presets/
├── docs/
│   └── app/
│       └── design/
│           ├── images/
│           ├── pages/
│           └── app-design-spec.md
├── scripts/
├── src/
│   ├── app/
│   │   ├── layouts/
│   │   ├── providers/
│   │   ├── routes/
│   │   └── styles/
│   ├── assets/
│   ├── components/
│   │   ├── charts/
│   │   ├── forms/
│   │   ├── layout/
│   │   ├── metrics/
│   │   └── tables/
│   ├── domain/
│   │   ├── model/
│   │   ├── performance/
│   │   ├── platform/
│   │   └── workload/
│   ├── engines/
│   │   ├── formula-strategies/
│   │   ├── model-registry/
│   │   └── performance-calculator/
│   ├── features/
│   │   ├── formula-notes/
│   │   │   ├── components/
│   │   │   └── services/
│   │   ├── model-structure/
│   │   │   ├── components/
│   │   │   └── services/
│   │   └── performance-calculator/
│   │       ├── components/
│   │       ├── hooks/
│   │       ├── services/
│   │       └── state/
│   ├── lib/
│   ├── pages/
│   │   ├── formula-notes/
│   │   ├── history/
│   │   ├── model-structure/
│   │   └── performance-calculator/
│   ├── test/
│   ├── types/
│   └── .gitkeep
└── src-tauri/
    ├── icons/
    └── src/
```

### 8.1 分层职责

- `src/app/`
  - 应用级入口层。
  - 放布局、路由、全局 provider、主题变量和应用级样式。

- `src/pages/`
  - 页面级装配层。
  - 负责把 feature 模块组合成完整页面，不直接承载复杂计算逻辑。

- `src/features/`
  - 面向具体业务页面或交互能力。
  - 例如 `performance-calculator` 负责参数面板、结果卡片、趋势图、状态管理。

- `src/domain/`
  - 放业务核心类型、领域对象和稳定接口定义。
  - 用来约束模型定义、平台输入、工作负载输入、结果对象。

- `src/engines/`
  - 纯计算和策略层。
  - 负责模型注册、公式策略选择、prefill/decode 性能计算，不依赖页面组件。

- `src/components/`
  - 可跨页面复用的 UI 组件库。
  - 如图表容器、指标卡、输入表单组件、通用表格。

- `data/models/`
  - 放模型静态定义或从文档整理出的结构化数据。
  - 首版先承载 `deepseek-v4` 家族，后续可继续增补其他模型族。

- `data/platform-presets/`
  - 放平台预设，如不同 GPU / 芯片的典型算力与带宽参数。

- `src-tauri/`
  - 桌面壳层。
  - 首版前端计算可以全部在 Web 层完成，Tauri 先只保留宿主结构。

- `scripts/`
  - 放数据整理、模型配置转换、校验脚本等工程辅助脚本。

### 8.2 扩展约束

- 新模型优先通过 `data/models/` + `src/engines/model-registry/` 扩展，不应把结构常量散落到页面组件。
- 新公式优先通过 `src/engines/formula-strategies/` 扩展，不应把公式分支直接写入页面层。
- 页面只消费统一的 `Result Model`，避免 UI 直接依赖底层中间公式实现细节。
- `src/features/` 可以引用 `domain`、`engines`、`components`，但 `engines` 不应反向依赖 `features` 或 `pages`。

- 应用标题：`LLM Perf Calculator`
- 当前模型家族标签：`DeepSeek V4`
- 应用状态标签：`Web / Desktop Ready`
- 右侧预留操作：
  - `导出`
  - `文档`

首版可以只实现展示，不必全部可用。

### 参数控制区

参数控制区由 4 个横向卡片组成。

#### A. 模型选择卡

内容：

- 模型家族选择器
  - 首版默认只有 `DeepSeek V4`
- 子模型切换
  - `DeepSeek-V4-Flash`
  - `DeepSeek-V4-Pro`
- 只读摘要字段：
  - `Layers`
  - `Hidden Size`
  - `Attention Heads`
  - `Experts`
  - `Context Limit`

#### B. 输入长度卡

内容：

- `Prompt Token Length`
- `Decode Output Tokens`
- `Token Sweep Range`
  - `Start`
  - `End`
  - `Step`
- 快捷长度按钮：
- `4K`
- `8K`
- `32K`
  - `128K`
  - `1M`

说明：

- `Prompt Token Length` 同时用于 prefill，并作为 decode 开始时 KV cache 中的上下文长度
- `Decode Output Tokens` 用于估算总生成时长，可与单 token TPS 区分显示
  - 默认为空；为空时使用 `Prompt Token Length`
- `Token Sweep Range` 用于生成趋势图
- 首版 `Token Sweep Range` 默认作用于上下文长度维度
- `Start <= End` 且 `Step > 0`

#### C. 平台参数卡

内容：

- `Compute Throughput (TFLOPS)`
- `Memory Bandwidth (GB/s)`
- `HBM / VRAM Capacity (GB)`
- 平台模板下拉
  - 首版可先提供 `Custom`
  - 后续可扩展预置硬件模板

#### D. 计算假设卡

内容：

- `Batch Size`
- `Compute Efficiency`
- `Bandwidth Efficiency`
- `Show Intermediate Metrics` 开关
- `Show Formula Trace` 开关
- 当估算内存超过容量时显示红色内存不足提示，不改变性能计算结果

#### 参数区底部操作条

包含：

- 主按钮：`计算性能`
- 次按钮：`重置`
- 次按钮：`复制当前配置`
- 状态提示：
  - `输入完整`
  - `可计算`
  - 或字段错误摘要

### 下方左列：性能结果主工作台

左列宽度约 60%。

#### A. 核心指标区

四张结果卡：

- `TTFT`
- `Prefill TPS`
- `Decode TPS`
- `Total Runtime Memory`

每张卡包含：

- 主数值
- 单位
- 简短说明
- 小标签：
  - `Compute-bound`
  - `Bandwidth-bound`
  - `Memory-cap limited`

#### B. Prefill / Decode 对比区

使用双列表格对比：

- `Dominant Cost`
- `Compute Demand`
- `Memory Traffic`
- `Effective Throughput`
- `Latency per Token / per Request`

每项显示：

- Prefill 值
- Decode 值
- 差异方向

#### C. Token 趋势图区

这是性能计算页的一等分析区，默认展开，放在“Prefill / Decode 对比区”之后。

用途：

- 展示关键性能指标随 token 数变化的趋势
- 观察不同长度区间下的拐点和瓶颈迁移
- 在图表下方以表格展示每个趋势点的完整数值与瓶颈分类

顶部控制条包含：

- 指标切换：
  - `Prefill TPS`
  - `Decode TPS`
  - `TTFT`
  - `Total Runtime Memory`
- 范围说明：
  - 当前 `Start / End / Step`
- 采样模式：
  - `Fixed Step`
  - 后续可扩展 `Log Scale`
- 显示开关：
  - `Show Bottleneck Background`
  - `Show Data Points`

图表要求：

- 横轴：`Token Length`
- 纵轴：当前选中指标
- 首版至少支持同时显示两条曲线：
  - `Prefill TPS`
  - `Decode TPS`
- 若展示 `TTFT` 或 `Total Runtime Memory`，允许切换为单指标模式
- 鼠标悬停时展示：
  - token 值
  - 当前指标值
  - 当前瓶颈分类
  - 关键中间量摘要

数据要求：

- 每个采样点必须由当前模型、当前平台参数和当前公式重新计算得到
- 不允许通过前端插值伪造结果点
- 首版默认按 `tokenRangeStart ~ tokenRangeEnd` 和 `tokenRangeStep` 生成离散点列

#### D. 瓶颈拆解区

使用可视化图表 + 表格组合。

必须覆盖的模块：

- `Attention Core`
- `Compressor`
- `Indexer`
- `MoE`
- `Output Projection`

图表支持在：

- `Prefill Breakdown`
- `Decode Breakdown`

之间切换。

#### E. 中间量结果表

默认展开。

列结构：

- `Metric`
- `Symbol`
- `Value`
- `Unit`
- `Source`

`Source` 取值至少包括：

- `config`
- `derived`
- `formula`

### 下方右列：结构摘要与显存分析

右列宽度约 40%。

#### A. 模型结构摘要卡

展示：

- `Decoder Layers`
- `Attention Heads`
- `Head Dim`
- `KV Heads`
- `MoE Experts`
- `Hash-MoE Layers`
- `Compression Schedule`

下方放一个小型结构示意图区域。

提供跳转按钮：`查看完整结构页`

#### B. 显存需求分析卡

分段展示：

- `Weights`
- `Persistent Decode Cache`
- `Peak Temp Working Set`
- `Runtime Overhead`
- `Estimated Total`

每段显示：

- 绝对值
- 占比
- 计算说明

若填写了容量参数，则必须额外显示：

- `Fits Capacity`
- 或 `Exceeds Capacity`

并给出超出量。

#### C. 公式追踪卡

支持标签切换：

- `Prefill`
- `Decode`
- `Memory`

每行展示：

- 公式名
- 代入表达式
- 当前结果

底部提供跳转按钮：`查看完整公式说明页`

#### D. 假设与备注卡

展示：

- 当前计算假设
- 当前模型来源
- 解析估算说明

### 页面状态

至少支持以下状态：

- 初始空态
- 已计算态
- 输入错误态
- 超显存态

### 页面约束

- 首版默认桌面断点按 `1440px` 宽设计
- 窄屏时允许右列下折，但不改变信息架构
- 页面切换时参数状态不能丢失
- token 趋势图与单点结果必须共享同一套模型、平台参数和假设上下文
- 修改 `Token Sweep Range` 后，趋势图与相关中间量应一起刷新

## 7.2 模型结构页

### 页面目标

完整展示当前模型的结构、层调度、关键超参和主要模块说明，为性能页提供解释上下文。

### 页面布局

采用：

- 顶部模型切换条
- 上部总览区
- 中部结构图区
- 下部表格与模块说明区

### 主要区域

#### A. 总览头部

展示：

- 模型名
- 一句结构摘要
- 关键数字标签：
  - `Layers`
  - `Heads`
  - `Hidden`
  - `Experts`
  - `Context`

#### B. 结构总图区

显示：

- 总体结构图
- 图例：
  - `Sliding`
  - `CSA`
  - `HCA`
  - `Hash-MoE`
  - `MoE`

首版可以使用静态结构图或简化块图。

#### C. 层调度区

必须包含两个表：

- `Attention Schedule`
- `MLP Schedule`

表中展示：

- 类型
- 层索引
- 数量
- 解释

#### D. 模块说明区

采用折叠卡片：

- `Attention`
- `Compressor`
- `Indexer`
- `Sparse MoE`
- `KV Cache`
- `RoPE`
- `mHC`

每张卡片统一格式：

- 作用
- 关键维度
- 关键公式
- 性能影响

#### E. 配置速查区

高密度表格，支持搜索：

- `field`
- `value`
- `meaning`

## 7.3 公式说明页

### 页面目标

完整解释性能页所用公式，明确符号、来源、代入方式以及适用范围。

### 页面布局

采用：

- 顶部目录与过滤器
- 下方公式分区
- 页面底部符号表

### 主要区域

#### A. 目录与过滤器

分类标签：

- `Prefill`
- `Decode`
- `Memory`
- `Weights`
- `Model-specific`

支持：

- 搜索公式名
- 搜索符号名
- 按模型过滤

#### B. 公式分区

每条公式卡片包含：

- `公式名`
- `用途`
- `符号定义`
- `通用形式`
- `DeepSeek V4 当前代入形式`
- `示例结果`
- `来源文档`

#### C. 符号表

统一列出：

- `symbol`
- `meaning`
- `source`
- `scope`

避免不同页面对同一符号解释不一致。

#### D. 适用范围说明

区分：

- `DeepSeek V4 专用公式`
- `通用 Transformer 公式`
- `工程近似公式`

## 8. 实现约束

- 所有页面共享一套全局工作台状态
- 首版默认仅支持本地计算
- 组件层不能把 `DeepSeek V4` 写死
- 模型应抽象成：
  - 配置定义
  - 公式策略
  - 结构摘要
  - 页面可展示字段

## 9. 首版验收标准

- 应用有 3 个一等页面：
  - `性能计算`
  - `模型结构`
  - `公式说明`
- `性能计算` 页能承载完整参数输入和结果展示
- `性能计算` 页必须支持指定 token 范围并生成趋势图
- `显存估算` 必须包含 `weights`
- `结构页` 和 `公式页` 不是弹窗，也不是附属区域
- 文档内容足够指导后续高保真设计和前端实现

## 10. Open Questions

以下问题后续仍需要在 spec 中继续收敛：

1. 平台模板首版是否内置具体 GPU 型号，如 `H100 / H20 / B200 / 4090`
2. 是否需要导出结果为图片、Markdown 或 JSON
3. `历史记录` 首版是否只保留本地会话内存态
4. 模型结构图首版是静态图、Mermaid 还是前端交互块图
5. 是否需要在性能页支持多模型横向对比
6. token 趋势图首版是否需要支持对数坐标或多指标叠图导出

## 11. Decision Log

### 2026-06-16

- 确定应用定位为内部研发用工程分析工具
- 确定使用 `React + Vite`
- 确定桌面打包使用 `Tauri`
- 确定首版支持 `DeepSeek-V4-Flash` 和 `DeepSeek-V4-Pro`
- 确定导航结构为左侧导航多页
- 确定首版页面为：
  - `性能计算`
  - `模型结构`
  - `公式说明`
- 确定首页采用高密度分析布局
- 确定设计文档保存在 `docs/app/design/`
- 确定效果图保存在 `docs/app/design/images/`
- 确定本轮先输出详细文字线框说明并写入 spec
- 确定性能计算页需要支持“随输入 token 数变化”的趋势图
- 确定趋势图支持由用户指定 token 范围
