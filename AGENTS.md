# AGENTS

## 项目概述

`llm-perf-calculator` 是一个面向内部研发与性能分析的应用，用于在选定模型与平台参数后，估算大语言模型在不同 token 长度下的：

- `TTFT`
- `Prefill TPS`
- `Decode TPS`
- 运行时内存需求
- 关键公式与中间推导结果

首版聚焦 `DeepSeek V4` 家族：

- `DeepSeek-V4-Flash`
- `DeepSeek-V4-Pro`

模型选择采用两段式：

- 先选 `family`
- 再在该 `family` 下选具体模型

当前 `DeepSeek V4` family 下的可选模型由模型注册表统一提供，不要在页面里手写模型列表。

交付形态同时覆盖：

- 浏览器应用
- `Tauri` 桌面应用

## 当前技术方向

- 前端：`React + Vite`
- 桌面壳：`Tauri`
- 计算方式：解析公式优先，本地计算，不依赖在线推理服务

## 事实源

设计与产品约定以以下文档为准：

- 主设计文档：[docs/app/design/app-design-spec.md](docs/app/design/app-design-spec.md)
- 架构设计文档：[docs/app/design/architecture-design.md](docs/app/design/architecture-design.md)
- 性能计算页分页设计：[docs/app/design/pages/performance-calculator.md](docs/app/design/pages/performance-calculator.md)

如讨论结果影响页面结构、交互、术语、结果字段或目录职责，应同步更新这些文档。

## 仓库结构

```text
.
├── AGENTS.md
├── data/
├── docs/
├── scripts/
├── src/
└── src-tauri/
```

各目录职责如下：

- `src/app/`
  - 应用入口、路由、布局、全局 provider、主题和全局样式。

- `src/pages/`
  - 页面级装配层。
  - 页面负责组合 feature，不直接承载复杂公式逻辑。
  - 性能计算页、模型结构页、公式说明页共享同一套计算状态，不要各自维护独立输入副本。

- `src/features/`
  - 面向页面的业务模块。
  - 例如 `performance-calculator`、`model-structure`、`formula-notes`。

- `src/components/`
  - 可跨页面复用的 UI 组件。

- `src/domain/`
  - 核心业务类型与稳定接口定义。

- `src/engines/`
  - 纯计算与策略层。
  - 包括模型注册、公式策略、性能估算逻辑。
  - 模型 family 和模型项统一从 `src/engines/model-registry/` 提供。
  - 不同架构的公式口径通过 `formulaStrategyId` 区分，不要把 DeepSeek V4 公式直接套到其他模型上。

- `data/models/`
  - 模型静态定义和结构化数据。

- `data/platform-presets/`
  - 平台参数预设。

- `src-tauri/`
  - Tauri 桌面宿主代码。

- `scripts/`
  - 数据整理、校验、转换等工程脚本。

## 分层约束

- 不要把模型结构常量直接散落在页面组件中。
- 新模型应优先通过 `data/models/` 和 `src/engines/model-registry/` 扩展。
- 新公式应优先通过 `src/engines/formula-strategies/` 扩展。
- `src/engines/` 不应依赖 `src/pages/` 或 `src/features/`。
- 页面层只消费统一结果对象，不直接耦合底层公式细节。
- 共享状态优先放在 `src/features/performance-calculator/state/`，由 `CalculatorProvider` 向多个页面提供一致的模型、输入、平台参数和计算假设。

## 实现原则

- 优先保证工程可追溯。
  - 所有关键结果应可回溯到输入参数、模型配置、公式和中间量。

- 优先保证模型扩展性。
  - 不能把 `DeepSeek V4` 写死成唯一支持对象。
  - 新增 family 时，先补模型注册表，再补对应公式策略，最后再让页面按 `architectureKind` 条件展示结构内容。

- 优先保证 Web / Desktop 统一体验。
  - 不要引入只能在浏览器站点式场景成立的核心交互假设。

- UI 应服务于内部分析工具场景。
  - 允许高密度，但必须层次清晰、可扫描。

## 文档同步规则

出现以下变更时，必须同步更新 `docs/app/design/` 下文档：

- 页面结构变化
- 核心输入项变化
- 结果字段变化
- 公式口径变化
- 模型支持范围变化
- 新增或调整效果图

效果图保存目录固定为：

- `docs/app/design/images/`

## 新增模型流程

新增 model family 或 model 时，先完成模型结构分析，再做工程适配。

### 1. 模型结构分析

- 前置材料：
  - 模型 `config.json`。没有时向用户索要。
  - 可用于核对结构的本地 `transformers` 推理代码。
- 产出文档：`docs/$model_family/$model_id.md`。
  - `$model_family` 与模型注册表中的 `family` 字段一致，例如 `deepseek_v4`。
  - `$model_id` 与 `ModelDefinition.id` 一致，例如 `DeepSeek-V4-Flash.md`。
  - `config.json` 保存到 `docs/$model_family/config/$model_id-config.json`。
  - 参考实例：`docs/deepseek_v4/DeepSeek-V4-Flash.md`。
- 文档应覆盖：
  - 顶层结构、核心超参、层类型 schedule。
  - 单层残差流、Attention、MoE、KV Cache、RoPE、量化或部署相关信息。
  - ASCII 字符图或 Mermaid 图，并标注关键维度。
  - Prefill 阶段 FLOPs 拆解与占比。
  - Decode 阶段权重常驻、持久 cache、单步临时工作集和 128K 场景显存估算。
- 架构文档和算力/显存估算需经用户确认后，再进入工程适配。

### 2. 工程适配

- 必要时在 `src/domain/model/types.ts` 补充 `architectureKind` 或 `formulaStrategyId`。
- 在 `src/engines/model-registry/` 新增或扩展 family 文件，并在 `index.ts` 注册模型与 `familyDisplayNames`。
- 必要时在 `src/engines/formula-strategies/` 新增公式策略，通过 `formulaStrategyId` 区分口径。
- 必要时把结构化静态数据放入 `data/models/$family/`。
- 页面只能按 `architectureKind` 做条件展示，不要手写模型列表或硬编码公式。

## 推荐开发顺序

1. 初始化前端工程基础文件。
2. 落地应用级路由、布局和左侧导航。
3. 定义 `domain` 层核心类型。
4. 建立模型注册表与公式策略接口。
5. 实现性能计算页静态骨架。
6. 接入计算引擎与趋势图。
7. 补模型结构页与公式说明页。
8. 最后接入 `Tauri` 桌面壳细节。

## 提交前检查

- 新代码是否放在正确分层。
- 公式是否可追溯，命名是否清晰。
- 页面是否仍然符合高密度工程工具定位。
- 设计文档是否需要同步更新。
- 新增模型或平台预设是否已进入结构化数据目录。
- 新增模型后，是否已经更新 family/model 两段式选择和相关文档。
