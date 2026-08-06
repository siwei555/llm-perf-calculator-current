# LLM Perf Calculator

大语言模型（LLM）性能估算工具。在选定模型与平台参数后，估算 **TTFT**、**Prefill TPS**、**Decode TPS**、**运行时内存**及关键公式推导结果。

> 定位：内部研发/性能分析工具。纯本地公式计算，不依赖在线推理服务。

## 支持模型

| Family | 模型 | 架构 | 参数 |
|--------|------|------|------|
| **DeepSeek V4** | DeepSeek-V4-Flash | Compressed MoE (MLA + CSA/HCA + mHC) | 200B / 6e active |
| | DeepSeek-V4-Pro | Compressed MoE (同上) | 1.5T / 6e active |
| **Gemma 4** | Gemma-4-12B-it | Dense Decoder (GQA/MQA + GeGLU) | 11.9B |
| | google/gemma-4-26B-A4B-it | Dense Decoder MoE (Sliding/Full + routed MoE) | 26B / A4B |
| **Qwen3.5** | Qwen3.5-35B-A3B | Hybrid (Gated DeltaNet + Full GQA + MoE) | 35B / 3B active |
| **Qwen3.6** | Qwen/Qwen3.6-35B-A3B | Hybrid (Gated DeltaNet + Full GQA + MoE) | 35B / 3B active |
| | Qwen/Qwen3.6-35B-A3B-FP8 | Hybrid (Gated DeltaNet + Full GQA + MoE), FP8 weights | 35B / 3B active |
| | Qwen/Qwen3.6-27B-FP8 | Hybrid (Gated DeltaNet + Full GQA + Dense SwiGLU), FP8 weights | 27B |

## 技术栈

- **前端**：React + Vite + TypeScript
- **桌面壳**：Tauri 2
- **计算方式**：纯客户端公式计算
- **无后端 / 无在线推理**

## 功能

- **性能计算**：输入模型、平台参数（算力/带宽/显存）、token 范围，输出 Prefill TPS、Decode TPS、TTFT、显存预算
- **模型推荐精度**：每个模型注册自己的权重、激活与专家字节数；切换模型时自动刷新，之后仍可手动覆盖
- **历史记录**：成功计算后保存日期、模型、参数快照与结果摘要，支持按模型筛选、时间排序和清空记录
- **Token 趋势**：序列长度变化下 TPS / 延迟 / 显存的趋势曲线
- **瓶颈分析**：Compute-bound vs Bandwidth-bound 判定
- **公式追溯**：每项结果均可追溯到输入参数 → 中间量 → 公式
- **模型结构**：模型超参速查、层 schedule、结构流图
- **HTML 报告**：按模型实际架构和算子分块导出理论计算推导报告
- **JSON 导出**：导出完整模型、平台、负载、结果、公式和趋势快照
- **桌面应用**：Tauri 打包为跨平台桌面 App

## 项目结构

```
llm-perf-calculator/
├── src/
│   ├── app/                # 入口、路由、布局、全局样式
│   ├── pages/              # 页面装配（性能计算 / 模型结构 / 公式说明）
│   ├── features/
│   │   └── performance-calculator/
│   │       ├── components/ # 计算器 UI 组件
│   │       ├── services/   # 性能计算引擎（核心公式）
│   │       └── state/      # 全局 CalculatorProvider
│   ├── components/         # 跨页面复用 UI 组件
│   ├── domain/             # 稳定业务类型定义
│   │   ├── model/          # ModelDefinition, FormulaStrategyId
│   │   ├── performance/    # PerformanceResult, BottleneckType
│   │   ├── platform/       # PlatformInput
│   │   └── workload/       # WorkloadInput
│   └── engines/
│       ├── model-registry/ # 模型注册表（按 family 组织）
│       └── formula-strategies/ # 公式策略（预留扩展点）
├── data/
│   ├── models/             # 模型结构化数据
│   └── platform-presets/   # 平台参数预设
├── docs/
│   ├── app/design/         # 设计文档、架构文档、效果图
│   ├── deepseek_v4/        # DeepSeek V4 架构分析
│   ├── gemma_4/            # Gemma 4 架构分析
│   └── Qwen_3.5/           # Qwen3.5 架构分析 + config
│   └── Qwen_3.6/           # Qwen3.6 架构分析 + BF16/FP8 config
├── src-tauri/              # Tauri 桌面宿主
└── scripts/                # 数据整理、校验脚本
```

### 分层约束

```
pages/  ← 页面装配，不承载公式逻辑
  ↓
features/  ← 业务模块，组合 domain + engines
  ↓
engines/   ← 纯计算与注册（不依赖 pages/features）
  ↓
domain/    ← 稳定类型定义
```

## 环境准备

完整开发环境包含两部分：

- Node.js / npm：用于安装前端依赖、运行 Vite、调用 Tauri CLI。
- Rust / Cargo + 系统依赖：用于编译和打包 Tauri 桌面应用。

只做 Web 开发时，安装 Node.js / npm 后执行 `npm install` 即可。要打包桌面应用，还需要安装 Rust / Cargo 和对应系统依赖。

### Windows 原生环境

Windows 上编译会产出 Windows 可运行的桌面应用。请在 PowerShell / CMD 中执行，不要在 WSL 路径下编译 Windows 包。

```powershell
# 1. 安装 Node.js LTS，内置 npm
winget install --id OpenJS.NodeJS.LTS

# 2. 安装 Rustup / Cargo
winget install --id Rustlang.Rustup

# 3. 安装 Visual Studio Build Tools
winget install --id Microsoft.VisualStudio.2022.BuildTools

# 4. 如系统缺少 WebView2 Runtime，再安装
winget install --id Microsoft.EdgeWebView2Runtime
```

安装 Visual Studio Build Tools 时，需要选择 `Desktop development with C++` 工作负载。安装完成后重新打开 PowerShell：

```powershell
# 设置 Rust MSVC 工具链
rustup default stable-msvc

# 验证基础环境
node --version
npm --version
rustc --version
cargo --version

# 安装项目依赖
npm install

# 验证 Web 构建
npm run build

# 开发运行桌面应用
npm run desktop:dev

# 打包 Windows 桌面应用
npm run desktop:build
```

Windows 产物位于：

```text
src-tauri\target\release\
src-tauri\target\release\bundle\
```

### Linux / WSL 环境

WSL 中编译会产出 Linux 应用包，不是 Windows `.exe`。Ubuntu / Debian 环境可执行：

```bash
# 1. 安装 Node.js / npm 和 Tauri Linux 系统依赖
sudo apt update

sudo apt install -y \
  nodejs \
  npm \
  libwebkit2gtk-4.1-dev \
  build-essential \
  curl \
  wget \
  file \
  libxdo-dev \
  libssl-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev

# 2. 安装 Rust / Cargo
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"

# 3. 验证基础环境
node --version
npm --version
rustc --version
cargo --version

# 4. 安装项目依赖
npm install

# 5. 验证 Web 构建
npm run build

# 6. WSL / Linux 下打包 deb 和 rpm
npm run desktop:build:linux
```

WSL 下建议使用 `npm run desktop:build:linux` 生成 `deb` / `rpm`，避免 AppImage 在部分 WSL 文件系统环境中失败。Linux 产物位于：

```text
src-tauri/target/release/bundle/deb/
src-tauri/target/release/bundle/rpm/
```

## 快速开始

```bash
# 安装依赖
npm install

# 开发模式
npm run dev

# 生产构建
npm run build

# 预览构建产物
npm run preview

# Tauri 桌面应用（开发）
npm run desktop:dev

# Tauri 桌面应用（打包）
npm run desktop:build

# WSL / Linux 下只打包 deb 和 rpm
npm run desktop:build:linux
```

打开浏览器访问 `http://localhost:5173`，默认进入**性能计算**页面。

桌面应用构建依赖本机 Rust / Cargo 和 Tauri 系统依赖。Web 构建不需要 Rust。`desktop:build` 会按当前操作系统生成对应桌面包；WSL / Linux 下如需避开 AppImage，可使用 `desktop:build:linux` 只生成 `deb` / `rpm`。

## 使用方式

完整操作说明参见 [LLM Perf Calculator 使用说明](docs/app/user-guide.md)。

### 1. 性能计算

1. 在左侧导航选择 **性能计算**
2. 选择模型家族 → 具体模型
3. 设置平台参数：
   - **算力吞吐**（TFLOPS）和 **算力利用率**
   - **显存带宽**（GB/s）和 **带宽利用率**
   - **显存容量**（GB）
   - **精度配置**：Bytes/Weight、Bytes/Activation、Bytes/Expert
4. 设置 Token 范围（Prefill 长度、Decode 上下文、输出长度、趋势范围）
5. 点击 **计算性能**
6. 需要归档时点击 **生成 HTML 报告** 或 **导出 JSON**

### 2. 模型结构

查看选中模型的层级结构、关键超参、层 schedule。

### 3. 公式说明

公式参考页，记录各架构的 FLOPs / Memory 公式口径。

## 计算口径

### Prefill FLOPs

根据不同架构拆解每层算力：

| 架构 | 核心公式 |
|------|----------|
| Compressed MoE (DeepSeek V4) | MLA-Q + Shared-KV + CSA/HCA Compressor + Indexer (O(S²)) + MoE |
| Dense Decoder (Gemma 4) | GQA/MQA + Sliding/Full attention + GeGLU MLP |
| Hybrid Linear MoE (Qwen3.5) | Gated DeltaNet (O(S·D)) + Full GQA causal (O(S²·n_h·c)) + MoE |

**Full attention causal 关键**：因果 mask 下有效 QK 对 ≈ S²/2，FLOPs 约减半。

### Decode Memory

```
M_total = M_weights + M_kv_cache + M_tmp_peak + M_overhead
```

- **M_weights**：由 `totalParamsB`、`bytesPerWeight`、`bytesPerExpert` 动态计算
- **M_kv_cache**：仅全注意层维护 KV cache；线性/压缩注意层只维护固定大小的状态
- **M_tmp_peak**：单步 decode 的 `repeat_kv` 瞬时峰值

### Decode TPS

```
TPS = min(compute_ceiling, bandwidth_ceiling)
compute_ceiling = effective_flops / flops_per_token
bandwidth_ceiling = effective_bandwidth / bytes_per_token
```

## 新增模型

参见 [AGENTS.md](./AGENTS.md) 中的两阶段流程：

1. **架构分析** — 分析 `config.json` + `transformers` 源码 → 生成 `docs/$family/$model.md`
2. **工程适配** — 补 domain 类型 → 补模型注册表 → 补公式策略 → 页面适配

关键原则：
- 不同架构通过 `formulaStrategyId` 区分，**不要**把 DeepSeek V4 公式套到其他模型
- 模型从 `engines/model-registry/` 统一注册，不在页面中手写模型列表

## 设计文档

- [主设计文档](docs/app/design/app-design-spec.md)
- [架构设计文档](docs/app/design/architecture-design.md)
- [性能计算页设计](docs/app/design/pages/performance-calculator.md)
- [用户使用说明](docs/app/user-guide.md)
- [模型结构页设计](docs/app/design/pages/model-structure.md)
- [公式说明页设计](docs/app/design/pages/formula-notes.md)

## 模型架构分析

- [DeepSeek-V4-Flash](docs/deepseek_v4/DeepSeek-V4-Flash.md)
- [Gemma-4-12B-it](docs/gemma_4/Gemma-4-12B-it.md)
- [google/gemma-4-26B-A4B-it](docs/gemma_4/gemma-4-26B-A4B-it.md)
- [Qwen3.5-35B-A3B](docs/Qwen_3.5/Qwen3.5-35B-A3B.md)
- [Qwen3.6-35B-A3B / FP8](docs/Qwen_3.6/Qwen3.6-35B-A3B.md)

## License

MIT
