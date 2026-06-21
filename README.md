# OMP Cache Optimizer

[![npm version](https://img.shields.io/npm/v/omp-cache-optimizer.svg)](https://www.npmjs.com/package/omp-cache-optimizer)
[![npm downloads](https://img.shields.io/npm/dm/omp-cache-optimizer.svg)](https://www.npmjs.com/package/omp-cache-optimizer)
[![license](https://img.shields.io/npm/l/omp-cache-optimizer.svg)](./LICENSE)

> 本项目是基于 [pi-cache-optimizer](https://github.com/jiangge/pi-cache-optimizer) 的二次开发（fork），适配 OMP（Oh My Pi）运行时。原项目由 [freescheme](https://github.com/jiangge) 开发，在此致谢。

用于提升 OMP 中 provider 侧 KV Cache / Prompt Cache 命中率的扩展：把稳定 prompt 内容前置，给 OpenAI-compatible 请求补保守的 `prompt_cache_key`，提示代理渠道常见缓存路由兼容问题，并在底部显示只读缓存统计。

> 本包从 `pi-cache-optimizer` fork 而来。已有底部统计会自动从旧状态目录 `~/.pi/agent/` 迁移到 `~/.omp/agent/`。正常运行时扩展不会触碰你的 `~/.omp/agent/models.yml`；`/cache-optimizer fix` 当前显示可复制的 YAML compat 片段供手动编辑（自动写入的外科 YAML 编辑器计划在后续版本实现）。

## 与原项目的关键差异

本 fork 针对 OMP 运行时做了以下适配：

- **状态目录**：`~/.pi/agent/` → `~/.omp/agent/`
- **模型配置**：`models.json` (JSONC) → `models.yml` (YAML)
- **包作用域**：`@earendil-works/pi-coding-agent` → `@oh-my-pi/pi-coding-agent`
- **扩展清单**：`package.json` 的 `pi.extensions` → `omp.extensions`（`pi.extensions` 仍兼容）
- **prompt 重写位置**：从 `before_agent_start`（直接改 systemPrompt）迁移到 `before_provider_request`（在 provider payload 层面修改），因为 OMP 的 `before_agent_start` 只支持注入消息，不支持修改 system prompt
- **compat 字段重映射**：
  - `forceAdaptiveThinking` → 移除（OMP 内置 catalog 自动设置）
  - `sendSessionAffinityHeaders` / `sendSessionIdHeader` → 移除（OMP 用多凭据 auth + `agent.db` 实现会话亲和性）
  - `supportsLongCacheRetention` → `supportsLongPromptCacheRetention`
  - `requiresReasoningContentOnAssistantMessages` → `requiresReasoningContentForToolCalls`
  - `thinkingFormat: "deepseek"` → 不再标记（OMP 自动检测 DeepSeek reasoning 格式）
- **事件适配**：`model_select` → `turn_start` + 模型变更检测（OMP 可能不支持 `model_select` 事件）
- **协议符号**：`Symbol.for("pi.routing.registry.v1")` → `Symbol.for("omp.routing.registry.v1")`，`Symbol.for("pi.cache.hints.v1")` → `Symbol.for("omp.cache.hints.v1")`
- **`/cache-optimizer fix` 当前状态**：降级为手动建议模式（显示可复制的 YAML compat 片段），自动写入的 YAML 外科编辑器待后续 PR 实现

## 目录

- [功能](#功能)
- [安装](#安装)
- [命令](#命令)
- [持久 Opt-out](#持久-opt-out)
- [OpenAI-compatible 代理配置](#openai-compatible-代理配置)
- [Anthropic adaptive thinking 模型](#anthropic-adaptive-thinking-模型)
- [使用 `/cache-optimizer fix` 手动修复](#使用-cache-optimizer-fix-手动修复)
- [Footer 统计](#footer-统计)
- [Router / Virtual-channel 扩展作者指南](#router--virtual-channel-扩展作者指南)
- [卸载](#卸载)
- [验证效果](#验证效果)
- [License](#license)

## 功能

- 将稳定的 system prompt 内容移动到动态上下文之前（在 `before_provider_request` 中对 provider payload 应用）。
- 压缩 OMP skill 列表，并移除 session-overview 中的易变字段。
- 通过 `PI_CACHE_RETENTION=long` 请求长缓存保留（OMP 沿用同一环境变量）。
- 对 `openai-completions` / `openai-responses` 请求，在没有有效 key 时使用 OMP session id 补 `prompt_cache_key`。
- 对缺少长缓存保留 compat 的第三方 OpenAI-compatible 代理给出一次性提醒。
- 检测 Anthropic adaptive thinking 模型（opus-4.6+、sonnet-4.6+、fable-5+）—— OMP 内置 catalog 已自动处理，此处仅作信息性提示。
- 为支持的模型家族显示按 session 隔离的底部缓存统计。
- 通过版本化全局协议（`Symbol.for("omp.routing.registry.v1")` 与 `Symbol.for("omp.cache.hints.v1")`）支持可选的 router extension 集成，而不导入任何 router 包。

缓存是 provider 侧的 best-effort 行为。第三方代理和 router extension 仍可能隐藏缓存 usage、拒绝不支持的参数，或把请求路由到多个上游。

## 安装

```bash
omp install npm:omp-cache-optimizer
```

如果之前安装过原版本：

```bash
omp remove npm:pi-cache-optimizer && omp install npm:omp-cache-optimizer
```

安装、更新或移除后，在 OMP 中运行 `/reload`，让 extension hooks 刷新。

OMP 0.79.7 及之后，`omp update` 默认只更新 OMP 本体。若要更新已安装的 OMP package（包括本扩展），请运行 `omp update --extensions`（只更新 packages）或 `omp update --all`（OMP 与 packages 一起更新）。

## 命令

| 命令 | 作用 |
|---|---|
| `/cache-optimizer` | UI 支持时打开交互菜单；否则打印帮助和当前状态。 |
| `/cache-optimizer enable` | 在当前 OMP 进程中开启运行时优化，清零当前 session 统计，并开始新的"开启状态"测量。 |
| `/cache-optimizer disable` | 在当前 OMP 进程中关闭优化，清零当前 session 统计，并继续以 disabled 对比模式采集 footer 统计。运行 `/reload` 或重启 OMP 后回到启动时行为。 |
| `/cache-optimizer doctor` | 显示当前模型 / provider / API / base URL / compat 与低命中诊断。 |
| `/cache-optimizer compat` | 对当前模型显示可复制的 compat 建议（如适用）。 |
| `/cache-optimizer stats` | 显示当前模型今天的 session-scoped 统计和近期趋势。 |
| `/cache-optimizer reset` | 只重置当前 session + 当前模型的本地统计；不会修改上游 provider 缓存。 |
| `/cache-optimizer fix` | **当前为手动建议模式**：显示可复制的 YAML compat 片段 + 手动编辑步骤。自动写入的 YAML 外科编辑器待后续实现。 |

`enable` / `disable` 是当前进程内开关。若要持久关闭某些能力，请使用下面的环境变量。

## 持久 Opt-out

| 环境变量 | 作用 |
|---|---|
| `PI_CACHE_OPTIMIZER_NO_PROMPT_REWRITE=1` | 只关闭 prompt 改写；footer 统计和 cache-key fallback 仍启用。 |
| `PI_CACHE_OPTIMIZER_NO_SKILL_COMPRESSION=1` | 保留 OMP 原始 verbose skill XML。 |
| `PI_CACHE_OPTIMIZER_NO_OPENAI_CACHE_KEY=1` | 关闭 OpenAI-compatible `prompt_cache_key` fallback。推荐使用这个显式 opt-out。 |

> OMP 会自动将 `OMP_CACHE_OPTIMIZER_*` 镜像为 `PI_CACHE_OPTIMIZER_*`，因此两个前缀都可使用。

## OpenAI-compatible 代理配置

LiteLLM / OneAPI / NewAPI / 类 OpenRouter 渠道等第三方 `openai-completions` 代理，常会把同一个 session 分散到多个上游后端，导致 provider 侧 prompt cache 被拆散。

**OMP 差异**：OMP 不再使用 `sendSessionAffinityHeaders` compat 字段（原项目中的旧字段），而是通过多凭据 auth + `agent.db` 中的会话亲和性实现上游粘性。长缓存保留改用 `supportsLongPromptCacheRetention` 字段。

`models.yml` 示例：

```yaml
providers:
  your-provider-id:
    api: openai-completions
    baseUrl: https://example.com/v1
    apiKey: YOUR_API_KEY
    compat:
      supportsLongPromptCacheRetention: true
    models:
      - id: gpt-5.5
        name: GPT-5.5
```

说明：

- `supportsLongPromptCacheRetention: true` 是可选项。只有 endpoint 明确支持 OpenAI long prompt cache retention 时才添加。
- 如果出现 `400 Unsupported parameter: prompt_cache_retention`，请为该渠道移除 / 避免 `supportsLongPromptCacheRetention`。
- 使用 `/cache-optimizer compat` 或 `/cache-optimizer doctor` 查看当前模型的具体建议。
- 对 DeepSeek 模型，OMP 自动检测 reasoning 格式，无需手动设置 `thinkingFormat`。
- 本扩展的 `doctor` 和 `compat` 命令只给建议，不会修改 `models.yml`。

## Anthropic adaptive thinking 模型

**OMP 差异**：OMP 的内置 model catalog 已为官方 Claude 模型自动设置 adaptive thinking（通过 `disableAdaptiveThinking` 字段，语义与原项目中的 `forceAdaptiveThinking` 相反），且不可从 `models.yml` 用户配置。因此本扩展对 adaptive thinking 的检测改为信息性提示，不再提供自动修复。

`/cache-optimizer doctor` 和 `/cache-optimizer compat` 会检测 adaptive thinking 模型并显示信息性说明。自定义渠道 fronting Anthropic 时，请确保模型 id 匹配官方发布版本，以便 OMP catalog 正确识别。

## 使用 `/cache-optimizer fix` 手动修复

**OMP 差异**：当前 `/cache-optimizer fix` 降级为手动建议模式。原项目中的自动写入安全协议（backup → 预览 + 确认 → 原子 temp+rename → 写入后自检 → 失败回滚）将在后续 PR 中为 YAML 重新实现。

当前行为：

1. 检测当前 active model 的 compat 缺失项
2. 显示可复制的 YAML compat 片段
3. 显示手动编辑步骤（文件路径、provider/model 位置、需添加的键值对）
4. 提示编辑后运行 `/reload`

可修复的 compat 问题：

- DeepSeek reasoning compat（`requiresReasoningContentForToolCalls: true` + `supportsLongPromptCacheRetention: true`）
- OpenAI-compatible proxy 长缓存保留（`supportsLongPromptCacheRetention: true`）

**范围：** 仅当前 active model。其他渠道需切换模型后再次运行 `fix`。

**运行：** 当 active model 检测到 compat 问题时执行 `/cache-optimizer fix`。compat 已完整时，命令显示"无需修复"。

### 没有 `models.yml` provider entry 的渠道

有些 OMP 渠道可用时，`~/.omp/agent/models.yml` 里可能还没有对应 provider block。保留现有认证方式，不要复制 credential、token 或 API key。只在 `models.yml` 里添加缓存兼容覆盖。

Provider 级最小 override：

```yaml
providers:
  your-provider-id:
    compat:
      supportsLongPromptCacheRetention: true
```

如果只想影响单个模型，用 `modelOverrides`：

```yaml
providers:
  your-provider-id:
    modelOverrides:
      gpt-5.5:
        compat:
          supportsLongPromptCacheRetention: true
```

## Footer 统计

统计是只读本地计数，保存在 `~/.omp/agent/omp-cache-optimizer-stats.json`，按 OMP session + provider/model 隔离。文件只包含日期和数字计数，不包含 API key、prompt、payload、headers、响应或模型输出。

OMP 0.79+ 已内置 footer `CH` 标记，用于显示最近一次 prompt cache hit rate。本扩展在此基础上补充持久化的 provider/model/session-scoped 计数，以及代理 compat 诊断。

示例 footer：

```text
OpenAI cache 3/10 · 0.002M/0.005M tok (40%) ⚠️ compat
```

格式：`<label> <命中请求数>/<总请求数> · <cached input tokens>/<total input tokens> tok (<token 命中率>)`。部分 adapter 还可能追加 `· write <tokens> tok`，运行时诊断可能追加 `⚠️ compat` 或 `⚠️ integrity`。

支持的 footer label 包括：DS、Claude、OpenAI、Gemini、Kimi、Qwen、GLM、MiniMax、Mimo、Hunyuan、Mistral、Grok、Llama、Nemotron、Cohere、Yi、Doubao、ERNIE、Baichuan、StepFun、Spark、InternLM、Gemma、Phi、Jamba、Solar、Sonar、Nova、Reka、Falcon、DBRX、MPT、StableLM、Aquila、EXAONE、HyperCLOVA、Luminous、Hermes、Granite、Arctic、Pangu、SenseNova、Zhinao、MiniCPM、XVERSE、Orion、OpenChat、Vicuna、Wizard、Zephyr、Dolphin、OpenOrca、Starling、BLOOM、RWKV、Aya。

Adapter 选择只看模型 id/name（以及 message_end 时 assistant message 的 model/name）。仅使用 OpenAI-shaped API 不会被当作 OpenAI-family，除非模型 id/name 匹配受支持的家族。

## Router / Virtual-channel 扩展作者指南

如果你的 OMP 扩展提供虚拟 routing provider（例如 `router/auto`、`router/smart`，或会转发到真实上游的 profile/channel），本扩展可以为真实上游 provider/model 显示缓存统计，而不是把统计记到虚拟外壳上。集成是可选、版本化的，并且**不需要导入本包**。

### 最小集成：最终 assistant message metadata

要无缝获得最终缓存统计归因，请在完成的 assistant message 上透传真实上游身份：

```ts
{
  role: "assistant",
  provider: "anthropic",              // 真实上游 provider
  responseModel: "claude-opus-4-8",   // 或 model: "..."
  api: "anthropic-messages",          // 已知时填写上游 OMP API id
  usage: {
    input: 1200,       // OMP-normalized 未缓存 input tokens，如可用
    cacheRead: 8000,   // 从 provider prompt cache 读取的 tokens
    cacheWrite: 500,   // 本次新写入 provider prompt cache 的 tokens
  },
}
```

`message_end` 会把这些 assistant-message 字段视为权威来源。只要存在 `provider` + `model`/`responseModel` + cache usage，即使 active model 仍是 `router/auto`，统计也会更新真实上游桶。如果上游 usage 没有 cache 字段，请保持缺失或为 0；本扩展不会伪造 cache hit。

### 可选：用于预响应 UX 的实时路由注册表

最终 message metadata 足以支持响应后的统计。若要支持响应前流程——首次响应前的 footer 显示、`/cache-optimizer doctor`、`/cache-optimizer compat`、`/cache-optimizer reset` 和 OpenAI-compatible `prompt_cache_key` fallback——请在 `Symbol.for("omp.routing.registry.v1")` 下注册 live route adapter。

协议形状：

```ts
type OmpRouteSnapshot = {
  virtualProvider: string;
  virtualModelId: string;
  provider: string;
  modelId: string;
  api?: string;
  canonicalModelId?: string;
  routeLabel?: string;
  status?: "planned" | "trying" | "selected" | "success" | "failed";
  sessionIdHash?: string;
  requestId?: string;
  timestamp: number;
};

type OmpRouterAdapterV1 = {
  virtualProvider: string;
  resolveActiveRoute(
    virtualModelId: string,
    hint?: { sessionIdHash?: string; requestId?: string },
  ): OmpRouteSnapshot | undefined;
  resolveCandidateRoutes?(virtualModelId: string): OmpRouteSnapshot[];
  subscribe?(listener: (event: OmpRouteSnapshot) => void): () => void;
};
```

注册模式：

```ts
const ROUTING = Symbol.for("omp.routing.registry.v1");
const registry = (globalThis as Record<symbol, unknown>)[ROUTING] as
  | { version: 1; registerRouter(adapter: OmpRouterAdapterV1): () => void }
  | undefined;

registry?.registerRouter({
  virtualProvider: "router",
  resolveActiveRoute(virtualModelId, hint) {
    return {
      virtualProvider: "router",
      virtualModelId,
      provider: "anthropic",
      modelId: "claude-opus-4-8",
      api: "anthropic-messages",
      timestamp: Date.now(),
    };
  },
});
```

cache hints 协议（`Symbol.for("omp.cache.hints.v1")`）形状与原项目一致，用于预响应阶段透传优化后的 system prompt / prompt cache key / cache retention hint。

## 卸载

```bash
omp remove npm:omp-cache-optimizer
```

卸载后运行 `/reload`。本地统计文件 `~/.omp/agent/omp-cache-optimizer-stats.json` 不会自动删除，可手动清理。

## 验证效果

1. 安装后运行 `/cache-optimizer doctor`，确认当前模型 / provider / API / compat 状态
2. 正常使用 OMP 几轮对话后，运行 `/cache-optimizer stats` 查看 session-scoped 命中率
3. 底部 footer 会显示实时 cache 统计（如 `OpenAI cache 3/10 · 0.002M/0.005M tok (40%)`）
4. 如命中率低，`/cache-optimizer doctor` 会给出低命中诊断和 compat 建议

## 致谢

本项目基于 [pi-cache-optimizer](https://github.com/jiangge/pi-cache-optimizer) 二次开发，感谢原作者 [freescheme](https://github.com/jiangge) 的工作。

## License

[MIT](./LICENSE)
