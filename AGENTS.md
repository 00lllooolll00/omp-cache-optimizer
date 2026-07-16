# Repository Guidelines

## Project Overview

OMP Cache Optimizer (npm: `omp-cache-optimizer`) is a single-file TypeScript OMP (Oh My Pi) extension that improves provider-side prompt/KV cache hit rates. It is a fork of `pi-cache-optimizer` adapted for OMP's runtime: YAML model config (`models.yml`), `@oh-my-pi/pi-coding-agent` package scope, `before_agent_start`-based prompt rewriting (OMP 17 string[] blocks), and remapped compat fields.

## Architecture & Data Flow

```
OMP Runtime (host, Bun-based)
  │
  ├─ session_start        → restore persisted stats, sync session hash
  ├─ turn_start           → model-change detection, publish footer status
  ├─ before_agent_start   → 主 prompt 重写（string[] 块：仅 <session-overview> churn strip，保持块顺序与内容）
  │                         + route snapshot + cache hint
  ├─ before_provider_request → 安全网：session-overview 兜底 strip + prompt_cache_retention 安全网 + prompt_cache_key 兜底注入（OpenAI-compatible）
  ├─ after_provider_response → detect 400 compat signals
  └─ message_end          → scrape OMP-normalized usage, persist stats, publish footer
```

- **Single-file monolith**: All logic lives in `index.ts` (~5,100 lines). No `src/` tree.
- **Extension entry**: `export default function (pi: ExtensionAPI) { … }` — registers hooks + `/cache-optimizer` command.
- **Prompt rewriting in `before_agent_start`**：OMP 17 适配。仅对 `event.systemPrompt: string[]` 块数组逐块调用 `stripSessionOverviewChurn()`，保持块顺序与 skill 描述逐字不变，不压缩、不重排。`before_provider_request` 做 payload 级 session-overview 兜底 strip、`prompt_cache_retention` 安全网，并为 OpenAI-compatible API（host 不注入的 openai-completions chat wire）兜底注入 body `prompt_cache_key`（读 header `providerPromptCacheKey` ?? sessionId，截断 64 字符，payload 已有则不覆盖）。`extractSystemPrompt()` / `setSystemPrompt()` 处理 `payload.system`（Anthropic）、`payload.systemInstruction`（Google）、`payload.messages[0].content`（OpenAI）形态。
- **Adapter pattern**: `CACHE_PROVIDER_ADAPTERS` array of ~50 adapter objects. Selected by token-matching on model id/name.
- **Stats persistence**: Session-scoped, versioned JSON at `~/.omp/agent/omp-cache-optimizer-stats.json` (v5 format). Atomic writes via temp + rename. Never persists prompts, payloads, or API keys.
- **Inter-extension protocol**: Two `Symbol.for` global registries — `omp.routing.registry.v1` (live routing) and `omp.cache.hints.v1` (pre-request hints).

## Key Directories

| Path | Purpose |
|---|---|
| `index.ts` | Entire extension source — types, adapters, stats, prompt optimization, command handling |
| `types/pi-coding-agent.d.ts` | Ambient type declarations for `@oh-my-pi/pi-coding-agent` host API |
| `types/node-shims.d.ts` | Minimal Node.js built-in shims (Bun-compatible) |
| `.trellis/` | Trellis workflow system — phases, specs, task management, verification scripts |
| `docs/agents/` | Agent onboarding docs — domain conventions, issue tracker, triage labels |

## Development Commands

| Command | Purpose |
|---|---|
| `bunx tsc --noEmit` | Type-check only (project uses `noEmit: true`) |
| `npm pack --dry-run` | Verify package contents before publish |
| `bun smoke-test.ts` | Run the smoke test (Bun) — exercises `__internals_for_tests` helpers + OMP 17 hook harness |

**No build step** — the extension is consumed as raw TypeScript by the OMP runtime. **No test framework** — verification is done via `smoke-test.ts`, which imports `__internals_for_tests` from `index.ts` and the default export, runs assertions with `expect(label, condition, message)`, and exits 0/1.

## Code Conventions & Common Patterns

### Formatting & Naming

- **camelCase** for functions, variables; **UPPER_SNAKE_CASE** for constants; **PascalCase** for types
- Single-file source; new files only with strong justification
- Inline types preferred; extensive JSDoc on non-trivial functions

### TypeScript Configuration

- `target: ES2022`, `module: NodeNext`, `strict: false`, `noEmit: true`
- Runtime validation via type guards (`isPiRouterAdapterV1()`, `asRecord()`)
- 环境变量主前缀为 `OMP_CACHE_OPTIMIZER_*` / `OMP_CACHE_RETENTION`；读取兼容旧 `PI_*`，长缓存写入时同步镜像 `PI_CACHE_RETENTION` 供宿主

### Compat Field Mapping (Legacy → OMP)

|Legacy field|OMP field|Notes|
|---|---|---|
| `forceAdaptiveThinking` | *(removed)* | OMP catalog sets `disableAdaptiveThinking` automatically; not user-configurable |
| `sendSessionAffinityHeaders` | *(removed)* | OMP uses multi-credential auth + session affinity in `agent.db` |
| `sendSessionIdHeader` | *(removed)* | Same as above |
| `supportsLongCacheRetention` | `supportsLongPromptCacheRetention` | Renamed |
| `requiresReasoningContentOnAssistantMessages` | `requiresReasoningContentForToolCalls` | Renamed |
| `thinkingFormat: "deepseek"` | *(not flagged)* | OMP auto-detects DeepSeek reasoning; `"deepseek"` is not a valid OMP value |

### Adapter Pattern

Each `CacheProviderAdapter` defines: `id`, `label`, `matchesModel(model)`, `matchesAssistantMessage(message, model)`, `normalizeUsage(message)`, `warningText(model)` (single argument — derives key internally via `modelKey(model)`), `showCacheWrite`.

### Prompt Optimization Pipeline (in `before_agent_start`)

1. `stripSessionOverviewChurn()` — 逐块移除 `<session-overview>` 中的易变字段（RECENT COMMITS、Working directory、Line count），保持块顺序与 skill 描述逐字不变。不压缩 skills、不重排块。

`before_provider_request` 安全网：若 `before_agent_start` 未跑或被覆盖，对 payload 中的 system prompt 兜底 strip session-overview churn；并按 400-history 与 compat 对 `prompt_cache_retention` 做安全 strip。另为 OpenAI-compatible API（openai-completions / openai-responses）在 payload 缺失时兜底注入 `prompt_cache_key`（读 header `providerPromptCacheKey` ?? sessionId，截断 64 字符防 400，payload 已有则不覆盖）。

### `/cache-optimizer fix` — Current Status

**OMP divergence**: The auto-write YAML surgical editor is not yet implemented. `/cache-optimizer fix` currently shows copyable YAML compat snippets + manual editing instructions. The legacy JSONC surgical editor code (`locateModelInJsonc` / `composeFixInsertion` / `selfCheckFix`) is preserved as dead code for a future YAML editor PR.

The original project's auto-write safety protocol (backup → preview + confirmation → atomic temp+rename → post-write self-check → restore-from-backup on failure) will be reimplemented for YAML in a follow-up.

### `OMP_CACHE_RETENTION` 机制

扩展加载时设置 `OMP_CACHE_RETENTION=long`，并同步写入 `PI_CACHE_RETENTION=long`（宿主 `@oh-my-pi/pi-ai` 的 `resolveCacheRetention` 仍读 `PI_`）。`/cache-optimizer disable` 恢复当前 OMP 进程启动时的值。

### 环境变量开关

| 变量 | 作用 |
|---|---|
| `OMP_CACHE_OPTIMIZER_NO_PROMPT_REWRITE=1` | 关闭 prompt 改写（session-overview churn strip） |

旧 `PI_CACHE_OPTIMIZER_*` 前缀读取时仍兼容。

## Important Files

| File | Role |
|---|---|
| `index.ts` | Entire extension — entry point, all types, all logic |
| `types/pi-coding-agent.d.ts` | Host API type contract (`ExtensionAPI`, `ExtensionContext`, `ExtensionModel`) — `@oh-my-pi/pi-coding-agent` scope |
| `package.json` | OMP extension metadata; `omp.extensions: ["./index.ts"]` (plus legacy `pi.extensions` fallback) |
| `.trellis/config.yaml` | Trellis workflow configuration |

## Runtime/Tooling Preferences

- **Runtime**: OMP coding agent (Bun-based; host loads extension as TypeScript)
- **TypeScript**: `tsc --noEmit` for type-checking; no bundler, no transpiler
- **Package manager**: npm / Bun (`omp install npm:omp-cache-optimizer`)
- **Node.js APIs**: `node:crypto` (SHA-256), `node:fs/promises` (atomic file I/O), `node:os` (homedir), `node:path` — all Bun-compatible
- **Peer dependency**: `@oh-my-pi/pi-coding-agent` (replaces the original project's `@earendil-works/pi-coding-agent` scope)
- **No external dependencies**: zero npm dependencies beyond the peer package

## Testing & QA

- **No test framework**: Tests live in `smoke-test.ts` (root), no `src/` tree.
- **Run with Bun**: `bun smoke-test.ts` (shebang `#!/usr/bin/env bun`).
- **Verification pattern**: `smoke-test.ts` imports `__internals_for_tests` from `index.ts` and the default export, constructs a minimal `ExtensionAPI` harness that captures real `before_agent_start` / `before_provider_request` handlers, and runs assertions with `expect(label, condition, message)`; exits 0/1.
- **Required checks before delivery**:
  1. `bun smoke-test.ts` — all assertions pass
  2. `bunx tsc --noEmit` — no type errors
  3. `npm pack --dry-run` — package includes correct files
- **Forbidden**: logging secrets, writing `models.yml` outside `/cache-optimizer fix` flow, adapter selection by provider/api alone, non-actionable startup warnings, `any` casts without runtime guards, throwing from hook paths
