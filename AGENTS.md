# Repository Guidelines

## Project Overview

OMP Cache Optimizer (npm: `omp-cache-optimizer`) is a single-file TypeScript OMP (Oh My Pi) extension that improves provider-side prompt/KV cache hit rates. It is a fork of `pi-cache-optimizer` adapted for OMP's runtime: YAML model config (`models.yml`), `@oh-my-pi/pi-coding-agent` package scope, `before_provider_request`-based prompt rewriting, and remapped compat fields.

## Architecture & Data Flow

```
OMP Runtime (host, Bun-based)
  │
  ├─ session_start        → restore persisted stats, sync session hash
  ├─ turn_start           → model-change detection, publish footer status
  │                         (replaces the original project’s `model_select` event)
  ├─ before_agent_start   → cache systemPromptOptions + route snapshot
  │                         (cannot mutate systemPrompt in OMP — returns {})
  ├─ before_provider_request → 3-step prompt rewrite on payload + inject
  │                              prompt_cache_key (OpenAI compat)
  ├─ after_provider_response → detect 400 compat signals
  └─ message_end          → scrape OMP-normalized usage, persist stats, publish footer
```

- **Single-file monolith**: All logic lives in `index.ts` (~6,500 lines). No `src/` tree.
- **Extension entry**: `export default function (pi: ExtensionAPI) { … }` — registers hooks + `/cache-optimizer` command.
- **Prompt rewriting in `before_provider_request`**: OMP-specific adaptation. The 3-step pipeline (churn strip → skill compression → stable-prefix reorder) runs on the provider payload's system prompt field, extracted via `extractSystemPrompt()` / `setSystemPrompt()` which handle `payload.system` (Anthropic), `payload.systemInstruction` (Google), and `payload.messages[0].content` (OpenAI) shapes.
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
| `bun .trellis/tasks/archive/<date>/<task>/verify.ts` | Run task verification scripts (Bun, not node/tsx) |

**No build step** — the extension is consumed as raw TypeScript by the OMP runtime. **No test framework** — verification is done via hand-written `verify.ts` scripts that import `__internals_for_tests` from `index.ts`, run assertions with `expect(label, condition, message)`, and exit 0/1.

## Code Conventions & Common Patterns

### Formatting & Naming

- **camelCase** for functions, variables; **UPPER_SNAKE_CASE** for constants; **PascalCase** for types
- Single-file source; new files only with strong justification
- Inline types preferred; extensive JSDoc on non-trivial functions

### TypeScript Configuration

- `target: ES2022`, `module: NodeNext`, `strict: false`, `noEmit: true`
- Runtime validation via type guards (`isPiRouterAdapterV1()`, `asRecord()`)
- Env var names keep `PI_CACHE_OPTIMIZER_*` prefix — OMP mirrors `OMP_*` → `PI_*` automatically

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

### Prompt Optimization Pipeline (in `before_provider_request`)

1. `stripSessionOverviewChurn()` — remove per-turn timestamps/task status from `<session-overview>`
2. `compressSkillsInSystemPrompt()` — replace verbose `<available_skills>` XML with compact index (min 4 skills)
3. `optimizeSystemPrompt()` — lift stable candidates (guidelines, tool snippets, context files) above dynamic content

Integrity guard: WORM-flag `promptTruncationDetected` detects if structural markers were lost during rewrite.

**Bypass**: All prompt mutations skipped for `openai-codex-responses` / `openai-responses` / `azure-openai-responses` APIs (server-managed caching + stricter content-safety filtering).

### `/cache-optimizer fix` — Current Status

**OMP divergence**: The auto-write YAML surgical editor is not yet implemented. `/cache-optimizer fix` currently shows copyable YAML compat snippets + manual editing instructions. The legacy JSONC surgical editor code (`locateModelInJsonc` / `composeFixInsertion` / `selfCheckFix`) is preserved as dead code for a future YAML editor PR.

The original project's auto-write safety protocol (backup → preview + confirmation → atomic temp+rename → post-write self-check → restore-from-backup on failure) will be reimplemented for YAML in a follow-up.

### `PI_CACHE_RETENTION` Mechanism

The extension sets `PI_CACHE_RETENTION=long` at load time (OMP honors this env var — see omp environment-variables.md §2). `/cache-optimizer disable` restores the startup value for the current OMP process.

### Environment Variable Gating

| Variable | Effect |
|---|---|
| `PI_CACHE_OPTIMIZER_NO_PROMPT_REWRITE=1` | Disable prompt mutations only |
| `PI_CACHE_OPTIMIZER_NO_SKILL_COMPRESSION=1` | Keep verbose skill XML |
| `PI_CACHE_OPTIMIZER_NO_OPENAI_CACHE_KEY=1` | Disable prompt_cache_key fallback |

OMP mirrors `OMP_CACHE_OPTIMIZER_*` → `PI_CACHE_OPTIMIZER_*` automatically, so either prefix works.

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

- **No test framework**: Tests are hand-written `verify.ts` scripts under `.trellis/tasks/archive/<date>/<task>/`
- **Run with Bun**: `bun .trellis/tasks/archive/<date>/<task>/verify.ts` (scripts use `#!/usr/bin/env bun` shebang)
- **Verification pattern**: Each script imports `__internals_for_tests` from `index.ts` (~150 exported helpers covering adapters, stats, JSONC/YAML editors, routing protocol, model detection, persistence, fix helpers), runs assertions with `expect(label, condition, message)`, and exits 0/1
- **Required checks before delivery**:
  1. `bunx tsc --noEmit` — no type errors
  2. `npm pack --dry-run` — package includes correct files
  3. Task-specific `verify.ts` — all assertions pass
- **Forbidden**: logging secrets, writing `models.yml` outside `/cache-optimizer fix` flow, adapter selection by provider/api alone, non-actionable startup warnings, `any` casts without runtime guards, throwing from hook paths
