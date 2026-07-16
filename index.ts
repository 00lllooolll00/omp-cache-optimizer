import { createHash } from "node:crypto";
import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";

type MutableEnv = Record<string, string | undefined>;

type CacheRetentionEnvSnapshot = {
  /** 启动时是否已设置 OMP_CACHE_RETENTION 或兼容的 PI_CACHE_RETENTION */
  wasSet: boolean;
  value?: string;
  /** 启动时是否存在 OMP_ 键（用于 disable 时精确恢复） */
  ompWasSet: boolean;
  ompValue?: string;
  /** 启动时是否存在 PI_ 键（宿主 pi-ai 仍读此名） */
  piWasSet: boolean;
  piValue?: string;
};

/** 扩展侧主环境变量名（用户应使用 OMP_ 前缀） */
const OMP_CACHE_RETENTION_ENV = "OMP_CACHE_RETENTION";
/** 宿主 @oh-my-pi/pi-ai 仍读取 PI_CACHE_RETENTION；写入时同步镜像 */
const PI_CACHE_RETENTION_ENV = "PI_CACHE_RETENTION";
const LONG_CACHE_RETENTION_VALUE = "long";

function readCacheRetentionValue(env: MutableEnv): string | undefined {
  // 优先 OMP_，兼容旧 PI_
  const omp = env[OMP_CACHE_RETENTION_ENV];
  if (typeof omp === "string" && omp.length > 0) return omp;
  const pi = env[PI_CACHE_RETENTION_ENV];
  if (typeof pi === "string" && pi.length > 0) return pi;
  return undefined;
}

function captureCacheRetentionEnv(env: MutableEnv = process.env): CacheRetentionEnvSnapshot {
  const ompWasSet = Object.prototype.hasOwnProperty.call(env, OMP_CACHE_RETENTION_ENV);
  const piWasSet = Object.prototype.hasOwnProperty.call(env, PI_CACHE_RETENTION_ENV);
  return {
    wasSet: ompWasSet || piWasSet,
    value: readCacheRetentionValue(env),
    ompWasSet,
    ompValue: env[OMP_CACHE_RETENTION_ENV],
    piWasSet,
    piValue: env[PI_CACHE_RETENTION_ENV],
  };
}

function requestLongCacheRetention(env: MutableEnv = process.env): void {
  // 同时写 OMP_（用户可见主名）与 PI_（宿主 resolveCacheRetention 依赖）
  if (env[OMP_CACHE_RETENTION_ENV] !== LONG_CACHE_RETENTION_VALUE) {
    env[OMP_CACHE_RETENTION_ENV] = LONG_CACHE_RETENTION_VALUE;
  }
  if (env[PI_CACHE_RETENTION_ENV] !== LONG_CACHE_RETENTION_VALUE) {
    env[PI_CACHE_RETENTION_ENV] = LONG_CACHE_RETENTION_VALUE;
  }
}

function restoreCacheRetentionEnv(snapshot: CacheRetentionEnvSnapshot, env: MutableEnv = process.env): void {
  if (snapshot.ompWasSet) {
    env[OMP_CACHE_RETENTION_ENV] = snapshot.ompValue;
  } else {
    delete env[OMP_CACHE_RETENTION_ENV];
  }
  if (snapshot.piWasSet) {
    env[PI_CACHE_RETENTION_ENV] = snapshot.piValue;
  } else {
    delete env[PI_CACHE_RETENTION_ENV];
  }
}

const STARTUP_CACHE_RETENTION_ENV = captureCacheRetentionEnv();

/**
 * OMP Cache Optimizer（pi-cache-optimizer 的 oh-my-pi 适配 fork）
 *
 * 功能：
 * 1. 重排 OMP system prompt，使稳定内容位于动态上下文之前。
 * 2. 加载时设置 OMP_CACHE_RETENTION=long（并镜像 PI_CACHE_RETENTION 供宿主读取）。
 * 3. 在信号保守的前提下，对 provider/model 缓存 compat 缺口做一次性提醒。
 * 4. 在 OMP footer 显示按 provider 区分的轻量持久化缓存统计。
 *
 * Provider 侧 prompt/KV 缓存为 best-effort。本扩展提高命中概率，
 * 尤其经代理时无法保证命中。
 */

// ============================================================
// 在 OMP 支持时自动请求长 prompt-cache 保留。
// /cache-optimizer disable 会恢复本 OMP 进程启动时的值。
// ============================================================
requestLongCacheRetention();

type PiModel = NonNullable<ExtensionContext["model"]>;
type UnknownRecord = Record<string, unknown>;
type CacheProviderId = "deepseek" | "openai" | "claude" | "gemini";

const LOG_PREFIX = "omp-cache-optimizer";
const STATUS_KEY = "omp-cache-stats";
const STATE_DIR = join(homedir(), ".omp", "agent");
const STATE_FILE_PATH = join(STATE_DIR, "omp-cache-optimizer-stats.json");
// 旧版源项目状态路径：仅用于单向迁移读取，从不写入。
const LEGACY_PI_STATE_FILE_PATH = join(homedir(), ".pi", "agent", "pi-cache-optimizer-stats.json");
const LEGACY_STATE_FILE_PATH = join(STATE_DIR, "deepseek-cache-optimizer-stats.json");
const CACHE_PROVIDER_IDS: CacheProviderId[] = ["deepseek", "openai", "claude", "gemini"];
// Prompt 改写默认关闭；仅在用户明确 opt-in 时启用。
const PROMPT_REWRITE_ENV = "OMP_CACHE_OPTIMIZER_PROMPT_REWRITE";
// 扩展间协议符号使用 omp.* 命名空间版本化。v1 形状与旧符号一致；
// OMP 上的 router/hints 集成方应注册 omp.routing.registry.v1 / omp.cache.hints.v1。
const PI_ROUTING_REGISTRY_SYMBOL = Symbol.for("omp.routing.registry.v1");
const PI_CACHE_HINTS_SYMBOL = Symbol.for("omp.cache.hints.v1");

let runtimeOptimizerEnabled = true;

const ASSISTANT_MESSAGE_MODEL_TOKEN_KEYS = ["model", "name"];
const OPENAI_REASONING_MODEL_PATTERN = /(^|[/\s:_-])o[1345]($|[-_.:/\s])/;
const XAI_MODEL_PATTERN = /(^|[/\s:_-])xai($|[-_.:/\s])/;
const MIMO_MODEL_PATTERN = /(^|[/\s:_-])mi-?mo($|[-_.:/\s])/i;
const PPLX_MODEL_PATTERN = /(^|[/\s:_-])pplx($|[-_.:/\s])/i;
const NOVA_MODEL_PATTERN = /(^|[/\s:_-])nova($|[-_.:/\s])/i;
const MPT_MODEL_PATTERN = /(^|[/\s:_-])mpt($|[-_.:/\s])/i;
const ALEPH_MODEL_PATTERN = /(^|[/\s:_-])aleph($|[-_.:/\s])/i;

// Safe-boundary patterns for models with short or ambiguous tokens
const ARCTIC_MODEL_PATTERN = /(^|[\/\s:_-])arctic($|[\-_.:\/\s])/i;
const AYA_MODEL_PATTERN = /(^|[\/\s:_-])aya($|[\-_.:\/\s])/i;
const ORION_MODEL_PATTERN = /(^|[\/\s:_-])orion($|[\-_.:\/\s])/i;

type CacheCompat = {
  // OMP compat fields (see omp models.md). Pi-era field names are remapped:
  //   sendSessionAffinityHeaders / sendSessionIdHeader  -> removed (use headers/extraBody)
  //   forceAdaptiveThinking                            -> removed (OMP catalog sets it internally)
  //   supportsLongCacheRetention                       -> supportsLongPromptCacheRetention
  //   requiresReasoningContentOnAssistantMessages      -> requiresReasoningContentForToolCalls
  supportsLongPromptCacheRetention?: boolean;
  thinkingFormat?: string;
  requiresReasoningContentForToolCalls?: boolean;
  cacheControlFormat?: string;
};

type CacheStats = {
  day: string;
  totalRequests: number;
  hitRequests: number;
  cachedInputTokens: number;
  cacheWriteInputTokens: number;
  totalInputTokens: number;
};

type PersistedCacheStatsV2 = {
  version: 2;
  statsByProvider: Partial<Record<CacheProviderId, CacheStats>>;
};

/** Per-model-key scoped state. Used in memory and for v3 persistence. */
type PersistedRoutedModelRef = {
  provider: string;
  id: string;
  name?: string;
};

type PiRouteSnapshot = {
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

type PiRouteResolveHint = {
  sessionIdHash?: string;
  requestId?: string;
};

type PiRouterAdapterV1 = {
  virtualProvider: string;
  resolveActiveRoute(
    virtualModelId: string,
    hint?: PiRouteResolveHint,
  ): PiRouteSnapshot | undefined;
  resolveCandidateRoutes?(virtualModelId: string): PiRouteSnapshot[];
  subscribe?(listener: (event: PiRouteSnapshot) => void): () => void;
};

type PiRoutingRegistryV1 = {
  version: 1;
  registerRouter(adapter: PiRouterAdapterV1): () => void;
  getRouter(virtualProvider: string): PiRouterAdapterV1 | undefined;
};

type PiCacheHintsInput = {
  sessionIdHash?: string;
  virtualProvider?: string;
  virtualModelId?: string;
  upstreamProvider?: string;
  upstreamModelId?: string;
  api?: string;
};

type PiCacheHintsOutput = {
  systemPrompt?: string;
  promptCacheKey?: string;
  cacheRetention?: "long";
};

type PiCacheHintSnapshot = PiCacheHintsInput & PiCacheHintsOutput & {
  timestamp: number;
};

type PiCacheHintsV1 = {
  version: 1;
  getHints(input: PiCacheHintsInput): PiCacheHintsOutput | undefined;
};

type ProtocolGlobal = typeof globalThis & Record<symbol, unknown> & {
  __ompCacheOptimizerRouter?: unknown;
  __ompCacheOptimizerCacheKey__?: unknown;
};

type ModelRegistryLike = {
  find?(provider: string, modelId: string): PiModel | undefined;
  getAvailable?(): PiModel[];
  getAll?(): PiModel[];
};

type ContextWithOptionalModelRegistry = Pick<ExtensionContext, "sessionManager"> & {
  modelRegistry?: ModelRegistryLike;
};

type CacheStatsState = {
  statsByModel: Record<string, CacheStats>;
  legacyFamily: Partial<Record<CacheProviderId, CacheStats>>;
  lastRoutedModelBySession?: Record<string, PersistedRoutedModelRef>;
};

type PersistedCacheStatsV3 = {
  version: 3;
  statsByModel: Record<string, CacheStats>;
  legacyFamily: Partial<Record<CacheProviderId, CacheStats>>;
};

/**
 * V4 format: session-scoped stats buckets.
 * Each session in the host runtime gets its own stats isolated by a hashed session id.
 *
 * sessions: sessionHash → modelKey (provider/id) → CacheStats
 * legacyFamily: unchanged from v3 (migration/fallback when ctx.model is unknown)
 */
type PersistedCacheStatsV4 = {
  version: 4;
  sessions: Record<string, Record<string, CacheStats>>;
  legacyFamily: Partial<Record<CacheProviderId, CacheStats>>;
};

type PersistedCacheStatsV5 = {
  version: 5;
  sessions: Record<string, Record<string, CacheStats>>;
  legacyFamily: Partial<Record<CacheProviderId, CacheStats>>;
  lastRoutedModelBySession?: Record<string, PersistedRoutedModelRef>;
};

type UsageSnapshot = {
  cacheRead: number;
  cacheWrite: number;
  totalInput: number;
};

/**
 * Per-request sample stored for trend analysis and usage-field-missing detection.
 * Contains only numeric counters and booleans — never message content, prompts,
 * payloads, headers, API keys, or model outputs.
 */
type CacheUsageSample = {
  timestamp: number;
  hit: boolean;
  cachedInputTokens: number;
  cacheWriteInputTokens: number;
  totalInputTokens: number;
  missingUsageFields: boolean;
  promptRewriteEnabled: boolean;
  systemPromptFingerprint?: string;
  promptCacheKeySource: PromptCacheKeySource;
  hintPayloadComparison: PromptComparison;
};

type PromptCacheKeySource = "header" | "session" | "unavailable";
type PromptComparison = "match" | "mismatch" | "unavailable";

type PromptRequestDiagnostics = {
  promptRewriteEnabled: boolean;
  systemPromptFingerprint?: string;
  promptCacheKeySource: PromptCacheKeySource;
  hintPayloadComparison: PromptComparison;
};

type PromptRewriteContext = {
  routeSnapshot?: PiRouteSnapshot;
  routedModel?: PiModel;
  timestamp: number;
};

const PROMPT_REWRITE_CONTEXT_TTL_MS = 10_000;

/** Maximum number of recent samples kept per model key (in-memory only, not persisted). */
const MAX_RECENT_SAMPLES = 50;

type CacheProviderAdapter = {
  id: CacheProviderId;
  label: string;
  showCacheWrite?: boolean;
  matchesModel(model: PiModel | undefined): boolean;
  matchesAssistantMessage(message: unknown, model: PiModel | undefined): boolean;
  normalizeUsage(message: unknown): UsageSnapshot | undefined;
  warningText?(model: PiModel): string | undefined;
};

/**
 * Strip per-turn churn from trellis `<session-overview>` block.
 *
 * Trellis injects a session-overview that includes `RECENT COMMITS`
 * (shifts on every git commit), `Working directory: Clean/N uncommitted`
 * (shifts on every edit/commit), and `Line count: N / 2000` (shifts on
 * every journal append). These fields are at the tail of the
 * session-overview and poison the prompt-prefix cache for everything
 * that follows.
 *
 * This function surgically removes those three churn fields from the
 * `<session-overview>...</session-overview>` block. The remaining
 * fields (DEVELOPER, GIT STATUS branch-only, CURRENT TASK, ACTIVE
 * TASKS, MY TASKS, JOURNAL FILE active-file-only, PACKAGES, PATHS)
 * are stable within a session and become cache-friendlier.
 *
 * No-op when the `<session-overview>` tag is not present (e.g.
 * trellis hook chose not to inject it, or a different extension
 * owns the prompt).
 */
function stripSessionOverviewChurn(prompt: string): string {
  const startTag = "<session-overview>";
  const endTag = "</session-overview>";

  const startIdx = prompt.indexOf(startTag);
  if (startIdx === -1) return prompt;

  const endIdx = prompt.indexOf(endTag, startIdx + startTag.length);
  if (endIdx === -1) return prompt;

  const before = prompt.slice(0, startIdx + startTag.length);
  const inner = prompt.slice(startIdx + startTag.length, endIdx);
  const after = prompt.slice(endIdx);

  let cleaned = inner
    // Drop the RECENT COMMITS section (from the heading through the
    // next heading or end of inner). The model sees commit history
    // via `git log`; carrying it in every system prompt is redundant.
    .replace(/\n## RECENT COMMITS\n[\s\S]*?(?=\n## |$)/, "")
    // Drop "Working directory: ..." (Git status tail churn).
    .replace(/\nWorking directory:[^\n]*/g, "")
    // Drop "Line count: N / NNNN" (Journal tail churn).
    .replace(/\nLine count:[^\n]*/g, "");

  return before + cleaned + after;
}

// ── OMP 17 系统 prompt 块数组工具 ──────────────────────────────
// 主重写路径操作 before_agent_start 给出的 string[] 块。
// join 仅用于 cache hint 展示，不用于再拆分。

function joinSystemPromptBlocks(blocks: string[]): string {
  return blocks.filter((b) => typeof b === "string" && b.length > 0).join("\n\n");
}

function mapSystemPromptBlocks(blocks: string[], mapFn: (block: string, index: number) => string): string[] {
  return blocks.map((block, index) => {
    if (typeof block !== "string") return block;
    const next = mapFn(block, index);
    // 防止整块被 map 成空串后误删。
    if (typeof next !== "string" || next.trim().length === 0) return block;
    return next;
  });
}

function systemPromptBlocksEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// OpenAI prompt_cache_key 的最大长度（API 限制）。超长会被 OpenAI 拒为 400；
// fallback 到原始 sessionId（可能为长 UUID/复合 id）时必须截断。
const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64;

/** Trim 并截断到 OpenAI prompt_cache_key 上限；空/空白返回 undefined。 */
function clampPromptCacheKey(key: string | undefined): string | undefined {
  const normalized = key?.trim();
  if (!normalized) return undefined;
  const chars = Array.from(normalized);
  if (chars.length <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH) return normalized;
  return chars.slice(0, OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH).join("");
}

function getSessionPromptCacheKey(ctx: ExtensionContext): string | undefined {
  // OMP 17：宿主解析 provider-facing cache key（header.providerPromptCacheKey ?? sessionId）。
  // cache hint 协议必须保留完整 key，防止不同长 key 因相同前缀被错误合并；仅 OpenAI
  // request body 受 64 字符限制，注入时再通过 clampPromptCacheKey 截断。
  const header = ctx.sessionManager.getHeader?.();
  const providerPromptCacheKey = asRecord(header)?.providerPromptCacheKey;
  if (isNonEmptyString(providerPromptCacheKey)) return (providerPromptCacheKey as string).trim();
  const sessionId = ctx.sessionManager.getSessionId().trim();
  return sessionId || undefined;
}

/**
 * Hash a session id for use as a non-reversible opaque scope key.
 * Returns a 16-character hex string (64 bits of SHA-256 digest prefix)
 * suitable for scoping stats buckets without exposing the raw session id.
 */
function hashSessionId(sessionId: string): string {
  return createHash("sha256").update(sessionId).digest("hex").slice(0, 16);
}

/** Returns a non-reversible diagnostic fingerprint; never persist the prompt text. */
function fingerprintPrompt(prompt: string | undefined): string | undefined {
  return typeof prompt === "string"
    ? createHash("sha256").update(prompt).digest("hex").slice(0, 16)
    : undefined;
}

function comparePromptFingerprints(
  expected: string | undefined,
  actual: string | undefined,
): PromptComparison {
  if (!expected || !actual) return "unavailable";
  return expected === actual ? "match" : "mismatch";
}

function getPromptCacheKeySource(header: unknown, sessionId: string | undefined): PromptCacheKeySource {
  if (isNonEmptyString(asRecord(header)?.providerPromptCacheKey)) return "header";
  return sessionId ? "session" : "unavailable";
}

function getProtocolGlobal(): ProtocolGlobal {
  return globalThis as ProtocolGlobal;
}

function firstNonEmptyString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (isNonEmptyString(value)) return value.trim();
  }
  return undefined;
}

function sessionHashFromContext(ctx: Pick<ExtensionContext, "sessionManager">): string | undefined {
  const sessionId = ctx.sessionManager.getSessionId();
  return sessionId ? hashSessionId(sessionId) : undefined;
}

function isPiRouterAdapterV1(value: unknown): value is PiRouterAdapterV1 {
  const record = asRecord(value);
  return !!record && isNonEmptyString(record.virtualProvider) && typeof record.resolveActiveRoute === "function";
}

function isRoutingRegistryV1(value: unknown): value is PiRoutingRegistryV1 {
  const record = asRecord(value);
  return !!record && record.version === 1 && typeof record.registerRouter === "function" && typeof record.getRouter === "function";
}

function createRoutingRegistry(): PiRoutingRegistryV1 {
  const routers = new Map<string, PiRouterAdapterV1>();
  return {
    version: 1,
    registerRouter(adapter: PiRouterAdapterV1): () => void {
      if (!isPiRouterAdapterV1(adapter)) return () => undefined;
      const key = adapter.virtualProvider.trim();
      routers.set(key, adapter);
      return () => {
        if (routers.get(key) === adapter) routers.delete(key);
      };
    },
    getRouter(virtualProvider: string): PiRouterAdapterV1 | undefined {
      return routers.get(virtualProvider);
    },
  };
}

function getRoutingRegistry(): PiRoutingRegistryV1 | undefined {
  const candidate = getProtocolGlobal()[PI_ROUTING_REGISTRY_SYMBOL];
  return isRoutingRegistryV1(candidate) ? candidate : undefined;
}

function ensureRoutingRegistry(): PiRoutingRegistryV1 {
  const existing = getRoutingRegistry();
  if (existing) return existing;

  const created = createRoutingRegistry();
  getProtocolGlobal()[PI_ROUTING_REGISTRY_SYMBOL] = created;
  return created;
}

function parseRouteStatus(value: unknown): PiRouteSnapshot["status"] | undefined {
  return value === "planned" || value === "trying" || value === "selected" || value === "success" || value === "failed"
    ? value
    : undefined;
}

function parseRouteSnapshot(
  value: unknown,
  fallbackVirtualProvider?: string,
  fallbackVirtualModelId?: string,
): PiRouteSnapshot | undefined {
  const record = asRecord(value);
  if (!record) return undefined;

  const virtualProvider = firstNonEmptyString(record.virtualProvider, fallbackVirtualProvider);
  const virtualModelId = firstNonEmptyString(record.virtualModelId, record.virtualModel, fallbackVirtualModelId);
  const provider = firstNonEmptyString(record.provider, record.upstreamProvider, record.targetProvider);
  const modelId = firstNonEmptyString(record.modelId, record.upstreamModelId, record.targetModelId, record.responseModel);
  if (!virtualProvider || !virtualModelId || !provider || !modelId) return undefined;

  const timestamp = getNumber(record.timestamp) ?? Date.now();
  return {
    virtualProvider,
    virtualModelId,
    provider,
    modelId,
    api: firstNonEmptyString(record.api),
    canonicalModelId: firstNonEmptyString(record.canonicalModelId),
    routeLabel: firstNonEmptyString(record.routeLabel, record.label),
    status: parseRouteStatus(record.status),
    sessionIdHash: firstNonEmptyString(record.sessionIdHash),
    requestId: firstNonEmptyString(record.requestId),
    timestamp,
  };
}

function resolveActiveRouteSnapshot(
  model: PiModel | undefined,
  ctx?: Pick<ExtensionContext, "sessionManager">,
): PiRouteSnapshot | undefined {
  if (!model) return undefined;
  const hint: PiRouteResolveHint | undefined = ctx ? { sessionIdHash: sessionHashFromContext(ctx) } : undefined;

  const adapter = getRoutingRegistry()?.getRouter(model.provider);
  if (adapter) {
    try {
      const snapshot = parseRouteSnapshot(
        adapter.resolveActiveRoute(model.id, hint),
        model.provider,
        model.id,
      );
      if (snapshot) return snapshot;
    } catch (error) {
      console.warn(`${LOG_PREFIX}: routing registry adapter failed`, error);
    }
  }

  // Temporary migration shim for the prototype global used by early router PRs.
  // New integrations should use Symbol.for("omp.routing.registry.v1") instead.
  const legacy = getProtocolGlobal().__ompCacheOptimizerRouter;
  if (!legacy || !lower(model.provider).includes("router")) return undefined;
  try {
    if (typeof legacy === "function") {
      return parseRouteSnapshot(legacy(model.provider, model.id, hint), model.provider, model.id);
    }
    const legacyRecord = asRecord(legacy);
    const resolver = legacyRecord?.resolveActiveRoute;
    if (typeof resolver === "function") {
      return parseRouteSnapshot(resolver.call(legacy, model.id, hint), model.provider, model.id);
    }
    return parseRouteSnapshot(legacy, model.provider, model.id);
  } catch (error) {
    console.warn(`${LOG_PREFIX}: legacy routing global failed`, error);
    return undefined;
  }
}

function routeSnapshotToPiModel(snapshot: PiRouteSnapshot, fallback?: PiModel): PiModel {
  return {
    ...(fallback ?? {}),
    id: snapshot.modelId,
    name: snapshot.canonicalModelId ?? snapshot.modelId,
    provider: snapshot.provider,
    api: snapshot.api ?? fallback?.api ?? "",
    baseUrl: fallback?.baseUrl ?? "",
    reasoning: fallback?.reasoning ?? false,
    input: fallback?.input ?? ["text"],
    cost: fallback?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: fallback?.contextWindow ?? 0,
    maxTokens: fallback?.maxTokens ?? 0,
    compat: fallback?.compat,
  } as PiModel;
}

function findModelInRegistry(registry: ModelRegistryLike | undefined, provider: string, id: string): PiModel | undefined {
  const found = registry?.find?.(provider, id);
  if (found) return found;

  const available = registry?.getAvailable?.() ?? [];
  const availableMatch = available.find((candidate) => candidate.provider === provider && candidate.id === id);
  if (availableMatch) return availableMatch;

  const all = registry?.getAll?.() ?? [];
  return all.find((candidate) => candidate.provider === provider && candidate.id === id);
}

function resolveRouteModel(
  model: PiModel | undefined,
  ctx?: ContextWithOptionalModelRegistry,
): PiModel | undefined {
  const snapshot = resolveActiveRouteSnapshot(model, ctx);
  if (!snapshot) return undefined;

  return findModelInRegistry(ctx?.modelRegistry, snapshot.provider, snapshot.modelId)
    ?? routeSnapshotToPiModel(snapshot, model);
}

function isVirtualRoutingModel(model: PiModel | undefined, ctx?: Pick<ExtensionContext, "sessionManager">): boolean {
  if (!model) return false;
  return isRouterModel(model) || !!getRoutingRegistry()?.getRouter(model.provider) || !!resolveActiveRouteSnapshot(model, ctx);
}

function isCacheHintsServiceV1(value: unknown): value is PiCacheHintsV1 {
  const record = asRecord(value);
  return !!record && record.version === 1 && typeof record.getHints === "function";
}

function getCacheHintsService(): PiCacheHintsV1 | undefined {
  const candidate = getProtocolGlobal()[PI_CACHE_HINTS_SYMBOL];
  return isCacheHintsServiceV1(candidate) ? candidate : undefined;
}

function installCacheHintsService(service: PiCacheHintsV1): () => void {
  const globals = getProtocolGlobal();
  const previous = globals[PI_CACHE_HINTS_SYMBOL];
  globals[PI_CACHE_HINTS_SYMBOL] = service;
  return () => {
    if (globals[PI_CACHE_HINTS_SYMBOL] !== service) return;
    if (previous === undefined) {
      delete globals[PI_CACHE_HINTS_SYMBOL];
    } else {
      globals[PI_CACHE_HINTS_SYMBOL] = previous;
    }
  };
}

/**
 * Build a session-scoped stats key from a session hash + provider/id.
 * Pure function (no closure dependency) for use by tests and internals.
 */
function makeSessionModelKey(sessionHash: string, provider: string, id: string): string {
  return `${sessionHash}:${provider}/${id}`;
}

/**
 * Extract the user-facing model key from a session-scoped key.
 * "abc123:otokapi/gpt-5.5" → "otokapi/gpt-5.5"
 */
function modelKeyFromSessionKey(sessionModelKey: string): string {
  const idx = sessionModelKey.indexOf(":");
  return idx >= 0 ? sessionModelKey.slice(idx + 1) : sessionModelKey;
}

function asRecord(value: unknown): UnknownRecord | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
  return value as UnknownRecord;
}

function lower(value: unknown): string {
  return typeof value === "string" ? value.toLowerCase() : "";
}

function getNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function getNonNegativeNumber(record: UnknownRecord, key: string): number | undefined {
  const value = getNumber(record[key]);
  return value !== undefined && value >= 0 ? value : undefined;
}

/**
 * Get effective compat for a model by merging provider-level and model-level compat.
 * Model-level compat takes precedence over provider-level compat for overlapping keys.
 * This matches OMP's model-registry.js mergeCompat behavior.
 */
function getCompat(model: PiModel | undefined): CacheCompat {
  if (!model) return {} as CacheCompat;

  const record = model as PiModel & { compatConfig?: Record<string, unknown> };
  return {
    ...((record.compatConfig ?? {}) as CacheCompat),
    ...((record.compat ?? {}) as CacheCompat),
  };
}

function makePromptRewriteContextKey(sessionHash: string | undefined, model: PiModel | undefined): string | undefined {
  if (!sessionHash || !model) return undefined;
  return `${sessionHash}:${modelKey(model)}`;
}

function rememberPromptRewriteContext(
  contexts: Map<string, PromptRewriteContext>,
  key: string | undefined,
  context: PromptRewriteContext,
): void {
  if (!key) return;
  contexts.set(key, context);
}

function getPromptRewriteContext(
  contexts: Map<string, PromptRewriteContext>,
  key: string | undefined,
  now = Date.now(),
  ttlMs = PROMPT_REWRITE_CONTEXT_TTL_MS,
): PromptRewriteContext | undefined {
  if (!key) return undefined;
  const context = contexts.get(key);
  if (!context) return undefined;
  if (now - context.timestamp > ttlMs) {
    contexts.delete(key);
    return undefined;
  }
  return context;
}

/**
 * 返回平台友好的 models.yml 展示路径（仅用于文案，不用于 I/O）。
 * Windows：`%USERPROFILE%\.omp\agent\models.yml`
 * 其它平台：`~/.omp/agent/models.yml`
 */
function getModelsYmlDisplayPath(platform: string = process.platform): string {
  if (platform.startsWith("win")) {
    return `%USERPROFILE%\\.omp\\agent\\models.yml`;
  }
  return "~/.omp/agent/models.yml";
}

function isEnabledEnv(value: string | undefined): boolean {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function isPromptRewriteEnabled(env: MutableEnv = process.env): boolean {
  return isEnabledEnv(env[PROMPT_REWRITE_ENV]);
}

function setRuntimeOptimizerEnabled(enabled: boolean, env: MutableEnv = process.env): void {
  runtimeOptimizerEnabled = enabled;
  if (enabled) {
    requestLongCacheRetention(env);
  } else {
    restoreCacheRetentionEnv(STARTUP_CACHE_RETENTION_ENV, env);
  }
}

function isRuntimeOptimizerEnabled(): boolean {
  return runtimeOptimizerEnabled;
}

function getOptimizerRuntimeModeLines(): string[] {
  const state = runtimeOptimizerEnabled ? "已启用" : "已关闭";
  const lines: string[] = [];
  const retention = readCacheRetentionValue(process.env);
  lines.push(`运行状态：${state}`);
  lines.push(`• Prompt 重写：${runtimeOptimizerEnabled && isPromptRewriteEnabled() ? "开启" : "关闭"}`);
  lines.push(`• Footer 缓存统计：开启${runtimeOptimizerEnabled ? "" : "（对比模式）"}`);
  lines.push(`• Compat 提示：${runtimeOptimizerEnabled ? "开启" : "关闭"}`);
  lines.push(`• ${OMP_CACHE_RETENTION_ENV}：${retention ?? "（未设置）"}`);
  if (!runtimeOptimizerEnabled) {
    lines.push("这是当前进程内开关。运行 /reload 或重启 OMP 可恢复到启动时行为。");
  } else if (!isPromptRewriteEnabled()) {
    lines.push(`可设置 ${PROMPT_REWRITE_ENV}=1 显式开启 prompt 重写。`);
  }
  return lines;
}

function formatOptimizerRuntimeMode(): string {
  return getOptimizerRuntimeModeLines().join("\n");
}

function isAssistantMessage(message: unknown): boolean {
  return asRecord(message)?.role === "assistant";
}

function getAssistantRecord(message: unknown): UnknownRecord | undefined {
  const record = asRecord(message);
  return record?.role === "assistant" ? record : undefined;
}

function getModelIdNameTokenValues(model: PiModel | undefined): string[] {
  if (!model) return [];
  return [model.id, model.name].map(lower).filter(Boolean);
}

function getAssistantMessageModelTokenValues(message: unknown): string[] {
  const record = asRecord(message);
  if (!record) return [];

  return ASSISTANT_MESSAGE_MODEL_TOKEN_KEYS.map((key) => lower(record[key])).filter(Boolean);
}

function hasAnyTokenContaining(tokens: string[], needles: string[]): boolean {
  return tokens.some((token) => needles.some((needle) => token.includes(needle)));
}

function modelOrAssistantMessageHas(message: unknown, model: PiModel | undefined, needles: string[]): boolean {
  return hasAnyTokenContaining([...getModelIdNameTokenValues(model), ...getAssistantMessageModelTokenValues(message)], needles);
}

function isDeepSeekLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["deepseek"]);
}

function isDeepSeekLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["deepseek"]);
}

function isOpenAICompatibleApi(api: unknown): boolean {
  const value = lower(api);
  return value === "openai-completions" || value === "openai-responses";
}

function isOpenAICompatibleProxyApi(api: unknown): boolean {
  return lower(api) === "openai-completions";
}

function isResponsesPromptRewriteBypassApi(api: unknown): boolean {
  const value = lower(api);
  return value === "openai-codex-responses" || value === "openai-responses" || value === "azure-openai-responses";
}

function isMistralConversationsApi(api: unknown): boolean {
  return lower(api) === "mistral-conversations";
}

function isOpenAIFamilyToken(token: string): boolean {
  return token.includes("gpt-") || token.includes("chatgpt") || OPENAI_REASONING_MODEL_PATTERN.test(token);
}

function isOpenAIFamilyModel(model: PiModel | undefined): boolean {
  return getModelIdNameTokenValues(model).some(isOpenAIFamilyToken);
}

function isOpenAIFamilyAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return [...getModelIdNameTokenValues(model), ...getAssistantMessageModelTokenValues(message)].some(isOpenAIFamilyToken);
}

function isClaudeLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["anthropic", "claude"]);
}

function isClaudeLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["anthropic", "claude"]);
}

function isGeminiLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["gemini", "vertex"]);
}

function isGeminiLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["gemini", "vertex"]);
}

// ── Adaptive generation model detection ────────────────────────────

/**
 * Check whether the model id uses Anthropic's adaptive generation (thinking)
 * that requires `forceAdaptiveThinking: true` in compat.
 *
 * Adaptive-generation models (from the bundled model catalog) include:
 *   claude-opus-4-6, claude-opus-4-7, claude-opus-4-8 (also dotted 4.6/4.7/4.8)
 *   claude-sonnet-4-6
 *   claude-fable-5
 *
 * We match broadly: opus >= 4-6, sonnet >= 4-6, fable >= 5.
 * Ids may carry date-stamp or size suffixes like "[1M]".
 */
const ADAPTIVE_OPUS_PATTERN = /(^|[\/\s:_-])(opus-4[.-][6-9]|opus-4-[1-9][0-9])($|[-_.:\/\s\[])/i;
const ADAPTIVE_SONNET_PATTERN = /(^|[\/\s:_-])(sonnet-4[.-][6-9]|sonnet-4-[1-9][0-9])($|[-_.:\/\s\[])/i;
const ADAPTIVE_FABLE_PATTERN = /(^|[\/\s:_-])fable-([5-9]|[1-9][0-9])($|[-_.:\/\s\[])/i;

function isAdaptiveGenerationModel(model: PiModel | undefined): boolean {
  if (!model) return false;
  const tokens = getModelIdNameTokenValues(model);
  return tokens.some((t) => ADAPTIVE_OPUS_PATTERN.test(t) || ADAPTIVE_SONNET_PATTERN.test(t) || ADAPTIVE_FABLE_PATTERN.test(t));
}

// OMP divergence: adaptive thinking is set automatically by the OMP built-in model
// catalog (via disableAdaptiveThinking, with reversed semantics) and is NOT
// user-configurable from models.yml (see omp models.md §Anthropic compatibility).
// The legacy `forceAdaptiveThinking` flag no longer exists. We keep model detection
// (isAdaptiveGenerationModel) for informational doctor output, but drop the fixable
// compat-suggestion path entirely.
function isAdaptiveThinkingCompatApplicable(_model: PiModel): boolean {
  return false;
}

function describeMissingAdaptiveThinkingCompat(_model: PiModel): string[] {
  return [];
}

function buildAdaptiveThinkingCompatSuggestion(_missing: string[]): Record<string, unknown> {
  return {};
}

function appendAdaptiveThinkingCompatAdviceLines(lines: string[], _missing: string[], placement: CompatAdvicePlacement = {}): void {
  lines.push("- 自适应思考：OMP 内置模型目录会为官方 Claude 模型自动设置。");
  lines.push("  自定义 Anthropic 渠道应依赖内置 catalog 元数据；");
  lines.push("  如果上游拒绝 adaptive thinking，请确认模型 id 是否匹配官方发布版本。");
  appendCredentialSafeProviderGuidance(lines, placement, {});
}

function buildAdaptiveThinkingCompatWarningText(key: string, _missing: string[]): string {
  const slashIdx = key.indexOf("/");
  const providerLabel = slashIdx > 0 ? key.slice(0, slashIdx) : key;
  const modelId = slashIdx > 0 ? key.slice(slashIdx + 1) : undefined;
  const modelsJsonPath = getModelsYmlDisplayPath();
  const lines: string[] = [
    `ℹ️ omp-cache-optimizer：${key} 是支持自适应生成的 Claude 模型。`,
    "OMP 内置 catalog 会自动处理自适应思考；官方模型不需要额外的 models.yml compat 键。",
    "如果是转发 Anthropic 的自定义渠道，可能仍需要显式 catalog 元数据。",
    `可参考 ${modelsJsonPath} -> providers["${providerLabel}"] -> models -> "${modelId ?? '<id>'}"。`,
    "",
  ];
  appendAdaptiveThinkingCompatAdviceLines(lines, [], { providerLabel, modelId });
  return lines.join("\n");
}

// ── Non-GPT OpenAI-compatible model detection ──────────────────────

function isKimiLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["kimi"]);
}
function isKimiLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["kimi"]);
}

function isQwenLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["qwen"]);
}
function isQwenLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["qwen"]);
}

function isGLMLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["glm"]);
}
function isGLMLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["glm"]);
}

function isMiniMaxLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["minimax"]);
}
function isMiniMaxLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["minimax"]);
}

function isMimoLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["xiaomimimo"]) || tokens.some((t) => MIMO_MODEL_PATTERN.test(t));
}
function isMimoLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["xiaomimimo"]) || allTokens.some((t) => MIMO_MODEL_PATTERN.test(t));
}

function isHunyuanLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["hunyuan"]);
}
function isHunyuanLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["hunyuan"]);
}

// ── Additional OpenAI-compatible model detection ──────────────────

function isMistralLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["mistral", "mixtral", "codestral"]);
}
function isMistralLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["mistral", "mixtral", "codestral"]);
}

function isGrokLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["grok"]) || tokens.some((t) => XAI_MODEL_PATTERN.test(t));
}
function isGrokLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["grok"]) || allTokens.some((t) => XAI_MODEL_PATTERN.test(t));
}

function isLlamaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["llama"]);
}
function isLlamaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["llama"]);
}

function isNemotronLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["nemotron"]);
}
function isNemotronLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["nemotron"]);
}

function isCohereLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["cohere", "command-r"]);
}
function isCohereLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["cohere", "command-r"]);
}

const YI_MODEL_PATTERN = /(^|[\/\s:_-])yi($|[\-_.:\/\s])/;

function isYiLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["yi-", "01-ai", "zero-one"]) || tokens.some((t) => YI_MODEL_PATTERN.test(t));
}
function isYiLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["yi-", "01-ai", "zero-one"]) || allTokens.some((t) => YI_MODEL_PATTERN.test(t));
}

// ── More OpenAI-compatible model detection (batch 2) ───────────────

const DOUBAO_SEED_PATTERN = /(^|[\/\s:_-])seed($|[\-_.:\/\s])/i;

function isDoubaoLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["doubao", "豆包", "volcengine", "bytedance", "byte-dance"]) ||
    tokens.some((t) => DOUBAO_SEED_PATTERN.test(t));
}
function isDoubaoLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["doubao", "豆包", "volcengine", "bytedance", "byte-dance"]) ||
    allTokens.some((t) => DOUBAO_SEED_PATTERN.test(t));
}

function isErnieLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["ernie", "wenxin", "文心", "yiyan", "一言", "baidu"]);
}
function isErnieLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["ernie", "wenxin", "文心", "yiyan", "一言", "baidu"]);
}

function isBaichuanLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["baichuan", "百川"]);
}
function isBaichuanLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["baichuan", "百川"]);
}

function isStepFunLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["stepfun", "step-"]);
}
function isStepFunLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["stepfun", "step-"]);
}

function isSparkLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["spark", "xinghuo", "星火", "iflytek", "讯飞"]);
}
function isSparkLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["spark", "xinghuo", "星火", "iflytek", "讯飞"]);
}

function isInternLMLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["internlm", "intern-lm", "书生"]);
}
function isInternLMLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["internlm", "intern-lm", "书生"]);
}

function isGemmaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["gemma"]);
}
function isGemmaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["gemma"]);
}

const PHI_MODEL_PATTERN = /(^|[\/\s:_-])phi($|[\-_.:\/\s])/i;

function isPhiLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["phi-"]) || tokens.some((t) => PHI_MODEL_PATTERN.test(t));
}
function isPhiLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["phi-"]) || allTokens.some((t) => PHI_MODEL_PATTERN.test(t));
}

function isJambaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["jamba", "ai21"]);
}
function isJambaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["jamba", "ai21"]);
}

function isSolarLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["solar", "upstage"]);
}
function isSolarLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["solar", "upstage"]);
}

// ── New OpenAI-compatible model detection (batch 3, 12 families) ──────

// Perplexity / Sonar
function isPerplexityLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["sonar", "perplexity"]) || tokens.some((t) => PPLX_MODEL_PATTERN.test(t));
}
function isPerplexityLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["sonar", "perplexity"]) || allTokens.some((t) => PPLX_MODEL_PATTERN.test(t));
}

// Amazon Nova
function isNovaLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["amazon-nova"]) || tokens.some((t) => NOVA_MODEL_PATTERN.test(t));
}
function isNovaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["amazon-nova"]) || allTokens.some((t) => NOVA_MODEL_PATTERN.test(t));
}

// Reka
function isRekaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["reka"]);
}
function isRekaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["reka"]);
}

// Falcon / TII
function isFalconLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["falcon", "tiiuae"]);
}
function isFalconLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["falcon", "tiiuae"]);
}

// Databricks DBRX
function isDbrxLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["dbrx", "databricks"]);
}
function isDbrxLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["dbrx", "databricks"]);
}

// MosaicML MPT
function isMptLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["mosaicml", "mpt-"]) || tokens.some((t) => MPT_MODEL_PATTERN.test(t));
}
function isMptLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["mosaicml", "mpt-"]) || allTokens.some((t) => MPT_MODEL_PATTERN.test(t));
}

// StableLM / Stability AI
function isStableLMLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["stablelm", "stable-lm", "stability-ai"]);
}
function isStableLMLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["stablelm", "stable-lm", "stability-ai"]);
}

// BAAI / Aquila
function isAquilaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["aquila", "baai"]);
}
function isAquilaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["aquila", "baai"]);
}

// LG EXAONE
function isExaoneLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["exaone"]);
}
function isExaoneLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["exaone"]);
}

// Naver HyperCLOVA X (conservative: hyperclova, clova-x only)
function isHyperCLOVALikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["hyperclova", "clova-x"]);
}
function isHyperCLOVALikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["hyperclova", "clova-x"]);
}

// Aleph Alpha Luminous
function isLuminousLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["luminous", "aleph-alpha"]) || tokens.some((t) => ALEPH_MODEL_PATTERN.test(t));
}
function isLuminousLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["luminous", "aleph-alpha"]) || allTokens.some((t) => ALEPH_MODEL_PATTERN.test(t));
}

// Nous / Hermes / OpenHermes
function isHermesLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["nous", "hermes", "openhermes"]);
}
function isHermesLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["nous", "hermes", "openhermes"]);
}

// ── More OpenAI-compatible model detection (batch 4, 18 families) ──

// IBM Granite
function isGraniteLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["granite", "ibm-granite"]);
}
function isGraniteLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["granite", "ibm-granite"]);
}

// Snowflake Arctic
function isArcticLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["snowflake-arctic"]) || tokens.some((t) => ARCTIC_MODEL_PATTERN.test(t));
}
function isArcticLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["snowflake-arctic"]) || allTokens.some((t) => ARCTIC_MODEL_PATTERN.test(t));
}

// Huawei Pangu / 盘古
function isPanguLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["pangu", "pan-gu", "盘古", "huawei-pangu"]);
}
function isPanguLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["pangu", "pan-gu", "盘古", "huawei-pangu"]);
}

// SenseTime SenseNova / 商汤
function isSenseNovaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["sensenova", "sense-nova", "sensechat", "商汤"]);
}
function isSenseNovaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["sensenova", "sense-nova", "sensechat", "商汤"]);
}

// 360 Zhinao / 智脑
function isZhinaoLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["360gpt", "360-gpt", "zhinao", "智脑"]);
}
function isZhinaoLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["360gpt", "360-gpt", "zhinao", "智脑"]);
}

// OpenBMB MiniCPM
function isMiniCPMLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["minicpm", "mini-cpm", "openbmb"]);
}
function isMiniCPMLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["minicpm", "mini-cpm", "openbmb"]);
}

// XVERSE
function isXVerseLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["xverse"]);
}
function isXVerseLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["xverse"]);
}

// OrionStar Orion
function isOrionLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["orionstar", "orion-star"]) || tokens.some((t) => ORION_MODEL_PATTERN.test(t));
}
function isOrionLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["orionstar", "orion-star"]) || allTokens.some((t) => ORION_MODEL_PATTERN.test(t));
}

// OpenChat
function isOpenChatLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["openchat"]);
}
function isOpenChatLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["openchat"]);
}

// Vicuna
function isVicunaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["vicuna"]);
}
function isVicunaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["vicuna"]);
}

// WizardLM / WizardCoder
function isWizardLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["wizardlm", "wizard-lm", "wizardcoder", "wizard-coder"]);
}
function isWizardLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["wizardlm", "wizard-lm", "wizardcoder", "wizard-coder"]);
}

// Zephyr
function isZephyrLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["zephyr"]);
}
function isZephyrLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["zephyr"]);
}

// Dolphin
function isDolphinLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["dolphin"]);
}
function isDolphinLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["dolphin"]);
}

// OpenOrca
function isOpenOrcaLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["openorca", "open-orca"]);
}
function isOpenOrcaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["openorca", "open-orca"]);
}

// Starling
function isStarlingLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["starling"]);
}
function isStarlingLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["starling"]);
}

// BLOOM / BigScience
function isBloomLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["bloom", "bigscience"]);
}
function isBloomLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["bloom", "bigscience"]);
}

// RWKV
function isRwkvLikeModel(model: PiModel | undefined): boolean {
  return hasAnyTokenContaining(getModelIdNameTokenValues(model), ["rwkv"]);
}
function isRwkvLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  return modelOrAssistantMessageHas(message, model, ["rwkv"]);
}

// Cohere Aya
function isAyaLikeModel(model: PiModel | undefined): boolean {
  const tokens = getModelIdNameTokenValues(model);
  return hasAnyTokenContaining(tokens, ["aya-expanse"]) || tokens.some((t) => AYA_MODEL_PATTERN.test(t));
}
function isAyaLikeAssistantMessage(message: unknown, model: PiModel | undefined): boolean {
  const allTokens = [
    ...getModelIdNameTokenValues(model),
    ...getAssistantMessageModelTokenValues(message),
  ];
  return hasAnyTokenContaining(allTokens, ["aya-expanse"]) || allTokens.some((t) => AYA_MODEL_PATTERN.test(t));
}

// ── Model key ──────────────────────────────────────────────────────

function modelKey(model: PiModel): string {
  return `${model.provider}/${model.id}`;
}

function isRouterModel(model: PiModel | undefined): boolean {
  return lower(model?.provider) === "router";
}

function modelFromAssistantMessage(message: unknown, fallback: PiModel | undefined): PiModel | undefined {
  const record = getAssistantRecord(message);
  if (!record) return fallback;

  const id = firstNonEmptyString(record.responseModel, record.model, fallback?.id);
  const provider = firstNonEmptyString(record.provider, fallback?.provider);
  const api = firstNonEmptyString(record.api, fallback?.api) ?? "";
  if (!id || !provider) return fallback;

  return {
    ...(fallback ?? {}),
    id,
    name: id,
    provider,
    api,
    baseUrl: fallback?.baseUrl ?? "",
    reasoning: fallback?.reasoning ?? false,
    input: fallback?.input ?? ["text"],
    cost: fallback?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: fallback?.contextWindow ?? 0,
    maxTokens: fallback?.maxTokens ?? 0,
  } as PiModel;
}

function keyForModelExt(model: { provider: string; id: string }): string {
  return `${model.provider}/${model.id}`;
}

function usageRecordFromAssistant(message: unknown): UnknownRecord | undefined {
  return asRecord(getAssistantRecord(message)?.usage);
}

function getNestedRecord(record: UnknownRecord | undefined, key: string): UnknownRecord | undefined {
  return asRecord(record?.[key]);
}

function getFirstNonNegativeNumber(...values: unknown[]): number | undefined {
  for (const value of values) {
    const number = getNumber(value);
    if (number !== undefined && number >= 0) return number;
  }
  return undefined;
}

function readCachedTokensFromDetails(details: UnknownRecord | undefined): number | undefined {
  return getFirstNonNegativeNumber(details?.cached_tokens, details?.cachedTokens);
}

function readCacheWriteFromDetails(details: UnknownRecord | undefined): number | undefined {
  return getFirstNonNegativeNumber(details?.cache_write_tokens, details?.cacheWriteTokens);
}

// The host runtime normalizes provider-specific raw usage (prompt_cache_hit_tokens, cached_tokens,
// cache_read_input_tokens, etc.) into a common shape:
//   input     = uncached prompt portion (total prompt minus cacheRead minus cacheWrite)
//   cacheRead = tokens read from a previously-cached prefix
//   cacheWrite= tokens newly written into cache in this request
//
// We reconstruct the total prompt-token count as input + cacheRead + cacheWrite.
// The host runtime guarantees that input, cacheRead, and cacheWrite are always present on
// assistant messages processed through its provider pipeline (at least as zero).
//
// Only DeepSeek sets allowInputOnly=true so that a cache miss (cacheRead=0) still
// contributes total input tokens to the denominator.
function getPiNormalizedUsage(message: unknown, allowInputOnly = false): UsageSnapshot | undefined {
  const usage = usageRecordFromAssistant(message);
  if (!usage) return undefined;

  const input = getNonNegativeNumber(usage, "input");
  const cacheRead = getNonNegativeNumber(usage, "cacheRead");
  const cacheWrite = getNonNegativeNumber(usage, "cacheWrite");
  const hasCacheSignal = cacheRead !== undefined || cacheWrite !== undefined;

  if (!hasCacheSignal && (input === undefined || !allowInputOnly)) return undefined;

  // Under healthy runtime normalization input is the uncached portion, so
  // totalInput = input + cacheRead + cacheWrite gives the full prompt token count.
  // Guard against degenerate reads where a broken proxy omits prompt_tokens and
  // normalized input falls to zero: totalInput must never be less than cacheRead + cacheWrite.
  const computed = (input ?? 0) + (cacheRead ?? 0) + (cacheWrite ?? 0);
  const floor = (cacheRead ?? 0) + (cacheWrite ?? 0);
  return {
    cacheRead: cacheRead ?? 0,
    cacheWrite: cacheWrite ?? 0,
    totalInput: computed >= floor ? computed : floor,
  };
}

// Raw fallback for DeepSeek responses that still carry their native usage fields.
// In practice the runtime normalizes usage before message_end fires, so this path is only
// reached when normalized fields are absent (e.g. custom/foreign providers).
function getDeepSeekRawUsage(message: unknown): UsageSnapshot | undefined {
  const usage = usageRecordFromAssistant(message);
  if (!usage) return undefined;

  const cacheRead = getFirstNonNegativeNumber(usage.prompt_cache_hit_tokens);
  if (cacheRead === undefined) return undefined;

  const cacheMiss = getFirstNonNegativeNumber(usage.prompt_cache_miss_tokens);
  const promptTokens = getFirstNonNegativeNumber(usage.prompt_tokens);
  // DeepSeek guarantees prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens.
  const totalInput = promptTokens ?? cacheRead + (cacheMiss ?? 0);

  return { cacheRead, cacheWrite: 0, totalInput };
}

// Raw fallback for OpenAI-family responses that still carry their native usage fields.
// In practice the runtime normalizes usage before message_end fires, so this path is only
// reached when normalized fields are absent (e.g. custom/foreign providers).
function getOpenAIRawUsage(message: unknown): UsageSnapshot | undefined {
  const usage = usageRecordFromAssistant(message);
  if (!usage) return undefined;

  const promptDetails = getNestedRecord(usage, "prompt_tokens_details") ?? getNestedRecord(usage, "promptTokensDetails");
  const inputDetails = getNestedRecord(usage, "input_tokens_details") ?? getNestedRecord(usage, "inputTokensDetails");
  const cacheRead = readCachedTokensFromDetails(promptDetails) ?? readCachedTokensFromDetails(inputDetails);
  if (cacheRead === undefined) return undefined;

  const cacheWrite = readCacheWriteFromDetails(promptDetails) ?? readCacheWriteFromDetails(inputDetails) ?? 0;
  const totalInput = getFirstNonNegativeNumber(
    usage.prompt_tokens,
    usage.promptTokens,
    usage.input_tokens,
    usage.inputTokens,
  ) ?? cacheRead + cacheWrite;

  return { cacheRead, cacheWrite, totalInput };
}

// Raw fallback for Anthropic/Claude responses that still carry their native usage fields.
// In practice the runtime normalizes usage before message_end fires, so this path is only
// reached when normalized fields are absent (e.g. custom/foreign providers).
function getAnthropicRawUsage(message: unknown): UsageSnapshot | undefined {
  const usage = usageRecordFromAssistant(message);
  if (!usage) return undefined;

  const cacheRead = getFirstNonNegativeNumber(usage.cache_read_input_tokens, usage.cacheReadInputTokens);
  const cacheWrite = getFirstNonNegativeNumber(usage.cache_creation_input_tokens, usage.cacheCreationInputTokens);
  if (cacheRead === undefined && cacheWrite === undefined) return undefined;

  // Anthropic input_tokens = tokens after the last cache breakpoint (neither read nor written).
  const input = getFirstNonNegativeNumber(usage.input_tokens, usage.inputTokens) ?? 0;
  return {
    cacheRead: cacheRead ?? 0,
    cacheWrite: cacheWrite ?? 0,
    totalInput: input + (cacheRead ?? 0) + (cacheWrite ?? 0),
  };
}

// Raw fallback for Gemini/Vertex responses that still carry their native usage fields.
// In practice the runtime normalizes usage before message_end fires, so this path is only
// reached when normalized fields are absent (e.g. custom/foreign providers).
function getGeminiRawUsage(message: unknown): UsageSnapshot | undefined {
  const record = getAssistantRecord(message);
  if (!record) return undefined;

  const usage = asRecord(record.usage);
  const metadata =
    getNestedRecord(record, "usageMetadata") ??
    getNestedRecord(record, "usage_metadata") ??
    getNestedRecord(usage, "usageMetadata") ??
    getNestedRecord(usage, "usage_metadata") ??
    usage;
  if (!metadata) return undefined;

  const cacheRead = getFirstNonNegativeNumber(
    metadata.cachedContentTokenCount,
    metadata.cached_content_token_count,
  );
  if (cacheRead === undefined) return undefined;

  const totalInput = getFirstNonNegativeNumber(
    metadata.promptTokenCount,
    metadata.prompt_token_count,
    metadata.inputTokenCount,
    metadata.input_token_count,
    usage?.input_tokens,
    usage?.inputTokens,
    usage?.prompt_tokens,
    usage?.promptTokens,
  ) ?? cacheRead;

  return { cacheRead, cacheWrite: 0, totalInput };
}

// Try normalized usage first (always present for messages that went through the runtime's
// provider pipeline). Fall back to provider-specific raw-field readers when normalized
// fields are absent (e.g. messages from custom/foreign providers whose raw usage shape
// matches the official API).
function normalizeWithFallback(
  message: unknown,
  rawNormalizer: (message: unknown) => UsageSnapshot | undefined,
  options: { allowInputOnlyPiUsage?: boolean } = {},
): UsageSnapshot | undefined {
  return getPiNormalizedUsage(message, options.allowInputOnlyPiUsage) ?? rawNormalizer(message);
}

// ── 系统 prompt 提取/写回（payload 安全网） ──
//
// OMP 17：主重写在 before_agent_start（string[] 块）。
// extractSystemPrompt / setSystemPrompt 仍用于 before_provider_request 的
// session-overview 安全网与冒烟测试。payload 形态：
//   - openai-completions / openai-responses: payload.messages[] role=system
//   - anthropic-messages: payload.system（字符串或块数组）
//   - google-generative-ai: payload.systemInstruction
function extractSystemPrompt(payload: unknown): string | undefined {
  const record = asRecord(payload);
  if (!record) return undefined;

  // anthropic-messages: payload.system (string or content blocks array)
  const systemField = record.system;
  if (typeof systemField === "string") return systemField;
  if (Array.isArray(systemField)) {
    return systemField
      .map((block) => {
        const r = asRecord(block);
        if (!r) return "";
        if (typeof r.text === "string") return r.text;
        return "";
      })
      .join("\n")
      .trim() || undefined;
  }

  // google-generative-ai: payload.systemInstruction
  const systemInstruction = asRecord(record.systemInstruction);
  if (systemInstruction) {
    const parts = systemInstruction.parts;
    if (Array.isArray(parts)) {
      const text = parts
        .map((p) => {
          const r = asRecord(p);
          return typeof r?.text === "string" ? r.text : "";
        })
        .join("\n")
        .trim();
      if (text) return text;
    }
  }

  // openai-completions / openai-responses: payload.messages[] first system message
  const messages = record.messages;
  if (Array.isArray(messages)) {
    for (const msg of messages) {
      const r = asRecord(msg);
      if (!r) continue;
      if (r.role === "system" || r.role === "developer") {
        if (typeof r.content === "string") return r.content;
        if (Array.isArray(r.content)) {
          const text = r.content
            .map((c) => {
              const cr = asRecord(c);
              return typeof cr?.text === "string" ? cr.text : "";
            })
            .join("\n")
            .trim();
          if (text) return text;
        }
      }
    }
  }

  return undefined;
}

function rewriteTextBlockArray(
  items: unknown[],
  text: string,
  makeFallback: (text: string) => Record<string, unknown>,
): unknown[] {
  const rewritten: unknown[] = [];
  let replacedText = false;

  for (const item of items) {
    const record = asRecord(item);
    if (typeof record?.text === "string") {
      if (!replacedText) {
        rewritten.push({ ...record, text });
        replacedText = true;
      }
      continue;
    }
    rewritten.push(item);
  }

  if (!replacedText) {
    return [makeFallback(text), ...items];
  }

  return rewritten;
}

function setSystemPrompt(payload: unknown, text: string): boolean {
  const record = asRecord(payload);
  if (!record) return false;

  // anthropic-messages: payload.system
  if (typeof record.system === "string") {
    record.system = text;
    return true;
  }
  if (Array.isArray(record.system) && record.system.length > 0) {
    record.system = rewriteTextBlockArray(record.system, text, (value) => ({ type: "text", text: value }));
    return true;
  }

  // google-generative-ai: payload.systemInstruction
  const systemInstruction = asRecord(record.systemInstruction);
  if (systemInstruction && Array.isArray(systemInstruction.parts) && systemInstruction.parts.length > 0) {
    systemInstruction.parts = rewriteTextBlockArray(systemInstruction.parts, text, (value) => ({ text: value }));
    return true;
  }

  // openai-completions / openai-responses: payload.messages[] first system/developer message
  const messages = record.messages;
  if (Array.isArray(messages)) {
    for (const msg of messages) {
      const r = asRecord(msg);
      if (!r) continue;
      if (r.role === "system" || r.role === "developer") {
        if (typeof r.content === "string") {
          r.content = text;
          return true;
        }
        if (Array.isArray(r.content) && r.content.length > 0) {
          r.content = rewriteTextBlockArray(r.content, text, (value) => ({ type: "text", text: value }));
          return true;
        }
      }
    }
  }

  return false;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isOfficialOpenAIBaseUrl(model: PiModel): boolean {
  const value = lower(model.baseUrl).trim();
  if (!value) {
    return lower(model.provider) === "openai";
  }

  try {
    return new URL(value).hostname === "api.openai.com";
  } catch {
    return value === "api.openai.com" || value.startsWith("api.openai.com/");
  }
}

function describeMissingOpenAIFamilyProxyCompat(_model: PiModel): string[] {
  // OMP divergence: the legacy `sendSessionAffinityHeaders` flag has no compat equivalent.
  // OMP achieves upstream stickiness via multi-credential auth + session affinity
  // in agent.db (see omp models.md §Auth). There is no required compat key for
  // OpenAI-family proxies on OMP, so this returns an empty list. Optional long
  // cache retention is reported separately by describeOptionalOpenAICompatibleProxyCompat.
  return [];
}

/**
 * Like describeMissingOpenAIFamilyProxyCompat but without the isOpenAIFamilyModel
 * gate. Warns for ANY model using openai-completions through a non-official base
 * URL — covers GPT, Kimi, Qwen, GLM, MiniMax, Mimo, Hunyuan, and any other
 * OpenAI-compatible proxy.
 */
function describeMissingOpenAICompatibleProxyCompat(_model: PiModel): string[] {
  // OMP divergence: no required compat key for OpenAI-compatible proxies.
  // See describeMissingOpenAIFamilyProxyCompat for rationale.
  return [];
}

function describeOptionalOpenAICompatibleProxyCompat(model: PiModel): string[] {
  const compat = getCompat(model);
  const optional: string[] = [];

  if (!isOpenAICompatibleProxyApi(model.api)) return optional;
  if (isOfficialOpenAIBaseUrl(model)) return optional;

  if (compat.supportsLongPromptCacheRetention !== true) {
    optional.push("supportsLongPromptCacheRetention");
  }

  return optional;
}

function buildSafeOpenAIProxyCompatSuggestion(_missing: string[]): Record<string, boolean> {
  // OMP divergence: no safe auto-fixable compat key for OpenAI-compatible proxies.
  // supportsLongPromptCacheRetention is optional and can cause 400s, so it is NOT
  // auto-fixed; users must add it manually after confirming upstream support.
  return {};
}

function getPromptCacheRetentionUnsupportedHint(): string {
  return "如果这个渠道返回 `400 Unsupported parameter: prompt_cache_retention`，请移除或避免 `supportsLongPromptCacheRetention`；扩展本身不会直接写这个字段，但当 compat 声明支持长缓存保留时，OMP 可能会发送它。";
}

function hasPromptCacheRetentionUnsupportedSignal(headers: Record<string, string> | undefined): boolean {
  if (!headers) return false;

  const normalized = Object.entries(headers)
    .map(([key, value]) => `${lower(key)}: ${lower(value)}`)
    .join("\n");
  if (!normalized.includes("prompt_cache_retention")) return false;

  return [
    "unsupported parameter",
    "unsupported_parameter",
    "unknown parameter",
    "not supported",
    "unsupported field",
  ].some((needle) => normalized.includes(needle));
}

type CompatAdvicePlacement = {
  providerLabel?: string;
  modelId?: string;
};

function buildProviderCompatOverride(providerLabel: string, compat: Record<string, unknown>): Record<string, unknown> {
  return {
    providers: {
      [providerLabel]: {
        compat,
      },
    },
  };
}

function buildModelCompatOverride(providerLabel: string, modelId: string, compat: Record<string, unknown>): Record<string, unknown> {
  return {
    providers: {
      [providerLabel]: {
        modelOverrides: {
          [modelId]: {
            compat,
          },
        },
      },
    },
  };
}

function appendCredentialSafeProviderGuidance(lines: string[], placement: CompatAdvicePlacement, compatSuggestion: Record<string, unknown>): void {
  const providerLabel = placement.providerLabel;
  if (!providerLabel) return;

  lines.push("");
  lines.push("如果这个渠道在 models.yml 里还没有 provider 配置：");
  lines.push("- 保留现有认证方式；不要复制 credential、token 或 API key。");
  lines.push(`- 只在 ${getModelsYmlDisplayPath()} 里添加缓存/路由 compat 覆盖。`);

  if (Object.keys(compatSuggestion).length === 0) {
    lines.push("- 上面这些缺失项目前没有安全可复制的 override。");
    return;
  }

  lines.push("Provider 级最小覆盖：");
  lines.push(JSON.stringify(buildProviderCompatOverride(providerLabel, compatSuggestion), null, 2));

  if (placement.modelId) {
    lines.push("单模型 override（只想影响当前模型时使用）：");
    lines.push(JSON.stringify(buildModelCompatOverride(providerLabel, placement.modelId, compatSuggestion), null, 2));
  }
}

function appendOpenAIProxyCompatAdviceLines(lines: string[], missing: string[], options: { includeJsonIntro?: boolean } & CompatAdvicePlacement = {}): void {
  const suggestion = buildSafeOpenAIProxyCompatSuggestion(missing);
  const hasSafeSuggestion = Object.keys(suggestion).length > 0;

  if (hasSafeSuggestion) {
    if (options.includeJsonIntro !== false) {
      lines.push("安全默认建议：");
    }
    lines.push(JSON.stringify(suggestion, null, 2));
  }

  appendCredentialSafeProviderGuidance(lines, options, suggestion);
}

function appendOptionalOpenAIProxyCompatAdviceLines(lines: string[], optional: string[]): void {
  if (!optional.includes("supportsLongPromptCacheRetention")) return;
  lines.push("");
  lines.push("可选项（非必需，不会自动修复）：");
  lines.push("- supportsLongPromptCacheRetention：仅当 endpoint / proxy 明确支持 OpenAI long prompt cache retention 时再开启。");
  lines.push(`- ${getPromptCacheRetentionUnsupportedHint()}`);
}

/**
 * Build the warning text displayed to users when an OpenAI-family third-party
 * proxy is missing one or more cache/session-affinity compat flags.
 *
 * The returned string contains a parseable JSON object (via JSON.stringify)
 * listing only the missing flags with recommended value `true`. Inline
 * explanations for each flag follow the JSON snippet as separate prose lines,
 * so the JSON remains valid and copyable.
 *
 * Expected use: the openai adapter's warningText calls this function; tests
 * exercise it via __internals_for_tests.
 */
function buildOpenAIProxyCompatWarningText(key: string, missing: string[]): string {
  const slashIdx = key.indexOf("/");
  const providerLabel = slashIdx > 0 ? key.slice(0, slashIdx) : key;
  const modelId = slashIdx > 0 ? key.slice(slashIdx + 1) : undefined;

  const modelsJsonPath = getModelsYmlDisplayPath();
  const lines: string[] = [
    `💡 omp-cache-optimizer：${key} 是第三方 GPT/OpenAI 兼容代理，但合并后的 compat 缺少 ${missing.join(" 和 ")}。`,
    `编辑 ${modelsJsonPath} -> providers["${providerLabel}"] -> compat（与 baseUrl/api/apiKey/models 同级）。`,
    "",
  ];

  appendOpenAIProxyCompatAdviceLines(lines, missing, { providerLabel, modelId });

  return lines.join("\n");
}

function describeMissingDeepSeekCompat(model: PiModel): string[] {
  const compat = getCompat(model);
  const missing: string[] = [];

  // OMP divergence: field names remapped (see CacheCompat).
  //   supportsLongCacheRetention                  -> supportsLongPromptCacheRetention
  //   requiresReasoningContentOnAssistantMessages -> requiresReasoningContentForToolCalls
  //   sendSessionAffinityHeaders / sendSessionIdHeader -> removed (OMP multi-credential auth)
  //   thinkingFormat: "deepseek"                  -> not a valid OMP value; OMP uses
  //     openai|openrouter|zai|qwen|qwen-chat-template. DeepSeek reasoning format is
  //     auto-detected by OMP's openai-completions transport, so we no longer flag it.
  if (compat.supportsLongPromptCacheRetention !== true) {
    missing.push("supportsLongPromptCacheRetention");
  }
  if (compat.requiresReasoningContentForToolCalls !== true) {
    missing.push("requiresReasoningContentForToolCalls");
  }

  return missing;
}

function isDeepSeekCompatCheckApplicable(model: PiModel): boolean {
  return isDeepSeekLikeModel(model) && isOpenAICompatibleApi(model.api);
}

function describeMissingCacheCompatForModel(model: PiModel): string[] {
  if (isAdaptiveThinkingCompatApplicable(model)) {
    return describeMissingAdaptiveThinkingCompat(model);
  }
  if (isDeepSeekCompatCheckApplicable(model)) {
    return describeMissingDeepSeekCompat(model);
  }
  return describeMissingOpenAICompatibleProxyCompat(model);
}

function buildDeepSeekCompatSuggestion(missing: string[]): Record<string, unknown> {
  const suggestion: Record<string, unknown> = {};

  if (missing.includes("supportsLongPromptCacheRetention")) {
    suggestion.supportsLongPromptCacheRetention = true;
  }
  if (missing.includes("requiresReasoningContentForToolCalls")) {
    suggestion.requiresReasoningContentForToolCalls = true;
  }

  return suggestion;
}

function appendDeepSeekCompatAdviceLines(lines: string[], missing: string[], placement: CompatAdvicePlacement = {}): void {
  const suggestion = buildDeepSeekCompatSuggestion(missing);
  if (Object.keys(suggestion).length > 0) {
    lines.push("推荐的 DeepSeek compat 片段：");
    lines.push(JSON.stringify(suggestion, null, 2));
  }

  if (missing.includes("requiresReasoningContentForToolCalls")) {
    lines.push("- requiresReasoningContentForToolCalls：保持带工具调用的 assistant 重放与 DeepSeek 的 reasoning_content 要求兼容。");
  }
  if (missing.includes("supportsLongPromptCacheRetention")) {
    lines.push("- supportsLongPromptCacheRetention：仅当 DeepSeek 兼容 endpoint 支持长缓存保留时再开启。");
  }

  appendCredentialSafeProviderGuidance(lines, placement, suggestion);
}

function buildDeepSeekCompatWarningText(key: string, missing: string[]): string {
  const slashIdx = key.indexOf("/");
  const providerLabel = slashIdx > 0 ? key.slice(0, slashIdx) : key;
  const modelId = slashIdx > 0 ? key.slice(slashIdx + 1) : undefined;
  const modelsJsonPath = getModelsYmlDisplayPath();
  const lines: string[] = [
    `💡 omp-cache-optimizer：${key} 看起来是 DeepSeek 风格模型，但合并后的 compat 缺少 ${missing.join(" 和 ")}。`,
    `这可能让代理降低或隐藏缓存命中。编辑 ${modelsJsonPath} -> providers["${providerLabel}"] -> compat（与 baseUrl/api/apiKey/models 同级）。`,
    "",
  ];

  appendDeepSeekCompatAdviceLines(lines, missing, { providerLabel, modelId });

  return lines.join("\n");
}

const CACHE_PROVIDER_ADAPTERS: CacheProviderAdapter[] = [
  {
    id: "deepseek",
    label: "DS cache",
    matchesModel: isDeepSeekLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isDeepSeekLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getDeepSeekRawUsage, { allowInputOnlyPiUsage: true });
    },
    warningText(model) {
      if (!isDeepSeekLikeModel(model) || !isOpenAICompatibleApi(model.api)) return undefined;

      const missing = describeMissingDeepSeekCompat(model);
      if (missing.length === 0) return undefined;

      const key = modelKey(model);
      return buildDeepSeekCompatWarningText(key, missing);
    },
  },
  {
    id: "claude",
    label: "Claude cache",
    showCacheWrite: true,
    matchesModel: isClaudeLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isClaudeLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getAnthropicRawUsage);
    },
    warningText(model) {
      if (!isClaudeLikeModel(model) || !isOpenAICompatibleApi(model.api)) return undefined;
      if (getCompat(model).cacheControlFormat === "anthropic") return undefined;

      return (
        `💡 omp-cache-optimizer：${modelKey(model)} 看起来是 Claude/Anthropic 风格模型，但 OpenAI 兼容 compat 缺少 cacheControlFormat: "anthropic"。` +
        "只有当 endpoint 支持并启用了这个 compat 字段时，OMP 才能放置 Anthropic 的 cache_control 断点。"
      );
    },
  },
  {
    id: "openai",
    label: "OpenAI cache",
    matchesModel: isOpenAIFamilyModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isOpenAIFamilyAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "gemini",
    label: "Gemini cache",
    matchesModel: isGeminiLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isGeminiLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getGeminiRawUsage);
    },
  },
  // ── Non-GPT OpenAI-compatible adapters ──────────────────────
  {
    id: "openai" as CacheProviderId,
    label: "Kimi cache",
    matchesModel: isKimiLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isKimiLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Qwen cache",
    matchesModel: isQwenLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isQwenLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "GLM cache",
    matchesModel: isGLMLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isGLMLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "MiniMax cache",
    matchesModel: isMiniMaxLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isMiniMaxLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Mimo cache",
    matchesModel: isMimoLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isMimoLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Hunyuan cache",
    matchesModel: isHunyuanLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isHunyuanLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  // ── More OpenAI-compatible adapters ──────────────────────────
  {
    id: "openai" as CacheProviderId,
    label: "Mistral cache",
    matchesModel: isMistralLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isMistralLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Grok cache",
    matchesModel: isGrokLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isGrokLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Llama cache",
    matchesModel: isLlamaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isLlamaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Nemotron cache",
    matchesModel: isNemotronLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isNemotronLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Cohere cache",
    matchesModel: isCohereLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isCohereLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Yi cache",
    matchesModel: isYiLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isYiLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  // ── More OpenAI-compatible adapters (batch 2) ───────────────────
  {
    id: "openai" as CacheProviderId,
    label: "Doubao cache",
    matchesModel: isDoubaoLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isDoubaoLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "ERNIE cache",
    matchesModel: isErnieLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isErnieLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Baichuan cache",
    matchesModel: isBaichuanLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isBaichuanLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "StepFun cache",
    matchesModel: isStepFunLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isStepFunLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Spark cache",
    matchesModel: isSparkLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isSparkLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "InternLM cache",
    matchesModel: isInternLMLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isInternLMLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Gemma cache",
    matchesModel: isGemmaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isGemmaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Phi cache",
    matchesModel: isPhiLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isPhiLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Jamba cache",
    matchesModel: isJambaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isJambaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Solar cache",
    matchesModel: isSolarLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isSolarLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  // ── New OpenAI-compatible adapters (batch 3, 12 families) ────────
  {
    id: "openai" as CacheProviderId,
    label: "Sonar cache",
    matchesModel: isPerplexityLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isPerplexityLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Nova cache",
    matchesModel: isNovaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isNovaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Reka cache",
    matchesModel: isRekaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isRekaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Falcon cache",
    matchesModel: isFalconLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isFalconLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "DBRX cache",
    matchesModel: isDbrxLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isDbrxLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "MPT cache",
    matchesModel: isMptLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isMptLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "StableLM cache",
    matchesModel: isStableLMLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isStableLMLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Aquila cache",
    matchesModel: isAquilaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isAquilaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "EXAONE cache",
    matchesModel: isExaoneLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isExaoneLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "HyperCLOVA cache",
    matchesModel: isHyperCLOVALikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isHyperCLOVALikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Luminous cache",
    matchesModel: isLuminousLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isLuminousLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Hermes cache",
    matchesModel: isHermesLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isHermesLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  // ── More OpenAI-compatible adapters (batch 4, 18 families) ────────
  {
    id: "openai" as CacheProviderId,
    label: "Granite cache",
    matchesModel: isGraniteLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isGraniteLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Arctic cache",
    matchesModel: isArcticLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isArcticLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Pangu cache",
    matchesModel: isPanguLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isPanguLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "SenseNova cache",
    matchesModel: isSenseNovaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isSenseNovaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Zhinao cache",
    matchesModel: isZhinaoLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isZhinaoLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "MiniCPM cache",
    matchesModel: isMiniCPMLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isMiniCPMLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "XVERSE cache",
    matchesModel: isXVerseLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isXVerseLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Orion cache",
    matchesModel: isOrionLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isOrionLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "OpenChat cache",
    matchesModel: isOpenChatLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isOpenChatLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Vicuna cache",
    matchesModel: isVicunaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isVicunaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Wizard cache",
    matchesModel: isWizardLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isWizardLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Zephyr cache",
    matchesModel: isZephyrLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isZephyrLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Dolphin cache",
    matchesModel: isDolphinLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isDolphinLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "OpenOrca cache",
    matchesModel: isOpenOrcaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isOpenOrcaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Starling cache",
    matchesModel: isStarlingLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isStarlingLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "BLOOM cache",
    matchesModel: isBloomLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isBloomLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "RWKV cache",
    matchesModel: isRwkvLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isRwkvLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
  {
    id: "openai" as CacheProviderId,
    label: "Aya cache",
    matchesModel: isAyaLikeModel,
    matchesAssistantMessage(message, model) {
      if (!isAssistantMessage(message)) return false;
      return isAyaLikeAssistantMessage(message, model);
    },
    normalizeUsage(message) {
      return normalizeWithFallback(message, getOpenAIRawUsage);
    },
    warningText(model) {
      const missing = describeMissingOpenAICompatibleProxyCompat(model);
      if (missing.length === 0) return undefined;
      return buildOpenAIProxyCompatWarningText(modelKey(model), missing);
    },
  },
];

function selectAdapterForModel(model: PiModel | undefined): CacheProviderAdapter | undefined {
  return CACHE_PROVIDER_ADAPTERS.find((adapter) => adapter.matchesModel(model));
}

function selectAdapterForAssistantMessage(message: unknown, model: PiModel | undefined): CacheProviderAdapter | undefined {
  // Assistant message metadata is request-local and authoritative for virtual
  // routing providers. Use it first for every model; direct providers normally
  // echo the same provider/model and therefore remain unchanged.
  const responseModel = modelFromAssistantMessage(message, model);
  return CACHE_PROVIDER_ADAPTERS.find((adapter) => adapter.matchesAssistantMessage(message, responseModel));
}

function notifyCacheCompatIfNeeded(
  model: PiModel | undefined,
  ctx: ExtensionContext,
  warnedModels: Set<string>,
): void {
  if (!model) return;

  // OMP divergence: adaptive thinking is set by the OMP built-in model catalog and
  // is not user-configurable, so the native anthropic-messages compat check is gone.
  // We only surface adapter warnings for OpenAI-compatible proxy compat gaps.
  const adapter = selectAdapterForModel(model);
  const text = adapter?.warningText?.(model);
  if (!adapter || !text) return;

  const key = `${adapter.id}:${modelKey(model)}`;
  if (warnedModels.has(key)) return;
  warnedModels.add(key);

  ctx.ui.notify(text, "warning");
}

function currentLocalDay(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function emptyCacheStats(day = currentLocalDay()): CacheStats {
  return {
    day,
    totalRequests: 0,
    hitRequests: 0,
    cachedInputTokens: 0,
    cacheWriteInputTokens: 0,
    totalInputTokens: 0,
  };
}

function emptyAllCacheStats(day = currentLocalDay()): Partial<Record<CacheProviderId, CacheStats>> {
  return Object.fromEntries(CACHE_PROVIDER_IDS.map((id) => [id, emptyCacheStats(day)])) as Partial<Record<CacheProviderId, CacheStats>>;
}

function addUsageToCacheStats(stats: CacheStats, usage: UsageSnapshot): void {
  stats.totalRequests += 1;
  if (usage.cacheRead > 0) stats.hitRequests += 1;
  stats.cachedInputTokens += usage.cacheRead;
  stats.cacheWriteInputTokens += usage.cacheWrite;
  stats.totalInputTokens += usage.totalInput;
}

function formatTokenCount(value: number): string {
  const n = Math.max(0, Math.round(value));
  if (n === 0) return "0";
  if (n < 1000) return String(n);
  if (n < 10_000) return `${(n / 1000).toFixed(2)}k`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  const millions = n / 1_000_000;
  if (millions < 10) return `${millions.toFixed(2)}M`;
  return `${millions.toFixed(1)}M`;
}

function localizeAdapterLabel(label: string): string {
  return label.endsWith(" cache") ? `${label.slice(0, -6)} Cache` : label;
}

/** token 命中率：cachedInputTokens / totalInputTokens，0–100 整数百分比。 */
function formatTokenHitPercent(stats: CacheStats): string {
  if (stats.totalInputTokens <= 0) return "—";
  return `${Math.round((stats.cachedInputTokens / stats.totalInputTokens) * 100)}%`;
}
/**
 * Footer 文案：三项均带中文说明，主指标为 token 命中率。
 * 例：`OpenAI Cache | 缓存命中率：40% | 缓存请求命中次数：1/2 次 | 缓存token/总输入：800/2.00k`
 */
function formatCacheStats(adapter: CacheProviderAdapter, stats: CacheStats): string {
  const tokenHit = formatTokenHitPercent(stats);
  const tok = `${formatTokenCount(stats.cachedInputTokens)}/${formatTokenCount(stats.totalInputTokens)}`;
  const req = `${stats.hitRequests}/${stats.totalRequests}`;
  const writeText = adapter.showCacheWrite && stats.cacheWriteInputTokens > 0
    ? ` | 写入token：${formatTokenCount(stats.cacheWriteInputTokens)}`
    : "";

  return (
    `${localizeAdapterLabel(adapter.label)}` +
    ` | 缓存命中率：${tokenHit}` +
    ` | 缓存请求命中次数：${req} 次` +
    ` | 缓存token/总输入：${tok}` +
    writeText
  );
}

function formatHitRatio(hits: number, total: number): string {
  if (total <= 0) return "无数据";
  return `${Math.round((hits / total) * 100)}%`;
}

function formatTokenM(value: number): string {
  const millions = Math.max(0, Math.round(value)) / 1_000_000;
  if (millions === 0) return "0";
  if (millions < 0.01) return millions.toFixed(4);
  if (millions >= 10) return millions.toFixed(1);
  return millions.toFixed(2);
}

function hasMissingUsageFields(message: unknown, adapter: CacheProviderAdapter): boolean {
  const usage = usageRecordFromAssistant(message);
  if (!usage) return true;

  const input = getNonNegativeNumber(usage, "input");
  const cacheRead = getNonNegativeNumber(usage, "cacheRead");
  const cacheWrite = getNonNegativeNumber(usage, "cacheWrite");

  if (cacheRead !== undefined || cacheWrite !== undefined || (input !== undefined && input > 0)) {
    return false;
  }

  const rawUsage = adapter.normalizeUsage(message);
  if (!rawUsage || (rawUsage.cacheRead === 0 && rawUsage.cacheWrite === 0 && rawUsage.totalInput === 0)) {
    return true;
  }

  return false;
}

function formatRecentPromptDiagnostics(samples: CacheUsageSample[], maxCount: number): string | undefined {
  const recent = samples.slice(-maxCount);
  if (recent.length === 0) return undefined;
  const rewriteEnabled = recent.filter((sample) => sample.promptRewriteEnabled).length;
  const fingerprints = new Set(recent.flatMap((sample) => sample.systemPromptFingerprint ? [sample.systemPromptFingerprint] : []));
  const headerKeys = recent.filter((sample) => sample.promptCacheKeySource === "header").length;
  const sessionKeys = recent.filter((sample) => sample.promptCacheKeySource === "session").length;
  const mismatches = recent.filter((sample) => sample.hintPayloadComparison === "mismatch").length;
  const unavailable = recent.filter((sample) => sample.hintPayloadComparison === "unavailable").length;
  let result = `Prompt 诊断（最近 ${recent.length} 次）：改写开启 ${rewriteEnabled}/${recent.length} · system 指纹 ${fingerprints.size} 组 · cache key 来源 header ${headerKeys} / session ${sessionKeys}`;
  if (mismatches > 0) result += ` · hint/payload 不一致 ${mismatches}`;
  if (unavailable > 0) result += ` · 不可比较 ${unavailable}`;
  return result;
}

function formatRecentTrendSummary(samples: CacheUsageSample[], maxCount: number): string {
  const recent = samples.slice(-maxCount);
  if (recent.length === 0) return `最近 ${maxCount} 次：暂无样本`;

  const hits = recent.filter((s) => s.hit).length;
  const totalCached = recent.reduce((sum, s) => sum + s.cachedInputTokens, 0);
  const totalInput = recent.reduce((sum, s) => sum + s.totalInputTokens, 0);
  const missingCount = recent.filter((s) => s.missingUsageFields).length;

  const tokenRatio = totalInput > 0 ? formatHitRatio(totalCached, totalInput) : "无数据";

  let result = `最近 ${recent.length}/${maxCount} 次：${hits}/${recent.length} 次命中 · ${tokenRatio} tok 已缓存`;
  if (missingCount > 0) {
    result += ` · ${missingCount} 条 usage 缺失`;
  }
  return result;
}

function buildStatsOutput(model: PiModel | undefined, adapter: CacheProviderAdapter | undefined, stats: CacheStats | undefined, recentSamples: CacheUsageSample[]): string {
  const lines: string[] = [];

  if (!model || !adapter) {
    lines.push("ℹ️ 当前活动模型未匹配到缓存适配器。请选择可识别模型家族后再查看统计。");
    return lines.join("\n");
  }

  const key = modelKey(model);
  const currentStats = stats ?? emptyCacheStats();

  lines.push(`模型键：${key}`);
  lines.push(`适配器：${localizeAdapterLabel(adapter.label)}`);
  lines.push("");
  lines.push("── 今日 ──");
  lines.push(`请求数：${currentStats.hitRequests} 次命中 / ${currentStats.totalRequests} 次总计 · ${formatHitRatio(currentStats.hitRequests, currentStats.totalRequests)}`);
  lines.push(`缓存 tokens：${formatTokenM(currentStats.cachedInputTokens)}M / ${formatTokenM(currentStats.totalInputTokens)}M 输入 · ${currentStats.totalInputTokens > 0 ? `${Math.round((currentStats.cachedInputTokens / currentStats.totalInputTokens) * 100)}%` : "无数据"}`);
  if (currentStats.cacheWriteInputTokens > 0) {
    lines.push(`缓存写入：${formatTokenM(currentStats.cacheWriteInputTokens)}M tok`);
  }

  const promptDiagnostics = formatRecentPromptDiagnostics(recentSamples, 10);
  if (promptDiagnostics) {
    lines.push("");
    lines.push(promptDiagnostics);
    lines.push("仅显示截断哈希和来源统计，不保存 prompt 或 cache key 原文。");
  }

  lines.push("");
  lines.push("── 近期趋势 ──");
  lines.push(formatRecentTrendSummary(recentSamples, 10));
  lines.push(formatRecentTrendSummary(recentSamples, 30));

  const missingAny = recentSamples.some((s) => s.missingUsageFields);
  if (missingAny) {
    lines.push("");
    lines.push("⚠️ 近期有响应缺少或返回了空的缓存 usage 字段，footer 命中率可能偏低。");
    lines.push("   代理可能没有返回 prompt_cache_hit_tokens，或没有返回 usage.input/cacheRead 等字段。");
  }

  return lines.join("\n");
}

function getErrorCode(error: unknown): string | undefined {
  return typeof error === "object" && error !== null && "code" in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}

function parseCacheStats(value: unknown): CacheStats | undefined {
  const stats = asRecord(value);
  if (!stats || typeof stats.day !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(stats.day)) {
    return undefined;
  }

  const totalRequests = getNonNegativeNumber(stats, "totalRequests");
  const hitRequests = getNonNegativeNumber(stats, "hitRequests");
  const cachedInputTokens = getNonNegativeNumber(stats, "cachedInputTokens");
  const cacheWriteInputTokens = getNonNegativeNumber(stats, "cacheWriteInputTokens") ?? 0;
  const totalInputTokens = getNonNegativeNumber(stats, "totalInputTokens");

  if (
    totalRequests === undefined ||
    hitRequests === undefined ||
    cachedInputTokens === undefined ||
    totalInputTokens === undefined ||
    hitRequests > totalRequests ||
    cachedInputTokens > totalInputTokens ||
    cacheWriteInputTokens > totalInputTokens
  ) {
    return undefined;
  }

  return {
    day: stats.day,
    totalRequests,
    hitRequests,
    cachedInputTokens,
    cacheWriteInputTokens,
    totalInputTokens,
  };
}

function parsePersistedRoutedModelRef(value: unknown): PersistedRoutedModelRef | undefined {
  const record = asRecord(value);
  const provider = record?.provider;
  const id = record?.id;
  const name = record?.name;
  if (!isNonEmptyString(provider) || !isNonEmptyString(id)) return undefined;

  return {
    provider: provider.trim(),
    id: id.trim(),
    name: isNonEmptyString(name) ? name.trim() : id.trim(),
  };
}

function routedModelRefToPiModel(ref: PersistedRoutedModelRef): PiModel {
  return {
    id: ref.id,
    name: ref.name ?? ref.id,
    provider: ref.provider,
    api: "",
    baseUrl: "",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 0,
    maxTokens: 0,
  } as PiModel;
}

function buildExactRouterStatusEntry(
  sessionHash: string | undefined,
  statsByModel: Record<string, CacheStats>,
  lastRoutedModel: PersistedRoutedModelRef | undefined,
): { model: PiModel; adapter: CacheProviderAdapter; stats: CacheStats } | undefined {
  if (!sessionHash || !lastRoutedModel) return undefined;

  const model = routedModelRefToPiModel(lastRoutedModel);
  const adapter = selectAdapterForModel(model);
  if (!adapter) return undefined;

  const key = makeSessionModelKey(sessionHash, lastRoutedModel.provider, lastRoutedModel.id);
  return { model, adapter, stats: statsByModel[key] ?? emptyCacheStats() };
}

function parsePersistedCacheStats(value: unknown): CacheStatsState | undefined {
  const record = asRecord(value);
  if (!record) return undefined;

  // version 4/5: session-scoped stats + legacy family fallback.
  // v5 additionally persists the last actual routed model per session so
  // router/auto can restore the exact upstream footer after /reload.
  if (record.version === 4 || record.version === 5) {
    const legacyFamily: Partial<Record<CacheProviderId, CacheStats>> = {};
    const rawFamily = asRecord(record.legacyFamily);
    if (rawFamily) {
      for (const id of CACHE_PROVIDER_IDS) {
        const stats = parseCacheStats(rawFamily[id]);
        if (stats) legacyFamily[id] = stats;
      }
    }

    // Collect all session entries into statsByModel with session-hash-prefixed keys
    // (e.g. "abc123:otokapi/gpt-5.5") so that writePersistedCacheStats can later
    // reconstruct individual sessions from the flat key format and other sessions'
    // data is not silently lost on round-trip.
    const statsByModel: Record<string, CacheStats> = {};
    const rawSessions = asRecord(record.sessions);
    if (rawSessions) {
      for (const [sessionHash, modelMap] of Object.entries(rawSessions)) {
        const parsedMap = asRecord(modelMap);
        if (parsedMap) {
          for (const [modelKey, val] of Object.entries(parsedMap)) {
            const parsed = parseCacheStats(val);
            if (parsed) statsByModel[`${sessionHash}:${modelKey}`] = parsed;
          }
        }
      }
    }

    const lastRoutedModelBySession: Record<string, PersistedRoutedModelRef> = {};
    const rawLastRoutedModels = asRecord(record.lastRoutedModelBySession);
    if (rawLastRoutedModels) {
      for (const [sessionHash, rawModel] of Object.entries(rawLastRoutedModels)) {
        const parsed = parsePersistedRoutedModelRef(rawModel);
        if (parsed) lastRoutedModelBySession[sessionHash] = parsed;
      }
    }

    return { statsByModel, legacyFamily, lastRoutedModelBySession };
  }

  // version 3: migrate to v4/v5 semantics by wrapping statsByModel into sessions
  if (record.version === 3) {
    const statsByModel: Record<string, CacheStats> = {};
    const rawModelMap = asRecord(record.statsByModel);
    if (rawModelMap) {
      for (const [key, val] of Object.entries(rawModelMap)) {
        const parsed = parseCacheStats(val);
        if (parsed) statsByModel[key] = parsed;
      }
    }

    const legacyFamily: Partial<Record<CacheProviderId, CacheStats>> = {};
    const rawFamily = asRecord(record.legacyFamily);
    if (rawFamily) {
      for (const id of CACHE_PROVIDER_IDS) {
        const stats = parseCacheStats(rawFamily[id]);
        if (stats) legacyFamily[id] = stats;
      }
    }

    return { statsByModel, legacyFamily };
  }

  // version 2: migrate statsByProvider into legacyFamily
  if (record.version === 2) {
    const statsByProvider = asRecord(record.statsByProvider);
    const legacyFamily: Partial<Record<CacheProviderId, CacheStats>> = {};
    if (statsByProvider) {
      for (const id of CACHE_PROVIDER_IDS) {
        const stats = parseCacheStats(statsByProvider[id]);
        if (stats) legacyFamily[id] = stats;
      }
    }
    return { statsByModel: {}, legacyFamily };
  }

  // version 1: single DeepSeek stats -> migrate to legacyFamily.deepseek
  if (record.version === 1) {
    const migrated = parseCacheStats(record.stats);
    return migrated ? { statsByModel: {}, legacyFamily: { deepseek: migrated } } : undefined;
  }

  return undefined;
}

async function readPersistedCacheStats(): Promise<CacheStatsState | undefined> {
  try {
    const raw = await readFile(STATE_FILE_PATH, "utf8");
    return parsePersistedCacheStats(JSON.parse(raw));
  } catch (error) {
    if (getErrorCode(error) !== "ENOENT") {
      console.warn(`${LOG_PREFIX}: failed to read persisted cache stats`, error);
      return undefined;
    }
  }

  // New path missing: try one-shot migration from the old (pre-rename) path.
  try {
    const raw = await readFile(LEGACY_STATE_FILE_PATH, "utf8");
    const parsed = parsePersistedCacheStats(JSON.parse(raw));
    if (parsed) {
      try {
        await writePersistedCacheStats(parsed);
        // Best-effort delete; if the unlink fails the new path is still authoritative.
        try {
          await unlink(LEGACY_STATE_FILE_PATH);
        } catch (unlinkError) {
          if (getErrorCode(unlinkError) !== "ENOENT") {
            console.warn(`${LOG_PREFIX}: failed to remove legacy stats file`, unlinkError);
          }
        }
      } catch (writeError) {
        console.warn(`${LOG_PREFIX}: failed to migrate legacy cache stats`, writeError);
      }
      return parsed;
    }
  } catch (error) {
    if (getErrorCode(error) !== "ENOENT") {
      console.warn(`${LOG_PREFIX}: failed to read legacy cache stats`, error);
    }
  }

  return undefined;
}

function filterRestorableStatsForSession(
  persisted: CacheStatsState | undefined,
  currentSessionHash?: string,
): Record<string, CacheStats> {
  if (!persisted || !currentSessionHash) return {};

  const prefix = `${currentSessionHash}:`;
  const filteredModelStats: Record<string, CacheStats> = {};
  for (const [fullKey, stats] of Object.entries(persisted.statsByModel)) {
    if (fullKey.startsWith(prefix)) {
      filteredModelStats[fullKey] = stats;
    } else if (!fullKey.includes(":")) {
      // Legacy v3-style key without session hash — migrate to current session.
      filteredModelStats[`${currentSessionHash}:${fullKey}`] = stats;
    } else if (fullKey.startsWith("_nosession:")) {
      // Transitional _nosession bucket — migrate to current session.
      filteredModelStats[`${currentSessionHash}:${fullKey.slice("_nosession:".length)}`] = stats;
    }
  }

  return filteredModelStats;
}

/**
 * The closure-internal writer. Since the closure has access to currentSessionHash,
 * it passes the hash and statsByModel here. This function wraps them in the v4
 * sessions format, combining with any previously-persisted sessions for safety.
 *
 * When called from the closure, `state.statsByModel` contains only the current
 * session's entries (keyed by `${sessionHash}:${provider}/${id}`). We extract
 * the model-key-only entries and store them under the session hash.
 */
/**
 * Merge in-memory stats state into an existing sessions map for persistence.
 *
 * When `currentSessionHash` is provided (explicit hash mode):
 *   - Current-session entries are extracted from `state.statsByModel` (keys
 *     prefixed with `currentSessionHash:`) and written under the session hash.
 *   - The transitional legacy `_nosession` bucket is DELETED — its entries
 *     were already consumed and migrated into memory by `restoreCacheStats`.
 *     Keeping `_nosession` on disk would allow resurrection of reset stats
 *     on the next reload (the reset-undo bug).
 *   - Other real session hashes are preserved intact.
 *
 * When `currentSessionHash` is undefined (no-hash mode):
 *   - Keys with a hash prefix (`hash:provider/model`) are grouped under their
 *     respective session hashes.
 *   - Keys without a hash prefix (legacy v3) are grouped under `_nosession` so
 *     `restoreCacheStats` can migrate them on the next load before the session
 *     id is known.
 *
 * Pure function (no I/O) — suitable for unit tests without touching the real
 * state file at `~/.omp/agent/omp-cache-optimizer-stats.json`.
 */
function mergeCacheSessions(
  existingSessions: Record<string, Record<string, CacheStats>>,
  state: CacheStatsState,
  currentSessionHash?: string,
): Record<string, Record<string, CacheStats>> {
  // Deep-copy to avoid mutating the caller's object.
  const sessions: Record<string, Record<string, CacheStats>> = {};
  for (const [hash, models] of Object.entries(existingSessions)) {
    sessions[hash] = { ...models };
  }

  if (currentSessionHash !== undefined) {
    // Explicit hash mode: extract this session's data from state.statsByModel.
    // When the session has no entries (e.g. after reset of sole bucket), this
    // still sets an empty map, ensuring the deleted bucket does not return.
    const prefix = `${currentSessionHash}:`;
    const currentModelStats: Record<string, CacheStats> = {};
    for (const [fullKey, stats] of Object.entries(state.statsByModel)) {
      if (fullKey.startsWith(prefix)) {
        currentModelStats[fullKey.slice(prefix.length)] = stats;
      }
    }
    sessions[currentSessionHash] = currentModelStats;

    // _nosession is a transitional legacy migration bucket — once we write
    // under an authoritative session hash, those entries have already been
    // consumed and migrated into memory by restoreCacheStats. Delete to
    // prevent resurrection of reset stats on the next reload.
    delete sessions["_nosession"];
  } else {
    // No-hash mode: group entries by their existing hash prefix to avoid
    // collapsing multiple sessions into one bucket. Keys without a hash
    // prefix (legacy v3) go under "_nosession" so restoreCacheStats can
    // migrate them to the current session on next load.
    const nosessionMap: Record<string, CacheStats> = {};
    for (const [fullKey, stats] of Object.entries(state.statsByModel)) {
      const idx = fullKey.indexOf(":");
      if (idx >= 0) {
        const hash = fullKey.slice(0, idx);
        const modelKey = fullKey.slice(idx + 1);
        if (!sessions[hash]) sessions[hash] = {};
        sessions[hash][modelKey] = stats;
      } else {
        // Key without hash prefix (legacy v3) — group under _nosession.
        nosessionMap[fullKey] = stats;
      }
    }
    if (Object.keys(nosessionMap).length > 0) {
      sessions["_nosession"] = nosessionMap;
    }
  }

  return sessions;
}

function mergeLastRoutedModels(
  existingLastRoutedModelBySession: Record<string, PersistedRoutedModelRef>,
  state: CacheStatsState,
  currentSessionHash?: string,
): Record<string, PersistedRoutedModelRef> {
  const merged: Record<string, PersistedRoutedModelRef> = { ...existingLastRoutedModelBySession };
  const incoming = state.lastRoutedModelBySession ?? {};

  if (currentSessionHash !== undefined) {
    const current = incoming[currentSessionHash];
    if (current) {
      merged[currentSessionHash] = current;
    } else {
      // Explicit deletion: when incoming state has no entry for current session,
      // remove any existing stale entry to reflect intentional reset.
      delete merged[currentSessionHash];
    }
    return merged;
  }

  for (const [sessionHash, ref] of Object.entries(incoming)) {
    merged[sessionHash] = ref;
  }
  return merged;
}

async function writePersistedCacheStats(state: CacheStatsState, currentSessionHash?: string): Promise<void> {
  await mkdir(STATE_DIR, { recursive: true });

  // Read existing file to preserve other sessions' data.
  let existingSessions: Record<string, Record<string, CacheStats>> = {};
  let existingLastRoutedModelBySession: Record<string, PersistedRoutedModelRef> = {};
  try {
    const raw = await readFile(STATE_FILE_PATH, "utf8");
    const parsed = parsePersistedCacheStats(JSON.parse(raw));
    if (parsed) {
      // Reconstruct sessions from statsByModel keys.
      // Each key has form `${hash}:${provider}/${id}`; group by hash.
      for (const [fullKey, stats] of Object.entries(parsed.statsByModel)) {
        const idx = fullKey.indexOf(":");
        if (idx >= 0) {
          const hash = fullKey.slice(0, idx);
          const modelKey = fullKey.slice(idx + 1);
          if (!existingSessions[hash]) existingSessions[hash] = {};
          existingSessions[hash][modelKey] = stats;
        }
      }
      existingLastRoutedModelBySession = { ...(parsed.lastRoutedModelBySession ?? {}) };
    }
  } catch {
    // Ignore read errors (file may not exist yet).
  }

  const sessions = mergeCacheSessions(existingSessions, state, currentSessionHash);
  const lastRoutedModelBySession = mergeLastRoutedModels(
    existingLastRoutedModelBySession,
    state,
    currentSessionHash,
  );

  const payload: PersistedCacheStatsV5 = {
    version: 5,
    sessions,
    legacyFamily: state.legacyFamily,
    ...(Object.keys(lastRoutedModelBySession).length > 0 ? { lastRoutedModelBySession } : {}),
  };
  const tempPath = `${STATE_FILE_PATH}.${process.pid}.${Date.now()}.tmp`;

  await writeFile(tempPath, JSON.stringify(payload, null, 2) + "\n", "utf8");
  await rename(tempPath, STATE_FILE_PATH);
}



function isCompatCheckApplicable(model: PiModel): boolean {
  return isOpenAICompatibleProxyApi(model.api) && !isOfficialOpenAIBaseUrl(model);
}

function isPromptCacheRetention400Applicable(model: PiModel): boolean {
  return isOpenAICompatibleApi(model.api) &&
    !isOfficialOpenAIBaseUrl(model) &&
    getCompat(model).supportsLongPromptCacheRetention === true;
}

/**
 * Detect router / channel profiles from a PiModel and return diagnostic notes.
 *
 * This function is advisory only — it does NOT participate in adapter selection,
 * prompt_cache_key injection, or footer stats. It inspects provider, api, baseUrl,
 * and compat to identify common proxy/router patterns where cache performance may
 * be degraded due to multi-backend routing.
 *
 * Known profiles (checked in order):
 *   1. OpenRouter — baseUrl or provider id matching openrouter.ai / openrouter
 *   2. Vercel AI Gateway — baseUrl matching ai-gateway.vercel.sh, or provider
 *      matching vercel / vercel-ai-gateway
 *   3. LiteLLM / OneAPI / NewAPI / VoAPI — baseUrl or provider matching litellm,
 *      oneapi, one-api, newapi, new-api, voapi, vo-api (self-hosted aggregation)
 *   4. Generic third-party OpenAI-compatible proxy — any openai-completions model
 *      with a non-official base URL that does not match a higher-profile above.
 *
 * Official OpenAI (api.openai.com) and custom transports (kiro-api, anthropic-messages,
 * bedrock-converse-stream) do NOT produce notes.
 */
function describeRouterChannelDiagnostics(model: PiModel): string[] {
  const notes: string[] = [];
  const api = lower(model.api);
  const baseUrl = lower(model.baseUrl || "");
  const provider = lower(model.provider);

  if (api === "azure-openai-responses" || isMistralConversationsApi(api) || !isOpenAICompatibleApi(api)) {
    return notes;
  }

  if (isOfficialOpenAIBaseUrl(model)) {
    return notes;
  }

  if (
    baseUrl.includes("openrouter.ai") ||
    baseUrl.includes("openrouter") ||
    provider.includes("openrouter")
  ) {
    const compat = getCompat(model);
    const routing = asRecord((compat as Record<string, unknown>)["openRouterRouting"]);
    const hasOnly = !!routing?.only;
    const hasOrder = !!routing?.order;

    notes.push(
      "🔀 路由/渠道：检测到 OpenRouter。OpenRouter 是多上游路由器；如果每一轮落到不同上游，缓存命中率偏低很常见。",
    );

    if (!hasOnly && !hasOrder) {
      notes.push(
        '   建议：添加 openRouterRouting，把上游固定住。位置：models.yml -> providers["<providerId>"] -> compat：',
      );
      notes.push(
        `   { "supportsLongPromptCacheRetention": true, ` +
        `"openRouterRouting": { "only": ["<provider-slug>"] } }`,
      );
      notes.push(
        '   把 <provider-slug> 替换成真实的 OpenRouter provider slug（如 "openai"、"anthropic"）。',
      );
      notes.push(
        '   也可以用 openRouterRouting.order: ["<provider-slug>", "..."] 作为回退顺序。只有在上游支持长缓存保留时才设置 supportsLongPromptCacheRetention。',
      );
    }

    return notes;
  }

  if (
    baseUrl.includes("ai-gateway.vercel.sh") ||
    provider.includes("vercel") ||
    provider.includes("vercel-ai-gateway")
  ) {
    const compat = getCompat(model);
    const routing = asRecord((compat as Record<string, unknown>)["vercelGatewayRouting"]);
    const hasOnly = !!routing?.only;
    const hasOrder = !!routing?.order;

    notes.push(
      "🔀 路由/渠道：检测到 Vercel AI Gateway。这个网关可能把不同请求分发到不同 provider endpoint，降低缓存局部性。",
    );

    if (!hasOnly && !hasOrder) {
      notes.push(
        '   建议：添加 vercelGatewayRouting，把上游固定住。位置：models.yml -> providers["<providerId>"] -> compat：',
      );
      notes.push(
        `   { "supportsLongPromptCacheRetention": true, ` +
        `"vercelGatewayRouting": { "only": ["<provider-id>"] } }`,
      );
      notes.push(
        '   把 <provider-id> 替换成真实的 Vercel provider ID（如 "openai"）。',
      );
      notes.push(
        "   只有在上游支持长缓存保留时才设置 supportsLongPromptCacheRetention。",
      );
    }

    return notes;
  }

  const aggregationPatterns = ["litellm", "oneapi", "one-api", "newapi", "new-api", "voapi", "vo-api"];
  if (
    aggregationPatterns.some((p) => baseUrl.includes(p)) ||
    aggregationPatterns.some((p) => provider.includes(p))
  ) {
    notes.push(
      "🔀 路由/渠道：检测到自建聚合代理（LiteLLM / OneAPI / NewAPI / VoAPI）。这类代理常把请求分到多个上游账号或实例，导致缓存被拆散。",
    );
    notes.push("   建议：");
    notes.push("   • 确保代理能按 session 固定到单一上游（session_id affinity）。");
    notes.push("   • 向上游透传 prompt_cache_key 与会话亲和性相关 header。");
    notes.push("   • 在响应里返回缓存 usage 字段（如 prompt_cache_hit_tokens）。");
    notes.push(`   可作为起点的 compat：{ "supportsLongPromptCacheRetention": true }`);
    notes.push("   只有在代理明确支持 prompt_cache_retention 时才加 supportsLongPromptCacheRetention。");

    return notes;
  }

  if (api === "openai-completions" && baseUrl) {
    const missing = describeMissingCacheCompatForModel(model);
    notes.push("🔀 路由/渠道：第三方 OpenAI 兼容代理。如果缓存命中率偏低：");
    notes.push("   • 确认代理会把同一 session 路由到同一个上游账号/实例。");
    notes.push("   • 确认代理会透传 prompt_cache_key，并发送会话亲和性相关 header。");
    notes.push("   • 确认代理会返回缓存 usage 字段（如 prompt_cache_hit_tokens）。");
    if (missing.length > 0) {
      notes.push(`   • 上面这些 compat 字段（${missing.join(", ")}）有助于提升缓存稳定性。`);
    }

    return notes;
  }

  return notes;
}

function getCompatCheckNotApplicableLines(model: PiModel): string[] {
  const api = lower(model.api);

  if (isMistralConversationsApi(api)) {
    return [
      "ℹ️ 当前模型不适用 compat 检查。",
      "   原生 Mistral `mistral-conversations` 使用 provider 原生传输；OpenAI 兼容代理 compat 不适用。",
    ];
  }

  if (api === "azure-openai-responses") {
    return [
      "ℹ️ 当前模型不适用 compat 检查。",
      "   原生 Azure OpenAI Responses 使用 Responses 传输；OpenAI 兼容代理 compat 不适用。",
    ];
  }

  if (api === "openai-codex-responses" || (api === "openai-responses" && isOfficialOpenAIBaseUrl(model))) {
    return [
      "ℹ️ 当前模型不适用 compat 检查。",
      "   原生 Responses 传输已经使用运行时核心请求链路；OpenAI 兼容代理 compat 不适用。",
    ];
  }

  return ["ℹ️ 当前模型不适用 compat 检查。"];
}

function buildDoctorDiagnosis(model: PiModel, options: { promptCacheRetention400?: boolean } = {}): string {
  const lines: string[] = [];
  lines.push(`提供方：${model.provider}`);
  lines.push(`模型：    ${model.id}`);
  if (model.name && model.name !== model.id) lines.push(`名称：    ${model.name}`);
  lines.push(`API：      ${model.api}`);
  lines.push(`Base URL： ${model.baseUrl || "（默认）"}`);

  const compat = getCompat(model);
  lines.push(`Compat：   ${JSON.stringify(compat)}`);

  const adaptiveThinkingApplicable = isAdaptiveThinkingCompatApplicable(model);
  const deepSeekCompatApplicable = isDeepSeekCompatCheckApplicable(model);
  const missing = describeMissingCacheCompatForModel(model);
  const optionalOpenAIProxyCompat = (!adaptiveThinkingApplicable && !deepSeekCompatApplicable)
    ? describeOptionalOpenAICompatibleProxyCompat(model)
    : [];
  const fixSug = buildFixSuggestion(model);
  const safeFixableMissing = fixSug ? Object.keys(fixSug.compatKeys) : [];
  const advisoryMissing = missing.filter(m => !safeFixableMissing.includes(m));

  if (safeFixableMissing.length > 0) {
    lines.push(`⚠️ 缺少 compat 字段：${safeFixableMissing.join(", ")}`);
  }
  if (advisoryMissing.length > 0) {
    lines.push(`ℹ️ 可选项：${advisoryMissing.join(", ")}（仅在确认支持时启用）`);
  }

  if (missing.length > 0) {
    const key = modelKey(model);
    const slashIdx = key.indexOf("/");
    const providerLabel = slashIdx > 0 ? key.slice(0, slashIdx) : key;
    const modelsJsonPath = getModelsYmlDisplayPath();
    lines.push(`编辑 ${modelsJsonPath} -> providers["${providerLabel}"] -> compat（与 baseUrl/api/apiKey/models 同级）。`);
    if (adaptiveThinkingApplicable) {
      appendAdaptiveThinkingCompatAdviceLines(lines, missing, { providerLabel, modelId: model.id });
    } else if (deepSeekCompatApplicable) {
      appendDeepSeekCompatAdviceLines(lines, missing, { providerLabel, modelId: model.id });
    } else {
      appendOpenAIProxyCompatAdviceLines(lines, missing, { providerLabel, modelId: model.id });
      appendOptionalOpenAIProxyCompatAdviceLines(lines, optionalOpenAIProxyCompat);
    }
  } else if (adaptiveThinkingApplicable || deepSeekCompatApplicable || isCompatCheckApplicable(model)) {
    lines.push("✅ compat 配置完整。");
    appendOptionalOpenAIProxyCompatAdviceLines(lines, optionalOpenAIProxyCompat);
  } else {
    lines.push(...getCompatCheckNotApplicableLines(model));
  }

  if (isPromptCacheRetention400Applicable(model)) {
    lines.push("");
    if (options.promptCacheRetention400) {
      lines.push("⚠️ 在启用 supportsLongPromptCacheRetention 时观测到一次 400 响应。");
      lines.push(`   ${getPromptCacheRetentionUnsupportedHint()}`);
    } else {
      lines.push(`ℹ️ 已启用长缓存保留。${getPromptCacheRetentionUnsupportedHint()}`);
    }
  }

  const routerNotes = describeRouterChannelDiagnostics(model);
  if (routerNotes.length > 0) {
    lines.push("");
    for (const note of routerNotes) {
      lines.push(note);
    }
  }

  return lines.join("\n");
}

function buildLowHitDiagnosis(
  model: PiModel,
  adapter: CacheProviderAdapter | undefined,
  stats: CacheStats | undefined,
  samples: CacheUsageSample[],
): string[] {
  const lines: string[] = [];

  const fixSugLHD = buildFixSuggestion(model);
  const safeFixableMissingLHD = fixSugLHD ? Object.keys(fixSugLHD.compatKeys) : [];
  const routerNotes = describeRouterChannelDiagnostics(model);
  const missingUsageSamples = samples.filter((s) => s.missingUsageFields).length;
  const recent10 = samples.slice(-10);
  const recent10Hits = recent10.filter((s) => s.hit).length;
  const recent10Total = recent10.length;
  const recent10Cached = recent10.reduce((sum, s) => sum + s.cachedInputTokens, 0);
  const recent10Input = recent10.reduce((sum, s) => sum + s.totalInputTokens, 0);
  const todayStats = stats ?? emptyCacheStats();
  const promptMismatchSamples = recent10.filter((sample) => sample.hintPayloadComparison === "mismatch").length;
  const promptFingerprints = new Set(recent10.flatMap((sample) => sample.systemPromptFingerprint ? [sample.systemPromptFingerprint] : []));

  const hasMissingCompat = safeFixableMissingLHD.length > 0;
  const hasRouterRisk = routerNotes.length > 0;

  const hasUsageMissing = missingUsageSamples > 0;
  const todayHitRatio = todayStats.totalInputTokens > 0
    ? Math.round((todayStats.cachedInputTokens / todayStats.totalInputTokens) * 100)
    : 0;

  const hasActualIssues = hasMissingCompat || hasUsageMissing ||
    (todayStats.totalRequests > 3 && todayStats.totalInputTokens > 0 &&
     (todayStats.cachedInputTokens / todayStats.totalInputTokens) < 0.3) ||
    (recent10Total >= 3 && recent10Hits === 0);

  if (!hasActualIssues && !(hasRouterRisk && (hasMissingCompat || hasUsageMissing))) {
    return lines;
  }

  lines.push("");
  lines.push("── 缓存诊断 ──");
  if (promptMismatchSamples > 0) {
    lines.push(`⚠️ 最近 ${recent10Total} 条请求中有 ${promptMismatchSamples} 条的发送 system prompt 与 cache hint 不一致。`);
    lines.push("   这通常表示后续 extension 或宿主在 hint 发布后修改了 prompt，可能导致 provider cache 前缀失效。");
  }
  if (promptFingerprints.size > 1) {
    lines.push(`ℹ️ 最近 ${recent10Total} 条请求观测到 ${promptFingerprints.size} 组 system prompt 指纹。`);
    lines.push("   若低命中持续，请避免在同一实验窗口混用不同 prompt 模式或动态 system 内容。");
  }

  if (hasMissingCompat) {
    lines.push(`⚠️ 缺少 compat 字段：${safeFixableMissingLHD.join(", ")}`);
    lines.push("   这些字段有助于稳定 prompt 缓存与上游路由粘性。");
    lines.push("   可运行 /cache-optimizer compat 查看编辑建议。");
  }

  if (hasRouterRisk && (hasMissingCompat || hasUsageMissing || hasActualIssues)) {
    lines.push("🔀 检测到路由/代理风险 —— 详见上方路由诊断。");
  }

  if (hasUsageMissing) {
    lines.push(`⚠️ 最近 ${samples.length} 条样本里有 ${missingUsageSamples} 条缺少或返回了空的 usage 字段。`);
    lines.push("   Footer 命中率可能会被低估。");
    lines.push("   请确认代理会返回 prompt 级 usage（如 prompt_tokens、input_tokens_details）。");
  }

  if (recent10Total > 0) {
    if (recent10Hits === 0 && todayStats.totalRequests > 3 && todayHitRatio < 30) {
      lines.push(`📉 今日缓存命中率偏低：${todayHitRatio}%（最近 ${recent10Total} 条样本）。`);
      lines.push("   常见原因：代理把请求路由到不同后端，或 prompt 前缀在各轮之间变化。");
      lines.push("   请检查上游路由粘性，以及 supportsLongPromptCacheRetention 配置是否正确。");
    } else if (todayHitRatio < 30 && todayStats.totalRequests > 3) {
      lines.push(`📉 今日缓存命中率偏低：${todayHitRatio}%（共 ${todayStats.totalRequests} 次请求）。`);
      lines.push("   请检查 compat 配置与代理上游路由。");
    }

    if (recent10Total >= 3) {
      const trend = formatRecentTrendSummary(samples, 10);
      lines.push(`📊 ${trend}`);
    }
  }

  if (!hasMissingCompat && !hasRouterRisk && todayStats.totalRequests > 3 && todayHitRatio < 30) {
    lines.push("💡 compat 已配置完整，但缓存命中率仍然偏低。");
    lines.push("   可能原因：");
    lines.push("   • 代理仍把请求分发到多个后端 —— 请检查代理侧的会话粘性。");
    lines.push("   • prompt 前缀每轮都在变化 —— 请检查 system prompt 中的动态上下文。");
    lines.push("   • provider 没有返回缓存 usage 字段 —— footer 无法准确测量命中。");
  }

  return lines;
}

function buildCompatDiagnosis(model: PiModel): string | undefined {
  const missing = describeMissingCacheCompatForModel(model);
  const fixSugC = buildFixSuggestion(model);
  const safeFixableMissingC = fixSugC ? Object.keys(fixSugC.compatKeys) : [];
  const advisoryMissingC = missing.filter(m => !safeFixableMissingC.includes(m));
  const adaptiveThinkingApplicable = isAdaptiveThinkingCompatApplicable(model);
  const deepSeekCompatApplicable = isDeepSeekCompatCheckApplicable(model);
  const optionalOpenAIProxyCompat = (!adaptiveThinkingApplicable && !deepSeekCompatApplicable)
    ? describeOptionalOpenAICompatibleProxyCompat(model)
    : [];
  const routerNotes = describeRouterChannelDiagnostics(model);

  if (missing.length === 0 && routerNotes.length === 0 && optionalOpenAIProxyCompat.length === 0) return undefined;

  const key = modelKey(model);
  const lines: string[] = [];

  if (missing.length > 0) {
    const slashIdx = key.indexOf("/");
    const providerLabel = slashIdx > 0 ? key.slice(0, slashIdx) : key;
    const modelsJsonPath = getModelsYmlDisplayPath();
    lines.push(`当前模型：${key}`);
    if (safeFixableMissingC.length > 0) {
      lines.push(`可安全修复：${safeFixableMissingC.join(", ")}`);
    }
    if (advisoryMissingC.length > 0) {
      lines.push(`可选项：${advisoryMissingC.join(", ")}（仅在确认支持时启用）`);
    }
    lines.push("");
    lines.push(`编辑 ${modelsJsonPath} -> providers["${providerLabel}"] -> compat`);
    lines.push("（与 baseUrl/api/apiKey/models 同级）。");
    if (adaptiveThinkingApplicable) {
      appendAdaptiveThinkingCompatAdviceLines(lines, missing, { providerLabel, modelId: model.id });
    } else if (deepSeekCompatApplicable) {
      appendDeepSeekCompatAdviceLines(lines, missing, { providerLabel, modelId: model.id });
    } else {
      appendOpenAIProxyCompatAdviceLines(lines, missing, { providerLabel, modelId: model.id });
      appendOptionalOpenAIProxyCompatAdviceLines(lines, optionalOpenAIProxyCompat);
    }
  }

  if ((routerNotes.length > 0 || optionalOpenAIProxyCompat.length > 0) && missing.length === 0) {
    if (adaptiveThinkingApplicable || deepSeekCompatApplicable || isCompatCheckApplicable(model)) {
      lines.push("✅ compat 配置完整。");
      if (isPromptCacheRetention400Applicable(model)) {
        lines.push(getPromptCacheRetentionUnsupportedHint());
      }
      appendOptionalOpenAIProxyCompatAdviceLines(lines, optionalOpenAIProxyCompat);
    } else {
      lines.push(...getCompatCheckNotApplicableLines(model));
    }
    lines.push("");
  }

  if (routerNotes.length > 0) {
    if (missing.length > 0) lines.push("");
    for (const note of routerNotes) {
      lines.push(note);
    }
  }

  return lines.join("\n");
}

// ============================================================
// Manual YAML compat suggestions for /cache-optimizer fix
// ============================================================

/** The real models.yml path used for I/O. OMP stores model config as YAML, not JSONC. */
const MODELS_YML_PATH = join(STATE_DIR, "models.yml");

interface FixSuggestion {
  providerLabel: string;
  modelId: string;
  compatKeys: Record<string, unknown>;
}

/**
 * Build the fix suggestion for the current active model.
 * Returns undefined if there is nothing to fix.
 */
function buildFixSuggestion(model: PiModel): FixSuggestion | undefined {
  const missing = describeMissingCacheCompatForModel(model);
  if (missing.length === 0) return undefined;

  let compatKeys: Record<string, unknown> = {};

  if (isAdaptiveThinkingCompatApplicable(model)) {
    compatKeys = buildAdaptiveThinkingCompatSuggestion(missing);
  } else if (isDeepSeekCompatCheckApplicable(model)) {
    compatKeys = buildDeepSeekCompatSuggestion(missing);
  } else {
    compatKeys = buildSafeOpenAIProxyCompatSuggestion(missing);
  }

  if (Object.keys(compatKeys).length === 0) return undefined;

  const key = modelKey(model);
  const slashIdx = key.indexOf("/");
  const providerLabel = slashIdx > 0 ? key.slice(0, slashIdx) : key;

  return {
    providerLabel,
    modelId: model.id,
    compatKeys,
  };
}

/**
 * Serialize a compat suggestion to YAML key/value lines for manual insertion.
 */
function formatCompatKeysForInsertion(compatKeys: Record<string, unknown>): string {
  return Object.entries(compatKeys)
    .map(([k, v]) => {
      return `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`;
    })
    .join(',\n');
}

// Internal helpers exported only so the task verification script
// (.trellis/tasks/.../verify.ts) can exercise them. They are not part of the
// extension's public API; the host runtime only invokes the default export below.
export const __internals_for_tests = {
  joinSystemPromptBlocks,
  mapSystemPromptBlocks,
  systemPromptBlocksEqual,
  stripSessionOverviewChurn,
  PROMPT_REWRITE_ENV,
  isEnabledEnv,
  fingerprintPrompt,
  comparePromptFingerprints,
  getPromptCacheKeySource,
  isNonEmptyString,
  isOpenAICompatibleApi,
  isOpenAICompatibleProxyApi,
  isResponsesPromptRewriteBypassApi,
  isMistralConversationsApi,
  isOpenAIFamilyModel,
  isOpenAIFamilyAssistantMessage,
  isOpenAIFamilyToken,
  describeMissingOpenAIFamilyProxyCompat,
  describeMissingOpenAICompatibleProxyCompat,
  describeOptionalOpenAICompatibleProxyCompat,
  describeMissingDeepSeekCompat,
  isDeepSeekCompatCheckApplicable,
  describeMissingCacheCompatForModel,
  buildDeepSeekCompatSuggestion,
  buildDeepSeekCompatWarningText,
  buildSafeOpenAIProxyCompatSuggestion,
  getPromptCacheRetentionUnsupportedHint,
  isOfficialOpenAIBaseUrl,
  isCompatCheckApplicable,
  isPromptCacheRetention400Applicable,
  hasPromptCacheRetentionUnsupportedSignal,
  // Non-GPT OpenAI-compatible model detection
  isKimiLikeModel,
  isKimiLikeAssistantMessage,
  isQwenLikeModel,
  isQwenLikeAssistantMessage,
  isGLMLikeModel,
  isGLMLikeAssistantMessage,
  isMiniMaxLikeModel,
  isMiniMaxLikeAssistantMessage,
  isMimoLikeModel,
  isMimoLikeAssistantMessage,
  isHunyuanLikeModel,
  isHunyuanLikeAssistantMessage,
  // Additional OpenAI-compatible model detection
  isMistralLikeModel,
  isMistralLikeAssistantMessage,
  isGrokLikeModel,
  isGrokLikeAssistantMessage,
  isLlamaLikeModel,
  isLlamaLikeAssistantMessage,
  isNemotronLikeModel,
  isNemotronLikeAssistantMessage,
  isCohereLikeModel,
  isCohereLikeAssistantMessage,
  isYiLikeModel,
  isYiLikeAssistantMessage,
  // More OpenAI-compatible model detection (batch 2)
  isDoubaoLikeModel,
  isDoubaoLikeAssistantMessage,
  isErnieLikeModel,
  isErnieLikeAssistantMessage,
  isBaichuanLikeModel,
  isBaichuanLikeAssistantMessage,
  isStepFunLikeModel,
  isStepFunLikeAssistantMessage,
  isSparkLikeModel,
  isSparkLikeAssistantMessage,
  isInternLMLikeModel,
  isInternLMLikeAssistantMessage,
  isGemmaLikeModel,
  isGemmaLikeAssistantMessage,
  isPhiLikeModel,
  isPhiLikeAssistantMessage,
  isJambaLikeModel,
  isJambaLikeAssistantMessage,
  isSolarLikeModel,
  isSolarLikeAssistantMessage,
  // New OpenAI-compatible model detection (batch 3, 12 families)
  isPerplexityLikeModel,
  isPerplexityLikeAssistantMessage,
  isNovaLikeModel,
  isNovaLikeAssistantMessage,
  isRekaLikeModel,
  isRekaLikeAssistantMessage,
  isFalconLikeModel,
  isFalconLikeAssistantMessage,
  isDbrxLikeModel,
  isDbrxLikeAssistantMessage,
  isMptLikeModel,
  isMptLikeAssistantMessage,
  isStableLMLikeModel,
  isStableLMLikeAssistantMessage,
  isAquilaLikeModel,
  isAquilaLikeAssistantMessage,
  isExaoneLikeModel,
  isExaoneLikeAssistantMessage,
  isHyperCLOVALikeModel,
  isHyperCLOVALikeAssistantMessage,
  isLuminousLikeModel,
  isLuminousLikeAssistantMessage,
  isHermesLikeModel,
  isHermesLikeAssistantMessage,
  // More OpenAI-compatible model detection (batch 4, 18 families)
  isGraniteLikeModel,
  isGraniteLikeAssistantMessage,
  isArcticLikeModel,
  isArcticLikeAssistantMessage,
  isPanguLikeModel,
  isPanguLikeAssistantMessage,
  isSenseNovaLikeModel,
  isSenseNovaLikeAssistantMessage,
  isZhinaoLikeModel,
  isZhinaoLikeAssistantMessage,
  isMiniCPMLikeModel,
  isMiniCPMLikeAssistantMessage,
  isXVerseLikeModel,
  isXVerseLikeAssistantMessage,
  isOrionLikeModel,
  isOrionLikeAssistantMessage,
  isOpenChatLikeModel,
  isOpenChatLikeAssistantMessage,
  isVicunaLikeModel,
  isVicunaLikeAssistantMessage,
  isWizardLikeModel,
  isWizardLikeAssistantMessage,
  isZephyrLikeModel,
  isZephyrLikeAssistantMessage,
  isDolphinLikeModel,
  isDolphinLikeAssistantMessage,
  isOpenOrcaLikeModel,
  isOpenOrcaLikeAssistantMessage,
  isStarlingLikeModel,
  isStarlingLikeAssistantMessage,
  isBloomLikeModel,
  isBloomLikeAssistantMessage,
  isRwkvLikeModel,
  isRwkvLikeAssistantMessage,
  isAyaLikeModel,
  isAyaLikeAssistantMessage,
  selectAdapterForModel,
  selectAdapterForAssistantMessage,
  buildOpenAIProxyCompatWarningText,
  getModelIdNameTokenValues,
  getAssistantMessageModelTokenValues,
  getCompat,
  modelKey,
  // Platform-friendly path helpers
  getModelsYmlDisplayPath,
  buildProviderCompatOverride,
  buildModelCompatOverride,
  captureCacheRetentionEnv,
  requestLongCacheRetention,
  restoreCacheRetentionEnv,
  setRuntimeOptimizerEnabled,
  isRuntimeOptimizerEnabled,
  getOptimizerRuntimeModeLines,
  formatOptimizerRuntimeMode,
  OMP_CACHE_RETENTION_ENV,
  PI_CACHE_RETENTION_ENV,
  LONG_CACHE_RETENTION_VALUE,
  // Diagnostic command helpers
  buildDoctorDiagnosis,
  buildCompatDiagnosis,
  describeRouterChannelDiagnostics,
  // Cache stats helpers (module-level, usable from verify script)
  addUsageToCacheStats,
  formatCacheStats,
  emptyCacheStats,
  emptyAllCacheStats,
  parseCacheStats,
  parsePersistedCacheStats,
  // Recent sample / stats output / diagnosis helpers
  MAX_RECENT_SAMPLES,
  buildStatsOutput,
  buildLowHitDiagnosis,
  formatRecentTrendSummary,
  formatHitRatio,
  formatTokenM,
  hasMissingUsageFields,
  keyForModelExt,
  // Session-scoped helpers
  hashSessionId,
  makeSessionModelKey,
  modelKeyFromSessionKey,
  makePromptRewriteContextKey,
  rememberPromptRewriteContext,
  getPromptRewriteContext,
  PROMPT_REWRITE_CONTEXT_TTL_MS,
  filterRestorableStatsForSession,
  parsePersistedRoutedModelRef,
  routedModelRefToPiModel,
  buildExactRouterStatusEntry,
  // Routing-provider protocol helpers
  PI_ROUTING_REGISTRY_SYMBOL,
  PI_CACHE_HINTS_SYMBOL,
  ensureRoutingRegistry,
  getRoutingRegistry,
  parseRouteSnapshot,
  resolveActiveRouteSnapshot,
  routeSnapshotToPiModel,
  resolveRouteModel,
  isVirtualRoutingModel,
  installCacheHintsService,
  getCacheHintsService,
  // Persistence helpers (for reload/reset tests)
  mergeCacheSessions,
  mergeLastRoutedModels,
  writePersistedCacheStats,
  readPersistedCacheStats,
  STATE_FILE_PATH,
  LEGACY_STATE_FILE_PATH,
  STATE_DIR,
  // Manual YAML compat suggestion helpers
  MODELS_YML_PATH,
  formatCompatKeysForInsertion,
  // Fix suggestion builder
  buildFixSuggestion,
  // Adaptive thinking compat helpers
  isAdaptiveGenerationModel,
  isAdaptiveThinkingCompatApplicable,
  describeMissingAdaptiveThinkingCompat,
  buildAdaptiveThinkingCompatSuggestion,
  buildAdaptiveThinkingCompatWarningText,
  appendAdaptiveThinkingCompatAdviceLines,
  // OMP migration: prompt rewrite helpers (new in fork)
  extractSystemPrompt,
  setSystemPrompt,
  // Additional model detection + helpers for smoke tests
  isDeepSeekLikeModel,
  isClaudeLikeModel,
  asRecord,
};

export default function (pi: ExtensionAPI) {
  const warnedModels = new Set<string>();
  const promptCacheRetention400Models = new Set<string>();
  const warnedPromptCacheRetention400Models = new Set<string>();
  let cacheStatsByModel: Record<string, CacheStats> = {};
  let cacheStatsLegacyFamily: Partial<Record<CacheProviderId, CacheStats>> = emptyAllCacheStats();
  let lastStatusText: string | undefined;
  let persistenceWarningShown = false;
  let persistTimer: ReturnType<typeof setTimeout> | null = null;
  let currentSessionId = "";
  let currentSessionHash = "";
  let currentSessionHashSet = false;
  let lastActualRoutedModel: PersistedRoutedModelRef | undefined;
  let latestCacheHint: PiCacheHintSnapshot | undefined;
  // OMP 17：主 prompt 重写在 before_agent_start（systemPrompt: string[]）。
  // promptRewriteContexts 仅保留路由快照供 routing/hints，不再缓存 options。
  const promptRewriteContexts = new Map<string, PromptRewriteContext>();
  const PERSIST_DEBOUNCE_MS = 2000;
  /** In-memory recent usage samples per model key (not persisted, cleared on reload). */
  const recentSamplesByModelKey = new Map<string, CacheUsageSample[]>();
  /** Per-request diagnostics awaiting the corresponding message_end event; never persisted. */
  const pendingRequestDiagnosticsByModelKey = new Map<string, PromptRequestDiagnostics>();

  function syncSessionHash(ctx: Pick<ExtensionContext, "sessionManager">): void {
    const sid = ctx.sessionManager.getSessionId();
    if (sid && (sid !== currentSessionId || !currentSessionHashSet)) {
      currentSessionId = sid;
      currentSessionHash = hashSessionId(sid);
      currentSessionHashSet = true;
      lastActualRoutedModel = undefined;
    }
  }

  const uninstallCacheHintsService = installCacheHintsService({
    version: 1,
    getHints(input: PiCacheHintsInput): PiCacheHintsOutput | undefined {
      if (!runtimeOptimizerEnabled) return undefined;
      const hint = latestCacheHint;
      if (!hint) return undefined;
      if (input.sessionIdHash && hint.sessionIdHash && input.sessionIdHash !== hint.sessionIdHash) return undefined;
      if (input.virtualProvider && hint.virtualProvider && input.virtualProvider !== hint.virtualProvider) return undefined;
      if (input.virtualModelId && hint.virtualModelId && input.virtualModelId !== hint.virtualModelId) return undefined;
      if (input.upstreamProvider && hint.upstreamProvider && input.upstreamProvider !== hint.upstreamProvider) return undefined;
      if (input.upstreamModelId && hint.upstreamModelId && input.upstreamModelId !== hint.upstreamModelId) return undefined;
      if (input.api && hint.api && input.api !== hint.api) return undefined;

      return {
        systemPrompt: hint.systemPrompt,
        promptCacheKey: hint.promptCacheKey,
        cacheRetention: hint.cacheRetention,
      };
    },
  });
  void uninstallCacheHintsService;

  /**
   * Build a session-scoped stats key from the current session hash + model key.
   * Returns `${sessionHash}:${provider}/${id}`.
   */
  function sessionModelKey(model: { provider: string; id: string }): string {
    const hash = currentSessionHash || "_nosession";
    return `${hash}:${model.provider}/${model.id}`;
  }

  /**
   * Extract the user-facing model key from a session-scoped key.
   * "abc123:otokapi/gpt-5.5" → "otokapi/gpt-5.5"
   */
  function modelKeyFromSessionScoped(sKey: string): string {
    const idx = sKey.indexOf(":");
    return idx >= 0 ? sKey.slice(idx + 1) : sKey;
  }

  function recordRecentSample(
    modelKeyStr: string,
    usage: UsageSnapshot,
    missingUsageFields: boolean,
    diagnostics: PromptRequestDiagnostics | undefined,
  ): void {
    let samples = recentSamplesByModelKey.get(modelKeyStr);
    if (!samples) {
      samples = [];
      recentSamplesByModelKey.set(modelKeyStr, samples);
    }
    samples.push({
      timestamp: Date.now(),
      hit: usage.cacheRead > 0,
      cachedInputTokens: usage.cacheRead,
      cacheWriteInputTokens: usage.cacheWrite,
      totalInputTokens: usage.totalInput,
      missingUsageFields,
      promptRewriteEnabled: diagnostics?.promptRewriteEnabled ?? isPromptRewriteEnabled(),
      systemPromptFingerprint: diagnostics?.systemPromptFingerprint,
      promptCacheKeySource: diagnostics?.promptCacheKeySource ?? "unavailable",
      hintPayloadComparison: diagnostics?.hintPayloadComparison ?? "unavailable",
    });
    if (samples.length > MAX_RECENT_SAMPLES) {
      samples.splice(0, samples.length - MAX_RECENT_SAMPLES);
    }
  }

  function getRecentSamples(modelKeyStr: string): CacheUsageSample[] {
    return recentSamplesByModelKey.get(modelKeyStr) ?? [];
  }

  function clearRecentSamples(): void {
    recentSamplesByModelKey.clear();
    pendingRequestDiagnosticsByModelKey.clear();
  }

  function getCacheStatsState(): CacheStatsState {
    return {
      statsByModel: cacheStatsByModel,
      legacyFamily: cacheStatsLegacyFamily,
      ...(currentSessionHashSet && lastActualRoutedModel
        ? { lastRoutedModelBySession: { [currentSessionHash]: lastActualRoutedModel } }
        : {}),
    };
  }

  /** Look up active stats for a model, falling back to legacy family. */
  function getStatsForModel(model: PiModel | undefined, adapter: CacheProviderAdapter): CacheStats {
    if (model) {
      const key = sessionModelKey(model);
      const existing = cacheStatsByModel[key];
      if (existing) return existing;
    }

    // Fallback: legacy family bucket — used when model key is unknown
    // or this model hasn't been seen yet in this session.
    const family = cacheStatsLegacyFamily[adapter.id];
    if (family) return family;

    const created = emptyCacheStats();
    cacheStatsLegacyFamily[adapter.id] = created;
    return created;
  }

  /** Get or create a stats entry for the given model key. */
  function getOrCreateStatsByModelKey(key: string): CacheStats {
    const existing = cacheStatsByModel[key];
    if (existing) return existing;

    const created = emptyCacheStats();
    cacheStatsByModel[key] = created;
    return created;
  }

  function resetStatsForModel(model: PiModel): void {
    const sk = sessionModelKey(model);
    delete cacheStatsByModel[sk];
    recentSamplesByModelKey.delete(sk);
    lastStatusText = undefined;
  }

  function resetCurrentSessionStats(): void {
    const prefix = `${currentSessionHash || "_nosession"}:`;
    for (const key of Object.keys(cacheStatsByModel)) {
      if (key.startsWith(prefix)) delete cacheStatsByModel[key];
    }
    for (const key of Array.from(recentSamplesByModelKey.keys())) {
      if (key.startsWith(prefix)) recentSamplesByModelKey.delete(key);
    }
    lastActualRoutedModel = undefined;
    lastStatusText = undefined;
  }

  async function persistCacheStats(ctx?: ExtensionContext): Promise<void> {
    try {
      await writePersistedCacheStats(getCacheStatsState(), currentSessionHashSet ? currentSessionHash : undefined);
    } catch (error) {
      console.warn(`${LOG_PREFIX}: failed to persist cache stats`, error);
      if (!persistenceWarningShown) {
        persistenceWarningShown = true;
        ctx?.ui.notify(
          `${LOG_PREFIX}: failed to persist footer stats; using in-memory stats for this process.`,
          "warning",
        );
      }
    }
  }

  /** Schedule a debounced persist. Coalesces rapid message_end writes
   *  into a single disk write after PERSIST_DEBOUNCE_MS of silence.
   *  In-memory stats remain instantly up-to-date for the footer; only
   *  the on-disk persistence is delayed. */
  function schedulePersistCacheStats(ctx?: ExtensionContext): void {
    if (persistTimer !== null) clearTimeout(persistTimer);
    persistTimer = setTimeout(() => {
      persistTimer = null;
      persistCacheStats(ctx).catch((err) => {
        console.warn(`${LOG_PREFIX}: debounced persist failed`, err);
      });
    }, PERSIST_DEBOUNCE_MS);
  }

  /** Flush any pending debounced persist immediately (cancels timer + writes).
   *  Used on reload and day-rollover where immediate durability matters. */
  async function flushPersistCacheStats(ctx?: ExtensionContext): Promise<void> {
    if (persistTimer !== null) {
      clearTimeout(persistTimer);
      persistTimer = null;
    }
    await persistCacheStats(ctx);
  }

  async function rollOverStatsIfNeeded(ctx?: ExtensionContext): Promise<void> {
    const day = currentLocalDay();
    let changed = false;

    // Roll over per-model entries.
    for (const key of Object.keys(cacheStatsByModel)) {
      const stats = cacheStatsByModel[key];
      if (stats && stats.day !== day) {
        cacheStatsByModel[key] = emptyCacheStats(day);
        changed = true;
      }
    }

    // Roll over legacy family entries.
    for (const id of CACHE_PROVIDER_IDS) {
      const stats = cacheStatsLegacyFamily[id];
      if (stats && stats.day !== day) {
        cacheStatsLegacyFamily[id] = emptyCacheStats(day);
        changed = true;
      }
    }

    if (changed) {
      lastStatusText = undefined;
      await persistCacheStats(ctx);
    }
  }

  async function restoreCacheStats(reason: string, ctx: ExtensionContext): Promise<void> {
    syncSessionHash(ctx);

    if (reason === "reload") {
      // /reload: preserve session-scoped stats (same session hash).
      // OMP extension reload creates a fresh closure, so cacheStatsByModel
      // starts empty. Read persisted data and filter for current session.
      lastStatusText = undefined;
      clearRecentSamples();

      const persisted = await readPersistedCacheStats();
      cacheStatsByModel = filterRestorableStatsForSession(
        persisted,
        currentSessionHashSet ? currentSessionHash : undefined,
      );
      cacheStatsLegacyFamily = persisted?.legacyFamily ?? emptyAllCacheStats();
      lastActualRoutedModel = currentSessionHashSet
        ? persisted?.lastRoutedModelBySession?.[currentSessionHash]
        : undefined;

      await rollOverStatsIfNeeded(ctx);
      return;
    }

    // First load / process start: read persisted stats and filter for
    // this session's entries. If the session hash is unavailable, start
    // fresh instead of loading all persisted session buckets.
    const persisted = await readPersistedCacheStats();
    cacheStatsByModel = filterRestorableStatsForSession(
      persisted,
      currentSessionHashSet ? currentSessionHash : undefined,
    );
    cacheStatsLegacyFamily = persisted?.legacyFamily ?? emptyAllCacheStats();
    lastActualRoutedModel = currentSessionHashSet
      ? persisted?.lastRoutedModelBySession?.[currentSessionHash]
      : undefined;
    lastStatusText = undefined;
    await rollOverStatsIfNeeded(ctx);
  }

  /**
   * Fallback for older persisted files that do not yet carry exact
   * last-routed-model metadata. When the current model is a router channel
   * (e.g. router/auto), restorable stats are stored under the real upstream
   * model's provider/id key, not under router/auto. Find the best valid entry
   * (highest totalRequests among adapter-detectable model keys) so we can show
   * meaningful footer content on session_start after reload.
   */
  function findBestRouterModelStats(): { adapter: CacheProviderAdapter; stats: CacheStats } | undefined {
    if (!currentSessionHash) return undefined;
    const prefix = `${currentSessionHash}:`;
    let best: { adapter: CacheProviderAdapter; stats: CacheStats; total: number } | undefined;

    for (const [key, stats] of Object.entries(cacheStatsByModel)) {
      if (!key.startsWith(prefix)) continue;

      // Extract provider/id from key like "abc123:run-claude/claude-opus-4-8"
      const modelKeyPart = key.slice(prefix.length);
      const slashIdx = modelKeyPart.indexOf("/");
      if (slashIdx < 0 || slashIdx >= modelKeyPart.length - 1) continue;
      const modelId = modelKeyPart.slice(slashIdx + 1);
      const providerName = modelKeyPart.slice(0, slashIdx);

      // Construct a minimal model for adapter detection.
      // Every is*LikeModel function only accesses model.id and model.name
      // via getModelIdNameTokenValues, so { id, name } is sufficient.
      const mockModel = {
        id: modelId,
        name: modelId,
        provider: providerName,
        api: "",
        baseUrl: "",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 0,
        maxTokens: 0,
      } as PiModel;

      const adapter = selectAdapterForModel(mockModel);
      if (!adapter) continue;

      if (!best || stats.totalRequests > best.total) {
        best = { adapter, stats, total: stats.totalRequests };
      }
    }

    return best ? { adapter: best.adapter, stats: best.stats } : undefined;
  }

  async function publishStatus(ctx: ExtensionContext, model: PiModel | undefined = ctx.model): Promise<void> {
    syncSessionHash(ctx);
    await rollOverStatsIfNeeded(ctx);

    const routedModel = resolveRouteModel(model, ctx);
    const displayModel = routedModel ?? model;
    const adapter = selectAdapterForModel(displayModel);
    const activeIsVirtualRoute = !!routedModel || isVirtualRoutingModel(model, ctx);
    let statusText: string | undefined;

    if (!adapter && !routedModel && activeIsVirtualRoute) {
      // On model_select (existing footer), keep the existing cache footer
      // visible instead of clearing it. On session_start (no footer yet
      // after reload/fresh start), restore the exact last actual routed model
      // for this session when available; fall back to older best-effort
      // heuristics only when no exact metadata exists.
      if (lastStatusText !== undefined) return;
      const realEntry = buildExactRouterStatusEntry(
        currentSessionHashSet ? currentSessionHash : undefined,
        cacheStatsByModel,
        lastActualRoutedModel,
      ) ?? findBestRouterModelStats();
      if (realEntry) {
        const statsText = formatCacheStats(realEntry.adapter, realEntry.stats);
        statusText = runtimeOptimizerEnabled
          ? statsText
          : `缓存优化已关闭 · ${statsText}`;
      }
    }

    if (adapter) {
      // Display session-scoped stats. A model that has never been used
      // in this session shows 0/0. The message_end hook populates
      // cacheStatsByModel[sessionModelKey(displayModel)] on first use.
      const sk = displayModel ? sessionModelKey(displayModel) : undefined;
      const stats = sk ? cacheStatsByModel[sk] : undefined;
      const statsText = formatCacheStats(adapter, stats ?? emptyCacheStats());
      statusText = runtimeOptimizerEnabled ? statsText : `缓存优化已关闭 · ${statsText}`;
    }

    // ⚠️ compat footer marker: if the active model has adapter-specific
    // missing compat (DeepSeek reasoning/cache compat, or a non-official
    // openai-completions model missing cache/session-affinity flags), append
    // the marker to indicate that compat configuration is incomplete.
    // Re-evaluated on every status update so the marker persists through stats
    // changes and day rollovers. Redundant setStatus calls are blocked by the
    // `lastStatusText` early return above.
    if (runtimeOptimizerEnabled && statusText !== undefined && displayModel) {
      // Only show ⚠️ compat when there are safe-fixable missing compat keys.
      // Optional/advisory-only flags (e.g. supportsLongPromptCacheRetention on generic
      // OpenAI-compatible proxies) do NOT trigger the marker — the doctor/compat
      // commands still mention them as optional guidance.
      if (buildFixSuggestion(displayModel) !== undefined) {
        statusText = statusText + " ⚠️ 配置";
      }
    }

    if (statusText === lastStatusText) return;

    lastStatusText = statusText;
    ctx.ui.setStatus(STATUS_KEY, statusText);
  }

  ensureRoutingRegistry();

  pi.on("session_start", async (event, ctx) => {
    await restoreCacheStats(event.reason, ctx);
    if (runtimeOptimizerEnabled) notifyCacheCompatIfNeeded(resolveRouteModel(ctx.model, ctx) ?? ctx.model, ctx, warnedModels);
    await publishStatus(ctx);
  });

  // OMP divergence: model_select event may not exist in OMP (not listed in
  // hooks.md/extensions.md). Use turn_start + model-change detection instead.
  let lastModelKeyForStatus: string | undefined;
  pi.on("turn_start", async (_event, ctx) => {
    const model = resolveRouteModel(ctx.model, ctx) ?? ctx.model;
    const key = model ? modelKey(model) : undefined;
    if (key === lastModelKeyForStatus) return;
    lastModelKeyForStatus = key;
    if (runtimeOptimizerEnabled && model) {
      notifyCacheCompatIfNeeded(model, ctx, warnedModels);
    }
    await publishStatus(ctx, model);
  });

  pi.on("before_agent_start", async (event, _ctx) => {
    latestCacheHint = undefined;
    const routeSnapshot = resolveActiveRouteSnapshot(_ctx.model, _ctx);
    const routedModel = routeSnapshot
      ? findModelInRegistry(_ctx.modelRegistry, routeSnapshot.provider, routeSnapshot.modelId) ?? routeSnapshotToPiModel(routeSnapshot, _ctx.model)
      : undefined;

    const model = routedModel ?? _ctx.model;
    const contextKey = makePromptRewriteContextKey(sessionHashFromContext(_ctx), model);
    rememberPromptRewriteContext(promptRewriteContexts, contextKey, {
      routeSnapshot,
      routedModel: model,
      timestamp: Date.now(),
    });

    const promptCacheKey = getSessionPromptCacheKey(_ctx);
    const cacheRetention = readCacheRetentionValue(process.env) === LONG_CACHE_RETENTION_VALUE ? LONG_CACHE_RETENTION_VALUE : undefined;

    const publishHintFromBlocks = (blocks: string[]): void => {
      latestCacheHint = {
        sessionIdHash: currentSessionHashSet ? currentSessionHash : sessionHashFromContext(_ctx),
        virtualProvider: routeSnapshot?.virtualProvider ?? _ctx.model?.provider,
        virtualModelId: routeSnapshot?.virtualModelId ?? _ctx.model?.id,
        upstreamProvider: routeSnapshot?.provider ?? model?.provider,
        upstreamModelId: routeSnapshot?.modelId ?? model?.id,
        api: model?.api,
        systemPrompt: joinSystemPromptBlocks(blocks),
        promptCacheKey,
        cacheRetention,
        timestamp: Date.now(),
      };
      const globals = getProtocolGlobal();
      globals.__ompCacheOptimizerCacheKey__ = promptCacheKey;
    };

    // 规范化 event.systemPrompt：OMP 17 为 string[]；兼容旧宿主的 string。
    const eventRecord = asRecord(event);
    const rawSp = eventRecord?.systemPrompt;
    const originalBlocks: string[] = Array.isArray(rawSp)
      ? rawSp.filter((b): b is string => typeof b === "string")
      : typeof rawSp === "string"
        ? [rawSp]
        : [];

    // Responses 族 API：服务端管理缓存 + 更严的内容安全过滤。
    // 跳过全部 prompt 变更；仍发布未修改的 hint。
    if (model && isResponsesPromptRewriteBypassApi(model.api)) {
      if (originalBlocks.length > 0) publishHintFromBlocks(originalBlocks);
      return {};
    }
    if (!runtimeOptimizerEnabled) {
      if (originalBlocks.length > 0) publishHintFromBlocks(originalBlocks);
      return {};
    }
    if (!isPromptRewriteEnabled()) {
      if (originalBlocks.length > 0) publishHintFromBlocks(originalBlocks);
      return {};
    }
    if (originalBlocks.length === 0) return {};

    // 块级改写：仅逐块清理 <session-overview> churn。不压缩 skills、不重排块顺序，
    // 保持 OMP 17 systemPrompt: string[] 的块内容与顺序逐字保真。
    const finalBlocks = mapSystemPromptBlocks(originalBlocks, stripSessionOverviewChurn);

    publishHintFromBlocks(finalBlocks);

    if (!systemPromptBlocksEqual(finalBlocks, originalBlocks)) {
      return { systemPrompt: finalBlocks };
    }
    return {};
  });

  pi.on("before_provider_request", (event, ctx) => {
    const requestModel = resolveRouteModel(ctx.model, ctx) ?? ctx.model;
    let mutated = false;
    let resultPayload = event.payload;

    // 仅安全网：若 before_agent_start 未剥离 session-overview churn
    //（钩子未跑 / 被后续覆盖），此处只 strip RECENT COMMITS，不做 skills/reorder。
    if (
      runtimeOptimizerEnabled &&
      isPromptRewriteEnabled() &&
      requestModel &&
      !isResponsesPromptRewriteBypassApi(requestModel.api)
    ) {
      const original = extractSystemPrompt(resultPayload);
      if (
        original &&
        original.includes("<session-overview>") &&
        original.includes("RECENT COMMITS")
      ) {
        const stripped = stripSessionOverviewChurn(original);
        if (stripped !== original && setSystemPrompt(resultPayload, stripped)) {
          mutated = true;
          if (latestCacheHint) latestCacheHint.systemPrompt = stripped;
        }
      }
    }

    const requestDiagnosticsModel = requestModel ?? ctx.model;
    if (requestDiagnosticsModel) {
      const payloadPrompt = extractSystemPrompt(resultPayload);
      const hintPrompt = latestCacheHint?.systemPrompt;
      const header = ctx.sessionManager.getHeader?.();
      pendingRequestDiagnosticsByModelKey.set(sessionModelKey(requestDiagnosticsModel), {
        promptRewriteEnabled: isPromptRewriteEnabled(),
        systemPromptFingerprint: fingerprintPrompt(payloadPrompt),
        promptCacheKeySource: getPromptCacheKeySource(header, ctx.sessionManager.getSessionId()),
        hintPayloadComparison: comparePromptFingerprints(
          fingerprintPrompt(hintPrompt),
          fingerprintPrompt(payloadPrompt),
        ),
      });
    }

    // ── prompt_cache_key injection (OpenAI-compatible APIs) ──
    // OMP 17 host injects prompt_cache_key into the body for openai-responses /
    // azure-openai-responses, but NOT for openai-completions (chat) wire. We inject
    // a cache key for ALL OpenAI-compatible APIs, but only when the payload doesn't
    // already carry one — so responses (host-injected) are automatically skipped,
    // and completions (host-skipped) get the key they need for proxy routing affinity.
    // The key is read from the host header (providerPromptCacheKey) or session id fallback,
    // and clamped to OpenAI's 64-char prompt_cache_key limit (see clampPromptCacheKey).
    // For Anthropic / Google / other non-OpenAI APIs, prompt_cache_key is not a valid
    // body parameter and isOpenAICompatibleApi excludes them.
    if (runtimeOptimizerEnabled && isOpenAICompatibleApi(requestModel?.api)) {
      const payloadRecord = asRecord(resultPayload);
      const cacheKey = clampPromptCacheKey(getSessionPromptCacheKey(ctx));
      if (
        payloadRecord &&
        isNonEmptyString(cacheKey) &&
        !isNonEmptyString(payloadRecord.prompt_cache_key) &&
        !isNonEmptyString(payloadRecord.promptCacheKey)
      ) {
        payloadRecord.prompt_cache_key = cacheKey;
        mutated = true;
      }
    }

    // ── Safety: strip prompt_cache_retention for 400-history models ──
    // OMP divergence: Pi defaults supportsLongCacheRetention to true for all
    // openai-completions models and runs a 4-gate safety check. OMP's pi-ai
    // only injects prompt_cache_retention when supportsLongPromptCacheRetention
    // is explicitly true in models.yml (openai-responses path). We therefore
    // need only two gates:
    //   Gate 1 – user opt-in (supportsLongPromptCacheRetention: true) → keep
    //   Gate 2 – 400 history (this process) → strip (overrides Gate 1)
    // Gate 2 is critical: if the user opted in but the endpoint returned 400,
    // we must strip — otherwise the 400 repeats every request.
    if (runtimeOptimizerEnabled) {
      const payloadRecord = asRecord(resultPayload);
      if (payloadRecord && typeof payloadRecord.prompt_cache_retention === "string") {
        const stripModel = resolveRouteModel(ctx.model, ctx) ?? ctx.model;
        if (stripModel) {
          if (promptCacheRetention400Models.has(modelKey(stripModel))) {
            // Gate 2: 400 history → strip (empirical evidence overrides user opt-in)
            delete payloadRecord.prompt_cache_retention;
            mutated = true;
          } else if (getCompat(stripModel).supportsLongPromptCacheRetention !== true) {
            // Safety net: no explicit user opt-in → strip.
            // pi-ai requires supportsLongPromptCacheRetention: true before it injects
            // prompt_cache_retention; if the field is present without that compat flag,
            // another source injected it — strip to prevent 400s on third-party proxies.
            delete payloadRecord.prompt_cache_retention;
            mutated = true;
          }
          // Implicit Gate 1: user opted in AND no 400 history → keep
        }
      }
    }

    return mutated ? resultPayload : undefined;
  });

  pi.on("after_provider_response", (event, ctx) => {
    const model = resolveRouteModel(ctx.model, ctx) ?? ctx.model;
    if (!runtimeOptimizerEnabled || !model) return;
    if (event.status !== 400) return;
    if (!isPromptCacheRetention400Applicable(model)) return;
    if (!hasPromptCacheRetentionUnsupportedSignal(event.headers)) return;

    const key = modelKey(model);
    promptCacheRetention400Models.add(key);
    if (warnedPromptCacheRetention400Models.has(key)) return;
    warnedPromptCacheRetention400Models.add(key);
    ctx.ui.notify(
      `⚠️ ${LOG_PREFIX}：${key} 在启用 supportsLongPromptCacheRetention 时返回了 HTTP 400。` +
      getPromptCacheRetentionUnsupportedHint() +
      ` 可运行 /cache-optimizer doctor 查看精确编辑位置。`,
      "warning",
    );
  });

  pi.on("message_end", async (event, ctx) => {
    syncSessionHash(ctx);
    const adapter = selectAdapterForAssistantMessage(event.message, ctx.model);
    if (!adapter) return;

    const usage = adapter.normalizeUsage(event.message);

    // Completed message metadata is request-local and authoritative for virtual
    // routing providers. Use it whenever it supplies provider/model identity;
    // fall back to the active context model for direct providers.
    const statsModel = modelFromAssistantMessage(event.message, ctx.model) ?? ctx.model;
    let routedModelChanged = false;
    if (isVirtualRoutingModel(ctx.model, ctx) && statsModel && !isVirtualRoutingModel(statsModel, ctx)) {
      const nextRoutedModel: PersistedRoutedModelRef = {
        provider: statsModel.provider,
        id: statsModel.id,
        name: statsModel.name || statsModel.id,
      };
      if (
        !lastActualRoutedModel ||
        lastActualRoutedModel.provider !== nextRoutedModel.provider ||
        lastActualRoutedModel.id !== nextRoutedModel.id ||
        (lastActualRoutedModel.name || lastActualRoutedModel.id) !== (nextRoutedModel.name || nextRoutedModel.id)
      ) {
        lastActualRoutedModel = nextRoutedModel;
        routedModelChanged = true;
      }
    }

    // Record recent sample (even when usage is missing, for trend diagnosis)
    if (statsModel) {
      const sk = sessionModelKey(statsModel);
      const missingFields = usage === undefined || (usage.cacheRead === 0 && usage.cacheWrite === 0 && usage.totalInput === 0)
        ? true
        : hasMissingUsageFields(event.message, adapter);
      const diagnostics = pendingRequestDiagnosticsByModelKey.get(sk)
        ?? (ctx.model ? pendingRequestDiagnosticsByModelKey.get(sessionModelKey(ctx.model)) : undefined);
      pendingRequestDiagnosticsByModelKey.delete(sk);
      if (ctx.model) pendingRequestDiagnosticsByModelKey.delete(sessionModelKey(ctx.model));
      recordRecentSample(sk, usage ?? { cacheRead: 0, cacheWrite: 0, totalInput: 0 }, missingFields, diagnostics);
    }

    if (!usage) {
      if (routedModelChanged) schedulePersistCacheStats(ctx);
      return;
    }

    await rollOverStatsIfNeeded(ctx);

    // Update stats scoped to current session + actual routed model.
    // Falls back to legacy family when no model is available.
    if (statsModel) {
      const sk = sessionModelKey(statsModel);
      addUsageToCacheStats(getOrCreateStatsByModelKey(sk), usage);
    } else {
      addUsageToCacheStats(getStatsForModel(undefined, adapter), usage);
    }

    schedulePersistCacheStats(ctx);
    await publishStatus(ctx, statsModel);
  });

  // ────────────────────────────────────────────────────────────────
  // Register /cache-optimizer command
  // Subcommands:
  //   enable  — enable runtime prompt/cache optimizations for this process
  //   disable — disable runtime prompt/cache optimizations for this process
  //   doctor  — show current model/provider/api/baseUrl/compat status
  //             with low-hit diagnosis
  //   stats   — show active model stats bucket, recent trend, usage
  //   compat  — show compat suggestion with file path
  //   fix     — auto-fix compat issues (writes models.yml, requires UI)
  //   reset   — reset current session model stats bucket (local only)
  //   (no args) — interactive menu (with UI) or help summary
  // ────────────────────────────────────────────────────────────────
  pi.registerCommand("cache-optimizer", {
    description: "Diagnose OMP cache configuration",
    handler: async (args: string, cmdCtx) => {
      syncSessionHash(cmdCtx);
      const selectedModel = cmdCtx.model;
      const model = resolveRouteModel(selectedModel, cmdCtx as unknown as ExtensionContext) ?? selectedModel;
      const subcommand = args.trim().toLowerCase().split(/\s+/)[0] || "help";

      if (subcommand === "enable") {
        setRuntimeOptimizerEnabled(true);
        resetCurrentSessionStats();
        await flushPersistCacheStats(cmdCtx as unknown as ExtensionContext);
        await publishStatus(cmdCtx as unknown as ExtensionContext, model);
        cmdCtx.ui.notify(`✅ 已为当前 OMP 进程开启缓存优化。已重置当前 session 统计，方便做前后对比。\n${formatOptimizerRuntimeMode()}`, "info");
      } else if (subcommand === "disable") {
        setRuntimeOptimizerEnabled(false);
        resetCurrentSessionStats();
        await flushPersistCacheStats(cmdCtx as unknown as ExtensionContext);
        await publishStatus(cmdCtx as unknown as ExtensionContext, model);
        cmdCtx.ui.notify(`⏸️ 已为当前 OMP 进程关闭缓存优化。已重置当前 session 统计，并会在关闭状态下继续采集用于对比。\n${formatOptimizerRuntimeMode()}`, "warning");
      } else if (subcommand === "doctor") {
        if (!model) {
          cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
          return;
        }
        const diagnosis = buildDoctorDiagnosis(model, { promptCacheRetention400: promptCacheRetention400Models.has(modelKey(model)) });
        const adapter = selectAdapterForModel(model);
        const sk = model ? sessionModelKey(model) : undefined;
        const statsState = sk ? cacheStatsByModel[sk] : undefined;
        const samples = sk ? getRecentSamples(sk) : [];
        const lowHitLines = buildLowHitDiagnosis(model, adapter, statsState, samples);
        const fullDiagnosis = lowHitLines.length > 0
          ? diagnosis + "\n" + lowHitLines.join("\n")
          : diagnosis;
        cmdCtx.ui.notify(fullDiagnosis, "info");
      } else if (subcommand === "stats") {
        if (!model) {
          cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
          return;
        }
        const adapter = selectAdapterForModel(model);
        const sk = model ? sessionModelKey(model) : undefined;
        const statsState = sk ? cacheStatsByModel[sk] : undefined;
        const samples = sk ? getRecentSamples(sk) : [];
        const output = buildStatsOutput(model, adapter, statsState, samples);
        cmdCtx.ui.notify(output, "info");
      } else if (subcommand === "compat") {
        if (!model) {
          cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
          return;
        }
        const compatResult = buildCompatDiagnosis(model);
        if (compatResult) {
          cmdCtx.ui.notify(compatResult, "warning");
        } else {
          cmdCtx.ui.notify(
            isAdaptiveThinkingCompatApplicable(model) || isDeepSeekCompatCheckApplicable(model) || isCompatCheckApplicable(model)
              ? "✅ compat 配置完整。"
              : getCompatCheckNotApplicableLines(model).join("\n"),
            "info",
          );
        }
      } else if (subcommand === "reset") {
        if (!model) {
          cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
          return;
        }
        const adapter = selectAdapterForModel(model);
        if (!adapter) {
          cmdCtx.ui.notify("ℹ️ 当前活动模型未匹配到缓存适配器，无需重置统计。", "info");
          return;
        }

        const displayKey = modelKey(model);

        // Reset session-scoped stats for the effective active model. If the
        // selected model is a virtual router and the protocol exposes a live
        // route, this clears the real upstream bucket, not the router shell.
        resetStatsForModel(model);

        // Persist immediately.
        await flushPersistCacheStats(cmdCtx as unknown as ExtensionContext);

        // Update footer to show 0/0.
        await publishStatus(cmdCtx as unknown as ExtensionContext, model);

        cmdCtx.ui.notify(
          `✅ 已重置 "${displayKey}" 的本地 session 缓存统计。` +
          "上游 provider 的 prompt cache 未被修改。" +
          "后续请求会为当前 OMP session 开始新的统计桶。",
          "info",
        );
      } else if (subcommand === "fix") {
        if (!model) {
          cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
          return;
        }

        const suggestion = buildFixSuggestion(model);
        if (!suggestion) {
          const key = modelKey(model);
          cmdCtx.ui.notify(`✅ "${key}" 当前无需修复，compat 已配置完成。`, "info");
          return;
        }

        // OMP divergence: auto-write YAML surgical editor is not yet implemented.
        // /cache-optimizer fix shows copyable YAML compat snippet + manual steps.
        // A future YAML editor must implement its own parser/self-check rather
        // than reusing the removed JSONC surgical editor from the source project.
        const compatResult = buildCompatDiagnosis(model);
        const yamlSnippet = formatCompatKeysForInsertion(suggestion.compatKeys);
        cmdCtx.ui.notify(
          `📝 ${getModelsYmlDisplayPath()} 的手动修复建议：\n\n` +
          `提供方：${suggestion.providerLabel}\n` +
          `模型：${suggestion.modelId}\n\n` +
          `在模型级 compat（模型条目下）添加这些键：\n\n` +
          `compat:\n${yamlSnippet}\n\n` +
          `或放到 provider 级（providers["${suggestion.providerLabel}"] 下）：\n\n` +
          `compat:\n${yamlSnippet}\n\n` +
          `编辑后运行 /reload。\n` +
          (compatResult ? `\n${compatResult}` : ""),
          "info",
        );
      } else {
        // Try interactive selection menu when UI supports it
        if (cmdCtx.hasUI) {
          const menuOptions = [
            "启用 —— 打开运行时优化",
            "关闭 —— 关闭运行时优化",
            "诊断 —— 查看缓存配置",
            "统计 —— 查看缓存统计与趋势",
            "兼容 —— 查看 compat 建议",
            "修复 —— 查看 compat 修复建议（会写 models.yml 时另行提示）",
            "重置 —— 重置本地 session 统计",
            "取消",
          ];
          const choice = await cmdCtx.ui.select("缓存优化器", menuOptions);
          if (choice === menuOptions[0]) {
            setRuntimeOptimizerEnabled(true);
            resetCurrentSessionStats();
            await flushPersistCacheStats(cmdCtx as unknown as ExtensionContext);
            await publishStatus(cmdCtx as unknown as ExtensionContext, model);
            cmdCtx.ui.notify(`✅ 已为当前 OMP 进程开启缓存优化。已重置当前 session 统计，方便做前后对比。\n${formatOptimizerRuntimeMode()}`, "info");
          } else if (choice === menuOptions[1]) {
            setRuntimeOptimizerEnabled(false);
            resetCurrentSessionStats();
            await flushPersistCacheStats(cmdCtx as unknown as ExtensionContext);
            await publishStatus(cmdCtx as unknown as ExtensionContext, model);
            cmdCtx.ui.notify(`⏸️ 已为当前 OMP 进程关闭缓存优化。已重置当前 session 统计，并会在关闭状态下继续采集用于对比。\n${formatOptimizerRuntimeMode()}`, "warning");
          } else if (choice === menuOptions[2]) {
            if (!model) {
              cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
            } else {
              const diagnosis = buildDoctorDiagnosis(model, { promptCacheRetention400: promptCacheRetention400Models.has(modelKey(model)) });
              const adapter = selectAdapterForModel(model);
              const sk = model ? sessionModelKey(model) : undefined;
              const statsState = sk ? cacheStatsByModel[sk] : undefined;
              const samples = sk ? getRecentSamples(sk) : [];
              const lowHitLines = buildLowHitDiagnosis(model, adapter, statsState, samples);
              const fullDiagnosis = lowHitLines.length > 0
                ? diagnosis + "\n" + lowHitLines.join("\n")
                : diagnosis;
              cmdCtx.ui.notify(fullDiagnosis, "info");
            }
          } else if (choice === menuOptions[3]) {
            if (!model) {
              cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
            } else {
              const adapter = selectAdapterForModel(model);
              const sk = model ? sessionModelKey(model) : undefined;
              const statsState = sk ? cacheStatsByModel[sk] : undefined;
              const samples = sk ? getRecentSamples(sk) : [];
              const output = buildStatsOutput(model, adapter, statsState, samples);
              cmdCtx.ui.notify(output, "info");
            }
          } else if (choice === menuOptions[4]) {
            if (!model) {
              cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
            } else {
              const compatResult = buildCompatDiagnosis(model);
              if (compatResult) {
                cmdCtx.ui.notify(compatResult, "warning");
              } else {
                cmdCtx.ui.notify(
                  isAdaptiveThinkingCompatApplicable(model) || isDeepSeekCompatCheckApplicable(model) || isCompatCheckApplicable(model)
                    ? "✅ compat 配置完整。"
                    : getCompatCheckNotApplicableLines(model).join("\n"),
                  "info",
                );
              }
            }
          } else if (choice === menuOptions[5]) {
            // Fix — auto-fix compat issues
            if (!model) {
              cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
              return;
            }
            const suggestion = buildFixSuggestion(model);
            if (!suggestion) {
              const key = modelKey(model);
              cmdCtx.ui.notify(`✅ "${key}" 当前无需修复，compat 已配置完成。`, "info");
              return;
            }

            // OMP divergence: auto-write YAML surgical editor is not yet implemented.
            const compatResult = buildCompatDiagnosis(model);
            const yamlSnippet = formatCompatKeysForInsertion(suggestion.compatKeys);
            cmdCtx.ui.notify(
              `📝 ${getModelsYmlDisplayPath()} 的手动修复建议：\n\n` +
              `提供方：${suggestion.providerLabel}\n` +
              `模型：${suggestion.modelId}\n\n` +
              `添加这些 compat 键：\n\n` +
              `compat:\n${yamlSnippet}\n\n` +
              `编辑后运行 /reload。\n` +
              (compatResult ? `\n${compatResult}` : ""),
              "info",
            );
          } else if (choice === menuOptions[6]) {
            if (!model) {
              cmdCtx.ui.notify("当前没有活动模型。请先用 /model 或 omp --model 选择模型。", "warning");
            } else {
              const adapter = selectAdapterForModel(model);
              if (!adapter) {
                cmdCtx.ui.notify("ℹ️ 当前活动模型未匹配到缓存适配器，无需重置统计。", "info");
              } else {
                const displayKey = modelKey(model);
                resetStatsForModel(model);
                await flushPersistCacheStats(cmdCtx as unknown as ExtensionContext);
                await publishStatus(cmdCtx as unknown as ExtensionContext, model);
                cmdCtx.ui.notify(
                  `✅ 已重置 "${displayKey}" 的本地 session 缓存统计。` +
                  "上游 provider 的 prompt cache 未被修改。",
                  "info",
                );
              }
            }
          }
          // choice === "cancel" or undefined → no action
          return;
        }

        // Fallback: text help when no interactive UI
        const diagnosis: string[] = [];
        diagnosis.push("📋 /cache-optimizer 命令：");
        diagnosis.push("  enable  —— 为当前 OMP 进程开启 prompt/cache 优化");
        diagnosis.push("  disable —— 为当前 OMP 进程关闭 prompt/cache 优化");
        diagnosis.push("  doctor  —— 查看当前模型/provider/api/baseUrl/compat 与低命中诊断");
        diagnosis.push("  stats   —— 查看当前活动模型的统计桶与近期趋势");
        diagnosis.push("  compat  —— 查看 compat 建议与编辑位置");
        diagnosis.push("  fix     —— 查看 compat 修复建议（需要 UI 时另有提示）");
        diagnosis.push("  reset   —— 重置当前模型的本地 session 统计（不影响上游）");
        diagnosis.push("");
        diagnosis.push(formatOptimizerRuntimeMode());
        diagnosis.push("");
        if (model) {
          const displayKey = modelKey(model);
          const missing = describeMissingCacheCompatForModel(model);
          if (missing.length > 0) {
            diagnosis.push(`⚠️ 当前模型 "${displayKey}" 缺少 compat：${missing.join(", ")}`);
            diagnosis.push('可运行 "/cache-optimizer compat" 查看编辑建议。');
          } else if (isAdaptiveThinkingCompatApplicable(model) || isDeepSeekCompatCheckApplicable(model) || isCompatCheckApplicable(model)) {
            diagnosis.push(`✅ 当前模型 "${displayKey}"：compat 配置完整。`);
          } else {
            diagnosis.push(`ℹ️ 当前模型 "${displayKey}"：不适用 compat 检查。`);
            const detailLines = getCompatCheckNotApplicableLines(model).slice(1);
            for (const line of detailLines) diagnosis.push(line);
          }
        } else {
          diagnosis.push("当前没有活动模型。");
        }
        cmdCtx.ui.notify(diagnosis.join("\n"), "info");
      }
    },
  });
}
