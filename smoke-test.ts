#!/usr/bin/env bun
/**
 * OMP Cache Optimizer 冒烟测试
 * 运行: bun smoke-test.ts
 *
 * 验证移植后的核心函数行为正确：
 * 1. 模型检测（DeepSeek/OpenAI/Claude）
 * 2. compat 字段重映射（DeepSeek 返回新字段名）
 * 3. 统计计算（addUsageToCacheStats + formatCacheStats）
 * 4. extractSystemPrompt / setSystemPrompt（三种 payload 格式）
 * 5. buildFixSuggestion（DeepSeek 模型返回 requiresReasoningContentForToolCalls）
 * 6. OMP 17 session-overview churn strip + hook 级回归
 */

// 默认不改写；仅 OMP_CACHE_OPTIMIZER_PROMPT_REWRITE=1 可显式启用。
// 历史 NO_PROMPT_REWRITE 变量不再影响行为。
delete process.env.OMP_CACHE_OPTIMIZER_PROMPT_REWRITE;
delete process.env.OMP_CACHE_OPTIMIZER_NO_PROMPT_REWRITE;
delete process.env.PI_CACHE_OPTIMIZER_NO_PROMPT_REWRITE;

import cacheOptimizer, { __internals_for_tests } from "./index.ts";

const {
  isDeepSeekLikeModel,
  isOpenAIFamilyModel,
  isClaudeLikeModel,
  isAdaptiveGenerationModel,
  describeMissingDeepSeekCompat,
  buildDeepSeekCompatSuggestion,
  buildFixSuggestion,
  addUsageToCacheStats,
  formatCacheStats,
  formatRecentTokenHitRate,
  emptyCacheStats,
  extractSystemPrompt,
  setSystemPrompt,
  asRecord,
  stripSessionOverviewChurn,
  mapSystemPromptBlocks,
  systemPromptBlocksEqual,
  fingerprintPrompt,
  getPromptCacheKeySource,
  isOpenAICompatibleApi,
  isOpenAIResponsesLongRetentionApi,
  buildProjectPromptCacheKey,
  normalizePromptCacheKeyForWire,
  configureCacheRetentionEnv,
  captureCacheRetentionEnv,
  restoreCacheRetentionEnv,
  describeOptionalOpenAIResponsesRetentionCompat,
  trackPendingOperation,
  waitForPendingOperations,
  sessionIdentityChanged,
  OMP_CACHE_RETENTION_ENV,
  PI_CACHE_RETENTION_ENV,
  LONG_CACHE_RETENTION_VALUE,
} = __internals_for_tests;

let passed = 0;
let failed = 0;

function expect(label: string, condition: boolean, message = ""): void {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`✗ ${label}: ${message}`);
  }
}

function makeModel(opts: {
  id: string;
  provider?: string;
  api?: string;
  name?: string;
  baseUrl?: string;
  compat?: Record<string, unknown>;
}) {
  return {
    provider: opts.provider ?? "test",
    id: opts.id,
    name: opts.name,
    api: opts.api,
    baseUrl: opts.baseUrl,
    compat: opts.compat,
  } as Parameters<typeof isDeepSeekLikeModel>[0];
}

// ── 1. 模型检测 ──────────────────────────────────────────────────

expect(
  "isDeepSeekLikeModel.deepseek-v4-pro",
  isDeepSeekLikeModel(makeModel({ id: "deepseek-v4-pro", provider: "opencode-go" })) === true,
  "deepseek-v4-pro 应被识别为 DeepSeek",
);

expect(
  "isDeepSeekLikeModel.glm-5.2",
  isDeepSeekLikeModel(makeModel({ id: "glm-5.2", provider: "opencode-go" })) === false,
  "glm-5.2 不应被识别为 DeepSeek",
);

expect(
  "isOpenAIFamilyModel.gpt-5.5",
  isOpenAIFamilyModel(makeModel({ id: "gpt-5.5", provider: "opencode-zen" })) === true,
  "gpt-5.5 应被识别为 OpenAI family",
);

expect(
  "isClaudeLikeModel.claude-opus-4-8",
  isClaudeLikeModel(makeModel({ id: "claude-opus-4-8", provider: "opencode-zen" })) === true,
  "claude-opus-4-8 应被识别为 Claude",
);

expect(
  "isAdaptiveGenerationModel.opus-4-6",
  isAdaptiveGenerationModel(makeModel({ id: "claude-opus-4-6", provider: "opencode-zen" })) === true,
  "opus-4-6 应被识别为 adaptive generation",
);

expect(
  "isAdaptiveGenerationModel.opus-4-1",
  isAdaptiveGenerationModel(makeModel({ id: "claude-opus-4-1", provider: "opencode-zen" })) === false,
  "opus-4-1 不应被识别为 adaptive generation",
);

// ── 2. compat 字段重映射 ─────────────────────────────────────────

const deepseekModel = makeModel({
  id: "deepseek-v4-pro",
  provider: "opencode-go",
  api: "openai-completions",
  baseUrl: "https://api.example.com/v1",
});

const deepseekMissing = describeMissingDeepSeekCompat(deepseekModel);
expect(
  "describeMissingDeepSeekCompat.returns-requiresReasoningContentForToolCalls",
  deepseekMissing.includes("requiresReasoningContentForToolCalls") === true,
  `应返回 requiresReasoningContentForToolCalls，实际: ${JSON.stringify(deepseekMissing)}`,
);
expect(
  "describeMissingDeepSeekCompat.no-long-retention-or-legacy-fields",
  deepseekMissing.includes("supportsLongPromptCacheRetention") === false &&
    deepseekMissing.includes("sendSessionAffinityHeaders") === false &&
    deepseekMissing.includes("sendSessionIdHeader") === false &&
    deepseekMissing.includes("supportsLongCacheRetention") === false &&
    deepseekMissing.includes("requiresReasoningContentOnAssistantMessages") === false &&
    deepseekMissing.includes("thinkingFormat") === false,
  `不应包含 long retention 或旧字段名，实际: ${JSON.stringify(deepseekMissing)}`,
);

const deepseekSuggestion = buildDeepSeekCompatSuggestion(deepseekMissing);
expect(
  "buildDeepSeekCompatSuggestion.has-new-keys",
  Object.prototype.hasOwnProperty.call(deepseekSuggestion, "requiresReasoningContentForToolCalls") === true &&
    Object.prototype.hasOwnProperty.call(deepseekSuggestion, "supportsLongPromptCacheRetention") === false,
  `建议应仅包含 reasoning 字段，实际: ${JSON.stringify(deepseekSuggestion)}`,
);
expect(
  "buildDeepSeekCompatSuggestion.no-legacy-keys",
  Object.prototype.hasOwnProperty.call(deepseekSuggestion, "sendSessionAffinityHeaders") === false &&
    Object.prototype.hasOwnProperty.call(deepseekSuggestion, "forceAdaptiveThinking") === false &&
    Object.prototype.hasOwnProperty.call(deepseekSuggestion, "thinkingFormat") === false,
  `建议不应包含旧字段名，实际: ${JSON.stringify(deepseekSuggestion)}`,
);

// buildFixSuggestion 对 DeepSeek 模型应返回非 undefined
const fixSuggestion = buildFixSuggestion(deepseekModel);
expect(
  "buildFixSuggestion.deepseek-returns-suggestion",
  fixSuggestion !== undefined,
  "DeepSeek 模型应返回 fix 建议",
);
if (fixSuggestion) {
  expect(
    "buildFixSuggestion.deepseek-provider-label",
    fixSuggestion.providerLabel === "opencode-go",
    `providerLabel 应为 opencode-go，实际: ${fixSuggestion.providerLabel}`,
  );
  expect(
    "buildFixSuggestion.deepseek-compat-keys-new",
    Object.prototype.hasOwnProperty.call(fixSuggestion.compatKeys, "requiresReasoningContentForToolCalls") === true &&
      Object.prototype.hasOwnProperty.call(fixSuggestion.compatKeys, "supportsLongPromptCacheRetention") === false,
    `compatKeys 应仅包含 reasoning 字段，实际: ${JSON.stringify(fixSuggestion.compatKeys)}`,
  );
}

// ── 3. 统计计算 ──────────────────────────────────────────────────

const stats = emptyCacheStats("2026-06-21");
addUsageToCacheStats(stats, { cacheRead: 800, cacheWrite: 0, totalInput: 1000 });
addUsageToCacheStats(stats, { cacheRead: 0, cacheWrite: 0, totalInput: 1000 });
expect(
  "addUsageToCacheStats.totalRequests",
  stats.totalRequests === 2,
  `应为 2 次请求，实际: ${stats.totalRequests}`,
);
expect(
  "addUsageToCacheStats.hitRequests",
  stats.hitRequests === 1,
  `应为 1 次命中，实际: ${stats.hitRequests}`,
);
expect(
  "addUsageToCacheStats.cachedInputTokens",
  stats.cachedInputTokens === 800,
  `应为 800 cached tokens，实际: ${stats.cachedInputTokens}`,
);
expect(
  "addUsageToCacheStats.totalInputTokens",
  stats.totalInputTokens === 2000,
  `应为 2000 total tokens，实际: ${stats.totalInputTokens}`,
);

// formatCacheStats 需要 adapter 参数
const formatted = formatCacheStats(
  { id: "openai", label: "OpenAI cache", showCacheWrite: false } as Parameters<typeof formatCacheStats>[0],
  stats,
);
expect(
  "formatCacheStats.token-hit-label",
  formatted.includes("缓存命中率：40%") === true,
  `应包含「缓存命中率：40%」，实际: "${formatted}"`,
);
expect(
  "formatCacheStats.request-hit-label",
  formatted.includes("缓存请求命中次数：1/2 次") === true,
  `应包含请求命中次数标签，实际: "${formatted}"`,
);
expect(
  "formatCacheStats.token-volume-label",
  formatted.includes("缓存token/总输入：800/2.00k") === true ||
    formatted.includes("缓存token/总输入：800/2000") === true,
  `应包含 token 量标签，实际: "${formatted}"`,
);
expect(
  "formatCacheStats.uses-cache-word",
  formatted.includes("OpenAI Cache") === true && formatted.includes(" | ") === true,
  `应包含 OpenAI Cache 与分隔符，实际: "${formatted}"`,
);

function makeUsageSample(partial: {
  cachedInputTokens: number;
  totalInputTokens: number;
  missingUsageFields?: boolean;
}): {
  timestamp: number;
  hit: boolean;
  cachedInputTokens: number;
  cacheWriteInputTokens: number;
  totalInputTokens: number;
  missingUsageFields: boolean;
} {
  return {
    timestamp: Date.now(),
    hit: partial.cachedInputTokens > 0,
    cachedInputTokens: partial.cachedInputTokens,
    cacheWriteInputTokens: 0,
    totalInputTokens: partial.totalInputTokens,
    missingUsageFields: partial.missingUsageFields ?? false,
  };
}

const recentWindowSamples = [
  makeUsageSample({ cachedInputTokens: 0, totalInputTokens: 1000 }),
  ...Array.from({ length: 9 }, () => makeUsageSample({ cachedInputTokens: 80, totalInputTokens: 100 })),
  makeUsageSample({ cachedInputTokens: 90, totalInputTokens: 100 }),
];
expect(
  "formatRecentTokenHitRate.uses-last-10-valid-samples",
  formatRecentTokenHitRate(recentWindowSamples, 10) === "最近10次token命中率：81%",
  `最近 11 个样本应只计最后 10 个，实际: ${JSON.stringify(formatRecentTokenHitRate(recentWindowSamples, 10))}`,
);
const mixedRecentSamples = [
  makeUsageSample({ cachedInputTokens: 0, totalInputTokens: 0, missingUsageFields: true }),
  makeUsageSample({ cachedInputTokens: 80, totalInputTokens: 100 }),
];
expect(
  "formatRecentTokenHitRate.excludes-missing-and-zero-input",
  formatRecentTokenHitRate(mixedRecentSamples, 10) === "最近1次token命中率：80%" &&
    formatRecentTokenHitRate([makeUsageSample({ cachedInputTokens: 0, totalInputTokens: 0, missingUsageFields: true })], 10) === undefined &&
    formatRecentTokenHitRate([makeUsageSample({ cachedInputTokens: 0, totalInputTokens: 0 })], 10) === undefined,
  `缺失/零输入样本应排除，实际: ${JSON.stringify({
    mixed: formatRecentTokenHitRate(mixedRecentSamples, 10),
    missingOnly: formatRecentTokenHitRate([makeUsageSample({ cachedInputTokens: 0, totalInputTokens: 0, missingUsageFields: true })], 10),
    zeroOnly: formatRecentTokenHitRate([makeUsageSample({ cachedInputTokens: 0, totalInputTokens: 0 })], 10),
  })}`,
);
const formattedWithRecent = formatCacheStats(
  { id: "openai", label: "OpenAI cache", showCacheWrite: false } as Parameters<typeof formatCacheStats>[0],
  stats,
  [
    makeUsageSample({ cachedInputTokens: 0, totalInputTokens: 100 }),
    makeUsageSample({ cachedInputTokens: 80, totalInputTokens: 100 }),
  ],
);
expect(
  "formatCacheStats.appends-recent-token-hit-rate",
  formattedWithRecent.includes("缓存命中率：40%") &&
    formattedWithRecent.includes(" | 最近2次token命中率：40%") &&
    formattedWithRecent.indexOf("缓存命中率：40%") < formattedWithRecent.indexOf(" | 最近2次token命中率：40%"),
  `累计命中率后应追加最近指标，实际: "${formattedWithRecent}"`,
);
expect(
  "formatCacheStats.omits-recent-when-no-samples",
  formatted === formatCacheStats(
    { id: "openai", label: "OpenAI cache", showCacheWrite: false } as Parameters<typeof formatCacheStats>[0],
    stats,
  ) &&
    !formatted.includes("最近"),
  `无样本时输出应与当前格式完全相同，实际: "${formatted}"`,
);

// ── 4. extractSystemPrompt / setSystemPrompt ─────────────────────

// OpenAI completions 格式: messages[0].content (string)
const openaiPayload = {
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello" },
  ],
};
const openaiPrompt = extractSystemPrompt(openaiPayload);
expect(
  "extractSystemPrompt.openai-string",
  openaiPrompt === "You are a helpful assistant.",
  `OpenAI 格式应提取 system prompt，实际: "${openaiPrompt}"`,
);

const openaiSet = setSystemPrompt(openaiPayload, "Modified system prompt.");
expect(
  "setSystemPrompt.openai-success",
  openaiSet === true,
  "OpenAI 格式应成功写入",
);
expect(
  "setSystemPrompt.openai-verify",
  extractSystemPrompt(openaiPayload) === "Modified system prompt.",
  `写入后应能读回，实际: "${extractSystemPrompt(openaiPayload)}"`,
);

// Anthropic 格式: payload.system (string)
const anthropicPayload = {
  system: "You are Claude.",
  messages: [{ role: "user", content: "Hi" }],
};
const anthropicPrompt = extractSystemPrompt(anthropicPayload);
expect(
  "extractSystemPrompt.anthropic-string",
  anthropicPrompt === "You are Claude.",
  `Anthropic 格式应提取 system prompt，实际: "${anthropicPrompt}"`,
);
setSystemPrompt(anthropicPayload, "Modified Claude prompt.");
expect(
  "setSystemPrompt.anthropic-verify",
  extractSystemPrompt(anthropicPayload) === "Modified Claude prompt.",
  `Anthropic 写入后应能读回，实际: "${extractSystemPrompt(anthropicPayload)}"`,
);

// Anthropic 格式: payload.system (content blocks array)
const anthropicBlocksPayload = {
  system: [{ type: "text", text: "Block-based system prompt." }],
  messages: [],
};
const blocksPrompt = extractSystemPrompt(anthropicBlocksPayload);
expect(
  "extractSystemPrompt.anthropic-blocks",
  blocksPrompt === "Block-based system prompt.",
  `Anthropic blocks 格式应提取 text，实际: "${blocksPrompt}"`,
);

// Google 格式: payload.systemInstruction.parts
const googlePayload = {
  systemInstruction: {
    parts: [{ text: "Google system instruction." }],
  },
  contents: [],
};
const googlePrompt = extractSystemPrompt(googlePayload);
expect(
  "extractSystemPrompt.google",
  googlePrompt === "Google system instruction.",
  `Google 格式应提取 systemInstruction，实际: "${googlePrompt}"`,
);


const anthropicStructuredPayload = {
  system: [
    { type: "text", text: "Old Anthropic prompt.", cache_control: { type: "ephemeral" }, foo: "bar" },
    { type: "image", source: { type: "base64", media_type: "image/png", data: "abc" } },
  ],
  messages: [],
};
const anthropicStructuredSet = setSystemPrompt(anthropicStructuredPayload, "New Anthropic prompt.");
const anthropicStructuredFirst = asRecord(anthropicStructuredPayload.system[0]);
expect(
  "setSystemPrompt.anthropic-block-array-success",
  anthropicStructuredSet === true,
  "Anthropic block array 应成功写入",
);
expect(
  "setSystemPrompt.anthropic-block-array-roundtrip",
  extractSystemPrompt(anthropicStructuredPayload) === "New Anthropic prompt.",
  `Anthropic block array 写入后应精确读回，实际: "${extractSystemPrompt(anthropicStructuredPayload)}"`,
);
expect(
  "setSystemPrompt.anthropic-block-array-keeps-array",
  Array.isArray(anthropicStructuredPayload.system) === true,
  "Anthropic block array 写入后仍应是数组",
);
expect(
  "setSystemPrompt.anthropic-block-array-keeps-metadata",
  asRecord(anthropicStructuredFirst?.cache_control)?.type === "ephemeral" && anthropicStructuredFirst?.foo === "bar",
  `Anthropic 文本块额外字段应保留，实际: ${JSON.stringify(anthropicStructuredFirst)}`,
);
expect(
  "setSystemPrompt.anthropic-block-array-keeps-non-text",
  asRecord(anthropicStructuredPayload.system[1])?.type === "image",
  `Anthropic 非文本块应保留，实际: ${JSON.stringify(anthropicStructuredPayload.system)}`,
);

const googleStructuredPayload = {
  systemInstruction: {
    parts: [{ text: "Old Google prompt.", thought: true }, { inlineData: { mimeType: "image/png", data: "abc" } }],
  },
  contents: [],
};
const googleStructuredSet = setSystemPrompt(googleStructuredPayload, "New Google prompt.");
const googleStructuredFirst = asRecord(googleStructuredPayload.systemInstruction.parts[0]);
expect(
  "setSystemPrompt.google-parts-array-success",
  googleStructuredSet === true,
  "Google parts array 应成功写入",
);
expect(
  "setSystemPrompt.google-parts-array-roundtrip",
  extractSystemPrompt(googleStructuredPayload) === "New Google prompt.",
  `Google parts array 写入后应精确读回，实际: "${extractSystemPrompt(googleStructuredPayload)}"`,
);
expect(
  "setSystemPrompt.google-parts-array-keeps-array",
  Array.isArray(googleStructuredPayload.systemInstruction.parts) === true,
  "Google parts 写入后仍应是数组",
);
expect(
  "setSystemPrompt.google-parts-array-keeps-metadata",
  googleStructuredFirst?.thought === true,
  `Google 文本 part 额外字段应保留，实际: ${JSON.stringify(googleStructuredFirst)}`,
);

const openaiArrayPayload = {
  messages: [
    {
      role: "developer",
      content: [
        { type: "text", text: "Old OpenAI prompt.", cache_control: { type: "ephemeral" } },
        { type: "input_image", image_url: "data:image/png;base64,abc" },
      ],
    },
    { role: "user", content: "Hello" },
  ],
};
const openaiArraySet = setSystemPrompt(openaiArrayPayload, "New OpenAI prompt.");
const openaiArrayContent = openaiArrayPayload.messages[0].content;
expect(
  "setSystemPrompt.openai-content-array-success",
  openaiArraySet === true,
  "OpenAI content array 应成功写入",
);
expect(
  "setSystemPrompt.openai-content-array-keeps-array",
  Array.isArray(openaiArrayContent) === true,
  "OpenAI content array 写入后不能塌缩为字符串",
);
expect(
  "setSystemPrompt.openai-content-array-roundtrip",
  extractSystemPrompt(openaiArrayPayload) === "New OpenAI prompt.",
  `OpenAI content array 写入后应精确读回，实际: "${extractSystemPrompt(openaiArrayPayload)}"`,
);
expect(
  "setSystemPrompt.openai-content-array-keeps-non-text",
  Array.isArray(openaiArrayContent) && asRecord(openaiArrayContent[1])?.type === "input_image",
  `OpenAI content array 非文本项应保留，实际: ${JSON.stringify(openaiArrayContent)}`,
);

const collapsePayload = {
  system: [
    { type: "text", text: "First old prompt.", marker: "first" },
    { type: "tool_result", id: "tool-1" },
    { type: "text", text: "Second stale prompt.", marker: "second" },
  ],
  messages: [],
};
setSystemPrompt(collapsePayload, "Collapsed prompt.");
expect(
  "setSystemPrompt.block-array-collapses-extra-text",
  collapsePayload.system.filter((item) => typeof asRecord(item)?.text === "string").length === 1,
  `后续文本块应被丢弃，实际: ${JSON.stringify(collapsePayload.system)}`,
);
expect(
  "setSystemPrompt.block-array-collapse-roundtrip",
  extractSystemPrompt(collapsePayload) === "Collapsed prompt.",
  `多文本块写入后应精确读回，实际: "${extractSystemPrompt(collapsePayload)}"`,
);

const emptyArrayPayload = { system: [] as unknown[], messages: [] };
expect(
  "setSystemPrompt.empty-array-remains-false",
  setSystemPrompt(emptyArrayPayload, "Ignored prompt.") === false,
  "空数组应保持写入失败语义",
);
// 无 system prompt 的 payload
const noSystemPayload = { messages: [{ role: "user", content: "Hello" }] };
expect(
  "extractSystemPrompt.none-returns-undefined",
  extractSystemPrompt(noSystemPayload) === undefined,
  "无 system prompt 应返回 undefined",
);

// ── 4b. OMP 17 hook 级回归 harness ──────────────────────────────

type HarnessCtx = {
  model?: { provider: string; id: string; name?: string; api?: string; baseUrl?: string; compat?: Record<string, unknown> };
  modelRegistry: { find(p: string, id: string): unknown; getAvailable(): unknown[]; getAll(): unknown[] };
  sessionManager: {
    getSessionId(): string;
    getHeader?(): { providerPromptCacheKey?: string } | null;
    getCwd?(): string;
  };
  ui: {
    notify(m: string, l?: string): void;
    setStatus(k: string, v?: string): void;
    confirm(t: string, m: string): Promise<boolean>;
    select(t: string, o: string[]): Promise<string | undefined>;
  };
  cwd?: string;
  hasUI?: boolean;
};

function makeContext(overrides: {
  model?: HarnessCtx["model"];
  sessionManager?: Partial<HarnessCtx["sessionManager"]>;
  cwd?: string;
} = {}): HarnessCtx {
  const sessionId = overrides.sessionManager?.getSessionId ? overrides.sessionManager.getSessionId() : "test-session-id";
  const defaultHeader = { providerPromptCacheKey: "host-cache-key" };
  const header = overrides.sessionManager?.getHeader
    ? overrides.sessionManager.getHeader()
    : defaultHeader;
  const getCwd =
    overrides.sessionManager?.getCwd ??
    (() => overrides.cwd ?? "/tmp");
  return {
    model: overrides.model ?? { provider: "test", id: "test-model", api: "openai-completions", baseUrl: "https://proxy.example/v1" },
    modelRegistry: { find: () => undefined, getAvailable: () => [], getAll: () => [] },
    sessionManager: {
      getSessionId: () => sessionId,
      getHeader: () => header,
      getCwd,
    },
    ui: { notify: () => {}, setStatus: () => {}, confirm: async () => true, select: async () => undefined },
    cwd: overrides.cwd ?? "/tmp",
    hasUI: false,
  };
}

function createExtensionHarness(): {
  handlers: Record<string, Function[]>;
  commands: Record<string, Function>;
  notifications: string[];
  runBeforeAgentStart(event: Record<string, unknown>, ctx?: HarnessCtx): Promise<unknown>;
  runBeforeProviderRequest(event: { payload: unknown }, ctx?: HarnessCtx): unknown;
  runCommand(name: string, args: string, ctx?: HarnessCtx): Promise<unknown>;
  runSessionStart(ctx?: HarnessCtx): Promise<unknown>;
  runSessionBeforeSwitch(event: Record<string, unknown>, ctx?: HarnessCtx): Promise<unknown>;
  runSessionSwitch(event: Record<string, unknown>, ctx?: HarnessCtx): Promise<unknown>;
  runMessageEnd(event: { message: unknown }, ctx?: HarnessCtx): Promise<unknown>;
} {
  const handlers: Record<string, Function[]> = {};
  const commands: Record<string, Function> = {};
  const notifications: string[] = [];
  const pi = {
    on(event: string, handler: Function) {
      (handlers[event] ??= []).push(handler);
    },
    registerCommand(name: string, command: { handler: Function }) {
      commands[name] = command.handler;
    },
  };
  cacheOptimizer(pi as unknown as Parameters<typeof cacheOptimizer>[0]);
  return {
    handlers,
    commands,
    notifications,
    async runBeforeAgentStart(event, ctx = makeContext()) {
      const fn = handlers["before_agent_start"]?.[0];
      if (!fn) throw new Error("before_agent_start handler not registered");
      return fn(event, ctx);
    },
    runBeforeProviderRequest(event, ctx = makeContext()) {
      const fn = handlers["before_provider_request"]?.[0];
      if (!fn) throw new Error("before_provider_request handler not registered");
      return fn(event, ctx);
    },
    async runSessionStart(ctx = makeContext()) {
      const fn = handlers["session_start"]?.[0];
      if (!fn) throw new Error("session_start handler not registered");
      return fn({ type: "session_start" }, ctx);
    },
    async runSessionBeforeSwitch(event, ctx = makeContext()) {
      const fn = handlers["session_before_switch"]?.[0];
      if (!fn) throw new Error("session_before_switch handler not registered");
      return fn(event, ctx);
    },
    async runSessionSwitch(event, ctx = makeContext()) {
      const fn = handlers["session_switch"]?.[0];
      if (!fn) throw new Error("session_switch handler not registered");
      return fn(event, ctx);
    },
    async runMessageEnd(event, ctx = makeContext()) {
      const fn = handlers["message_end"]?.[0];
      if (!fn) throw new Error("message_end handler not registered");
      return fn(event, ctx);
    },
    async runCommand(name, args, ctx) {
      const handler = commands[name];
      if (!handler) throw new Error(`command ${name} not registered`);
      const base = ctx ?? makeContext();
      const cmdCtx = {
        ...base,
        ui: {
          ...base.ui,
          notify(message: string, _level?: string) {
            notifications.push(message);
          },
        },
      };
      return handler(args, cmdCtx);
    },
  };
}

const harness = createExtensionHarness();

const primaryBlock = "# Primary system prompt with enough content to be stable";
const overviewBlock = [
  "<session-overview>",
  "## DEVELOPER",
  "dev notes",
  "## RECENT COMMITS",
  "abc123 fix stuff",
  "def456 more stuff",
  "## CURRENT TASK",
  "do the thing",
  "Working directory: 3 uncommitted",
  "Line count: 12 / 2000",
  "</session-overview>",
].join("\n");
const skillsBlock = [
  "<skills>",
  "- alpha: Alpha 技能说明很长",
  "- beta: Beta 描述内容",
  "- gamma: Gamma 描述内容",
  "- delta: Delta 描述内容",
  "</skills>",
].join("\n");
const unknownBlock = "<unknown-extension>\nsecret payload must survive\n</unknown-extension>";

// 1. before_agent_start：默认保持原始 system prompt。
const defaultRewriteResult = await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock, overviewBlock, skillsBlock, unknownBlock] });
const defaultRewriteBlocks = (asRecord(defaultRewriteResult)?.systemPrompt as string[] | undefined) ?? [primaryBlock, overviewBlock, skillsBlock, unknownBlock];
expect(
  "before_agent_start.default-does-not-rewrite",
  defaultRewriteBlocks[1] === overviewBlock,
  `未设置 OMP_CACHE_OPTIMIZER_PROMPT_REWRITE 时应保留原始 session-overview，实际: ${JSON.stringify(defaultRewriteBlocks[1])}`,
);

// 用户从 v1.1.0 沿用的两个 PI_* 禁用开关不应改变当前默认的逐块保真行为。
process.env.PI_CACHE_OPTIMIZER_NO_SKILL_COMPRESSION = "1";
process.env.PI_CACHE_OPTIMIZER_NO_PROMPT_REWRITE = "1";
const legacyDisabledBlocks = [primaryBlock, overviewBlock, skillsBlock, unknownBlock];
const legacyDisabledEvent = { type: "before_agent_start", prompt: "hi", systemPrompt: [...legacyDisabledBlocks] };
const legacyDisabledResult = await harness.runBeforeAgentStart(legacyDisabledEvent);
const legacyDisabledPayload = { system: [...legacyDisabledBlocks], messages: [] };
harness.runBeforeProviderRequest({ payload: legacyDisabledPayload });
expect(
  "before_agent_start.legacy-disable-flags-are-no-op",
  !Object.hasOwn(asRecord(legacyDisabledResult) ?? {}, "systemPrompt") &&
    systemPromptBlocksEqual(legacyDisabledEvent.systemPrompt, legacyDisabledBlocks) &&
    systemPromptBlocksEqual(legacyDisabledPayload.system, legacyDisabledBlocks),
  `旧 PI_* 禁用开关存在时 hook 与 provider payload 应逐块保真，实际: ${JSON.stringify({ result: legacyDisabledResult, event: legacyDisabledEvent.systemPrompt, payload: legacyDisabledPayload.system })}`,
);
delete process.env.PI_CACHE_OPTIMIZER_NO_SKILL_COMPRESSION;
delete process.env.PI_CACHE_OPTIMIZER_NO_PROMPT_REWRITE;

// 2. 显式 opt-in 时：保留块数量、顺序、skill 描述、未知块；仅清理 session-overview churn。
process.env.OMP_CACHE_OPTIMIZER_PROMPT_REWRITE = "1";
const result1 = await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock, overviewBlock, skillsBlock, unknownBlock] });
const out1 = (asRecord(result1)?.systemPrompt as string[] | undefined) ?? [primaryBlock, overviewBlock, skillsBlock, unknownBlock];
expect(
  "before_agent_start.preserve-block-count-and-order",
  Array.isArray(out1) && out1.length === 4 &&
    out1[0].startsWith("# Primary") &&
    out1[1].includes("<session-overview>") &&
    out1[2].includes("<skills>") &&
    out1[3].includes("<unknown-extension>"),
  `块数量和顺序应保持 4 块且原序，实际: ${JSON.stringify(out1)}`,
);
expect(
  "before_agent_start.preserve-skill-descriptions",
  out1[2].includes("Alpha 技能说明很长") && out1[2].includes("Beta 描述内容"),
  `skill 描述应逐字保留，实际: ${JSON.stringify(out1[2])}`,
);
expect(
  "before_agent_start.preserve-unknown-block",
  out1[3] === unknownBlock,
  `未知块应逐字保留，实际: ${JSON.stringify(out1[3])}`,
);
expect(
  "before_agent_start.strip-session-overview-churn",
  !out1[1].includes("RECENT COMMITS") &&
    !out1[1].includes("Working directory:") &&
    !out1[1].includes("Line count:") &&
    out1[1].includes("CURRENT TASK"),
  `session-overview churn 应被清理但保留 CURRENT TASK，实际: ${JSON.stringify(out1[1])}`,
);

// 3. 历史 NO_PROMPT_REWRITE 变量已不再受支持，不能覆盖新变量的显式 opt-in。
process.env.OMP_CACHE_OPTIMIZER_NO_PROMPT_REWRITE = "1";
process.env.PI_CACHE_OPTIMIZER_NO_PROMPT_REWRITE = "1";
const legacyEnvResult = await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock, overviewBlock] });
const legacyEnvBlocks = (asRecord(legacyEnvResult)?.systemPrompt as string[] | undefined) ?? [primaryBlock, overviewBlock];
expect(
  "before_agent_start.ignores-legacy-no-rewrite-env",
  !legacyEnvBlocks[1].includes("RECENT COMMITS"),
  `历史 NO_PROMPT_REWRITE 变量不应影响显式 prompt 改写，实际: ${JSON.stringify(legacyEnvBlocks[1])}`,
);
delete process.env.OMP_CACHE_OPTIMIZER_NO_PROMPT_REWRITE;
delete process.env.PI_CACHE_OPTIMIZER_NO_PROMPT_REWRITE;

// 2. before_agent_start：无变化时返回 {}（非 { systemPrompt: [...] }）
const result2 = await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: ["stable block one with enough text", "stable block two with enough text"] });
expect(
  "before_agent_start.no-change-returns-empty-object",
  result2 !== undefined && typeof result2 === "object" && !("systemPrompt" in (asRecord(result2) ?? {})),
  `无变化时应返回 {}，实际: ${JSON.stringify(result2)}`,
);

// 3. before_provider_request：对 openai-completions 代理注入 prompt_cache_key
const providerCtx = makeContext({ model: { provider: "proxy", id: "gpt-test", api: "openai-completions", baseUrl: "https://proxy.example/v1" } });
const providerPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
const result3 = harness.runBeforeProviderRequest({ payload: providerPayload }, providerCtx);
expect(
  "before_provider_request.injects-prompt-cache-key-for-proxy",
  providerPayload.prompt_cache_key === "host-cache-key",
  `openai-completions 代理应注入 host-cache-key，实际: ${JSON.stringify(providerPayload)}`,
);
// KnownApi openrouter 会在 OMP 内部分派到 Responses 或 Completions，同样需要稳定 key。
const openRouterCtx = makeContext({
  model: { provider: "openrouter", id: "openai/gpt-test", api: "openrouter", baseUrl: "https://openrouter.ai/api/v1" },
});
const openRouterPayload: Record<string, unknown> = { model: "openai/gpt-test", input: [] };
harness.runBeforeProviderRequest({ payload: openRouterPayload }, openRouterCtx);
expect(
  "before_provider_request.injects-prompt-cache-key-for-openrouter-api",
  openRouterPayload.prompt_cache_key === "host-cache-key",
  `openrouter API 应注入 host-cache-key，实际: ${JSON.stringify(openRouterPayload)}`,
);
// no-op API 不注入
const noApiCtx = makeContext({ model: { provider: "test", id: "no-api", api: undefined } });
const noApiPayload: Record<string, unknown> = { messages: [] };
harness.runBeforeProviderRequest({ payload: noApiPayload }, noApiCtx);
expect(
  "before_provider_request.does-not-inject-for-unknown-api",
  noApiPayload.prompt_cache_key === undefined && noApiPayload.promptCacheKey === undefined,
  `未知 API 不应注入，实际: ${JSON.stringify(noApiPayload)}`,
);
// 3b. openai-responses 也注入（宿主未注入时兜底）
const responsesCtx = makeContext({ model: { provider: "proxy", id: "gpt-test", api: "openai-responses", baseUrl: "https://proxy.example/v1" } });
const responsesPayload: Record<string, unknown> = { model: "gpt-test", input: [] };
harness.runBeforeProviderRequest({ payload: responsesPayload }, responsesCtx);
expect(
  "before_provider_request.injects-for-openai-responses-when-missing",
  responsesPayload.prompt_cache_key === "host-cache-key",
  `openai-responses 无已有 key 时应注入，实际: ${JSON.stringify(responsesPayload)}`,
);
// 3c. openai-responses 宿主已注入时不覆盖
const responsesHostPayload: Record<string, unknown> = { model: "gpt-test", input: [], prompt_cache_key: "host-responses-key" };
harness.runBeforeProviderRequest({ payload: responsesHostPayload }, responsesCtx);
expect(
  "before_provider_request.preserves-host-injected-responses-key",
  responsesHostPayload.prompt_cache_key === "host-responses-key",
  `宿主已注入的 responses key 不应覆盖，实际: ${JSON.stringify(responsesHostPayload)}`,
);

// 4. before_provider_request：已有非 session 的 cache key 字段不被覆盖
const snakePayload: Record<string, unknown> = { prompt_cache_key: "custom-snake-key", messages: [] };
expect(
  "before_provider_request.keeps-existing-snake-key",
  harness.runBeforeProviderRequest({ payload: snakePayload }, providerCtx) === undefined && snakePayload.prompt_cache_key === "custom-snake-key",
  `已有非 session snake_case key 不应被覆盖，实际: ${JSON.stringify(snakePayload)}`,
);
const camelPayload: Record<string, unknown> = { promptCacheKey: "camel-key", messages: [] };
expect(
  "before_provider_request.keeps-existing-camel-key",
  harness.runBeforeProviderRequest({ payload: camelPayload }, providerCtx) === undefined && camelPayload.promptCacheKey === "camel-key",
  `已有 camelCase key 不应被覆盖，实际: ${JSON.stringify(camelPayload)}`,
);
// 3d. 无 host header 时注入项目级稳定 key，而不是 sessionId。
const longSessionId = "x".repeat(80);
const fallbackModel = { provider: "proxy", id: "gpt-test", api: "openai-completions", baseUrl: "https://proxy.example/v1" };
const fallbackCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: { getSessionId: () => longSessionId, getHeader: () => ({}) },
});
const expectedProjectKey = buildProjectPromptCacheKey("/tmp/project-a", fallbackModel as never);
const fallbackPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: fallbackPayload }, fallbackCtx);
expect(
  "before_provider_request.injects-project-key-when-no-header",
  fallbackPayload.prompt_cache_key === expectedProjectKey,
  `无稳定 header key 时应注入项目 key，实际: ${JSON.stringify(fallbackPayload.prompt_cache_key)}`,
);
const blankHeaderCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: { getSessionId: () => "fallback-session-id", getHeader: () => ({ providerPromptCacheKey: "   " }) },
});
const blankHeaderPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: blankHeaderPayload }, blankHeaderCtx);
expect(
  "before_provider_request.injects-project-key-for-blank-header",
  blankHeaderPayload.prompt_cache_key === expectedProjectKey,
  `空白 header key 时应注入项目 key，实际: ${JSON.stringify(blankHeaderPayload.prompt_cache_key)}`,
);
// session 级 body key 应被替换为项目 key
const sessionBodyCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: {
    getSessionId: () => "session-body-id",
    getHeader: () => ({}),
  },
});
const sessionBodyPayload: Record<string, unknown> = { model: "gpt-test", messages: [], prompt_cache_key: "session-body-id" };
harness.runBeforeProviderRequest({ payload: sessionBodyPayload }, sessionBodyCtx);
expect(
  "before_provider_request.replaces-session-id-body-key",
  sessionBodyPayload.prompt_cache_key === expectedProjectKey,
  `sessionId body key 应被替换为项目 key，实际: ${JSON.stringify(sessionBodyPayload.prompt_cache_key)}`,
);
const dualSessionPayload: Record<string, unknown> = {
  model: "gpt-test",
  messages: [],
  prompt_cache_key: "session-body-id",
  promptCacheKey: "session-body-id",
};
harness.runBeforeProviderRequest({ payload: dualSessionPayload }, sessionBodyCtx);
expect(
  "before_provider_request.replaces-matching-session-snake-and-camel",
  dualSessionPayload.prompt_cache_key === expectedProjectKey && dualSessionPayload.promptCacheKey === expectedProjectKey,
  `相同 session fallback 的 snake/camel 应同步替换，实际: ${JSON.stringify(dualSessionPayload)}`,
);
const conflictPayload: Record<string, unknown> = {
  model: "gpt-test",
  messages: [],
  prompt_cache_key: "custom-a",
  promptCacheKey: "custom-b",
};
const conflictResult = harness.runBeforeProviderRequest({ payload: conflictPayload }, sessionBodyCtx);
expect(
  "before_provider_request.keeps-conflicting-custom-keys",
  conflictResult === undefined &&
    conflictPayload.prompt_cache_key === "custom-a" &&
    conflictPayload.promptCacheKey === "custom-b",
  `冲突的非空自定义 key 应保持不变，实际: ${JSON.stringify(conflictPayload)}`,
);
// 真实 header key 即使文本等于本地 session ID 也保留
const sessionLikeHeaderCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: {
    getSessionId: () => "same-as-header",
    getHeader: () => ({
      providerPromptCacheKey: "same-as-header",
    }),
  },
});
const sessionLikeHeaderPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: sessionLikeHeaderPayload }, sessionLikeHeaderCtx);
expect(
  "before_provider_request.prefers-real-header-even-if-session-like",
  sessionLikeHeaderPayload.prompt_cache_key === "same-as-header",
  `真实 header key 应优先，实际: ${JSON.stringify(sessionLikeHeaderPayload.prompt_cache_key)}`,
);
// 未知非空 body key 保留
const unknownBodyCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: { getSessionId: () => "session-body-id", getHeader: () => ({}) },
});
const unknownBodyPayload: Record<string, unknown> = {
  model: "gpt-test",
  messages: [],
  prompt_cache_key: "unknown-custom-identity",
};
harness.runBeforeProviderRequest({ payload: unknownBodyPayload }, unknownBodyCtx);
expect(
  "before_provider_request.keeps-unknown-custom-body-key",
  unknownBodyPayload.prompt_cache_key === "unknown-custom-identity",
  `未知自定义 body key 应保留，实际: ${JSON.stringify(unknownBodyPayload.prompt_cache_key)}`,
);
// 四种 OpenAI API 在无 header 时都注入项目 key
for (const api of ["openai-completions", "openai-responses", "openai-codex-responses", "azure-openai-responses"] as const) {
  const apiModel = { provider: "proxy", id: "gpt-test", api, baseUrl: "https://proxy.example/v1" };
  const apiCtx = makeContext({
    model: apiModel,
    cwd: "/tmp/project-a",
    sessionManager: { getSessionId: () => "session-for-api", getHeader: () => ({}) },
  });
  const apiPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
  harness.runBeforeProviderRequest({ payload: apiPayload }, apiCtx);
  expect(
    `before_provider_request.injects-project-key-for-${api}`,
    apiPayload.prompt_cache_key === buildProjectPromptCacheKey("/tmp/project-a", apiModel as never),
    `${api} 应注入项目 key，实际: ${JSON.stringify(apiPayload.prompt_cache_key)}`,
  );
}
// 3e. 超长 host header key：OpenAI body 用无碰撞哈希归一化，cache hint 保留完整 key。
const longHeaderKey = `header-${"y".repeat(80)}`;
const expectedLongWireKey = normalizePromptCacheKeyForWire(longHeaderKey);
const longHeaderCtx = makeContext({
  sessionManager: { getSessionId: () => "fallback-session-id", getHeader: () => ({ providerPromptCacheKey: longHeaderKey }) },
});
const longHeaderPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: longHeaderPayload }, longHeaderCtx);
expect(
  "before_provider_request.normalizes-long-header-key-for-openai-body",
  longHeaderPayload.prompt_cache_key === expectedLongWireKey &&
    typeof expectedLongWireKey === "string" &&
    expectedLongWireKey.startsWith("pc_") &&
    expectedLongWireKey.length === 51,
  `OpenAI body key 应归一化为 pc_+48hex，实际: ${JSON.stringify(longHeaderPayload.prompt_cache_key)}`,
);
await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, longHeaderCtx);
const longHeaderHint = __internals_for_tests.getCacheHintsService()?.getHints({});
expect(
  "cache-hints.preserve-full-long-header-key",
  longHeaderHint?.promptCacheKey === longHeaderKey,
  `cache hint 应保留完整 host key，实际: ${JSON.stringify(longHeaderHint?.promptCacheKey)}`,
);
// 跨会话相同项目稳定 key；不同项目隔离
const projectACtx1 = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: { getSessionId: () => "session-a-1", getHeader: () => ({}) },
});
const projectACtx2 = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: { getSessionId: () => "session-a-2", getHeader: () => ({}) },
});
const projectBCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-b",
  sessionManager: { getSessionId: () => "session-b-1", getHeader: () => ({}) },
});
await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, projectACtx1);
const projectAHint1 = __internals_for_tests.getCacheHintsService()?.getHints({});
const projectAPayload1: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: projectAPayload1 }, projectACtx1);
await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, projectACtx2);
const projectAHint2 = __internals_for_tests.getCacheHintsService()?.getHints({});
const projectAPayload2: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: projectAPayload2 }, projectACtx2);
await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, projectBCtx);
const projectBHint = __internals_for_tests.getCacheHintsService()?.getHints({});
const projectBPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: projectBPayload }, projectBCtx);
expect(
  "stable-project-key.same-project-cross-session",
  projectAHint1?.promptCacheKey === expectedProjectKey &&
    projectAHint2?.promptCacheKey === expectedProjectKey &&
    projectAPayload1.prompt_cache_key === expectedProjectKey &&
    projectAPayload2.prompt_cache_key === expectedProjectKey &&
    typeof expectedProjectKey === "string" &&
    /^omp-cache-v1-[a-f0-9]{48}$/.test(expectedProjectKey) &&
    expectedProjectKey.length === 61 &&
    !expectedProjectKey.includes("project-a") &&
    !expectedProjectKey.includes("proxy") &&
    !expectedProjectKey.includes("gpt-test"),
  `同项目跨会话应得到 61 字符不可逆项目 key，实际: ${JSON.stringify({
    expectedProjectKey,
    hint1: projectAHint1?.promptCacheKey,
    hint2: projectAHint2?.promptCacheKey,
    body1: projectAPayload1.prompt_cache_key,
    body2: projectAPayload2.prompt_cache_key,
  })}`,
);
expect(
  "stable-project-key.different-project-isolated",
  projectBHint?.promptCacheKey !== expectedProjectKey &&
    projectBPayload.prompt_cache_key !== expectedProjectKey &&
    projectBPayload.prompt_cache_key === buildProjectPromptCacheKey("/tmp/project-b", fallbackModel as never),
  `不同项目应隔离，实际: ${JSON.stringify({
    projectBHint: projectBHint?.promptCacheKey,
    projectBBody: projectBPayload.prompt_cache_key,
  })}`,
);
// 3e. 运行时禁用（/cache-optimizer disable）时不注入
const prevEnabled = __internals_for_tests.isRuntimeOptimizerEnabled();
__internals_for_tests.setRuntimeOptimizerEnabled(false);
const disabledPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: disabledPayload }, providerCtx);
expect(
  "before_provider_request.no-inject-when-runtime-disabled",
  disabledPayload.prompt_cache_key === undefined && disabledPayload.promptCacheKey === undefined,
  `运行时禁用不应注入 prompt_cache_key，实际: ${JSON.stringify(disabledPayload)}`,
);
const disabledNoHeaderCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: { getSessionId: () => "disabled-session-id", getHeader: () => ({}) },
});
const disabledNoHeaderPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: disabledNoHeaderPayload }, disabledNoHeaderCtx);
expect(
  "before_provider_request.disabled-does-not-inject-project-key",
  disabledNoHeaderPayload.prompt_cache_key === undefined &&
    disabledNoHeaderPayload.promptCacheKey === undefined,
  `禁用时即使无 header 也不应注入项目 key，实际: ${JSON.stringify(disabledNoHeaderPayload)}`,
);
await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, disabledNoHeaderCtx);
const disabledHint = __internals_for_tests.getCacheHintsService()?.getHints({});
expect(
  "cache-hints.disabled-service-hides-hints",
  disabledHint === undefined,
  `禁用时 getHints 应返回 undefined（现有服务契约），实际: ${JSON.stringify(disabledHint)}`,
);
__internals_for_tests.setRuntimeOptimizerEnabled(prevEnabled);

await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, providerCtx);
// 5. cache hints：优先使用 OMP 17 header 的 providerPromptCacheKey，缺失时使用项目 key
const hintService = __internals_for_tests.getCacheHintsService();
const hint1 = hintService?.getHints({});
expect(
  "cache-hints.uses-providerPromptCacheKey-before-session-id",
  hint1?.promptCacheKey === "host-cache-key",
  `hint 应使用 header 的 providerPromptCacheKey，实际: ${JSON.stringify(hint1?.promptCacheKey)}`,
);
const noHeaderCtx = makeContext({
  model: fallbackModel,
  cwd: "/tmp/project-a",
  sessionManager: { getSessionId: () => "fallback-session-id", getHeader: () => ({}) },
});
await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, noHeaderCtx);
const hint2 = hintService?.getHints({});
expect(
  "cache-hints.uses-project-key-when-no-header-key",
  hint2?.promptCacheKey === expectedProjectKey,
  `无 header key 时应使用项目 key，实际: ${JSON.stringify(hint2?.promptCacheKey)}`,
);

// ── 4b. 项目 key 纯函数确定性 ───────────────────────────────────
expect(
  "isOpenAICompatibleApi.covers-four-openai-shapes",
  isOpenAICompatibleApi("openai-completions") &&
    isOpenAICompatibleApi("openai-responses") &&
    isOpenAICompatibleApi("openai-codex-responses") &&
    isOpenAICompatibleApi("azure-openai-responses") &&
    !isOpenAICompatibleApi("anthropic-messages"),
  "应覆盖四种 OpenAI API 且排除 anthropic-messages",
);
const baseKeyModel = { provider: "proxy", id: "gpt-test", api: "openai-completions", baseUrl: "https://proxy.example/v1" };
const baseKey = buildProjectPromptCacheKey("/tmp/project-a", baseKeyModel as never);
expect(
  "buildProjectPromptCacheKey.stable-and-redacted",
  typeof baseKey === "string" &&
    baseKey === buildProjectPromptCacheKey("/tmp/project-a", baseKeyModel as never) &&
    /^omp-cache-v1-[a-f0-9]{48}$/.test(baseKey) &&
    baseKey.length === 61 &&
    !baseKey.includes("project-a") &&
    !baseKey.includes("proxy") &&
    !baseKey.includes("gpt-test") &&
    !baseKey.includes("proxy.example"),
  `项目 key 应稳定且不可逆，实际: ${JSON.stringify(baseKey)}`,
);
expect(
  "buildProjectPromptCacheKey.isolates-scope-dimensions",
  baseKey !== buildProjectPromptCacheKey("/tmp/project-b", baseKeyModel as never) &&
    baseKey !== buildProjectPromptCacheKey("/tmp/project-a", { ...baseKeyModel, api: "openai-responses" } as never) &&
    baseKey !== buildProjectPromptCacheKey("/tmp/project-a", { ...baseKeyModel, provider: "other" } as never) &&
    baseKey !== buildProjectPromptCacheKey("/tmp/project-a", { ...baseKeyModel, id: "other-model" } as never) &&
    baseKey !== buildProjectPromptCacheKey("/tmp/project-a", { ...baseKeyModel, baseUrl: "https://other.example/v1" } as never) &&
    buildProjectPromptCacheKey("/tmp/project-a", { ...baseKeyModel, api: "anthropic-messages" } as never) === undefined &&
    buildProjectPromptCacheKey("/tmp/project-a", undefined) === undefined,
  "cwd/API/provider/model/baseUrl 变化应隔离，非兼容 API 或缺失 model 应返回 undefined",
);

// ── 5. asRecord 类型守卫 ─────────────────────────────────────────

expect(
  "asRecord.object",
  asRecord({ a: 1 }) !== undefined,
  "对象应返回 record",
);
expect(
  "asRecord.null",
  asRecord(null) === undefined,
  "null 应返回 undefined",
);
expect(
  "asRecord.string",
  asRecord("hello") === undefined,
  "字符串应返回 undefined",
);

// ── 6. OMP 17 session-overview churn ──────────────────────────

const overviewWithChurn = [
  "<session-overview>",
  "## DEVELOPER",
  "dev",
  "## RECENT COMMITS",
  "abc123 fix stuff",
  "## CURRENT TASK",
  "task",
  "Working directory: 3 uncommitted",
  "Line count: 12 / 2000",
  "</session-overview>",
].join("\n");
const strippedBlocks = mapSystemPromptBlocks([overviewWithChurn], stripSessionOverviewChurn);
expect(
  "stripSessionOverviewChurn.block-map",
  !strippedBlocks[0].includes("RECENT COMMITS") &&
    !strippedBlocks[0].includes("Working directory:") &&
    !strippedBlocks[0].includes("Line count:") &&
    strippedBlocks[0].includes("CURRENT TASK"),
  `块映射 strip 应去掉 churn 字段，实际: ${JSON.stringify(strippedBlocks[0])}`,
);

// ── 7. Prompt cache 诊断（仅哈希，不保留 prompt 或 cache key 原文） ─────

const diagnosticPrompt = "system prompt with stable content";
const fingerprint = fingerprintPrompt(diagnosticPrompt);
expect(
  "prompt-diagnostics.fingerprint-is-stable-and-redacted",
  typeof fingerprint === "string" &&
    /^[a-f0-9]{16}$/.test(fingerprint) &&
    !fingerprint.includes("stable content") &&
    fingerprintPrompt(diagnosticPrompt) === fingerprint &&
    fingerprintPrompt("changed prompt") !== fingerprint &&
    fingerprintPrompt(undefined) === undefined,
  `指纹应为稳定且不包含 prompt 原文的 16 位哈希，实际: ${JSON.stringify(fingerprint)}`,
);
expect(
  "prompt-diagnostics.cache-key-source-reflects-effective-body",
  getPromptCacheKeySource(
    { prompt_cache_key: "host-cache-key" },
    "openai-completions",
    { key: "host-cache-key", source: "header" },
  ) === "header" &&
    getPromptCacheKeySource(
      { prompt_cache_key: "omp-cache-v1-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
      "openai-completions",
      { key: "omp-cache-v1-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "project" },
    ) === "project" &&
    getPromptCacheKeySource(
      { prompt_cache_key: "existing-other-key" },
      "openai-completions",
      { key: "host-cache-key", source: "header" },
    ) === "custom" &&
    getPromptCacheKeySource({}, "openai-completions", { key: "host-cache-key", source: "header" }) === "unavailable" &&
    getPromptCacheKeySource(
      { prompt_cache_key: "host-cache-key" },
      "anthropic-messages",
      { key: "host-cache-key", source: "header" },
    ) === "unavailable" &&
    getPromptCacheKeySource(
      { prompt_cache_key: expectedLongWireKey },
      "openai-completions",
      { key: longHeaderKey, source: "header" },
    ) === "header",
  "OpenAI-compatible body key 应按 header/project/custom/unavailable 分类，超长 header 经 wire 归一化后仍计为 header",
);

// ── 8. wire key 归一化碰撞回归 ───────────────────────────────────

const longWireA = normalizePromptCacheKeyForWire("x".repeat(64) + "A");
const longWireB = normalizePromptCacheKeyForWire("x".repeat(64) + "B");
expect(
  "normalizePromptCacheKeyForWire.stable-collision-free",
  typeof longWireA === "string" &&
    typeof longWireB === "string" &&
    longWireA !== longWireB &&
    longWireA.startsWith("pc_") &&
    longWireB.startsWith("pc_") &&
    longWireA.length === 51 &&
    longWireB.length === 51 &&
    normalizePromptCacheKeyForWire("short-key") === "short-key" &&
    normalizePromptCacheKeyForWire("   ") === undefined &&
    normalizePromptCacheKeyForWire(undefined) === undefined,
  `超长 key 应生成不同 pc_ wire key，短 key 原样返回，实际: ${JSON.stringify({ longWireA, longWireB })}`,
);

// ── 9. retention 真值表 ──────────────────────────────────────────

function withTempEnv(mutator: (env: Record<string, string | undefined>) => void, assertFn: (env: Record<string, string | undefined>) => void): void {
  const env: Record<string, string | undefined> = {};
  mutator(env);
  const snapshot = captureCacheRetentionEnv(env);
  configureCacheRetentionEnv(env);
  assertFn(env);
  restoreCacheRetentionEnv(snapshot, env);
}

withTempEnv(
  (env) => {
    env[OMP_CACHE_RETENTION_ENV] = "none";
    env[PI_CACHE_RETENTION_ENV] = "long";
  },
  (env) => {
    expect(
      "configureCacheRetentionEnv.omp-none-overrides-pi-long",
      env[OMP_CACHE_RETENTION_ENV] === "none" && env[PI_CACHE_RETENTION_ENV] === "none",
      `OMP=none 应优先，实际: ${JSON.stringify(env)}`,
    );
  },
);
withTempEnv(
  (env) => {
    env[OMP_CACHE_RETENTION_ENV] = "short";
  },
  (env) => {
    expect(
      "configureCacheRetentionEnv.omp-short-mirrors-to-pi",
      env[OMP_CACHE_RETENTION_ENV] === "short" && env[PI_CACHE_RETENTION_ENV] === "short",
      `OMP=short 应镜像到 PI，实际: ${JSON.stringify(env)}`,
    );
  },
);
withTempEnv(
  (env) => {
    env[PI_CACHE_RETENTION_ENV] = "none";
  },
  (env) => {
    expect(
      "configureCacheRetentionEnv.pi-none-when-omp-missing",
      env[OMP_CACHE_RETENTION_ENV] === "none" && env[PI_CACHE_RETENTION_ENV] === "none",
      `OMP 缺失时 PI=none 应写入两者，实际: ${JSON.stringify(env)}`,
    );
  },
);
withTempEnv(
  (_env) => {},
  (env) => {
    expect(
      "configureCacheRetentionEnv.default-long-when-both-empty",
      env[OMP_CACHE_RETENTION_ENV] === LONG_CACHE_RETENTION_VALUE &&
        env[PI_CACHE_RETENTION_ENV] === LONG_CACHE_RETENTION_VALUE,
      `两者缺失应默认 long，实际: ${JSON.stringify(env)}`,
    );
  },
);
{
  const env: Record<string, string | undefined> = {
    [OMP_CACHE_RETENTION_ENV]: "invalid",
    [PI_CACHE_RETENTION_ENV]: "long",
  };
  const snapshot = captureCacheRetentionEnv(env);
  configureCacheRetentionEnv(env);
  expect(
    "configureCacheRetentionEnv.invalid-omp-leaves-both-unchanged",
    env[OMP_CACHE_RETENTION_ENV] === "invalid" && env[PI_CACHE_RETENTION_ENV] === "long",
    `非法 OMP 候选应保持原值，实际: ${JSON.stringify(env)}`,
  );
  restoreCacheRetentionEnv(snapshot, env);
  expect(
    "configureCacheRetentionEnv.restore-preserves-presence-and-values",
    env[OMP_CACHE_RETENTION_ENV] === "invalid" && env[PI_CACHE_RETENTION_ENV] === "long",
    `restore 应恢复存在性与原值，实际: ${JSON.stringify(env)}`,
  );
}
{
  const env: Record<string, string | undefined> = {};
  const snapshot = captureCacheRetentionEnv(env);
  env[OMP_CACHE_RETENTION_ENV] = "long";
  env[PI_CACHE_RETENTION_ENV] = "long";
  restoreCacheRetentionEnv(snapshot, env);
  expect(
    "configureCacheRetentionEnv.restore-deletes-previously-missing",
    !Object.prototype.hasOwnProperty.call(env, OMP_CACHE_RETENTION_ENV) &&
      !Object.prototype.hasOwnProperty.call(env, PI_CACHE_RETENTION_ENV),
    `原先缺失的变量 restore 后应删除，实际: ${JSON.stringify(env)}`,
  );
}

// ── 10. 动态 getCwd 项目 key ─────────────────────────────────────

const dynamicCwdModel = { provider: "proxy", id: "gpt-test", api: "openai-completions", baseUrl: "https://proxy.example/v1" };
const dynamicCwdCtx = makeContext({
  model: dynamicCwdModel,
  cwd: "/project-a",
  sessionManager: {
    getSessionId: () => "cwd-session",
    getHeader: () => ({}),
    getCwd: () => "/project-b",
  },
});
const dynamicCwdPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
harness.runBeforeProviderRequest({ payload: dynamicCwdPayload }, dynamicCwdCtx);
const projectBKey = buildProjectPromptCacheKey("/project-b", dynamicCwdModel as never);
const projectAKey = buildProjectPromptCacheKey("/project-a", dynamicCwdModel as never);
expect(
  "before_provider_request.uses-sessionManager-getCwd-for-project-key",
  dynamicCwdPayload.prompt_cache_key === projectBKey &&
    dynamicCwdPayload.prompt_cache_key !== projectAKey,
  `应使用 getCwd=/project-b，实际: ${JSON.stringify({
    body: dynamicCwdPayload.prompt_cache_key,
    projectBKey,
    projectAKey,
  })}`,
);

// ── 11. 请求级 prompt 诊断 + stats（不依赖 message_end） ──────────

const diagModel = { provider: "proxy", id: "gpt-5.5", api: "openai-completions", baseUrl: "https://proxy.example/v1" };
const diagCtx = makeContext({
  model: diagModel,
  cwd: "/tmp/diag-project",
  sessionManager: { getSessionId: () => "diag-session", getHeader: () => ({}) },
});
const diagPayload1: Record<string, unknown> = {
  model: "gpt-5.5",
  messages: [{ role: "system", content: "stable system A" }],
};
const diagPayload2: Record<string, unknown> = {
  model: "gpt-5.5",
  messages: [{ role: "system", content: "stable system B" }],
};
harness.runBeforeProviderRequest({ payload: diagPayload1 }, diagCtx);
harness.runBeforeProviderRequest({ payload: diagPayload2 }, diagCtx);
const notifyBefore = harness.notifications.length;
await harness.runCommand("cache-optimizer", "stats", diagCtx);
const statsNotify = harness.notifications.slice(notifyBefore).join("\n");
expect(
  "cache-optimizer.stats.prompt-diagnostics-without-message-end",
  statsNotify.includes("Prompt 诊断（最近 2 次 provider 请求）") &&
    statsNotify.includes("请求数：0 次命中 / 0 次总计"),
  `stats 应显示 2 次 prompt 诊断且 usage 为 0，实际: ${JSON.stringify(statsNotify)}`,
);
expect(
  "extension.does-not-register-after_provider_response",
  harness.handlers["after_provider_response"] === undefined,
  "插件不应注册 after_provider_response handler",
);

// ── 11b. OMP 17 会话生命周期与全零 usage ───────────────────────

expect(
  "extension.registers-session-switch-lifecycle",
  harness.handlers["session_before_switch"]?.length === 1 && harness.handlers["session_switch"]?.length === 1,
  `应同时注册 session_before_switch/session_switch，实际: ${JSON.stringify(Object.keys(harness.handlers))}`,
);

const pendingOperations = new Set<Promise<void>>();
let releasePendingOperation: (() => void) | undefined;
const delayedOperation = trackPendingOperation(
  pendingOperations,
  new Promise<void>((resolve) => {
    releasePendingOperation = resolve;
  }),
);
let switchBarrierSettled = false;
const switchBarrier = waitForPendingOperations(pendingOperations).then(() => {
  switchBarrierSettled = true;
});
await Promise.resolve();
expect(
  "session-switch.waits-for-in-flight-message-end",
  switchBarrierSettled === false && pendingOperations.size === 1,
  `message_end 未结束时切换屏障不应放行，实际: ${JSON.stringify({ switchBarrierSettled, size: pendingOperations.size })}`,
);
releasePendingOperation?.();
await delayedOperation;
await switchBarrier;
expect(
  "session-switch-clears-completed-message-end",
  switchBarrierSettled === true && pendingOperations.size === 0,
  `message_end 结束后屏障应放行并清理集合，实际: ${JSON.stringify({ switchBarrierSettled, size: pendingOperations.size })}`,
);

expect(
  "session-identity-detects-host-rollback",
  sessionIdentityChanged("target-session", "previous-session") === true &&
    sessionIdentityChanged("previous-session", "previous-session") === false,
  "宿主回滚 SessionManager 后应检测到已加载统计与活动 session 不一致",
);

const switchModelHarness = createExtensionHarness();
const switchStatuses: Array<string | undefined> = [];
const oldModelCtx = makeContext({
  model: { provider: "openai", id: "gpt-old", api: "openai-responses", baseUrl: "https://api.openai.com/v1" },
  sessionManager: { getSessionId: () => "switch-target-session" },
});
oldModelCtx.ui.setStatus = (_key, value) => switchStatuses.push(value);
await switchModelHarness.runSessionSwitch(
  { type: "session_switch", reason: "resume", previousSessionFile: "previous.jsonl" },
  oldModelCtx,
);
expect(
  "session_switch.defers-model-status-until-turn-start",
  switchStatuses.length === 0,
  `session_switch 发生时目标模型尚未恢复，不应发布旧模型状态，实际: ${JSON.stringify(switchStatuses)}`,
);
const targetModelCtx = makeContext({
  model: { provider: "anthropic", id: "claude-sonnet-4-6", api: "anthropic-messages", baseUrl: "https://api.anthropic.com" },
  sessionManager: { getSessionId: () => "switch-target-session" },
});
targetModelCtx.ui.setStatus = (_key, value) => switchStatuses.push(value);
await switchModelHarness.handlers["turn_start"]?.[0]?.(
  { type: "turn_start", turnIndex: 0, timestamp: Date.now() },
  targetModelCtx,
);
expect(
  "turn_start.publishes-restored-target-model-status",
  switchStatuses.some((status) => status?.includes("Claude")),
  `turn_start 应在目标模型恢复后发布状态，实际: ${JSON.stringify(switchStatuses)}`,
);

const rollbackStatusCountBefore = switchStatuses.length;
const rolledBackSameModelCtx = makeContext({
  model: targetModelCtx.model,
  sessionManager: { getSessionId: () => "previous-session" },
});
rolledBackSameModelCtx.ui.setStatus = (_key, value) => switchStatuses.push(value);
await switchModelHarness.handlers["turn_start"]?.[0]?.(
  { type: "turn_start", turnIndex: 1, timestamp: Date.now() },
  rolledBackSameModelCtx,
);
expect(
  "turn_start.republishes-status-after-same-model-session-rollback",
  switchStatuses.length > rollbackStatusCountBefore,
  `session 回滚后即使模型相同也应刷新 footer，实际: ${JSON.stringify(switchStatuses)}`,
);

const zeroUsageHarness = createExtensionHarness();
const zeroUsageModel = {
  provider: "openai",
  id: "gpt-5.5",
  name: "GPT 5.5",
  api: "openai-responses",
  baseUrl: "https://api.openai.com/v1",
};
const zeroUsageCtx = makeContext({ model: zeroUsageModel });
await zeroUsageHarness.runMessageEnd(
  {
    message: {
      role: "assistant",
      api: "openai-responses",
      provider: "openai",
      model: "gpt-5.5",
      stopReason: "error",
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 },
    },
  },
  zeroUsageCtx,
);
const zeroUsageNotifyBefore = zeroUsageHarness.notifications.length;
await zeroUsageHarness.runCommand("cache-optimizer", "stats", zeroUsageCtx);
const zeroUsageStats = zeroUsageHarness.notifications.slice(zeroUsageNotifyBefore).join("\n");
expect(
  "message_end.does-not-count-zero-input-error-as-request",
  zeroUsageStats.includes("请求数：0 次命中 / 0 次总计"),
  `全零 error usage 不应计为 provider 请求，实际: ${JSON.stringify(zeroUsageStats)}`,
);
await zeroUsageHarness.runMessageEnd(
  {
    message: {
      role: "assistant",
      api: "openai-responses",
      provider: "openai",
      model: "gpt-5.5",
      stopReason: "stop",
      usage: { input: 100, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 110 },
    },
  },
  zeroUsageCtx,
);
const nonZeroUsageNotifyBefore = zeroUsageHarness.notifications.length;
await zeroUsageHarness.runCommand("cache-optimizer", "stats", zeroUsageCtx);
const nonZeroUsageStats = zeroUsageHarness.notifications.slice(nonZeroUsageNotifyBefore).join("\n");
expect(
  "message_end.counts-nonzero-input-request",
  nonZeroUsageStats.includes("请求数：0 次命中 / 1 次总计"),
  `非零输入请求仍应计数，实际: ${JSON.stringify(nonZeroUsageStats)}`,
);
const missingUsageHarness = createExtensionHarness();
await missingUsageHarness.runMessageEnd(
  {
    message: {
      role: "assistant",
      api: "openai-completions",
      provider: "proxy",
      model: "gpt-5.5",
      stopReason: "stop",
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 },
    },
  },
  makeContext({ model: { ...zeroUsageModel, provider: "proxy", api: "openai-completions" } }),
);
const missingUsageNotifyBefore = missingUsageHarness.notifications.length;
await missingUsageHarness.runCommand(
  "cache-optimizer",
  "stats",
  makeContext({ model: { ...zeroUsageModel, provider: "proxy", api: "openai-completions" } }),
);
const missingUsageStats = missingUsageHarness.notifications.slice(missingUsageNotifyBefore).join("\n");
expect(
  "message_end.counts-successful-request-with-missing-usage",
  missingUsageStats.includes("请求数：0 次命中 / 1 次总计") && missingUsageStats.includes("usage 缺失"),
  `正常 stop 的全零 usage 应计请求并标缺失，实际: ${JSON.stringify(missingUsageStats)}`,
);

// ── 12. session-overview 兜底：仅 Working directory / 仅 Line count ─

process.env.OMP_CACHE_OPTIMIZER_PROMPT_REWRITE = "1";
const workingOnlyPayload: Record<string, unknown> = {
  system: "<session-overview>\n## CURRENT TASK\nwork\nWorking directory: dirty\n</session-overview>",
  messages: [],
};
const workingOnlyResult = harness.runBeforeProviderRequest(
  { payload: workingOnlyPayload },
  makeContext({ model: { provider: "proxy", id: "gpt-5.5", api: "openai-completions", baseUrl: "https://proxy.example/v1" } }),
);
expect(
  "before_provider_request.strips-working-directory-only-overview",
  workingOnlyResult !== undefined &&
    typeof workingOnlyPayload.system === "string" &&
    !workingOnlyPayload.system.includes("Working directory:") &&
    workingOnlyPayload.system.includes("CURRENT TASK"),
  `仅 Working directory 应被清理，实际: ${JSON.stringify(workingOnlyPayload.system)}`,
);
const lineCountOnlyPayload: Record<string, unknown> = {
  system: "<session-overview>\n## CURRENT TASK\nwork\nLine count: 1 / 2\n</session-overview>",
  messages: [],
};
const lineCountOnlyResult = harness.runBeforeProviderRequest(
  { payload: lineCountOnlyPayload },
  makeContext({ model: { provider: "proxy", id: "gpt-5.5", api: "openai-completions", baseUrl: "https://proxy.example/v1" } }),
);
expect(
  "before_provider_request.strips-line-count-only-overview",
  lineCountOnlyResult !== undefined &&
    typeof lineCountOnlyPayload.system === "string" &&
    !lineCountOnlyPayload.system.includes("Line count:") &&
    lineCountOnlyPayload.system.includes("CURRENT TASK"),
  `仅 Line count 应被清理，实际: ${JSON.stringify(lineCountOnlyPayload.system)}`,
);
delete process.env.OMP_CACHE_OPTIMIZER_PROMPT_REWRITE;

// ── 13. Responses instructions 指纹 + retention 安全网 ───────────

const responsesInstructionsPayload: Record<string, unknown> = {
  model: "gpt-test",
  instructions: "responses system instructions for fingerprint only",
  input: [],
};
const responsesInstrText = extractSystemPrompt(responsesInstructionsPayload);
expect(
  "extractSystemPrompt.reads-responses-instructions",
  responsesInstrText === "responses system instructions for fingerprint only",
  `应读取顶层 instructions，实际: ${JSON.stringify(responsesInstrText)}`,
);
const responsesCtxNoCompat = makeContext({
  model: {
    provider: "proxy",
    id: "gpt-responses",
    api: "openai-responses",
    baseUrl: "https://proxy.example/v1",
  },
});
const responsesRetentionPayload: Record<string, unknown> = {
  model: "gpt-responses",
  instructions: "stable",
  prompt_cache_retention: "24h",
  input: [],
};
const beforeInstructions = responsesRetentionPayload.instructions;
harness.runBeforeProviderRequest({ payload: responsesRetentionPayload }, responsesCtxNoCompat);
expect(
  "before_provider_request.strips-retention-without-responses-opt-in",
  responsesRetentionPayload.prompt_cache_retention === undefined &&
    responsesRetentionPayload.instructions === beforeInstructions,
  `无 opt-in 应删除 retention 且不改写 instructions，实际: ${JSON.stringify(responsesRetentionPayload)}`,
);
const responsesCtxWithCompat = makeContext({
  model: {
    provider: "proxy",
    id: "gpt-responses",
    api: "openai-responses",
    baseUrl: "https://proxy.example/v1",
    compat: { supportsLongPromptCacheRetention: true },
  },
});
const responsesKeepPayload: Record<string, unknown> = {
  model: "gpt-responses",
  instructions: "stable",
  prompt_cache_retention: "24h",
  input: [],
};
harness.runBeforeProviderRequest({ payload: responsesKeepPayload }, responsesCtxWithCompat);
expect(
  "before_provider_request.keeps-retention-with-responses-opt-in",
  responsesKeepPayload.prompt_cache_retention === "24h",
  `Responses 显式 opt-in 应保留 retention，实际: ${JSON.stringify(responsesKeepPayload)}`,
);
const completionsWithRetentionPayload: Record<string, unknown> = {
  model: "gpt-test",
  messages: [],
  prompt_cache_retention: "24h",
};
harness.runBeforeProviderRequest(
  { payload: completionsWithRetentionPayload },
  makeContext({
    model: {
      provider: "proxy",
      id: "gpt-test",
      api: "openai-completions",
      baseUrl: "https://proxy.example/v1",
      compat: { supportsLongPromptCacheRetention: true },
    },
  }),
);
expect(
  "before_provider_request.strips-retention-on-completions-even-with-compat",
  completionsWithRetentionPayload.prompt_cache_retention === undefined,
  `openai-completions 即使误设 compat 也应删除 retention，实际: ${JSON.stringify(completionsWithRetentionPayload)}`,
);

// ── 14. optional Responses retention compat ─────────────────────

expect(
  "isOpenAIResponsesLongRetentionApi.literal-only",
  isOpenAIResponsesLongRetentionApi("openai-responses") === true &&
    isOpenAIResponsesLongRetentionApi("openai-completions") === false &&
    isOpenAIResponsesLongRetentionApi("openai-codex-responses") === false,
  "仅字面 openai-responses 支持 long retention API",
);
const optionalResponses = describeOptionalOpenAIResponsesRetentionCompat(
  makeModel({
    id: "gpt-test",
    provider: "proxy",
    api: "openai-responses",
    baseUrl: "https://proxy.example/v1",
  }) as never,
);
const optionalCompletions = describeOptionalOpenAIResponsesRetentionCompat(
  makeModel({
    id: "gpt-test",
    provider: "proxy",
    api: "openai-completions",
    baseUrl: "https://proxy.example/v1",
  }) as never,
);
expect(
  "describeOptionalOpenAIResponsesRetentionCompat.responses-only",
  optionalResponses.includes("supportsLongPromptCacheRetention") &&
    optionalCompletions.length === 0,
  `仅第三方 openai-responses 应建议 long retention，实际: ${JSON.stringify({ optionalResponses, optionalCompletions })}`,
);

// ── 结果汇总 ─────────────────────────────────────────────────────

console.log(`\n${"=".repeat(60)}`);
console.log(`冒烟测试结果: ${passed} 通过, ${failed} 失败`);
console.log(`${"=".repeat(60)}`);

if (failed > 0) {
  process.exit(1);
}
