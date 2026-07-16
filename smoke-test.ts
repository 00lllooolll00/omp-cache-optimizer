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

// 保证本进程不受宿主 prompt 重写开关影响（OMP_ 主名 + 旧 PI_ 兼容名）。
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
  emptyCacheStats,
  extractSystemPrompt,
  setSystemPrompt,
  asRecord,
  stripSessionOverviewChurn,
  mapSystemPromptBlocks,
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
  "describeMissingDeepSeekCompat.returns-new-field-names",
  deepseekMissing.includes("supportsLongPromptCacheRetention") === true,
  `应返回 supportsLongPromptCacheRetention，实际: ${JSON.stringify(deepseekMissing)}`,
);
expect(
  "describeMissingDeepSeekCompat.returns-requiresReasoningContentForToolCalls",
  deepseekMissing.includes("requiresReasoningContentForToolCalls") === true,
  `应返回 requiresReasoningContentForToolCalls，实际: ${JSON.stringify(deepseekMissing)}`,
);
expect(
  "describeMissingDeepSeekCompat.no-legacy-fields",
  deepseekMissing.includes("sendSessionAffinityHeaders") === false &&
    deepseekMissing.includes("sendSessionIdHeader") === false &&
    deepseekMissing.includes("supportsLongCacheRetention") === false &&
    deepseekMissing.includes("requiresReasoningContentOnAssistantMessages") === false &&
    deepseekMissing.includes("thinkingFormat") === false,
  `不应包含旧字段名，实际: ${JSON.stringify(deepseekMissing)}`,
);

const deepseekSuggestion = buildDeepSeekCompatSuggestion(deepseekMissing);
expect(
  "buildDeepSeekCompatSuggestion.has-new-keys",
  Object.prototype.hasOwnProperty.call(deepseekSuggestion, "supportsLongPromptCacheRetention") === true &&
    Object.prototype.hasOwnProperty.call(deepseekSuggestion, "requiresReasoningContentForToolCalls") === true,
  `建议应包含新字段名，实际: ${JSON.stringify(deepseekSuggestion)}`,
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
    Object.prototype.hasOwnProperty.call(fixSuggestion.compatKeys, "supportsLongPromptCacheRetention") === true,
    `compatKeys 应包含 supportsLongPromptCacheRetention，实际: ${JSON.stringify(fixSuggestion.compatKeys)}`,
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
  sessionManager: { getSessionId(): string; getHeader?(): Record<string, unknown> };
  ui: { notify(m: string, l?: string): void; setStatus(k: string, v?: string): void; confirm(t: string, m: string): Promise<boolean>; select(t: string, o: string[]): Promise<string | undefined> };
  cwd?: string;
  hasUI?: boolean;
};

function makeContext(overrides: { model?: HarnessCtx["model"]; sessionManager?: Partial<HarnessCtx["sessionManager"]>; } = {}): HarnessCtx {
  const sessionId = overrides.sessionManager?.getSessionId ? overrides.sessionManager.getSessionId() : "test-session-id";
  const defaultHeader = { providerPromptCacheKey: "host-cache-key", providerSessionId: "provider-session-id" };
  const header = overrides.sessionManager?.getHeader ? overrides.sessionManager.getHeader() : defaultHeader;
  return {
    model: overrides.model ?? { provider: "test", id: "test-model", api: "openai-completions", baseUrl: "https://proxy.example/v1" },
    modelRegistry: { find: () => undefined, getAvailable: () => [], getAll: () => [] },
    sessionManager: { getSessionId: () => sessionId, getHeader: () => header },
    ui: { notify: () => {}, setStatus: () => {}, confirm: async () => true, select: async () => undefined },
    cwd: "/tmp",
    hasUI: false,
  };
}

function createExtensionHarness(): {
  handlers: Record<string, Function[]>;
  runBeforeAgentStart(event: Record<string, unknown>, ctx?: HarnessCtx): Promise<unknown>;
  runBeforeProviderRequest(event: { payload: unknown }, ctx?: HarnessCtx): unknown;
} {
  const handlers: Record<string, Function[]> = {};
  const pi = {
    on(event: string, handler: Function) { (handlers[event] ??= []).push(handler); },
    registerCommand() {},
  };
  cacheOptimizer(pi as unknown as Parameters<typeof cacheOptimizer>[0]);
  return {
    handlers,
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

// 1. before_agent_start：保留块数量、顺序、skill 描述、未知块；仅清理 session-overview churn
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

// 2. before_agent_start：无变化时返回 {}（非 { systemPrompt: [...] }）
const result2 = await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: ["stable block one with enough text", "stable block two with enough text"] });
expect(
  "before_agent_start.no-change-returns-empty-object",
  result2 !== undefined && typeof result2 === "object" && !("systemPrompt" in (asRecord(result2) ?? {})),
  `无变化时应返回 {}，实际: ${JSON.stringify(result2)}`,
);

// 3. before_provider_request：不向 provider body 注入 prompt_cache_key
const providerCtx = makeContext({ model: { provider: "proxy", id: "gpt-test", api: "openai-completions", baseUrl: "https://proxy.example/v1" } });
const providerPayload: Record<string, unknown> = { model: "gpt-test", messages: [] };
const result3 = harness.runBeforeProviderRequest({ payload: providerPayload }, providerCtx);
expect(
  "before_provider_request.does-not-inject-prompt-cache-key",
  result3 === undefined && providerPayload.prompt_cache_key === undefined && providerPayload.promptCacheKey === undefined,
  `不应注入 prompt_cache_key，实际 result=${JSON.stringify(result3)} payload=${JSON.stringify(providerPayload)}`,
);

// 4. before_provider_request：已有 cache key 字段不被覆盖
const snakePayload: Record<string, unknown> = { prompt_cache_key: "raw-session-id", messages: [] };
expect(
  "before_provider_request.keeps-existing-snake-key",
  harness.runBeforeProviderRequest({ payload: snakePayload }, providerCtx) === undefined && snakePayload.prompt_cache_key === "raw-session-id",
  `已有 snake_case key 不应被覆盖，实际: ${JSON.stringify(snakePayload)}`,
);
const camelPayload: Record<string, unknown> = { promptCacheKey: "camel-key", messages: [] };
expect(
  "before_provider_request.keeps-existing-camel-key",
  harness.runBeforeProviderRequest({ payload: camelPayload }, providerCtx) === undefined && camelPayload.promptCacheKey === "camel-key",
  `已有 camelCase key 不应被覆盖，实际: ${JSON.stringify(camelPayload)}`,
);

// 5. cache hints：优先使用 OMP 17 header 的 providerPromptCacheKey，缺失时 fallback session id
const hintService = __internals_for_tests.getCacheHintsService();
const hint1 = hintService?.getHints({});
expect(
  "cache-hints.uses-providerPromptCacheKey-before-session-id",
  hint1?.promptCacheKey === "host-cache-key",
  `hint 应使用 header 的 providerPromptCacheKey，实际: ${JSON.stringify(hint1?.promptCacheKey)}`,
);
const noHeaderCtx = makeContext({ sessionManager: { getSessionId: () => "fallback-session-id", getHeader: () => ({}) } });
await harness.runBeforeAgentStart({ type: "before_agent_start", prompt: "hi", systemPrompt: [primaryBlock] }, noHeaderCtx);
const hint2 = hintService?.getHints({});
expect(
  "cache-hints.fallback-to-session-id-when-no-header-key",
  hint2?.promptCacheKey === "fallback-session-id",
  `无 header key 时应 fallback 到 session id，实际: ${JSON.stringify(hint2?.promptCacheKey)}`,
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

// ── 结果汇总 ─────────────────────────────────────────────────────

console.log(`\n${"=".repeat(60)}`);
console.log(`冒烟测试结果: ${passed} 通过, ${failed} 失败`);
console.log(`${"=".repeat(60)}`);

if (failed > 0) {
  process.exit(1);
}
