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
 */

import { __internals_for_tests } from "./index.ts";

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
  "formatCacheStats.includes-hit-ratio",
  formatted.includes("1/2") === true,
  `应包含 1/2 命中率，实际: "${formatted}"`,
);
expect(
  "formatCacheStats.includes-40-percent",
  formatted.includes("40%") === true,
  `应包含 40% 百分比，实际: "${formatted}"`,
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

// 无 system prompt 的 payload
const noSystemPayload = { messages: [{ role: "user", content: "Hello" }] };
expect(
  "extractSystemPrompt.none-returns-undefined",
  extractSystemPrompt(noSystemPayload) === undefined,
  "无 system prompt 应返回 undefined",
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

// ── 结果汇总 ─────────────────────────────────────────────────────

console.log(`\n${"=".repeat(60)}`);
console.log(`冒烟测试结果: ${passed} 通过, ${failed} 失败`);
console.log(`${"=".repeat(60)}`);

if (failed > 0) {
  process.exit(1);
}
