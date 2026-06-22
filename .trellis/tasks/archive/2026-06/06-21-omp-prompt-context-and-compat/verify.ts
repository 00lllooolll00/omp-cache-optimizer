// Verification script for OMP compat fallback and prompt rewrite context matching.
//
// Run from the repo root with:
//   bun .trellis/tasks/archive/2026-06/06-21-omp-prompt-context-and-compat/verify.ts
//
// Exits 0 on success, 1 on any failed assertion.

import { __internals_for_tests } from "../../../../../index.ts";

const {
  getCompat,
  makePromptRewriteContextKey,
  rememberPromptRewriteContext,
  getPromptRewriteContext,
  PROMPT_REWRITE_CONTEXT_TTL_MS,
} = __internals_for_tests;

type Failure = { name: string; detail: string };
const failures: Failure[] = [];

function expect(name: string, cond: boolean, detail: string): void {
  if (!cond) failures.push({ name, detail });
}

function expectEq(name: string, actual: unknown, expected: unknown): void {
  expect(name, actual === expected, `actual=${JSON.stringify(actual)}, expected=${JSON.stringify(expected)}`);
}

{
  const model = {
    provider: "eflowcode_cn",
    id: "deepseek-v4-pro",
    compatConfig: { supportsLongPromptCacheRetention: true },
    compat: { requiresReasoningContentForToolCalls: true },
  };
  const compat = getCompat(model as any);

  expectEq("compat-fallback-uses-sparse-config", compat.supportsLongPromptCacheRetention, true);
  expectEq("compat-fallback-keeps-model-compat", compat.requiresReasoningContentForToolCalls, true);
}

{
  const model = {
    provider: "eflowcode_cn",
    id: "deepseek-v4-pro",
    compatConfig: { supportsLongPromptCacheRetention: false },
    compat: { supportsLongPromptCacheRetention: true },
  };
  const compat = getCompat(model as any);

  expectEq("compat-resolved-model-wins", compat.supportsLongPromptCacheRetention, true);
}

{
  expectEq(
    "prompt-key-direct-model",
    makePromptRewriteContextKey("session-a", { provider: "p", id: "m" } as any),
    "session-a:p/m",
  );
  expectEq(
    "prompt-key-missing-model",
    makePromptRewriteContextKey("session-a", undefined),
    undefined,
  );
}

{
  const contexts = new Map<string, any>();
  const optionsA = { promptGuidelines: ["stable A"] };
  const optionsB = { promptGuidelines: ["stable B"] };
  rememberPromptRewriteContext(contexts, "session-a:p/model-a", { options: optionsA, timestamp: 1000 });
  rememberPromptRewriteContext(contexts, "session-a:p/model-b", { options: optionsB, timestamp: 2000 });
  expectEq("prompt-context-a-is-isolated", getPromptRewriteContext(contexts, "session-a:p/model-a", 2000)?.options, optionsA);
  expectEq("prompt-context-b-is-isolated", getPromptRewriteContext(contexts, "session-a:p/model-b", 2000)?.options, optionsB);
  expectEq("prompt-context-missing-key", getPromptRewriteContext(contexts, "session-a:p/model-c", 2000), undefined);
}

{
  const contexts = new Map<string, any>();
  const options = { promptGuidelines: ["stable"] };
  rememberPromptRewriteContext(contexts, "session-a:p/model-a", { options, timestamp: 1000 });

  expectEq(
    "prompt-context-within-ttl",
    getPromptRewriteContext(contexts, "session-a:p/model-a", 1000 + PROMPT_REWRITE_CONTEXT_TTL_MS)?.options,
    options,
  );
  expectEq(
    "prompt-context-expired",
    getPromptRewriteContext(contexts, "session-a:p/model-a", 1001 + PROMPT_REWRITE_CONTEXT_TTL_MS),
    undefined,
  );
  expectEq("prompt-context-expiry-deletes-entry", contexts.has("session-a:p/model-a"), false);
}

if (failures.length === 0) {
  console.log("✅ OMP prompt context and compat verification passed");
  process.exit(0);
}

console.error(`❌ ${failures.length} assertion(s) failed:`);
for (const failure of failures) {
  console.error(`- ${failure.name}: ${failure.detail}`);
}
process.exit(1);
