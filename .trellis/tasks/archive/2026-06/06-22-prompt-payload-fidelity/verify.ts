// Verification script for prompt payload array write-back fidelity.
//
// Run from the repo root with:
//   bun .trellis/tasks/archive/2026-06/06-22-prompt-payload-fidelity/verify.ts
//
// Exits 0 on success, 1 on any failed assertion.

import { __internals_for_tests } from "../../../../../index.ts";

const { extractSystemPrompt, setSystemPrompt, asRecord } = __internals_for_tests;

type Failure = { name: string; detail: string };
const failures: Failure[] = [];

function expect(name: string, cond: boolean, detail: string): void {
  if (!cond) failures.push({ name, detail });
}

function expectEq(name: string, actual: unknown, expected: unknown): void {
  expect(name, actual === expected, `actual=${JSON.stringify(actual)}, expected=${JSON.stringify(expected)}`);
}

{
  const nonTextBlock = { type: "image", source: { type: "base64", media_type: "image/png", data: "abc" } };
  const payload = {
    system: [
      { type: "text", text: "old anthropic", cache_control: { type: "ephemeral" }, foo: "bar" },
      nonTextBlock,
    ],
    messages: [],
  };

  expectEq("anthropic-set-returns-true", setSystemPrompt(payload, "new anthropic"), true);
  expectEq("anthropic-roundtrip", extractSystemPrompt(payload), "new anthropic");
  expectEq("anthropic-keeps-array", Array.isArray(payload.system), true);
  expectEq("anthropic-keeps-metadata", asRecord(payload.system[0])?.foo, "bar");
  expectEq("anthropic-keeps-cache-control", asRecord(asRecord(payload.system[0])?.cache_control)?.type, "ephemeral");
  expectEq("anthropic-keeps-non-text-sibling", payload.system[1], nonTextBlock);
}

{
  const nonTextPart = { inlineData: { mimeType: "image/png", data: "abc" } };
  const payload = {
    systemInstruction: {
      parts: [{ text: "old google", thought: true }, nonTextPart],
    },
    contents: [],
  };

  expectEq("google-set-returns-true", setSystemPrompt(payload, "new google"), true);
  expectEq("google-roundtrip", extractSystemPrompt(payload), "new google");
  expectEq("google-keeps-array", Array.isArray(payload.systemInstruction.parts), true);
  expectEq("google-keeps-metadata", asRecord(payload.systemInstruction.parts[0])?.thought, true);
  expectEq("google-keeps-non-text-sibling", payload.systemInstruction.parts[1], nonTextPart);
}

{
  const nonTextContent = { type: "input_image", image_url: "data:image/png;base64,abc" };
  const payload = {
    messages: [
      {
        role: "developer",
        content: [{ type: "text", text: "old openai", cache_control: { type: "ephemeral" } }, nonTextContent],
      },
      { role: "user", content: "Hello" },
    ],
  };

  expectEq("openai-set-returns-true", setSystemPrompt(payload, "new openai"), true);
  expectEq("openai-roundtrip", extractSystemPrompt(payload), "new openai");
  expectEq("openai-keeps-array", Array.isArray(payload.messages[0].content), true);
  expectEq("openai-keeps-cache-control", asRecord(asRecord(payload.messages[0].content[0])?.cache_control)?.type, "ephemeral");
  expectEq("openai-keeps-non-text-sibling", payload.messages[0].content[1], nonTextContent);
}

{
  const payload = {
    system: [{ type: "tool_result", id: "tool-1" }],
    messages: [],
  };

  expectEq("anthropic-fallback-returns-true", setSystemPrompt(payload, "fallback anthropic"), true);
  expectEq("anthropic-fallback-roundtrip", extractSystemPrompt(payload), "fallback anthropic");
  expectEq("anthropic-fallback-inserted-first-type", asRecord(payload.system[0])?.type, "text");
  expectEq("anthropic-fallback-inserted-first-text", asRecord(payload.system[0])?.text, "fallback anthropic");
  expectEq("anthropic-fallback-keeps-original-second", asRecord(payload.system[1])?.id, "tool-1");
}

{
  const payload = {
    systemInstruction: {
      parts: [{ inlineData: { mimeType: "image/png", data: "abc" } }],
    },
    contents: [],
  };

  expectEq("google-fallback-returns-true", setSystemPrompt(payload, "fallback google"), true);
  expectEq("google-fallback-roundtrip", extractSystemPrompt(payload), "fallback google");
  expectEq("google-fallback-inserted-first-text", asRecord(payload.systemInstruction.parts[0])?.text, "fallback google");
  expectEq("google-fallback-keeps-original-second", Boolean(asRecord(payload.systemInstruction.parts[1])?.inlineData), true);
}

{
  const payload = {
    messages: [
      {
        role: "system",
        content: [{ type: "input_image", image_url: "data:image/png;base64,abc" }],
      },
    ],
  };

  expectEq("openai-fallback-returns-true", setSystemPrompt(payload, "fallback openai"), true);
  expectEq("openai-fallback-keeps-array", Array.isArray(payload.messages[0].content), true);
  expectEq("openai-fallback-roundtrip", extractSystemPrompt(payload), "fallback openai");
  expectEq("openai-fallback-inserted-first-type", asRecord(payload.messages[0].content[0])?.type, "text");
  expectEq("openai-fallback-inserted-first-text", asRecord(payload.messages[0].content[0])?.text, "fallback openai");
  expectEq("openai-fallback-keeps-original-second", asRecord(payload.messages[0].content[1])?.type, "input_image");
}

if (failures.length === 0) {
  console.log("✅ prompt payload fidelity verification passed");
  process.exit(0);
}

console.error(`❌ ${failures.length} assertion(s) failed:`);
for (const failure of failures) {
  console.error(`- ${failure.name}: ${failure.detail}`);
}
process.exit(1);
