declare module "@oh-my-pi/pi-coding-agent" {
  export type ExtensionModel = {
    provider: string;
    id: string;
    name: string;
    api: string;
    baseUrl: string;
    compat: Record<string, unknown>;
    reasoning: boolean;
    input: ("text" | "image")[];
    cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
    contextWindow: number | null;
    maxTokens: number | null;
  };

  export type ExtensionContext = {
    model?: ExtensionModel;
    modelRegistry: {
      find(provider: string, modelId: string): ExtensionModel | undefined;
      getAvailable(): ExtensionModel[];
      getAll(): ExtensionModel[];
    };
    sessionManager: {
      getSessionId(): string;
      /** OMP 17.0.5 ReadonlySessionManager：会话工作目录。 */
      getCwd(): string;
      getHeader(): {
        type: "session";
        id: string;
        timestamp: string;
        cwd: string;
        providerPromptCacheKey?: string;
      } | null;
    };
    ui: {
      notify(message: string, level?: "info" | "warning" | "error"): void;
      setStatus(key: string, value: string | undefined): void;
      confirm(title: string, message: string): Promise<boolean>;
      select(title: string, options: string[]): Promise<string | undefined>;
    };
    hasUI: boolean;
    cwd: string;
    /** OMP 17：当前生效的系统 prompt（有序块）。 */
    getSystemPrompt(): string[];
  };

  export type CommandContext = ExtensionContext;

  export type BeforeAgentStartEvent = {
    type: "before_agent_start";
    prompt: string;
    images?: unknown[];
    /** OMP 17：有序系统 prompt 块（必填）。 */
    systemPrompt: string[];
  };

  export type BeforeAgentStartEventResult = {
    message?: unknown;
    /** 本 turn 完整替换系统 prompt（多扩展链式合并）。 */
    systemPrompt?: string[];
  };

  // OMP 17：before_agent_start 再次支持返回 systemPrompt: string[] 做整段替换。
  // 主 prompt 重写在该钩子执行。before_provider_request 仅做 payload 级
  // session-overview 兜底 strip 与 prompt_cache_retention 安全网。turn_start 替代 Pi 的 model_select 做 footer 更新。

  export type ExtensionAPI = {
    on(event: "session_start", handler: (event: { type: "session_start" }, ctx: ExtensionContext) => unknown): void;
    on(
      event: "session_before_switch",
      handler: (
        event: { type: "session_before_switch"; reason: "new" | "resume" | "fork" | "handoff"; targetSessionFile?: string },
        ctx: ExtensionContext,
      ) => unknown,
    ): void;
    on(
      event: "session_switch",
      handler: (
        event: { type: "session_switch"; reason: "new" | "resume" | "fork" | "handoff"; previousSessionFile: string | undefined },
        ctx: ExtensionContext,
      ) => unknown,
    ): void;
    on(
      event: "before_agent_start",
      handler: (
        event: BeforeAgentStartEvent,
        ctx: ExtensionContext,
      ) => BeforeAgentStartEventResult | void | Promise<BeforeAgentStartEventResult | void>,
    ): void;
    on(event: "turn_start", handler: (event: { type: "turn_start"; turnIndex: number; timestamp: number }, ctx: ExtensionContext) => unknown): void;
    on(event: "before_provider_request", handler: (event: { type: "before_provider_request"; payload: unknown }, ctx: ExtensionContext) => unknown): void;
    // 保留官方真实事件 overload；本插件不再注册该 handler（非 2xx 不会触发）。
    on(event: "after_provider_response", handler: (event: { status: number; headers?: Record<string, string> }, ctx: ExtensionContext) => unknown): void;
    on(event: "message_end", handler: (event: { message: unknown }, ctx: ExtensionContext) => unknown): void;
    registerCommand(name: string, command: { description?: string; handler: (args: string, ctx: CommandContext) => unknown }): void;
  };
}
