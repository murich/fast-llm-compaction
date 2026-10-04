import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions, type CompactRunOptions } from '../src/compact.js';
import { goalFromMessages, truncate } from '../src/state.js';
import { buildSummarizePrompt, httpSummarizer, SUMMARIZE_SYSTEM } from '../src/summarize.js';
import type {
  CompactResult,
  Message,
  SummarizeInput,
  Summarizer,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  provider: 'engine',
  model: 'haiku',
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  preSummarize: true,
  preSummarizeAtPercent: 0,
  concurrency: 4,
};

export type HookConfig = CompactRunOptions & {
  /** `engine` uses the session's own model provider; `http` an OpenAI-compatible endpoint. */
  provider: 'engine' | 'http';
  /** Model for the summarizer (`haiku` for `engine`; a model id for `http`). */
  model: string;
  /** OpenAI-compatible base URL, `http` only. */
  baseUrl?: string;
  /** API key for the endpoint, `http` only. */
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  /** Summarize tool results as they land, so compaction reuses ready sentences. */
  preSummarize: boolean;
  /** Context percentage from which `tool.call` starts pre-summarizing. Default 0. */
  preSummarizeAtPercent: number;
  concurrency: number;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function optionBoolean(options: PluginOptions, key: string, fallback: boolean): boolean {
  const value = options[key];
  return typeof value === 'boolean' ? value : fallback;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<CompactRunOptions> = {};
  for (const key of [
    'preserveRecentMessages',
    'minResultChars',
    'maxToolInputChars',
    'maxPromptResultChars',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const provider = optionString(options, 'provider');
  const config: HookConfig = {
    ...numbers,
    provider: provider === 'http' ? 'http' : 'engine',
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    preSummarize: optionBoolean(options, 'preSummarize', HOOK_DEFAULTS.preSummarize),
    preSummarizeAtPercent: optionNumber(
      options,
      'preSummarizeAtPercent',
      HOOK_DEFAULTS.preSummarizeAtPercent,
    ),
    concurrency: Math.max(1, optionNumber(options, 'concurrency', HOOK_DEFAULTS.concurrency)),
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const baseUrl = optionString(options, 'baseUrl');
  if (baseUrl) config.baseUrl = baseUrl;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** What the engine's `$.model.complete` needs; the hook passes the real one. */
export type ModelComplete = (request: {
  model: string;
  prompt: string;
  system?: string;
  maxTokens?: number;
}) => Promise<string>;

/** A `Summarizer` on the session's own model provider — no second API key. */
export function engineSummarizer(
  complete: ModelComplete,
  model: string,
  maxPromptResultChars: number,
): Summarizer {
  return {
    async summarize(input: SummarizeInput): Promise<string> {
      const text = await complete({
        model,
        prompt: buildSummarizePrompt(input, maxPromptResultChars),
        system: SUMMARIZE_SYSTEM,
        maxTokens: 200,
      });
      const summary = text.trim();
      if (!summary) throw new Error('summarizer model returned an empty reply');
      return summary;
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when the summarizer fails outright. */
export async function compactSession(
  messages: readonly SessionMessage[],
  summarizer: Summarizer,
  config: HookConfig,
  cache?: Map<string, string>,
): Promise<SessionCompaction> {
  const result = await compact(messages, summarizer, {
    ...config,
    cache,
    concurrency: config.concurrency,
  });
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.summarized > 0 ? `${stats.summarized} summarized` : '',
    stats.truncated > 0 ? `${stats.truncated} truncated (summarizer failed)` : '',
    stats.kept > 0 ? `${stats.kept} kept verbatim` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; ${stats.requests} summarizer call(s) in ${stats.ms} ms`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map((d) => `${d.id}:${d.tool}:${d.action}/${d.charsBefore}→${d.charsAfter}ch`)
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

type HostEnv = {
  env: { get: (name: string) => Promise<string | undefined> };
  settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
};

function settingsEnv(
  settings: Readonly<Record<string, unknown>>,
  name: string,
): string | undefined {
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)[name];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

/** Resolves the `http` provider's endpoint and key from userConfig, then env, then settings. */
export async function getHttpConfig(
  $: HostEnv,
  config: HookConfig,
): Promise<{ baseUrl: string; apiKey?: string; model: string }> {
  const settings = await $.settings.read();
  const baseUrl =
    config.baseUrl ??
    (await $.env.get('LLM_BASE_URL')) ??
    settingsEnv(settings, 'LLM_BASE_URL');
  if (!baseUrl) throw new Error('provider http needs a baseUrl (or LLM_BASE_URL)');
  const apiKey = config.apiKey ?? (await $.env.get('LLM_API_KEY')) ?? settingsEnv(settings, 'LLM_API_KEY');
  const model =
    config.model !== HOOK_DEFAULTS.model
      ? config.model
      : ((await $.env.get('LLM_MODEL')) ?? settingsEnv(settings, 'LLM_MODEL') ?? config.model);
  return { baseUrl, ...(apiKey ? { apiKey } : {}), model };
}

type Host = HostEnv & {
  http: {
    fetch: (
      url: string,
      init?: { method?: string; headers?: Record<string, string>; body?: string },
    ) => Promise<{ status: number; ok: boolean; text: string }>;
  };
  model: { complete: ModelComplete };
  ui: {
    log: (text: string) => void;
    toast: (text: string, options?: { timeoutMs?: number }) => void;
  };
  session: {
    usage: () => Promise<{ context: { percent?: number } }>;
    compact: () => Promise<unknown>;
    messages: () => Promise<readonly SessionMessage[]>;
  };
};

async function buildSummarizer($: Host, config: HookConfig): Promise<Summarizer> {
  if (config.provider === 'http') {
    const http = await getHttpConfig($, config);
    return httpSummarizer({
      baseUrl: http.baseUrl,
      apiKey: http.apiKey,
      model: http.model,
      maxPromptResultChars: config.maxPromptResultChars,
      fetch: (url, init) => $.http.fetch(url, init),
    });
  }
  return engineSummarizer(
    (request) => $.model.complete(request),
    config.model,
    config.maxPromptResultChars ?? 6000,
  );
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

/** The tool's input as the model gave it: `e` flattens the arguments beside `tool` and ids. */
export function inputFromEvent(event: Record<string, unknown>): Record<string, unknown> {
  const input: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(event)) {
    if (key === 'tool' || key === 'tool_use_id' || key === 'agentId') continue;
    input[key] = value;
  }
  return input;
}

/** The task context for a summary made outside compaction; refreshed lazily. */
type GoalState = { goal: string; at: number };

async function recentGoal($: Host, config: HookConfig, state: GoalState): Promise<string> {
  if (config.goal) return config.goal;
  if (state.goal && Date.now() - state.at < 30_000) return state.goal;
  try {
    state.goal = goalFromMessages(await $.session.messages());
    state.at = Date.now();
  } catch {
    /* a missing goal costs detail, not correctness */
  }
  return state.goal;
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  const cache = new Map<string, string>();
  const goalState: GoalState = { goal: configured.goal ?? '', at: 0 };
  let compacting = false;
  let inFlight = 0;

  on('session.compact', async ($, event, next) => {
    try {
      const summarizer = await buildSummarizer($ as unknown as Host, configured);
      const { result, messages } = await compactSession(
        event.messages,
        summarizer,
        configured,
        cache,
      );
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < configured.minReductionRatio) {
        notify(
          $,
          `fallback to built-in summary (below ${percent(configured.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        return next(event);
      }
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, tool results as one-liners (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      notify(
        $,
        `fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`,
      );
      return next(event);
    }
  });

  on('tool.call', async ($, event, next) => {
    const result = await next(event);
    if (!configured.preSummarize || compacting) return result;
    try {
      const text = result?.text;
      const toolUseId = (event as unknown as { tool_use_id?: string }).tool_use_id;
      if (
        typeof text !== 'string' ||
        typeof toolUseId !== 'string' ||
        text.length <= (configured.minResultChars ?? 200) ||
        cache.has(toolUseId) ||
        inFlight >= configured.concurrency
      ) {
        return result;
      }
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < configured.preSummarizeAtPercent) return result;

      const summarizer = await buildSummarizer($ as unknown as Host, configured);
      const input: SummarizeInput = {
        id: toolUseId,
        tool_use_id: toolUseId,
        tool: String((event as unknown as { tool?: unknown }).tool ?? 'tool'),
        input: inputFromEvent(event as unknown as Record<string, unknown>),
        resultText: text,
        isError: false,
        goal: await recentGoal($ as unknown as Host, configured, goalState),
      };
      inFlight += 1;
      void summarizer
        .summarize(input)
        .then((summary) => {
          if (summary.trim()) cache.set(toolUseId, summary.trim());
        })
        .catch((error) => {
          $.ui.log(
            `pre-summarize skipped for ${truncate(input.tool, 40)} (${truncate(
              error instanceof Error ? error.message : String(error),
              120,
            )})`,
          );
        })
        .finally(() => {
          inFlight -= 1;
        });
    } catch {
      /* never disturb the tool call for the sake of a summary */
    }
    return result;
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < configured.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
