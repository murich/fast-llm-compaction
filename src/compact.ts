import { collectToolCalls, goalFromMessages } from './state.js';
import { condenseInput, pooledMap } from './summarize.js';
import type {
  CallDecision,
  CompactOptions,
  CompactResult,
  Message,
  ResolvedCompactOptions,
  SummarizeInput,
  Summarizer,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  preserveRecentMessages: 6,
  minResultChars: 200,
  maxToolInputChars: 200,
  maxPromptResultChars: 6000,
  truncateHeadChars: 300,
};

/** How many summarizer calls may be in flight at once. */
export const DEFAULT_CONCURRENCY = 4;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    minResultChars: Math.max(
      0,
      Math.floor(finite(options.minResultChars, DEFAULT_OPTIONS.minResultChars)),
    ),
    maxToolInputChars: Math.max(
      1,
      Math.floor(finite(options.maxToolInputChars, DEFAULT_OPTIONS.maxToolInputChars)),
    ),
    maxPromptResultChars: Math.max(
      200,
      Math.floor(finite(options.maxPromptResultChars, DEFAULT_OPTIONS.maxPromptResultChars)),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
  };
}

/** What a tool result becomes when its summarization failed. */
export function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-llm-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

export interface CompactRunOptions extends CompactOptions {
  /**
   * Sentences already made, keyed by `tool_use_id` — filled incrementally as
   * tool calls land, read here, updated with whatever is computed. A summary
   * is paid for once.
   */
  cache?: Map<string, string>;
  /** Summarizer calls in flight at once. Default 4. */
  concurrency?: number;
}

/**
 * Compacts a transcript by replacing every unpinned tool result over
 * `minResultChars` characters with one sentence: what the call found, why it
 * matters for the task, what follows from it. The call itself stays (its input
 * condensed when long), so nothing in the transcript refers to work whose
 * record is gone. Results at or below the limit stay verbatim.
 *
 * Throws when the summarizer fails for every candidate; the caller decides
 * whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  summarizer: Summarizer,
  options: CompactRunOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const goal = options.goal || goalFromMessages(messages);
  const cache = options.cache;
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const candidates = calls.filter(
    (call) => !call.pinned && call.resultChars > resolved.minResultChars,
  );
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let requests = 0;
  const outcomes = new Map<string, { summary?: string; error?: Error }>();
  const pending = candidates.filter((call) => !cache?.has(call.tool_use_id));
  const answered = await pooledMap(
    pending,
    options.concurrency ?? DEFAULT_CONCURRENCY,
    async (call) => {
      requests += 1;
      const input: SummarizeInput = {
        id: call.id,
        tool_use_id: call.tool_use_id,
        tool: call.tool,
        input: call.input,
        resultText: call.resultText,
        isError: call.isError,
        goal,
      };
      return summarizer.summarize(input);
    },
  );
  answered.forEach((outcome, index) => {
    const call = pending[index] as ToolCall;
    if (outcome instanceof Error) outcomes.set(call.tool_use_id, { error: outcome });
    else {
      const summary = outcome.trim();
      outcomes.set(call.tool_use_id, { summary });
      cache?.set(call.tool_use_id, summary);
    }
  });

  const successes = candidates.filter(
    (call) =>
      (cache?.get(call.tool_use_id) ?? outcomes.get(call.tool_use_id)?.summary) !== undefined,
  ).length;
  if (candidates.length > 0 && successes === 0) {
    const first = [...outcomes.values()].find((outcome) => outcome.error)?.error;
    throw new Error(
      `summarizer failed for all ${candidates.length} tool calls: ${
        first ? first.message : 'no error captured'
      }`,
    );
  }

  const decisions = calls.map((call): CallDecision => {
    const base = { id: call.id, tool: call.tool, charsBefore: call.resultChars };
    if (call.pinned) return { ...base, action: 'keep', reason: 'pinned', charsAfter: call.resultChars };
    if (call.resultChars <= resolved.minResultChars) {
      return { ...base, action: 'keep', reason: 'small', charsAfter: call.resultChars };
    }
    const summary = cache?.get(call.tool_use_id) ?? outcomes.get(call.tool_use_id)?.summary;
    if (summary !== undefined) {
      return {
        ...base,
        action: 'summarize',
        reason: 'summarized',
        charsAfter: summary.length,
        summary,
      };
    }
    const text = truncatedResultText(
      call.resultText,
      call.isError,
      resolved.truncateHeadChars,
    );
    return {
      ...base,
      action: 'truncate',
      reason: 'summarize_failed',
      charsAfter: text.length,
    };
  });

  const kept = applyDecisions(messages, decisions, calls, resolved);
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'small') + count(decisions, 'pinned'),
      summarized: count(decisions, 'summarized'),
      truncated: count(decisions, 'summarize_failed'),
      pinned: count(decisions, 'pinned'),
      requests,
      failures: count(decisions, 'summarize_failed'),
      ms: Date.now() - started,
    },
  };
}

/**
 * Rebuilds the conversation from the decisions: a summarized result becomes
 * its sentence, a failed one a bounded head and note, a long tool input its
 * condensed skeleton. Messages that lose all their content are removed;
 * untouched messages are returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  options: ResolvedCompactOptions,
): Message[] {
  const byToolUseId = new Map<string, CallDecision>();
  const byCallId = new Map(calls.map((call) => [call.id, call]));
  for (const decision of decisions) {
    const call = byCallId.get(decision.id);
    if (call) byToolUseId.set(call.tool_use_id, decision);
  }

  const kept: Message[] = [];
  for (const message of messages) {
    const toolUses = message.toolUses.map((tool) => {
      const decision = byToolUseId.get(tool.tool_use_id);
      const input =
        decision && decision.reason !== 'pinned'
          ? condenseInput(tool.input, options.maxToolInputChars)
          : tool.input;
      const text = decisionText(tool, decision, undefined, tool.isError ?? false, options.truncateHeadChars);
      const inputChanged = input !== tool.input;
      const textChanged = (tool.text ?? '') !== (text ?? '');
      if (!inputChanged && !textChanged) return tool;
      const copy: ToolUse = {
        tool_use_id: tool.tool_use_id,
        tool: tool.tool,
        input,
      };
      if (text !== undefined) copy.text = text;
      if (tool.isError) copy.isError = true;
      return copy;
    });
    const toolResults = (message.toolResults ?? []).map((result) => {
      const decision = byToolUseId.get(result.tool_use_id);
      const text = decisionText(
        undefined,
        decision,
        result.text,
        result.isError ?? false,
        options.truncateHeadChars,
      );
      if (text === undefined || text === result.text) return result;
      return { tool_use_id: result.tool_use_id, text, isError: result.isError };
    });

    const usesChanged = toolUses.some((tool, index) => tool !== message.toolUses[index]);
    const resultsChanged = toolResults.some(
      (result, index) => result !== message.toolResults?.[index],
    );
    if (!usesChanged && !resultsChanged) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/**
 * The replacement text for a call's result (and its `toolUse.text` mirror):
 * the sentence on success, a bounded head on failure, `undefined` when the
 * result stands as it is.
 */
function decisionText(
  tool: ToolUse | undefined,
  decision: CallDecision | undefined,
  resultText?: string,
  isError = false,
  headChars = 300,
): string | undefined {
  const original = tool ? (tool.text ?? '') : (resultText ?? '');
  if (!decision || decision.action === 'keep') return tool ? tool.text : resultText;
  if (decision.action === 'summarize') return decision.summary;
  return truncatedResultText(original, isError, headChars);
}
