export type Role = 'user' | 'assistant';

/**
 * A tool_use block of an assistant message. `text` and `isError` mirror the
 * outcome once the transcript holds it (Claude Code attaches them).
 */
export interface ToolUse {
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  text?: string;
  isError?: boolean;
}

/** A tool_result block of a user message. */
export interface ToolResult {
  tool_use_id: string;
  text: string;
  isError?: boolean;
}

/**
 * One transcript message. The shape is a subset of Claude Code's
 * `SessionMessage`, so a session transcript can be passed in as is.
 */
export interface Message {
  role: Role;
  text: string;
  toolUses: ToolUse[];
  toolResults?: ToolResult[];
}

/** A tool call paired with its result by `tool_use_id`. */
export interface ToolCall {
  /** Short id used in the decision log (`t1`, `t2`, ...). */
  id: string;
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  /** Index of the message holding the tool_use block. */
  callIndex: number;
  /** Index of the message holding the tool_result block. */
  resultIndex: number;
  /** The full result text; what gets replaced by the summary. */
  resultText: string;
  resultChars: number;
  isError: boolean;
  /** In the first or the newest preserved messages; never a candidate. */
  pinned: boolean;
}

/** What one summarization is asked about. */
export interface SummarizeInput {
  /** Short id of the call (`t1`), for logs. */
  id: string;
  tool_use_id: string;
  tool: string;
  /** The call's input (already condensed when it was long). */
  input: Record<string, unknown>;
  /** The tool result to distil, full text; the prompt builder caps it. */
  resultText: string;
  isError: boolean;
  /** The ongoing task, so the summary can say why a finding matters. */
  goal: string;
}

/**
 * The one-sentence distiller. Replaces the upstream `JevAsker`: instead of
 * scoring whether a call still matters, it states what the call found, why it
 * matters and what follows from it. The implementation of record is
 * `forkSummarizer` — the session's own model answering over its own context.
 */
export interface Summarizer {
  summarize(input: SummarizeInput): Promise<string>;
}

export type CallAction = 'keep' | 'summarize' | 'truncate';

export interface CallDecision {
  id: string;
  tool: string;
  action: CallAction;
  reason: 'pinned' | 'small' | 'summarized' | 'summarize_failed';
  /** Result characters before and after the action. */
  charsBefore: number;
  charsAfter: number;
  /** The sentence kept, when one was produced. */
  summary?: string;
}

export interface CompactOptions {
  /** Ongoing task description; defaults to the last few user prompts. */
  goal?: string;
  /** Newest messages never touched (the first message is always kept). Default 6. */
  preserveRecentMessages?: number;
  /** Tool results at or below this many characters stay verbatim. Default 200. */
  minResultChars?: number;
  /** Serialized tool inputs above this many characters are condensed. Default 200. */
  maxToolInputChars?: number;
  /** Characters of a tool result sent to the summarizer. Default 6000. */
  maxPromptResultChars?: number;
  /** Characters of a tool result kept when summarization fails. Default 300. */
  truncateHeadChars?: number;
}

export interface ResolvedCompactOptions {
  goal: string;
  preserveRecentMessages: number;
  minResultChars: number;
  maxToolInputChars: number;
  maxPromptResultChars: number;
  truncateHeadChars: number;
}

export interface CompactResult {
  /** The compacted transcript; untouched messages are the input objects. */
  messages: Message[];
  decisions: CallDecision[];
  stats: {
    messagesBefore: number;
    messagesAfter: number;
    charsBefore: number;
    charsAfter: number;
    calls: number;
    kept: number;
    summarized: number;
    truncated: number;
    pinned: number;
    /** Summarizer calls actually made (cache hits excluded). */
    requests: number;
    /** Calls whose summarization failed and fell back to a truncated head. */
    failures: number;
    ms: number;
  };
}
