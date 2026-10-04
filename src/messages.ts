import { compact, type CompactRunOptions } from './compact.js';
import type { CompactResult, Message, Summarizer } from './types.js';

export type SummarizeMessagesOptions = CompactRunOptions;

/** `compact` under its public name: one sentence per tool result worth keeping. */
export function summarizeMessages(
  messages: readonly Message[],
  summarizer: Summarizer,
  options: SummarizeMessagesOptions = {},
): Promise<CompactResult> {
  return compact(messages, summarizer, options);
}
