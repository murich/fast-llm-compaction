import type { SummarizeInput, Summarizer } from './types.js';
import { truncate } from './state.js';

/**
 * The side question the plugin puts to the session's own model about every
 * tool call worth asking about. The model that answers is the one that did
 * the work — same model, same conversation context — so the sentence carries
 * its judgement of what the result means for the task. The reply is what
 * survives into the compacted transcript.
 */
export const SUMMARIZE_PREAMBLE =
  'Side question from the context-compaction plugin. Do not continue the task, do not use ' +
  'tools, and do not comment on this question; answer it directly.';

export function buildSummarizePrompt(input: SummarizeInput, maxPromptResultChars: number): string {
  const goal = input.goal.trim();
  const args = Object.entries(input.input)
    .map(([key, value]) => {
      const text = typeof value === 'string' ? value : JSON.stringify(value);
      return `${key}=${truncate(text ?? 'null', 200).replace(/\s+/g, ' ')}`;
    })
    .join(' ');
  const result = truncate(input.resultText, maxPromptResultChars);
  return [
    SUMMARIZE_PREAMBLE,
    goal ? `\nTask context: ${goal}` : '',
    `Tool call ${input.tool}(${truncate(args, 600)}).`,
    `Tool result (${input.resultText.length} chars${
      input.isError ? ', tool reported an error' : ''
    }):`,
    result,
    '',
    'In one sentence, summarize which key elements and values were found in the tool response, ' +
      'why these key elements are important for us, and which conclusion you make in the context ' +
      'of our task based on this response.',
    'Answer with that one sentence only.',
  ]
    .filter((line) => line !== '')
    .join('\n');
}

const STRING_VALUE_CHARS = 80;
const BULK_VALUE_CHARS = 120;

/**
 * Shrinks a tool input to its informative skeleton: long strings get an
 * elision marker, bulky structures become a size note. The call itself always
 * survives — a compacted transcript must never narrate work whose call is gone.
 */
export function condenseInput(
  input: Record<string, unknown>,
  maxChars: number,
): Record<string, unknown> {
  let json = '';
  try {
    json = JSON.stringify(input) ?? '';
  } catch {
    return { _input: '[unserializable input]' };
  }
  if (json.length <= maxChars) return input;

  const condensed: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (typeof value === 'string') {
      condensed[key] =
        value.length <= STRING_VALUE_CHARS
          ? value
          : `${value.slice(0, STRING_VALUE_CHARS)}… [${value.length - STRING_VALUE_CHARS} more chars]`;
      continue;
    }
    if (value === null || typeof value === 'number' || typeof value === 'boolean') {
      condensed[key] = value;
      continue;
    }
    let inner = '';
    try {
      inner = JSON.stringify(value) ?? '';
    } catch {
      inner = '';
    }
    if (inner.length <= BULK_VALUE_CHARS) condensed[key] = value;
    else {
      const kind = Array.isArray(value) ? 'array' : typeof value;
      condensed[key] = `[${kind}, ${inner.length} chars omitted]`;
    }
  }

  let out = '';
  try {
    out = JSON.stringify(condensed) ?? '';
  } catch {
    return { _input: '[unserializable input]' };
  }
  if (out.length <= maxChars * 3) return condensed;

  // Still oversized: keep the first keys that fit the budget and say how much went.
  const kept: Record<string, unknown> = {};
  const keys = Object.keys(condensed);
  for (const key of keys) {
    const next = { ...kept, [key]: condensed[key] };
    let size = 0;
    try {
      size = JSON.stringify(next)?.length ?? 0;
    } catch {
      size = maxChars * 3;
    }
    if (size > maxChars * 2) break;
    Object.assign(kept, next);
  }
  kept._input_omitted = `${keys.length - Object.keys(kept).length} of ${keys.length} arguments`;
  return kept;
}

/** What `$.model.fork` does: one completion over the session's own transcript. */
export type ModelFork = (request: { prompt: string }) => Promise<{ text: string } | null>;

/**
 * A `Summarizer` on the session's own model and context. The question is put
 * to the model that did the work — `complete` would be a stranger to the task,
 * and a smaller model would not know what matters.
 */
export function forkSummarizer(
  fork: ModelFork,
  maxPromptResultChars: number,
): Summarizer {
  return {
    async summarize(input: SummarizeInput): Promise<string> {
      const reply = await fork({ prompt: buildSummarizePrompt(input, maxPromptResultChars) });
      const summary = reply?.text.trim() ?? '';
      if (!summary) throw new Error('the session model returned no summary');
      return summary;
    },
  };
}

/** Runs `worker` over `items` with at most `limit` in flight. */
export async function pooledMap<T, R>(
  items: readonly T[],
  limit: number,
  worker: (item: T) => Promise<R>,
): Promise<Array<R | Error>> {
  const results: Array<R | Error> = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    for (;;) {
      const index = next++;
      if (index >= items.length) return;
      try {
        results[index] = await worker(items[index] as T);
      } catch (error) {
        results[index] = error instanceof Error ? error : new Error(String(error));
      }
    }
  });
  await Promise.all(runners);
  return results;
}
