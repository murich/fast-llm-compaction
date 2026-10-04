import type {
  HttpFetch,
  SummarizeInput,
  Summarizer,
} from './types.js';
import { truncate } from './state.js';

/**
 * The question the plugin asks about every tool call worth asking about. The
 * answer is what survives into the compacted transcript.
 */
export const SUMMARIZE_SYSTEM =
  'You distil tool outputs for a coding agent whose context window is being compacted. ' +
  'You answer with exactly one sentence and nothing else: no preamble, no label, no quotes.';

export function buildSummarizePrompt(input: SummarizeInput, maxPromptResultChars: number): string {
  const goal = input.goal.trim() || '(no user request captured yet)';
  const args = Object.entries(input.input)
    .map(([key, value]) => {
      const text = typeof value === 'string' ? value : JSON.stringify(value);
      return `${key}=${truncate(text ?? 'null', 200).replace(/\s+/g, ' ')}`;
    })
    .join(' ');
  const result = truncate(input.resultText, maxPromptResultChars);
  return [
    `Task context: ${goal}`,
    '',
    `Tool call: ${input.tool}(${truncate(args, 600)})`,
    `Tool result (${input.resultText.length} chars${input.isError ? ', tool reported an error' : ''}):`,
    result,
    '',
    'In one sentence, summarize which key elements and values were found in the tool response, ' +
      'why these key elements are important for us, and which conclusion you make in the context ' +
      'of our task based on this response.',
    'Answer with that one sentence only.',
  ].join('\n');
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

export interface HttpSummarizerOptions {
  /** OpenAI-compatible base URL, e.g. `https://llm.example.com/v1`. */
  baseUrl: string;
  apiKey?: string;
  model: string;
  maxTokens?: number;
  /** Characters of the tool result shown to the model. Default 6000. */
  maxPromptResultChars?: number;
  fetch: HttpFetch;
  /** Extra request body fields (temperature, etc.). */
  extraBody?: Record<string, unknown>;
}

/**
 * A `Summarizer` over any OpenAI-compatible `/chat/completions` endpoint —
 * a local model, a gateway, or a hosted one.
 */
export function httpSummarizer(options: HttpSummarizerOptions): Summarizer {
  const base = options.baseUrl.replace(/\/+$/, '');
  return {
    async summarize(input) {
      const body = {
        model: options.model,
        max_tokens: options.maxTokens ?? 128,
        messages: [
          { role: 'system', content: SUMMARIZE_SYSTEM },
          {
            role: 'user',
            content: buildSummarizePrompt(input, options.maxPromptResultChars ?? 6000),
          },
        ],
        ...options.extraBody,
      };
      const response = await options.fetch(`${base}/chat/completions`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
        },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(`summarizer HTTP ${response.status}: ${truncate(response.text, 200)}`);
      }
      let parsed: {
        choices?: Array<{ message?: { content?: unknown } }>;
        error?: { message?: unknown };
      };
      try {
        parsed = JSON.parse(response.text);
      } catch {
        throw new Error(`summarizer returned non-JSON: ${truncate(response.text, 200)}`);
      }
      const content = parsed.choices?.[0]?.message?.content;
      if (typeof content !== 'string' || content.trim().length === 0) {
        throw new Error(
          `summarizer returned no content: ${truncate(response.text, 200)}`,
        );
      }
      return content.trim();
    },
  };
}

/** Replaces a result with its cached sentence; on a miss, asks and caches. */
export function cachedSummarizer(
  inner: Summarizer,
  cache: Map<string, string>,
  counter?: { requests: number },
): Summarizer {
  return {
    async summarize(input) {
      const hit = cache.get(input.tool_use_id);
      if (hit !== undefined) return hit;
      const summary = await inner.summarize(input);
      cache.set(input.tool_use_id, summary);
      if (counter) counter.requests += 1;
      return summary;
    },
  };
}

/** Runs `worker` over `items` with at most `limit` in flight, in input order of completion. */
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
