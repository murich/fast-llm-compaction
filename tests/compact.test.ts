import { describe, expect, it } from 'vitest';
import {
  compact,
  condenseInput,
  messageChars,
  reductionRatio,
  resolveOptions,
  summarizeMessages,
  type Message,
  type SummarizeInput,
  type Summarizer,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(
  id: string,
  tool: string,
  input: Record<string, unknown>,
  text: string,
): Message {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
  });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const fileA = 'export const a = 1;\n'.repeat(50); // 1000 chars
const fileB = 'const b = 2;\n'.repeat(60); // 720 chars

/** A transcript with two big tool results, one small one, and pinned edges. */
function transcript(): Message[] {
  return [
    message('user', 'Fix the failing test in src/a.ts.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL b.test.ts: expected 2 to be 3'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    call('tool-3', 'Read', { file_path: 'src/b.ts' }, fileB),
    result('tool-3', fileB),
    message('assistant', 'Fixing now.'),
    message('user', 'go ahead'),
    message('assistant', 'On it.'),
    message('user', 'status?'),
    message('assistant', 'Working.'),
    message('user', 'continue'),
  ];
}

function fakeSummarizer(
  sentence: (input: SummarizeInput) => string = (i) => `Found in ${i.tool}: the key values and the conclusion.`,
  seen: SummarizeInput[] = [],
): Summarizer {
  return {
    async summarize(input) {
      seen.push(input);
      return sentence(input);
    },
  };
}

describe('resolveOptions', () => {
  it('applies defaults and floors', () => {
    expect(resolveOptions()).toMatchObject({
      preserveRecentMessages: 6,
      minResultChars: 200,
      maxToolInputChars: 200,
      maxPromptResultChars: 6000,
      truncateHeadChars: 300,
    });
    expect(resolveOptions({ minResultChars: -5, preserveRecentMessages: 2.7 })).toMatchObject({
      minResultChars: 0,
      preserveRecentMessages: 2,
    });
  });
});

describe('compact', () => {
  it('replaces long tool results with one sentence and keeps short ones verbatim', async () => {
    const seen: SummarizeInput[] = [];
    const result = await compact(transcript(), fakeSummarizer(undefined, seen));

    const texts = result.messages.flatMap((m) => (m.toolResults ?? []).map((r) => r.text));
    expect(texts).toContain('Found in Read: the key values and the conclusion.');
    expect(texts).toContain('FAIL b.test.ts: expected 2 to be 3'); // 35 chars: untouched
    expect(texts.some((t) => t.includes('export const a'))).toBe(false);

    // only the two long results were sent to the summarizer
    expect(seen.map((s) => s.tool_use_id).sort()).toEqual(['tool-1', 'tool-3']);
    // and the full text went with them, not a stub
    expect(seen[0]!.resultText).toBe(fileA);
    expect(seen[0]!.goal).toBe('go ahead\nstatus?\ncontinue'); // last three user prompts
    expect(result.stats.summarized).toBe(2);
    expect(result.stats.kept).toBe(1);
    expect(result.stats.requests).toBe(2);
  });

  it('keeps tool calls and mirrors the sentence onto the call text', async () => {
    const result = await compact(transcript(), fakeSummarizer());
    const uses = result.messages.flatMap((m) => m.toolUses);
    expect(uses.map((u) => u.tool_use_id)).toEqual(['tool-1', 'tool-2', 'tool-3']);
    expect(uses[0]!.text).toBe('Found in Read: the key values and the conclusion.');
  });

  it('never touches the first or the newest messages', async () => {
    const seen: SummarizeInput[] = [];
    const messages = transcript();
    messages.push(
      call('tool-4', 'Read', { file_path: 'src/c.ts' }, fileB),
      result('tool-4', fileB),
    );
    const run = await compact(messages, fakeSummarizer(undefined, seen), {
      preserveRecentMessages: 6,
    });
    expect(seen.map((s) => s.tool_use_id)).not.toContain('tool-4');
    expect(run.stats.pinned).toBeGreaterThan(0);
  });

  it('condenses long tool inputs and leaves short ones alone', async () => {
    const longInput = { command: 'x'.repeat(500), timeout: 30 };
    const messages = [
      message('user', 'run it'),
      call('tool-1', 'Bash', longInput, fileA),
      result('tool-1', fileA),
      call('tool-2', 'Read', { file_path: 'src/a.ts' }, fileA),
      result('tool-2', fileA),
      ...transcript().slice(-6),
    ];
    const run = await compact(messages, fakeSummarizer());
    const uses = run.messages.flatMap((m) => m.toolUses);
    const bash = uses.find((u) => u.tool_use_id === 'tool-1')!;
    expect(JSON.stringify(bash.input).length).toBeLessThan(JSON.stringify(longInput).length);
    expect(bash.input.timeout).toBe(30);
    expect(uses.find((u) => u.tool_use_id === 'tool-2')!.input).toEqual({ file_path: 'src/a.ts' });
  });

  it('falls back to a truncated head when one summarization fails', async () => {
    const summarizer: Summarizer = {
      async summarize(input) {
        if (input.tool_use_id === 'tool-1') throw new Error('boom');
        return 'Sentence.';
      },
    };
    const result = await compact(transcript(), summarizer);
    const texts = result.messages.flatMap((m) => (m.toolResults ?? []).map((r) => r.text));
    const truncated = texts.find((t) => t.includes('[fast-llm-compaction truncated'))!;
    expect(truncated).toContain('export const a');
    expect(result.stats.truncated).toBe(1);
    expect(result.stats.failures).toBe(1);
  });

  it('throws when the summarizer fails for every candidate', async () => {
    await expect(
      compact(transcript(), {
        async summarize() {
          throw new Error('endpoint down');
        },
      }),
    ).rejects.toThrow(/endpoint down/);
  });

  it('reuses cached sentences and counts only real requests', async () => {
    const cache = new Map([['tool-1', 'From the cache.']]);
    const seen: SummarizeInput[] = [];
    const result = await compact(transcript(), fakeSummarizer(undefined, seen), { cache });
    expect(seen.map((s) => s.tool_use_id)).toEqual(['tool-3']);
    expect(result.stats.requests).toBe(1);
    const texts = result.messages.flatMap((m) => (m.toolResults ?? []).map((r) => r.text));
    expect(texts).toContain('From the cache.');
    expect(cache.get('tool-3')).toBe('Found in Read: the key values and the conclusion.');
  });

  it('uses an explicit goal over the inferred one', async () => {
    const seen: SummarizeInput[] = [];
    await compact(transcript(), fakeSummarizer(undefined, seen), {
      goal: 'Migrate the parser.',
    });
    expect(seen[0]!.goal).toBe('Migrate the parser.');
  });

  it('reports a positive reduction ratio and rebuilds only touched messages', async () => {
    const messages = transcript();
    const untouched = messages[7]; // assistant prose
    const result = await summarizeMessages(messages, fakeSummarizer());
    expect(reductionRatio(result)).toBeGreaterThan(0.6);
    expect(result.messages).toContain(untouched);
    expect(result.stats.messagesAfter).toBeLessThanOrEqual(result.stats.messagesBefore);
    expect(messageChars(result.messages[2]!)).toBeLessThan(messageChars(messages[2]!));
  });

  it('does nothing to a transcript without tool calls', async () => {
    const seen: SummarizeInput[] = [];
    const messages = [message('user', 'hi'), message('assistant', 'hello')];
    const result = await compact(messages, fakeSummarizer(undefined, seen));
    expect(seen).toHaveLength(0);
    expect(result.messages).toEqual(messages);
    expect(reductionRatio(result)).toBe(0);
  });
});

describe('condenseInput', () => {
  it('returns the input unchanged when it fits', () => {
    const input = { file_path: 'src/a.ts' };
    expect(condenseInput(input, 200)).toBe(input);
  });

  it('elides long strings and bulky structures but keeps scalars', () => {
    const input = {
      command: 'x'.repeat(500),
      timeout: 30,
      dry: true,
      files: Array.from({ length: 40 }, (_, i) => `file-${i}.ts`),
    };
    const out = condenseInput(input, 200);
    expect(String(out.command)).toMatch(/… \[420 more chars\]$/);
    expect(out.timeout).toBe(30);
    expect(out.dry).toBe(true);
    expect(String(out.files)).toMatch(/\[array, \d+ chars omitted\]/);
    expect(JSON.stringify(out).length).toBeLessThan(JSON.stringify(input).length);
  });
});
