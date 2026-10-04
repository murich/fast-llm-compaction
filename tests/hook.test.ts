import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLogLines,
  buildSummarizer,
  flushPending,
  inputFromEvent,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-llm.ts';
import {
  buildBatchPrompt,
  collectToolCalls,
  forkSummarizer,
  parseBatchReply,
  type Message,
  type SummarizeInput,
  type Summarizer,
} from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
    message('assistant', 'Done.', { handle: 'h-7' }),
    message('user', 'thanks', { handle: 'h-8' }),
    message('assistant', 'Welcome.', { handle: 'h-9' }),
    message('user', 'one more', { handle: 'h-10' }),
    message('assistant', 'Sure.', { handle: 'h-11' }),
    message('user', 'now', { handle: 'h-12' }),
    message('assistant', 'Yes.', { handle: 'h-13' }),
  ];
}

const sentence: Summarizer = {
  async summarize(input: SummarizeInput) {
    return `Summary of ${input.tool_use_id}.`;
  },
};

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      preSummarize: true,
      preSummarizeAtPercent: 0,
      concurrency: 4,
      batchSize: 10,
    });
    expect(
      resolveHookConfig({
        minResultChars: 500,
        preSummarize: false,
        concurrency: 2,
        batchSize: 3,
        goal: 'g',
        compactAtPercent: 'no',
      }),
    ).toEqual({
      minResultChars: 500,
      preSummarize: false,
      preSummarizeAtPercent: 0,
      concurrency: 2,
      batchSize: 3,
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      goal: 'g',
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', async () => {
    const messages = transcript();
    const before = new Map(messages.map((m) => [m.role + m.text, m]));
    const { messages: out } = await compactSession(messages, sentence, resolveHookConfig({}), new Map());

    // pinned prose keeps its handle
    const prose = out.find((m) => m.text === 'Fixing now.')!;
    expect(prose.handle).toBe('h-5');

    // rebuilt tool messages have no handle and carry the summary
    const rebuiltCall = out.find((m) => m.toolUses.some((u) => u.tool_use_id === 'tool-1'))!;
    expect(rebuiltCall.handle).toBeUndefined();
    expect(rebuiltCall.toolUses[0]!.text).toBe('Summary of tool-1.');
    expect(before.get('usergo ahead')!.handle).toBe('h-6');
  });
});

describe('compactSession', () => {
  it('carries the stats and the decisions out', async () => {
    const { result } = await compactSession(
      transcript(),
      sentence,
      resolveHookConfig({ preserveRecentMessages: 0 }),
      new Map(),
    );
    expect(result.stats.summarized).toBe(1); // tool-2's result is 35 chars
    expect(result.stats.requests).toBe(1);
    expect(summarize(result)).toContain('1 summarized');
    expect(decisionLogLines(result)[0]).toContain('t1:Read:summarize/1000→');
  });

  it('reuses the pre-summarization cache', async () => {
    const cache = new Map([['tool-1', 'Cached sentence.']]);
    const { result, messages } = await compactSession(
      transcript(),
      {
        async summarize() {
          throw new Error('must not be called for tool-1');
        },
      },
      resolveHookConfig({ preserveRecentMessages: 0 }),
      cache,
    );
    expect(result.stats.requests).toBe(0);
    const texts = messages.flatMap((m) => (m.toolResults ?? []).map((r) => r.text));
    expect(texts).toContain('Cached sentence.');
  });
});

describe('the session model as summarizer', () => {
  const input: SummarizeInput = {
    id: 't1',
    tool_use_id: 'tool-1',
    tool: 'Read',
    input: { file_path: 'src/a.ts' },
    resultText: fileA,
    isError: false,
    goal: 'Fix the test',
  };

  it('forkSummarizer asks the session model and returns the trimmed reply', async () => {
    const prompts: string[] = [];
    const summarizer = forkSummarizer(async (request) => {
      prompts.push(request.prompt);
      return { text: '  One sentence.  ' };
    }, 6000);
    const summary = await summarizer.summarize(input);
    expect(summary).toBe('One sentence.');
    expect(prompts[0]).toContain('Side question from the context-compaction plugin');
    expect(prompts[0]).toContain('In one sentence, summarize which key elements and values');
    expect(prompts[0]).toContain('Tool call Read(file_path=src/a.ts)');
    expect(prompts[0]).toContain('Tool result (1000 chars)');
    expect(prompts[0]).toContain('Task context: Fix the test');
    expect(prompts[0]).toContain(fileA.slice(0, 40));
    expect(prompts[0]).toContain('Answer with that one sentence only.');
  });

  it('puts the tool result in front of the model even when the fork snapshot is stale', async () => {
    const prompts: string[] = [];
    const summarizer = forkSummarizer(async (request) => {
      prompts.push(request.prompt);
      return { text: 'Sentence.' };
    }, 6000);
    await summarizer.summarize({ ...input, resultText: 'x'.repeat(10_000) });
    // capped at maxPromptResultChars, but the length note tells the whole size
    expect(prompts[0]).toContain('Tool result (10000 chars)');
    expect(prompts[0]!.length).toBeLessThan(10_000);
  });

  it('fails loudly when the fork has nothing to say', async () => {
    const summarizer = forkSummarizer(async () => null, 6000);
    await expect(summarizer.summarize(input)).rejects.toThrow(/no summary/);
    const blank = forkSummarizer(async () => ({ text: '   ' }), 6000);
    await expect(blank.summarize(input)).rejects.toThrow(/no summary/);
  });

  it('buildSummarizer wires $.model.fork', async () => {
    const calls: Array<{ prompt: string }> = [];
    const summarizer = buildSummarizer(
      {
        model: {
          fork: async (request) => {
            calls.push(request);
            return { text: 'Via fork.' };
          },
        },
        ui: { log: () => {}, toast: () => {} },
        session: {
          usage: async () => ({ context: {} }),
          compact: async () => ({}),
          messages: async () => [],
        },
      },
      resolveHookConfig({ maxPromptResultChars: 500 }),
    );
    expect(await summarizer.summarize(input)).toBe('Via fork.');
    expect(calls[0]!.prompt).toContain('In one sentence, summarize');
  });
});

describe('batch question and reply', () => {
  const inputs: SummarizeInput[] = [
    {
      id: 't1',
      tool_use_id: 'tool-1',
      tool: 'Read',
      input: { file_path: 'src/a.ts' },
      resultText: fileA,
      isError: false,
      goal: 'Fix the test',
    },
    {
      id: 't2',
      tool_use_id: 'tool-2',
      tool: 'Bash',
      input: { command: 'npm test' },
      resultText: 'FAIL b.test.ts',
      isError: true,
      goal: 'Fix the test',
    },
  ];

  it('buildBatchPrompt lists every call with its result and the strict reply format', () => {
    const prompt = buildBatchPrompt(inputs, 6000);
    expect(prompt).toContain('Side question from the context-compaction plugin');
    expect(prompt).toContain('2 tool call(s) to summarize');
    expect(prompt).toContain('--- tool_use_id=tool-1');
    expect(prompt).toContain('--- tool_use_id=tool-2');
    expect(prompt).toContain('Tool result (14 chars, tool reported an error)');
    expect(prompt).toContain('`tool_use_id: sentence`');
  });

  it('parseBatchReply pulls sentences and ignores junk', () => {
    const parsed = parseBatchReply(
      [
        'Here are the summaries:',
        '1. tool-1: The file defines the parser and it is the one under test.',
        '- `tool-2` — the test fails on line 3, so the fixture is wrong.',
        'tool-9: not one of our calls.',
        'tool-1: second line for the same call is ignored',
      ].join('\n'),
      ['tool-1', 'tool-2'],
    );
    expect(parsed.get('tool-1')).toBe('The file defines the parser and it is the one under test.');
    expect(parsed.get('tool-2')).toBe('the test fails on line 3, so the fixture is wrong.');
    expect(parsed.has('tool-9')).toBe(false);
    expect(parsed.size).toBe(2);
  });

  it('flushPending puts batch answers into the cache and reports failures', async () => {
    const cache = new Map<string, string>();
    const errors: string[] = [];
    const summarizer: Summarizer = {
      async summarize() {
        throw new Error('not used');
      },
      async summarizeBatch(items) {
        if (items[0]!.tool_use_id === 'tool-1') throw new Error('fork died');
        return new Map(items.map((i) => [i.tool_use_id, `  ${i.tool_use_id} sentence.  `]));
      },
    };
    await flushPending(inputs, summarizer, cache, 1, 2, (error) => errors.push(error.message));
    expect(cache.get('tool-2')).toBe('tool-2 sentence.');
    expect(cache.has('tool-1')).toBe(false);
    expect(errors).toEqual(['fork died']);
  });

  it('flushPending falls back to single questions without a batch method', async () => {
    const cache = new Map<string, string>();
    const summarizer: Summarizer = {
      async summarize(input) {
        return `${input.tool_use_id} sentence.`;
      },
    };
    await flushPending(inputs, summarizer, cache, 10, 2);
    expect(cache.get('tool-1')).toBe('tool-1 sentence.');
    expect(cache.get('tool-2')).toBe('tool-2 sentence.');
  });
});

describe('inputFromEvent', () => {
  it('strips the envelope keys and keeps the arguments', () => {
    expect(
      inputFromEvent({ tool: 'Bash', tool_use_id: 't', agentId: 'a', command: 'npm test', timeout: 5 }),
    ).toEqual({ command: 'npm test', timeout: 5 });
  });
});

describe('collectToolCalls', () => {
  it('pairs calls with results and marks pinned ones', () => {
    const calls = collectToolCalls(transcript(), 6);
    expect(calls.map((c) => c.id)).toEqual(['t1', 't2']);
    expect(calls[0]!.resultChars).toBe(fileA.length);
    expect(calls.every((c) => !c.pinned)).toBe(true);
    expect(collectToolCalls(transcript(), 20).every((c) => c.pinned)).toBe(true);
  });
});
