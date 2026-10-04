import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLogLines,
  engineSummarizer,
  getHttpConfig,
  inputFromEvent,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-llm.ts';
import {
  collectToolCalls,
  httpSummarizer,
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
      provider: 'engine',
      model: 'haiku',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      preSummarize: true,
      preSummarizeAtPercent: 0,
      concurrency: 4,
    });
    expect(
      resolveHookConfig({
        provider: 'http',
        model: 'llm-x',
        minResultChars: 500,
        preSummarize: false,
        concurrency: 2,
        baseUrl: 'https://llm.example.com/v1',
        apiKey: 'k',
        goal: 'g',
        compactAtPercent: 'no',
      }),
    ).toEqual({
      provider: 'http',
      model: 'llm-x',
      minResultChars: 500,
      preSummarize: false,
      preSummarizeAtPercent: 0,
      concurrency: 2,
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      baseUrl: 'https://llm.example.com/v1',
      apiKey: 'k',
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

describe('providers', () => {
  it('engineSummarizer asks the session model for one sentence', async () => {
    const requests: Array<{ model: string; prompt: string; system?: string }> = [];
    const summarizer = engineSummarizer(async (request) => {
      requests.push(request);
      return '  One sentence.  ';
    }, 'haiku', 6000);
    const summary = await summarizer.summarize({
      id: 't1',
      tool_use_id: 'tool-1',
      tool: 'Read',
      input: { file_path: 'src/a.ts' },
      resultText: fileA,
      isError: false,
      goal: 'Fix the test',
    });
    expect(summary).toBe('One sentence.');
    expect(requests[0]!.model).toBe('haiku');
    expect(requests[0]!.prompt).toContain('In one sentence, summarize which key elements and values');
    expect(requests[0]!.prompt).toContain('Tool call: Read(file_path=src/a.ts)');
  });

  it('httpSummarizer posts to /chat/completions and reads the reply', async () => {
    const calls: Array<{ url: string; body: string; headers: Record<string, string> }> = [];
    const summarizer = httpSummarizer({
      baseUrl: 'https://llm.example.com/v1/',
      apiKey: 'secret',
      model: 'llm-x',
      fetch: async (url, init) => {
        calls.push({
          url,
          body: init?.body ?? '',
          headers: init?.headers ?? {},
        });
        return {
          status: 200,
          ok: true,
          text: JSON.stringify({ choices: [{ message: { content: 'From http.' } }] }),
        };
      },
    });
    const summary = await summarizer.summarize({
      id: 't1',
      tool_use_id: 'tool-1',
      tool: 'Read',
      input: {},
      resultText: 'a'.repeat(10_000),
      isError: false,
      goal: '',
    });
    expect(summary).toBe('From http.');
    expect(calls[0]!.url).toBe('https://llm.example.com/v1/chat/completions');
    expect(calls[0]!.headers.authorization).toBe('Bearer secret');
    expect(JSON.parse(calls[0]!.body).model).toBe('llm-x');
  });

  it('httpSummarizer fails loudly on an error status', async () => {
    const summarizer = httpSummarizer({
      baseUrl: 'https://llm.example.com/v1',
      model: 'llm-x',
      fetch: async () => ({ status: 403, ok: false, text: '<html>blocked</html>' }),
    });
    await expect(
      summarizer.summarize({
        id: 't1',
        tool_use_id: 'tool-1',
        tool: 'Read',
        input: {},
        resultText: 'text',
        isError: false,
        goal: '',
      }),
    ).rejects.toThrow(/HTTP 403/);
  });

  it('getHttpConfig reads config first, then env, then settings', async () => {
    const $ = {
      env: { get: async (name: string) => (name === 'LLM_MODEL' ? 'env-model' : undefined) },
      settings: {
        read: async () => ({ env: { LLM_BASE_URL: 'https://from-settings/v1' } }),
      },
    };
    const config = resolveHookConfig({ provider: 'http' });
    expect(await getHttpConfig($, config)).toEqual({
      baseUrl: 'https://from-settings/v1',
      model: 'env-model',
    });
    expect(
      await getHttpConfig($, {
        ...config,
        baseUrl: 'https://explicit/v1',
        model: 'explicit-model',
        apiKey: 'k',
      }),
    ).toEqual({ baseUrl: 'https://explicit/v1', model: 'explicit-model', apiKey: 'k' });
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
