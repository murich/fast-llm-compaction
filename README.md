# fast-llm-compaction

Claude Code plugin that replaces the compaction summary with one-sentence
summaries of tool results. Instead of asking the model to rewrite the session
into a paraphrase (or asking a decision model which calls to delete), every
tool result worth keeping becomes **one LLM-written sentence**: which key
elements and values were found, why they matter for the task, and what follows
from them. Everything else in the transcript stays word for word.

A fork of [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction)
with the Jev decision layer swapped for plain LLM summarization. Same seam,
same triggers, same fallbacks; no TypeSafe account, no third-party scoring API.

## What it does

- **`tool.call` (pre-summarization).** As tool calls land, any result over
  `minResultChars` (200) characters is sent to a cheap model with the
  summarization question. The sentence is cached per `tool_use_id`, so it is
  paid for once and compaction later is instant.
- **`session.compact` (the compaction).** When `/compact`, auto-compaction or
  the plugin's own trigger fires, the transcript is rebuilt:
  - the first message and the newest `preserveRecentMessages` (6) are pinned;
  - tool results at or below 200 characters stay verbatim;
  - longer tool results are replaced by their one sentence (from cache, or
    summarized on the spot);
  - long tool calls are condensed to their informative skeleton (short strings
    elided, bulky structures become size notes) — but **the call itself always
    survives**, so the assistant never narrates work whose record is gone;
  - user and assistant text is never touched, and no text is ever rewritten.
- **`turn.complete` (the trigger).** At `compactAtPercent` (60%) context usage
  the plugin requests a compaction, same as upstream.
- **Fallback.** If the estimated reduction is below `minReductionRatio` (25%),
  or the summarizer fails outright, Claude Code's built-in summary runs
  instead — the standard `/compact` path. You never end up with no compaction.

## The question

Each summarized tool call is asked this, with the task context, the call's
input and its result:

> In one sentence, summarize which key elements and values were found in the
> tool response, why these key elements are important for us, and which
> conclusion you make in the context of our task based on this response.

Only the answer survives into the compacted transcript; the question itself
never enters the context.

## Install

Function hooks are early-access, so the opt-in flag is required wherever
Claude Code runs — in `~/.claude/settings.json`:

```json
{ "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1" } }
```

Then:

```bash
claude plugin marketplace add murich/fast-llm-compaction
claude plugin install fast-llm-compaction@fast-llm-compaction
```

Restart Claude Code or run `/reload-plugins`. From a checkout without
installing: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`

## Summarizer providers

| Provider | How it runs | What you need |
|---|---|---|
| `engine` (default) | `$.model.complete` on the session's own model provider, `haiku` by default | nothing — no second API key |
| `http` | any OpenAI-compatible `/chat/completions` endpoint (local model, gateway, hosted) | `baseUrl` + optional `apiKey`, or `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` |

For `http`, the endpoint and key come from the plugin's `userConfig` first,
then the environment, then `settings.json`'s `env` block.

## Options

| Option | Default | What it does |
|---|---|---|
| `provider` | `engine` | `engine` or `http` summarizer |
| `model` | `haiku` | Summarizer model (`LLM_MODEL` overrides the default for `http`) |
| `baseUrl` / `apiKey` | — | OpenAI-compatible endpoint and key (`http` only) |
| `minResultChars` | `200` | Results at or below this stay verbatim |
| `maxToolInputChars` | `200` | Serialized tool inputs above this are condensed |
| `maxPromptResultChars` | `6000` | Result characters shown to the summarizer |
| `truncateHeadChars` | `300` | Head kept when summarization fails |
| `preserveRecentMessages` | `6` | Newest messages pinned |
| `compactAtPercent` | `60` | Context percentage that triggers compaction |
| `minReductionRatio` | `0.25` | Below this reduction the built-in summary runs |
| `preSummarize` | `true` | Summarize results as they land |
| `preSummarizeAtPercent` | `0` | Context percentage from which to pre-summarize |
| `concurrency` | `4` | Summarizer calls in flight at once |

## As a library

```ts
import { summarizeMessages, httpSummarizer, reductionRatio, type Message } from 'fast-llm-compaction';

const summarizer = httpSummarizer({
  baseUrl: 'https://llm.example.com/v1',
  model: 'llm-x',
  fetch: globalThis.fetch as never,
});

const result = await summarizeMessages(transcript, summarizer, { cache: new Map() });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`, so a session JSONL
goes in as is. To bring your own transport or model, implement `Summarizer`
(one `summarize(input)` method returning a sentence). The building blocks
(`collectToolCalls`, `condenseInput`, `applyDecisions`) are exported
individually.

## Development

```bash
npm install
npm test          # vitest, fake summarizers, no network
npm run typecheck # library + hooks against types/claude-code.d.ts
npm run build
```

## Differences from upstream

| | fast-jev-compaction | fast-llm-compaction |
|---|---|---|
| Decision maker | TypeSafe Jev (cloud, per-call keep/drop probabilities) | any LLM, one sentence per result |
| What happens to a result | kept verbatim, truncated to 300 chars, or deleted with its call | replaced by its one-sentence summary |
| Tool calls | can be deleted wholesale | always kept, condensed when long |
| Requires | `TYPESAFE_API_KEY` | nothing (`engine`) or your own endpoint (`http`) |
| Pre-summarization | no | yes, on `tool.call`, cached per call |
| Fallback | built-in summary | built-in summary, same thresholds |

Deleting calls wholesale (upstream's `drop_call`) is what leaves the assistant
narrating work it can no longer see. Keeping every call — just smaller — is
deliberate.

## Limitations

- A one-sentence summary is lossy by design. It keeps conclusions and key
  values, not full contents; re-run the tool if the details are needed again.
- The pre-summarization cache is per process. A summary made early reflects
  the task context known at that moment; set `preSummarizeAtPercent` if you
  only want summaries once the session is under way.
- Each summarized result costs one small-model call. Results under 200
  characters cost nothing.
- `--resume` and compaction-boundary behavior are Claude Code function-hook
  territory and identical to upstream's — nothing here fixes what the engine
  does not expose.
- With `provider: engine` the summarizer bills the session's own account.
  With `provider: http` your tool results go to the endpoint you configure.

## Credits

Built on [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction)
(MIT). The pairing, pinning, rebuild and fallback design is theirs; the
decision layer is this fork's.

## License

MIT
