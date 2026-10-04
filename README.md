# fast-llm-compaction

Claude Code plugin that replaces the compaction summary with one-sentence
summaries of tool results — written by **the session's own model**, the one
that did the work. Instead of asking the model to rewrite the whole session
into a paraphrase (or asking a decision model which calls to delete), every
tool result worth keeping becomes **one sentence**: which key elements and
values were found, why they matter for the task, and what follows from them.
Everything else in the transcript stays word for word.

A fork of [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction)
with the Jev decision layer swapped for self-summarization. Same seam, same
triggers, same fallbacks; no TypeSafe account, no third-party model — only the
session's own model and context.

## Why the session model and not a cheap one

Only the agent knows what a tool result means *for the task*. A small
sidecar model can say what the text contains; it cannot say which values
matter or what conclusion follows. So the question is put to the session's
model through `$.model.fork`: one completion over the session's own
transcript, with the session's own model and system prompt, sharing the main
thread's prompt cache. The agent is simply forced to verbalize what it found.

## What it does

- **`tool.call` (asking as work happens).** As tool calls land, any result over
  `minResultChars` (200) characters joins the pending series; the series goes
  to the session model as **one question per `batchSize` (10) calls**, flushed
  when the batch fills or the turn ends. Sentences are cached per
  `tool_use_id`, so they are asked once and compaction later is instant. Each
  tool result is put in front of the model with the question, so a stale fork
  snapshot cannot blind it.
- **`session.compact` (the compaction).** When `/compact`, auto-compaction or
  the plugin's own trigger fires, the transcript is rebuilt:
  - the first message and the newest `preserveRecentMessages` (6) are pinned;
  - tool results at or below 200 characters stay verbatim;
  - longer tool results are replaced by their one sentence (from cache, or
    asked on the spot — batched the same way, one question per series);
  - long tool calls are condensed to their informative skeleton (short strings
    elided, bulky structures become size notes) — but **the call itself always
    survives**, so the assistant never narrates work whose record is gone;
  - user and assistant text is never touched, and no text is ever rewritten.
- **`turn.complete` (the trigger).** At `compactAtPercent` (60%) context usage
  the plugin requests a compaction, same as upstream.
- **Fallback.** If the estimated reduction is below `minReductionRatio` (25%),
  or the model fails outright, Claude Code's built-in summary runs instead —
  the standard `/compact` path. You never end up with no compaction.

## The question

Each summarized tool call is put to the session model like this (the exact
question the plugin asks):

> Side question from the context-compaction plugin. Do not continue the task,
> do not use tools, and do not comment on this question; answer it directly.
>
> Task context: …
>
> Tool call Read(file_path=src/a.ts). Tool result (1000 chars): …
>
> In one sentence, summarize which key elements and values were found in the
> tool response, why these key elements are important for us, and which
> conclusion you make in the context of our task based on this response.
> Answer with that one sentence only.

The fork question is transient — it is not a message in the visible
conversation, so there is nothing to purge later. Only the agent's answer is
kept, and it is kept where the tool result was.

For a series of calls the same question is asked once for the whole series,
with a strict reply format (`tool_use_id: sentence` per line). Batching is
what makes this cheap: the context behind the question is paid **once per
series**, not once per tool call — a completion forks over the session's
transcript, so without batching every question would re-read the whole
context.

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

## Options

| Option | Default | What it does |
|---|---|---|
| `minResultChars` | `200` | Results at or below this stay verbatim |
| `maxToolInputChars` | `200` | Serialized tool inputs above this are condensed |
| `maxPromptResultChars` | `6000` | Result characters put in front of the model with the question |
| `truncateHeadChars` | `300` | Head kept when no summary could be obtained |
| `preserveRecentMessages` | `6` | Newest messages pinned |
| `compactAtPercent` | `60` | Context percentage that triggers compaction |
| `minReductionRatio` | `0.25` | Below this reduction the built-in summary runs |
| `preSummarize` | `true` | Ask as tool results land |
| `preSummarizeAtPercent` | `0` | Context percentage from which to ask |
| `concurrency` | `4` | Questions in flight at once |
| `batchSize` | `10` | Tool calls per batch question |

## As a library

```ts
import { summarizeMessages, forkSummarizer, reductionRatio, type Message } from 'fast-llm-compaction';

const summarizer = forkSummarizer($.model.fork, 6000);
const result = await summarizeMessages(transcript, summarizer, { cache: new Map() });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`, so a session JSONL
goes in as is. To bring your own transport or model, implement `Summarizer`
(one `summarize(input)` method returning a sentence). The building blocks
(`collectToolCalls`, `condenseInput`, `applyDecisions`, `buildSummarizePrompt`)
are exported individually.

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
| Decision maker | TypeSafe Jev (cloud, per-call keep/drop probabilities) | the session's own model, one sentence per result |
| What happens to a result | kept verbatim, truncated to 300 chars, or deleted with its call | replaced by the agent's own one-sentence summary |
| Tool calls | can be deleted wholesale | always kept, condensed when long |
| Requires | `TYPESAFE_API_KEY` | nothing |
| When the decision is made | at compaction time | as the work happens (and on demand at compaction) |
| Fallback | built-in summary | built-in summary, same thresholds |

Deleting calls wholesale (upstream's `drop_call`) is what leaves the assistant
narrating work it can no longer see. Keeping every call — just smaller — is
deliberate.

## Limitations

- A one-sentence summary is lossy by design. It keeps conclusions and key
  values, not full contents; re-run the tool if the details are needed again.
- Each question is one completion over the session's cached transcript
  (cheap in cache-read tokens, but not free). Batching keeps it to one
  completion per `batchSize` calls; set `preSummarizeAtPercent` to ask only
  once the session is under way, or `preSummarize: false` to ask only at
  compaction time.
- A fork is null on a cold transcript (before the first turn) and on API
  errors; those calls fall back to a bounded truncated head.
- The one-sentence answer reflects the task understanding at the moment it
  was asked; a mid-session pivot can make earlier sentences stale.
- `--resume` and compaction-boundary behavior are Claude Code function-hook
  territory and identical to upstream's — nothing here fixes what the engine
  does not expose.

## Credits

Built on [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction)
(MIT). The pairing, pinning, rebuild and fallback design is theirs; the
decision layer is this fork's.

## License

MIT
