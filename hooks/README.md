# fast-llm-compaction Claude Code mod

`hooks/fast-llm.ts` is a thin adapter around the `fast-llm-compaction` library
in `src/` (the plugin folder is the repository root). It registers three
function hooks:

- `session.compact` — hands the transcript to `compact()`, which replaces every
  unpinned tool result over `minResultChars` characters with one sentence from
  the session model, and maps the result back onto `SessionMessage`s (unchanged
  messages keep their engine handles, rebuilt ones come back fresh). When the
  reduction is below `minReductionRatio` or the model fails outright it calls
  `next(event)` and Claude Code's built-in summary runs instead.
- `tool.call` — asking as work happens: after `next(event)` resolves, a result
  over `minResultChars` characters is put to the session model in the
  background through `$.model.fork` (same model, same transcript, shared prompt
  cache), and the sentence is cached per `tool_use_id`, so `session.compact`
  reuses ready answers. Never blocks or disturbs the tool call itself.
- `turn.complete` — at `compactAtPercent` context usage, requests a compaction
  through `$.session.compact()`.

The question is transient: `$.model.fork` completes over the session's
transcript without putting the question or the reply into the visible
conversation, so nothing has to be purged later. Only the agent's sentence is
kept, and it lands where the tool result was.

User and assistant text is never touched; tool calls always survive, condensed
when long. The plugin toast reads
`fast-llm-compaction: kept N/M messages, tool results as one-liners (…)` when
the rebuilt history replaced the summary, or `fallback to built-in summary (…)`.

## Install

```json
{ "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1" } }
```

```bash
claude plugin marketplace add murich/fast-llm-compaction
claude plugin install fast-llm-compaction@fast-llm-compaction
```

See the root [README](../README.md) for the options table and the library API.
