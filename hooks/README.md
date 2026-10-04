# fast-llm-compaction Claude Code mod

`hooks/fast-llm.ts` is a thin adapter around the `fast-llm-compaction` library
in `src/` (the plugin folder is the repository root). It registers three
function hooks:

- `session.compact` — hands the transcript to `compact()`, which replaces every
  unpinned tool result over `minResultChars` characters with one sentence from
  the summarizer, and maps the result back onto `SessionMessage`s (unchanged
  messages keep their engine handles, rebuilt ones come back fresh). When the
  reduction is below `minReductionRatio` or the summarizer fails outright it
  calls `next(event)` and Claude Code's built-in summary runs instead.
- `tool.call` — pre-summarization: after `next(event)` resolves, a result over
  `minResultChars` characters is summarized in the background and cached per
  `tool_use_id`, so `session.compact` reuses ready sentences. Never blocks or
  disturbs the tool call itself.
- `turn.complete` — at `compactAtPercent` context usage, requests a compaction
  through `$.session.compact()`.

The summarizer is either the engine's own model provider
(`$.model.complete`, provider `engine`, default) or an OpenAI-compatible
`/chat/completions` endpoint (provider `http`; `baseUrl`/`apiKey` from
userConfig, else `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`).

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
