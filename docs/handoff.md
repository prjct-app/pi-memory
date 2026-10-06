# Model-switch context handoff

Pi 0.85.1 `context` can replace `AgentMessage[]` before each LLM call. `model_select` fires on set/cycle/restore. This module does **not** start the memory daemon or run hidden analysis. It never cancels compaction: automatic, overflow and manual `/compact` all belong to Pi.

Provider prompt cache (Anthropic/OpenAI cache keys) is **not** pi-memory. Handoff only rewrites the messages Pi sends; it does not claim to flush provider caches. Cache notices remain ordinary retained messages when their complete turn fits.

## Behavior

Every `context` event is budgeted, including before the first model switch. A never-initialized checkout uses a transient window without registry/database writes. Durable memory supplies an optional checkpoint, not permission to optimize context. A present binding is validated; after successful authority opening, a missing or changed binding returns the safe refusal. A pending open is not proof of prior ownership. Concurrent context events await activation and stale session completions cannot activate a replacement session.

The session-local window preserves message bytes while they fit. Once full, it evicts whole older turns toward 75% of the limits to leave growth room. If one agent turn grows across many tool rounds, it keeps the operator request and newest complete tool rounds that fit instead of aborting merely because earlier rounds accumulated. Tool calls and results remain atomic. A watermark prevents discarded whole turns from reappearing when overhead decreases; host history replacement resets that watermark. Identical recall blocks keep their first retained copy, preserving the cacheable prefix, while changed recall is appended. Context dedupe is based on retained messages, not a session-global delivery flag. No provider cache key, retention setting or routing identifier is changed.

These are cache-friendly inputs, not guaranteed cache hits: provider/model changes, cache expiry, tool/system changes and necessary eviction can still incur input charges. System/tool overhead is not removed. If the newest complete tool round exceeds the hard budget, oversized strings are truncated in place before anything is refused: first tool-result/custom/bash output text, then image blocks (replaced by a text note), then assistant text and tool-call string arguments. One shared per-field cap (minimum 512 chars) is binary-searched so the largest strings are cut first and the pack uses the available room; each cut keeps the head and tail with an explicit omission marker. No message is removed, so call/result pairs stay atomic, and signed thinking blocks are never modified. The operator request, explicit checkpoint and Pi summaries are never clipped. Continuation refuses only if overhead, the request, or the message count still cannot fit after truncation; it never purchases a summary. A truncated pack is not further compacted toward 75%. The host session transcript is not deleted or summarized by local selection.

The bounded context contains one continuity prefix followed by the newest complete turns that fit:

1. A newer explicit `/memory checkpoint` wins.
2. Otherwise the latest Pi `compactionSummary` or `branchSummary` message is the deterministic fallback.
3. The current user request and newest complete tool round are mandatory; additional current-turn tool rounds and older complete turns are added newest-first.

The newest user requirement, including qualifications and evidence references in that request or checkpoint, is therefore retained. Pi 0.85.1 tool calls are assistant `content` blocks whose type is `toolCall`. Calls and `toolResult` messages must form a one-to-one set; missing, orphaned, or duplicate results refuse the handoff. Multi-call loops are kept whole even when results arrive in a different order.

If the continuity prefix, current request, newest complete tool round (after truncation), and provider overhead cannot fit, the handler aborts, notifies, and returns a known safe instruction only if that instruction also fits. Otherwise it returns no messages. Thrown errors are caught because Pi swallows `context` exceptions (fail-open). There is **no** documented cancel-return that proves the HTTP request was prevented; `ctx.abort()` is best-effort. If Pi continues transport, the original history is never eligible. Fixed system/tool overhead can itself exceed the budget and cannot be removed safely by this message handler: this is not a guaranteed total-cost or transport-cancellation boundary.

## Observation masking

Old tool outputs were 70-86% of every request in real sessions and were re-sent on each of ~35 LLM calls per prompt. Before selection, the window replaces tool results that are older than the newest `keepRounds` tool rounds and larger than `minTokens` with a short stub: `read` names the path and range and says to read again, `bash` keeps the command and its last 15 lines (failures print last), `grep`/`find`/`ls` keep the arguments and first 10 lines. Tool calls, call/result pairing, user and assistant text, and signed thinking are untouched.

Masking is a pure function of the host history and a session-local frontier. The frontier only advances when stale unmasked output behind the keep window reaches `advanceTokens`, then masks everything up to the keep boundary at once, so the serialized prefix is identical between advances. It resets with the watermark when history is replaced. When either masking or history retirement advances, the other flushes its pending work in the same request (masking down to the keep boundary, retirement down to the target without the batch margin), so one prefix rewrite pays for both instead of two full re-bills a few calls apart. Defaults are `keepRounds` 8, `minTokens` 300, and `advanceTokens` 24,000: batching amortizes prefix rewrites instead of repeatedly invalidating the prefix for one stale output. This is not a provider retention guarantee. `MemoryExtensionOptions.observations` overrides them; `{ enabled: false }` turns masking off.

Only the fixed memory policy enters the system prompt (L0). Digest snapshots and overflow recall enter the message tail (L1). Retained-context deduplication keeps the first available copy of an unchanged snapshot and appends distinct recall without repeating the digest. New revisions, including empty snapshots, supersede earlier automatic memory; A → B → A delivers A again after B. Evicted material can be delivered again. `memory_context` provides pull-on-demand detail (L2).

When the window preserves every host message's identity and order, the `context` handler returns `{}` without a `messages` property to avoid unnecessary replacement. Object identity alone does not affect serialized provider bytes. Masking, eviction, truncation, synthetic checkpoints, and recall deduplication that drops a message still return replacement messages. Batched masking, retained-context deduplication and a separate soft history target reduce retained input without lowering the hard model-window budget. Cold requests can still rebill retained history, system prompt and tools; a six-minute gap does not establish a universal provider cache TTL.

## Output caps

`bash` output above 16,000 characters keeps its first 4,000 and last 12,000 characters; `grep`/`find` output above 12,000 characters keeps its first matches. The full text is written first (Pi's own `fullOutputPath` when present, otherwise `$TMPDIR/pi-memory-tool-output/<toolCallId>.txt`) and the result names that file. If the write fails the output is left whole. `read` is never capped. `MemoryExtensionOptions.outputCaps` configures or disables this.

## Budgets

`MemoryExtensionOptions.history` sets a soft completed-history target: `targetTokens` 32,000 and `advanceTokens` 24,000 by default. Once historical input reaches 56,000 estimated tokens, eligible completed interactions retire in a batch toward the target. When the original invocation is backed by the active session branch, a scoped `sr1:` reference replaces its arguments and results; `memory_context` with `action: "inspect"` retrieves the redacted data without rerunning actions. Read-only interactions without session backing keep re-fetch pointers. Operator messages, conclusions, checkpoints, media, unbacked side-effect or unknown interactions, and the entire current turn remain. Those protected items can exceed the target. `{ enabled: false }` disables this retirement policy.

Unless `MemoryExtensionOptions.handoff` is set, the ceiling is derived on every request from the active model: `contextWindow` minus a response reserve (the model's `maxTokens`, clamped to 16,384–32,768 and at most a quarter of the window), 16 bytes per budgeted token, and 4,096 messages. For a 272k model that is 239,232 estimated tokens. Only when the model exposes no usable window does the conservative fallback of 16,000 tokens, 262,144 bytes, and 48 messages apply. A fixed 16k ceiling applied to every request previously aborted or truncated ordinary document reads, so it is no longer the default. Every ceiling includes measured system/tool overhead. `MemoryExtensionOptions.handoff` configures an explicit ceiling:

- `maxTokens`: system prompt + explicit tool-schema reserve + selected messages
- `maxBytes`: measured system prompt + active tool definitions + serialized selected messages, excluding host-only `details` that Pi never sends to the provider
- `maxMessages`: selected message count
- `toolSchemaReserveTokens`: fallback reserve when active tool definitions are unavailable

The effective system prompt is measured on every request through public `ctx.getSystemPrompt()`. Active definitions from `getAllTools()`/`getActiveTools()` replace the default 1,500-token fallback reserve. Token counts are estimates, not provider billing telemetry; serialization by the provider or later extensions can change actual request size. Diagnostics itemize system tokens/bytes, tool reserve, message tokens/bytes, and totals. If overhead alone consumes the budget, handoff refuses and returns no messages.

## Operational checkpoints

`/memory checkpoint {json}` transactionally writes a maximum 4,000-byte operational record into the `operational_checkpoints` table of the **current project's** `memory.sqlite`. The row is keyed by project and session.

Operational checkpoints never enter documents, chunks, FTS, vectors, the semantic journal, curation, default or all-namespace recall, evaluation evidence, or semantic item counts. Opening a current database remains read-only at schema/ownership validation; the table is created only by normal fresh-schema creation or an explicit schema upgrade.
