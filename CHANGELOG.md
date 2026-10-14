## Unreleased

- Breaking: pi-memory is now session-history memory. It indexes Pi's session journals (contentless SQLite FTS5 plus byte pointers; the journals stay the only source of truth) and offers one on-demand tool, `memory_context`: `search` (words, local-day or ISO `since`/`until`, `project`/`session`/`global` scope, kind), `read` (an entry by id, paged) and `summaries` (past compaction summaries).
- Removed: curated facts and `memory_record`, the curation pipeline, the daemon, retention and sweep, the handoff and observation masking, embeddings and sqlite-vec, the child view and the eval scripts. Old databases under `~/.prjct/pi-memory` are left untouched.
- `/memory gc` lists junk journals (temporary folders, no instruction at all, two replies or fewer; never compacted; quiet 24 h); `/memory gc confirm` deletes exactly those, with their index rows.
- Why: sessions run for weeks and compact up to 23 times; a rule stated once was gone from the summary by compaction 21. The journal keeps everything, so the agent needs a way to find it, not a second store of curated facts (85% of lookups came back empty, 46% of writes failed).
- Measured on 2.37 GB of real journals: 102 s first pass, 270 MB index, search p50 21 ms; 21 of 26 real `context_history` searches fully reproduced and none missed.

## 0.3.1 (2026-10-12)

- Team discovery reads team markers from `${PRJCT_HOME:-~/.prjct}/teams`, where pi-team writes them, instead of Pi's own folder.
- Tests run with temporary Pi and prjct folders.

## 0.3.0 (2026-10-09)

- Memory is on demand. Before a turn nothing is injected: no snapshot, no recall, and no restore after compaction or a tree move. The agent reads memory with `memory_context`, whose description now says what memory holds and when to look it up.
- Failed tool results are no longer tagged with an evidence handle. In 94 recent `memory_record` calls, 2 cited one, while the tag was added to 994 failed results.
- Remove the automatic recall search and its encoder warm-up from the prompt path.
- Remove the link to pi-context-prune (`/memory prune`, the panel's prune key and context line): with nothing injected, there is no superseded memory to prune.

## 0.2.3 (2026-10-09)

- Send the memory snapshot only when the retained context lacks it: an unchanged snapshot is no longer repeated every turn, a new recall rides alone, and an empty memory with nothing to revoke sends nothing.
- Never return `systemPrompt` from `before_agent_start`. Pi treats any returned prompt, even an unchanged one, as a forced prompt for that turn only.

## 0.2.2 (2026-10-07)

## 0.2.0 (2026-10-06)

- Remove semantic reranking, classifier curation, turn judging and credential setup.
- Preserve default conversation history on memory failures; keep memory ownership checks independent of conversation retention.
- Preserve source language and remove English preference in duplicate retention.
- Isolate tests from personal Pi/prjct state.

## 0.1.5 (2026-10-06)

- Protect direct SDK and optional Jev requests with the published pi-secrets outbound guard.

# Changelog

## 0.1.4 — 2026-10-06

- Preserve long histories and original-language instructions. Disable implicit classifier activation, evidence removal and session-close curation. Update Transformers to 4.3.0.
