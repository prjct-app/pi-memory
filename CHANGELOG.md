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
