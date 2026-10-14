# pi-memory

[![pi-memory — for PI Agent](https://raw.githubusercontent.com/prjct-app/pi-memory/main/docs/cover.png)](https://pi.dev)

Session-history memory for Pi. Long sessions compact many times, and every compaction replaces older turns with a summary. Pi still keeps every entry in the session journal. pi-memory indexes those journals, so the agent can find what was compacted away, or what happened in an earlier session of the same project, and read it verbatim.

- **On demand.** One tool, `memory_context`. Nothing is injected: no recall, no rewritten context, no retired turns. The model calls it when it decides it needs something from the past.
- **The journal is the source of truth.** The index stores search terms (SQLite FTS5, contentless) and a byte pointer into the journal line. Text is always read from the journal and redacted on the way out. Delete the index and nothing is lost: `/memory rebuild` recreates it.
- **No model calls, no daemon, no embeddings.** Indexing is incremental and mechanical.

## Install

```sh
pi install npm:@prjct.app/pi-memory
```

Requires Pi and Node.js 22.19 or later (`node:sqlite` with FTS5). No runtime dependencies.

## `memory_context`

| Action | Use it to | Parameters |
| --- | --- | --- |
| `search` | find entries by words, newest-relevant first | `query` (words in any order; empty lists entries newest first), `since` / `until` (`YYYY-MM-DD` = your local day, or ISO), `scope`, `kind`, `offset` |
| `read` | get the original text of one entry | `entryId`, optional `sessionId` when an id exists in several sessions, `offset` for 16,000-character pages |
| `summaries` | list past compaction and branch summaries by date | `since` / `until`, `scope`, `sessionId`, `offset` |

`scope` is `project` (default), `session` or `global`. A project is the outermost folder above the cwd that has sessions of its own, below a direct child of home: from `~/Apps/prjct/prjct-be` it covers `prjct-be`, `prjct-fe`, `prjct-landing` and `~/Apps/prjct`, but not `~/Apps/fluenty`. `kind` is `user`, `assistant`, `tool_result`, `bash`, `custom` (team and extension messages), `compaction` or `branch_summary`.

Search terms are matched together first; when no entry has all of them, entries with any of them are returned and the result says `match: "any"`. Relevance is BM25, with newer entries ranked higher among equally relevant ones. Accents do not matter. FTS syntax in a query is treated as literal text.

Recipes:

- *What did the user say about deploys last week?* `search` with `query: "deploy"`, `kind: "user"`, `since: "2026-10-06"`.
- *What did the summary before the last one say?* `summaries`, then `read` with its `entryId`.
- *A compaction summary points at "session entry 7a1c03f2".* `read` with `entryId: "7a1c03f2"`.
- *Did another terminal of this project already hit this error?* `search` with the error text; `scope: "global"` to include every project.

Encrypted reasoning and images are not indexed. Secrets are redacted before terms are indexed and again when text is returned, so a key never becomes a search term.

## `/memory`

| Command | Does |
| --- | --- |
| `/memory` or `/memory status` | Sessions, entries and summaries indexed. |
| `/memory index` | Index every journal that grew since it was last seen. |
| `/memory rebuild` | Delete the index and build it again from the journals. |
| `/memory gc` | List junk journals: never compacted, quiet for 24 hours, and either run in a temporary folder (benchmarks, probes), received no instruction at all (no user, team or extension message), or got two replies or fewer. |
| `/memory gc confirm` | Delete for good exactly the journals the last `/memory gc` listed, if unchanged since, with their index rows. |

Indexing starts a few seconds after a session starts. One Pi process at a time scans other journals (a lease in the index); every process indexes its own journal as it grows. Measured on 2.37 GB of real journals (709 sessions): 102 s for the first pass, a 270 MB index, event-loop pauses up to 187 ms (p99 97 ms) during that first pass only, and search p50 21 ms.

The index lives in `${PI_MEMORY_HOME:-${PRJCT_HOME:-~/.prjct}}/pi-memory/history.sqlite`.

## Development

```sh
npm run check
npm test
npm run check:package
```
