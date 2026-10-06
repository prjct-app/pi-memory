# pi-memory

[![pi-memory — for PI Agent](https://raw.githubusercontent.com/prjct-app/pi-memory/main/docs/cover.png)](https://pi.dev)

Pi-native temporal memory and hybrid retrieval for agents. The extension supplies
durable evidence, indexing, retrieval, and bounded garbage collection.

**Architecture:** the Pi extension retrieves and records; a standalone daemon
(`npm run daemon -- once|start|stop|status`) analyzes changed sources while Pi is
closed. Default `/memory sync` fingerprints publishers and enqueues work — it does
not copy raw source bodies. Configure `PI_MEMORY_ANALYSIS_PROVIDER` and
`PI_MEMORY_ANALYSIS_MODEL`. Do not install a persistent service unless explicitly
authorized. See [the design](docs/curated-memory-plan.md).

## Install

Requires Pi, Node.js 22.19 or later, macOS or Linux.

```sh
pi install npm:@prjct.app/pi-memory
```

Memory belongs to a git repository. The first thing worth keeping — a
`memory_record` call or a "remember/recuerda…" declaration — initializes the
repository root automatically, wherever inside it Pi was started. Reading never
initializes: a lookup in a repository without memory abstains. Outside a git
repository nothing is created and the memory tools are hidden. `/memory init`
still binds a checkout explicitly.

`/memory status` and extension startup are read-only while a checkout is
uninitialized. Initialization writes a checksummed binding to pi-memory's own
registry and creates exactly one owner-bound project database. The default
local multilingual encoder is downloaded once into `<memory-home>/pi-memory/models`
and shared by every project, and only when memory outgrows the prompt block.

## Agent tools

- `memory_context` searches the active project with up to four agent-authored
  query expansions, inspects ids, proposes consolidation candidates, and records
  useful/wrong/stale feedback. It never opens a team, shared or other-project
  authority. `scopes: ['project']` is the only accepted scope selector.
- `memory_record` stores a selective temporal fact, appends a resolution instead
  of rewriting history, or indexes a generic source document.

The extension keeps a bounded, session-local staging window for host tool
results. A failed tool result carries its staged evidence handle so the agent
can cite it with `memory_record`; successful results are left untouched.
Exact, secret-free user declarations beginning with `remember`, `recuerda`, or
`acuérdate` are also stored directly as supported lexical procedures after the
turn, preserving their original wording and language; corrections
follow the same declared-evidence path. Memory's own tools
are excluded to prevent self-citation. The agent cannot mint native provenance.
An explicit user statement is accepted only when `userQuote` occurs verbatim in
the current prompt.

The agent should remember decisions, corrections, stable constraints,
preferences, verified failures, and reusable procedures—not routine reads,
progress narration, secrets, or generic summaries.

### Preserve the original language

User declarations, `memory_record` statements and rationales are stored in their
original language. Storage never depends on an auxiliary translation call.
Analysis and curation preserve meaning, qualifications and unresolved conflicts;
a different language alone is no reason to reject or rewrite a rule.

Evidence remains verbatim. A `userQuote` must occur in the current prompt, and
an excerpt must cite its source. These grounding checks apply in every language.
Memory identity remains keyed on what was actually said.

In a repository with memory, every request carries a `<project_memory>` block in
the system prompt: active memories ordered by kind (corrections, constraints,
preferences and decisions first), escaped as data, capped at 4KB and stable
between prompts, so it stays in the provider's cached prefix. The model itself
matches paraphrases and other languages, with no encoder loaded and nothing to
wait for. Agent-recorded memories without a user quote are marked
`(unconfirmed)`. Only when memory outgrows the block are the remaining memories
searched per prompt (the prompt and each of its sentences); their vectors are
built and the local encoder is loaded in the background, and a prompt never
waits for it. A dense-only match must then be close (cosine ≥ 0.6) and clearly
ahead of the other candidates (margin ≥ 0.2).

## Commands

```text
/memory init            # explicitly bind this checkout and create/adopt its authority
/memory setup           # set or rotate the optional TypeSafe evaluator key
/memory status          # read-only when this checkout has not been initialized
/memory sources          # counters, last run per adapter, queued jobs, and what is due
/memory sync [adapter]   # scan and enqueue now; does not copy raw source bodies
/memory replay
/memory rebuild
/memory gc
/memory migrate-curated  # checkpoint raw journal/projection and enqueue legacy documents
```

Interactive Pi shows a framed overlay (esc/enter to close) for every `/memory`
action. If the overlay is dismissed, the same card is notified — never a JSON
dump, never "Operation aborted". RPC and other non-TUI hosts use notify only.
`/memory sync` states scanned / new / queued rather than adapter payloads. The
last durable source error stays on `/memory status` until a later command succeeds.

Daemon (separate process, never started by the extension):

```sh
npm run daemon -- status
npm run daemon -- once --home "$PI_MEMORY_HOME" --provider anthropic --model claude-sonnet-4-5
npm run daemon -- start --home "$PI_MEMORY_HOME" --provider anthropic --model claude-sonnet-4-5
npm run daemon -- stop
```

## Optional semantic reranking

Memory works without a classifier. Reranking requires an explicit project
`config.json` setting and a TypeSafe key; a shared credential alone never enables
it, and `/memory init` preserves a previous opt-out.

```json
{ "rerank": { "enabled": true, "model": "jev-1.13.0", "candidates": 24, "timeoutMs": 15000 } }
```

When enabled, `memory_context` may ask Jev to order its shortlist. Scores are
advisory: low relevance, weak evidence, missing judgements, or instruction flags
never delete candidates. The active Pi model assesses their meaning and validity.
Missing credentials, timeout, and offline mode preserve the fused order.
`PI_MEMORY_RERANK=0` disables this stage; `PI_MEMORY_OFFLINE=1` prevents network use.
The key stays in the OS keyring, never in project configuration.

Automatic hooks do not ask Jev to remove turns or promote tool failures into
rules. Observations remain available as evidence; the active model can record a
verified lesson explicitly. Configured daemon curation uses its Pi model through
the public SDK. Session-close curation requires `PI_MEMORY_CURATE_ON_CLOSE=1`.

## Long sessions

Pi owns normal compaction. Observation masking and history retirement are off by
default. The fallback context ceiling follows the actual model window and output
reserve, without an arbitrary message-count cap. A newer rule snapshot does not
declare earlier retrieved evidence obsolete.

## Sources

`/memory sync` scans configured publishers for the active project and enqueues
changed identities for explicitly configured maintenance. Source bodies remain
transient; only validated selected knowledge, fingerprints and citations enter
the project authority. The only built-in source is pi-memory's own Pi session
log: selected failed tool results (native host provenance), exact corrections,
and explicit remember/recuerda declarations stated in the prompt. Routine
successes are not written. A failure is stored as its diagnosis — up to three
distinct error lines without runner framing or stack frames — and only in a
repository that already has memory; red tests during development are not
stored. The extension never calls a model; the standalone daemon analyzes
fingerprints while Pi is closed.

Each adapter declares its owner. The production registry installs only the
active project's `pi-session` adapter by default and rejects a team/shared
adapter or a document whose project id differs from the open engine. It does not inspect prjct observation
trees unless the compatibility adapter is explicitly enabled. Standalone
discovery helpers are not part of runtime sync. There is no federated
cross-project fallback.

Within the project, candidate legs use one ranking with corpus-wide lexical
statistics and measured cosine similarity. Unrelated sources receive no bonus or
reserved slot. When no sufficient signal exists, retrieval abstains. Evidence
windows preserve the continuation of matching headings, and byte-limited
excerpts are explicitly marked as shortened. See [real-data evaluation](docs/real-data-evaluation.md)
for reproducible private-snapshot checks with the real encoder.

### When sources are re-read

Sync is not on a schedule and does not run at start-up. Automatic sync only
queues analysis for the daemon, so it is skipped entirely until an analysis
provider is configured (`PI_MEMORY_ANALYSIS_PROVIDER` or the shared memory
config); `/memory sync` still runs on demand. Each turn adds to a
watermark table in the project's projection — turns taken, context tokens
consumed, memories written — and a source is re-read only once the work done
since its last run crosses a threshold:

| | `pi-session` | optional adapters |
|---|---:|---:|
| turns since last run | 8 | 20 |
| context tokens since last run | 16,000 | 40,000 |
| memories written since last run | 4 | 10 |
| minimum time between runs | 1 minute | 5 minutes |

Any one threshold is enough; the minimum interval overrides all of them, so a
burst of activity cannot re-scan project-local source trees every few seconds.
The run happens in the background, so a turn never waits on it, and never twice
at once.

`/memory sources` shows the counters, each adapter's last run, and why it is or
is not due. `/memory sync` ignores all of it and runs anyway. Configure optional
adapter thresholds with `installMemory(pi, { sync: { everyTurns: 50 } })`, or
disable all automatic source sync with `installMemory(pi, { sync: { enabled:
false } })`. The first-party `pi-session` cadence is fixed apart from that global
enable switch.

A failed run is recorded like a successful one, so a source that throws every
time is visible as failing rather than looking like one that has never run.

Source selection distinguishes questions from answers. Ordinary recall
suppresses non-user instructions independently of publisher or namespace; an
explicit namespace lookup can still inspect them. Eligible artifacts retain
their full bounded content instead of an 8,000-character preview that might
omit the answer.

### Optional prjct compatibility

The package still provides a data-only adapter for existing prjct observation
streams, but never registers it implicitly. A custom extension entry point can
opt in without importing or depending on the prjct package:

```ts
import { installMemory } from '@prjct.app/pi-memory';

export default pi => installMemory(pi, {
  sources: { prjct: {} },
});
```

Use `sources: { prjct: { home: '/path/to/publisher/home' } }` when the publisher
root differs from memory's home. Uninstalling the prjct package or omitting this
option does not affect pi-memory's first-party session learning and retrieval.
During explicit initialization only, a legacy `.prjct/prjct.config.json` locator
may be adopted when the legacy checksummed identity index confirms the exact
canonical checkout binding. A locator alone is untrusted.

### Connecting anything else

Sources are described, not coded. `JsonRecordAdapter` walks a tree of `.json`
and `.jsonl` files and maps records with a `RecordMapping`; when a field is not
declared it is found under the conventional names, so an ordinary publisher
needs no mapping at all:

```ts
import { JsonRecordAdapter, SourceRegistry } from '@prjct.app/pi-memory/sources';

const adapter = new JsonRecordAdapter({
  id: 'my-source',
  scope: { kind: 'project', id: projectId },
  root: '/path/to/records',
  mapping: {
    namespace: 'my.source',
    container: 'envelope.items',          // where records live inside a .json
    id: ['ref'], text: ['blurb'], observedAt: ['when'],
    kind: { rules: [{ when: [{ field: 'level', equals: 'high' }], kind: 'decision' }], fallback: 'note' },
    trust: { from: 'level', when: { high: 'host' }, fallback: 'agent' },
    metadata: { level: 'level' },
    select: { keep: [{ field: 'level', oneOf: ['high', 'medium'] }] },
  },
});
```

Paths support nesting and `*` fan-out (`replies.*.state`). Timestamps are
accepted as ISO strings, epoch seconds or epoch milliseconds. Selection is a
rule set — `keep` is a disjunction, `drop` vetoes — so what a source contributes
is configuration, not a code change. Custom registries may add adapters, but
every production adapter and returned document must identify the active project.

## One product, reusable vector layer

Vector retrieval is part of this package, not a separate service or extension.
Other applications can index arbitrary `SourceDocument` records through the
agent tool or import the same implementation:

```ts
import { openVectorIndex, TransformerEmbeddingProvider } from '@prjct.app/pi-memory/vector';

const vectors = openVectorIndex({
  path: '/absolute/path/to/rebuildable-index.sqlite',
  provider: new TransformerEmbeddingProvider(),
});
```

The default provider is a quantized local multilingual sentence encoder. An
OpenAI-compatible embedding endpoint can be selected in the project's
`memory/config.json`; credentials are read from the host environment and are
never persisted by pi-memory. A configured local `cacheDir` must remain inside
that project memory root; escaping and symlinked cache paths are rejected.

Known supply-chain caveat: `@huggingface/transformers` currently brings
`onnxruntime-node` and image-processing dependencies whose audit advisories may
report no fixed release. pi-memory uses the text feature-extraction path only;
review `npm audit --omit=dev` before publishing or deploying. `@napi-rs/keyring`
is a second native dependency, loaded only when the optional evaluator key is
read or written; include it in the same review.

## Storage

Each initialized project owns exactly one database at
`<memory-home>/<projectId>/memory/memory.sqlite`. Home resolution is: explicit
`installMemory({ home })`, then `PI_MEMORY_HOME`, then the temporary compatibility
fallbacks `PRJCT_HOME` and `~/.prjct`. Selecting a new home never moves live data
implicitly.

The memory-owned project registry is
`<memory-home>/pi-memory/projects.json`. It is checksummed, updated atomically
under an exclusive lock, and contains canonical checkout bindings only. A
missing or corrupt registry never causes path inference during status, recall,
or daemon discovery. `/memory init` is the sole interactive creation path. The
engine refuses team/shared authorities and foreign owners; there is no shared
database or cross-project fallback.

New small projects use a compact authority: hash-chained history, domain state,
chunks and packed vectors commit together in one bounded SQLite snapshot, with
no journal or checkpoint sidecar. Retrieval scans that bounded state directly.
Existing indexed stores remain indexed and open without implicit migration. A
large discovered source selects the indexed layout before its first mutation;
if an active compact store reaches capacity, indexed tables are staged while the
compact marker remains authoritative, all logical state is copied in one
transaction, and the authority mode switches last. Indexed mode retains the
hash-chained `events/<YYYYMMDD>/<writer>.jsonl` recovery log plus FTS5,
temporal-graph and sqlite-vec tables in the same project directory.

See [Architecture](docs/architecture.md) for retrieval, concurrency, retention,
and provenance details.

## Development

```sh
npm run check
npm test
npm run test:integration
npm run eval -- --suite tests/fixtures/retrieval-gold.jsonl
npm run bench -- --documents 5000 --queries 1000
npm pack --dry-run --ignore-scripts

# Real Pi RPC load, read-only status, explicit init, and reopen checks.
npm run test:integration
```

The MiniLM run is a diagnostic promotion gate, not evidence that hybrid retrieval
is better by default. A superiority claim is allowed only when the real-encoder
report reaches 1.2× the BM25 nDCG@10 score without Recall@10 or MRR regression.
Until an authorized run records that evidence, BM25 remains the supported quality
baseline. The diagnostic scores the system **without** fixture-authored query
expansions; see [Architecture](docs/architecture.md) for why.

### Measured on an M-series laptop

Documents of ~1.05 KB drawn from a Zipf-like vocabulary of ~5,000 terms, one
chunk each, with a deterministic stand-in encoder:

| | 5,000 docs | 100,000 docs |
|---|---|---|
| ingest, `index()` one at a time | 187 docs/s | 199 docs/s |
| ingest, `indexAll()` in batches | 4,244 docs/s | 2,847 docs/s |
| KNN p95 | 0.64 ms | 12.6 ms |
| whole hybrid query p50 | 13.6 ms | 74.4 ms |
| whole hybrid query p95 | 15.2 ms | 105.7 ms |
| resting size | 7.19 KB/chunk | 6.82 KB/chunk |

These scale figures exercise the indexed layout. Indexed single-document ingest
is bounded by one `fsync` per journal entry (~3.8 ms); `indexAll()` uses one
`fsync` per batch. Compact history and projection instead commit in the same
FULL-synchronous SQLite transaction, eliminating the append/apply crash window.
Real encoder inference is not included in these figures and will dominate them.

Whole-query latency is dominated by FTS5 bm25 scoring and grows with corpus
size. Three things keep that in hand: no search leg joins `documents`;
`lexicalSearch` keeps at most the twelve most selective terms of a query; and
above 5,000 chunks it drops terms appearing in more than 5% of the corpus
outright, because their bm25 contribution is near zero while the cost of
scoring every chunk they appear in is not.

That last one matters most for the queries the automatic hook actually sends —
a whole user prompt, mostly ordinary words around a few real ones. Measured with
eight known documents buried in 100,000 of filler and queried in prompt form,
recall@10, MRR and nDCG@10 are identical with the ceiling and without it, while
p50 goes from 88.6 ms to 2.9 ms. The benchmark's own queries are slices of
corpus text and so carry far more mid-frequency terms than a real prompt, which
is why its end-to-end figure improves by less.

Corpus vocabulary matters as much as corpus size: the same 100,000 documents
drawn from a 60-word vocabulary put the query at 568 ms, because every term then
matches nearly every chunk and there is nothing selective to choose.


### Freshness validation

Source sync detects validity/metadata-only changes and retires missing documents
only after a complete, ownership-scoped scan. Failed source checks preserve the
index with a freshness warning. Recall exposes dates; historical proposals are
not automatically certified as current. See [temporal semantics](docs/architecture.md#source-freshness-and-retirement)
and [real-content lifecycle validation](docs/real-data-evaluation.md#real-content-lifecycle-fault-injection).

### Storage and abstention boundaries

See [WAL maintenance, default abstention, and the corrected offline benchmark](docs/storage-and-abstention.md).
The r18 compact run stores the exact 68,857-byte tiny workload in 59,000 B at
peak-live, 45,568 B quiescent/reopened, and 12,800 B closed. The 7,381,450-byte
source workload selects indexed mode and remains a measured storage win. These
are workload-specific mechanical results; semantic answer quality remains
blocked pending authorized model evaluation.
