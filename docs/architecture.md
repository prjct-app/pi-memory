# Architecture

## Interactive extension and autonomous memory maintenance

The confirmed target separates interactive retrieval from an autonomous memory
analysis daemon. The daemon must maintain curated knowledge while Pi is closed;
a timer that runs only inside Pi does not meet that requirement. See
[curated memory and refresh jobs](curated-memory-plan.md#confirmed-execution-model-autonomous-daemon)
for the job contract and acceptance gates. The standalone daemon is
`scripts/memory-daemon.ts` (`once|start|stop|status|run`). Operational jobs live
in `curation.sqlite`. The following still describes the extension/storage substrate.

pi-memory follows the pi-team extension pattern. It registers tools, commands,
renderers, and documented lifecycle hooks in the Pi process. The extension starts
no MCP server, graph database or implicit daemon/model loop. Default source sync
fingerprints publishers and enqueues analysis; it does not copy raw bodies.

The active agent currently performs the cognitive work:

1. turn a task into up to four standalone retrieval queries;
2. rerank returned candidates against the actual task and inspect evidence;
3. extract only durable knowledge and choose entities and temporal bounds;
4. decide whether related facts duplicate, contradict, or supersede one another;
5. compose the final context or answer.

The extension performs mechanical work: provenance enforcement, hashing,
chunking, embedding, BM25/KNN lookup, temporal filtering, graph adjacency,
storage, byte budgets, and garbage collection. The embedding encoder only maps
text to numbers; it cannot promote a claim or decide what is true.

## Scope and identity

Pi-memory owns checkout binding in the checksummed, atomically replaced
`<memory-home>/pi-memory/projects.json` registry. Startup, ordinary prompts and
`/memory status` do not infer or create a binding. `/memory init` canonicalizes
the checkout, serializes concurrent initialization with an exclusive registry
lock, publishes the binding, and opens the owner-bound database. A partial claim
is rolled back when database ownership validation fails. Missing databases are
reported as incomplete rather than recreated by status or daemon discovery.

A new binding uses `p_` plus the first twelve hexadecimal characters of SHA-256
over the canonical path. For migration only, initialization may adopt a
`.prjct/prjct.config.json` id when the legacy checksummed identity index confirms
the same canonical location and project id. A locator by itself is never trusted.
This compatibility read imports no prjct package and does not let prjct open the
memory database.

The home is selected by an explicit API/CLI option, then `PI_MEMORY_HOME`, then
the compatibility fallbacks `PRJCT_HOME` and `~/.prjct`. Changing the selected
home does not move or duplicate live storage automatically. The authority remains
`<memory-home>/<projectId>/memory/memory.sqlite`. MemoryEngine accepts only
project scopes. Team and shared publishers may be discovered as sources, but
cannot open an authority or route documents outside the engine's project owner.
The standalone daemon unions ready ids from the memory-owned registry and the
legacy verified index; it never scans arbitrary `p_*` directories.

The old sibling `vector/` component is not used. `src/vector/` is a public module
inside the pi-memory npm package.

## Compact and indexed authority

Every project has one canonical `memory.sqlite`. New bounded stores start in
compact mode. One owner-bound, checksummed, Brotli-compressed row contains typed
domain maps, hash-chained events, deterministic chunks, packed int8 vectors,
curation state and operational checkpoints. Append and domain reduction share
one revision CAS and one FULL-synchronous SQLite commit. Compact mode creates no
JSONL, second database or checkpoint sidecar; lexical/dense retrieval scans only
its capacity-bounded state.

Capacity is admitted before the authority row is mutated. Busy checkpoints and
stale revisions are explicit failures, never dropped writes. A pinned reader may
hold one bounded WAL generation; later writes back off rather than accumulating
unbounded frames. Current-schema opens perform only connection-local setup and
reads.

Existing indexed stores stay indexed. Indexed mode appends bounded,
hash-chained `MemoryEvent` streams under `events/<UTC-day>/<writerId>.jsonl`,
then applies them idempotently to `memory.sqlite`, which holds source documents,
deterministic chunks, FTS5, sqlite-vec, temporal graph state and curation state.
A crash in the indexed append/apply window is repaired by `replay()`.

Large sources select indexed mode before the first durable mutation. A non-empty
compact store that exhausts capacity first materializes its verified history,
stages additive indexed schema while compact mode remains authoritative, then
copies facts, documents, vectors, curation, sync and checkpoint state in one
SQLite transaction. The mode marker switches last. Readers therefore observe a
complete compact authority or a complete indexed authority, never partially
populated indexed tables. Existing indexed data is never implicitly rewritten
as compact.

## Provenance and time

An `EvidenceRef` distinguishes host observation, exact user statement, agent
report, and imported source. Credential-shaped text is redacted before source
content or evidence enters the journal. Only a `tool_result` hook can construct
`native_observation`; tool arguments cannot. A user declaration must quote text
from the current prompt exactly. Agent-only facts remain `needs_review` unless
later supported.

Facts carry valid time (`validAt`, `invalidAt`) and transaction time
(`recordedAt`, `expiredAt`). Superseding or contradicting a fact appends a
resolution event, sets its terminal standing, and closes its valid interval
without erasing it. Historical `asOf` retrieval may still return the earlier
fact when the query time falls inside that interval. Without an explicit valid
start, the observation/recording date is the lower bound, not negative infinity.
All scopes in a search use the same query clock and half-open `[start, end)`
intervals. Empty intervals represent cancelled plans; malformed or inverted
source dates are rejected.

A fact that supersedes another closes it at the replacement's `validAt` (falling
back to its `recordedAt`); resolution transaction time comes from the journal.
Thus a future-effective replacement does not retire the current answer early.
Terminal intervals cannot be reopened in place: record a new fact so the gap in
validity is not erased. Fact ids are immutable. GC's seven-day grace starts no
earlier than both resolution time and the end of validity. Retrieval includes
observation, declared validity and resolution dates; automatic recall explicitly
warns that publication dates do not establish present applicability.

### Source freshness and retirement

JSON adapters map `validFrom` and `validTo` (custom field paths are supported).
Sync fingerprints include version, observation/validity dates and metadata, not
only body hashes. Metadata-only amendments therefore reach the projection and
journal. Schema v2 adds journal-recoverable source ownership and revision columns;
older projections migrate in place and re-sync to establish ownership.

A JSON adapter offers an authoritative snapshot in addition to `scan()`. Only a
complete snapshot may retire previously owned documents absent from it. A missing
root, missing blob, malformed JSON, invalid timestamps or failed indexing cannot
be interpreted as a withdrawal. Retained results report a source freshness gap;
a rebuild also reports unverified source state until a successful scan. Legacy
unowned rows are never guessed to belong to an adapter, and ordinary adapters
that only implement `scan()` remain additive. An adapter cannot overwrite another
adapter's tracked identity. Latest-per-id selection happens on raw revisions,
before content selection, so an excluded newer revision cannot resurrect an old
one. Ownership and tombstones survive rebuild.

These checks establish freshness only as of the last successful scan. They do
not infer semantic supersession between different document ids or filenames.
External documents retain the latest indexed revision, not a queryable version
archive: `asOf` applies its declared validity (or observation lower bound), and
cannot reconstruct deleted/overwritten source bodies. Facts retain their own
closed intervals until GC. Publisher writes should be atomic; a directory scan
is not a transactional snapshot of a publisher's entire store.

## Vector indexing

The default provider lazily loads the
`Xenova/paraphrase-multilingual-MiniLM-L12-v2` feature-extraction pipeline with
q8 model weights and mean-pooled normalized embeddings. Model files are cached
inside the active project's memory root; no model-cache path is shared across
projects. A configured OpenAI-compatible endpoint implements the same
`EmbeddingProvider` contract.

Vectors are scalar-quantized into sqlite-vec `int8` collections, reducing vector
storage fourfold relative to Float32. Model name and dimensions select a
separate collection; old collections are rebuildable and GC removes them after a
provider change. The lexical chunks are committed before embedding, so a cold
model download or provider outage yields explicit partial service rather than a
silent empty index.

## Retrieval

`hybridSearch` evaluates up to four active-agent query expansions through three
independent legs:

1. exact chunk-id and external-id lookups, plus a URI and substring scan for
   queries short enough to plausibly be a literal;
2. FTS5 BM25 over the twelve most selective terms of the query, chosen by
   document frequency;
3. dense sqlite-vec KNN.

None of the three joins the `documents` table. Deleting a document physically
removes its chunks, FTS rows and vectors, so there is no deleted row left to
filter out.

Weighted reciprocal-rank fusion combines incomparable ranks. Evidence trust,
fact confidence, and observed usefulness provide bounded priors. Valid-time and
standing filters run before output. One adjacency hop over shared entities adds
related temporal facts, subject to the same score threshold as everything else.
Lexical overlap removes redundant candidates, and a per-source cap applies only
when the candidates actually span more than one source — enforcing it in a
single-source scope just truncates the answer. The final list is serialized
under a caller-supplied hard byte budget.

`omitted` counts everything the limit, the diversity filter, or the byte budget
left out. `status` is narrower: `partial` means a retrieval leg failed or the
byte budget cut the answer short. Matching more than the limit is ordinary and
does not make an answer partial.

Public retrieval is project-local. The ranking coordinator accepts one or more
legs only when every engine has the same project owner; mixed-project, team and
shared inputs are rejected before any search. It applies namespace, kind and
temporal eligibility before ranking. Filtered legs replenish their candidate
budget up to 1,000 hits; exhausting that bound reports a gap rather than claiming
complete recall.

Lexical scoring uses BM25 with one set of full-project document frequencies and
average length across the eligible legs. It does not compare local FTS5
magnitudes or derive IDF from the query's candidate pool. Titles contribute
alongside body text, common function words are removed, and named identifiers
found in titles/URIs constrain their query's candidates. The lexical quality is
normalized against the query's theoretical saturated BM25 score, not its best
observed hit. A weak corpus winner must not become a perfect match by definition.

The vector collection's `distance` is **L2 over quantized int8 values**, not
cosine. KNN still uses that existing collection; candidates additionally expose
cosine similarity computed on the stored vectors. Rank fusion uses this
similarity and refuses to compare different model/dimension spaces, returning
lexical results with a gap instead. No re-embedding is needed for this change.

One quality-weighted RRF combines the global lexical and cosine lists. Exact
ids, URIs and short literals get an explicit signal; a prose substring does not
get an exact-answer bonus just because an earlier prompt repeated the question.
There is no scope prior or source quota. Confidence and provenance remain visible
but cannot promote an unrelated observed failure over a relevant imported answer.
A query with no sufficient lexical or semantic signal abstains. These relevance
floors are heuristics checked against unrelated-query controls, not probabilities
or a guarantee that every returned claim is true. Supported queries keep weaker
candidates for the active agent's reranking rather than losing multi-answer recall.

### Model-owned interpretation

The fused evidence windows go directly to the active Pi model. Retrieval makes no
external classifier calls and does not remove candidates based on classifier scores.
Semantic interpretation and any explicit memory write belong to the active model.

Document identity includes scope and namespace. The best chunk per document
brings up to two following chunks (at most 2,400 characters), so a matching heading
can carry its actual decision or procedure. The returned `contextChunkIds` identify
that evidence window. Redundancy filtering, scoped graph expansion, the final
item limit and byte budget are applied once. A shortened excerpt is marked with
`excerptTruncated`, and the result is `partial`; an oversized hit cannot silently
turn useful retrieval into an empty answer. The single-scope `hybridSearch` API
retains its local RRF scoring as a regression reference.

The project engine builds an embedding provider. Concurrent components in the
same process reuse one read-only encoder instance per model; the load is
reference counted and released when the last holder lets go. Persistent model
files and every data-bearing cache remain under the project root.

`memory_record`, retrieval, source sync, checkpoints and maintenance all target
the same active project authority.

Automatic recall appends a reference-data message; it does not rewrite the system
prompt. The recall threshold defaults to zero. The active agent can call
`memory_context` for further evidence or query expansion.

## Language

Stored statements and evidence preserve their original language and wording.
Memory writes do not require a translation call. User quotes and excerpts retain
verbatim provenance; the active model interprets them in context.

## Selective capture

The host stages a bounded window of tool results, excluding memory's own tools,
to the newest 64 and appends the resulting evidence id to the tool result seen by
the active agent. Staging is not memory and disappears with the session. Learnable
session observations are redacted, deduplicated by semantic key plus summary hash,
and written in one JSONL append per turn. Generic failures such as a missing npm
script fail the automatic-source precision gate.
`memory_record` promotes evidence only when the active agent identifies a
reusable decision, correction, constraint, preference, failure, or procedure.
Routine reads and progress are never automatically vectorized.

`SourceAdapter` is the application-neutral ingress boundary, and the built-in
sources are mappings over one generic adapter rather than a class per
publisher. `JsonRecordAdapter` walks a tree of JSON or JSONL records and reads
each through a `RecordMapping`: dot paths with `*` fan-out say where the id,
text, title, timestamp and metadata live, rules derive the kind and trust from
the record's own values, and `keep`/`drop` rules decide what is worth storing.
Undeclared fields fall back to the conventional names, so an ordinary publisher
needs no mapping and an unusual one needs configuration rather than code.

No publisher package is imported. `pi-session` is the only adapter installed by
default and is written and read entirely by pi-memory. The data-only prjct
observation preset remains an explicitly enabled compatibility adapter; default
source sync and daemon cycles do not inspect its observation tree. Identity and
storage retain the legacy shared-home contract during this migration stage.
Standalone discovery helpers can describe other publisher formats for
diagnostics, but the production registry never invokes them and routing cannot
open their authorities or ingest them into the project.

Every adapter declares the owner it belongs to. The project-only engine rejects
a team/shared adapter rather than writing it into a project database. Project
adapters are checked again against every returned document; a mismatch fails
loudly before any ingest.

Re-reading a source is driven by work done, not by a clock. A clock re-scans an
idle session for nothing and leaves a busy one stale; the signal that a sibling
may have published something is that this session has been doing things. Two
tables in the project's projection carry it: `sync_activity` counts turns,
context tokens and memories written, monotonically and across restarts, and
`sync_state` records for each adapter when it last ran, what it found, and the
activity watermark at that moment. An adapter is due when any counter has moved
past its threshold since that watermark, subject to a minimum interval that
stops a burst from re-scanning every few seconds.

Both records are operational, not retrieval candidates. Indexed mode keeps them
in ordinary SQLite tables; compact mode keeps them in separate typed maps inside
the same authority snapshot. Rebuild and promotion preserve them. The host
reports total context size rather than growth, so the per-turn delta is computed
by the hook; a context that shrank has been compacted and its new size is counted
as the growth since.

The run is fired without being awaited. A source scan must never sit between the
user's prompt and the agent starting, and a second run cannot begin while one is
in flight.

`pi-session` uses an independent 8-turn/60-second sync cadence. Exact declared
corrections and secret-free `remember`/`recuerda`/`acuérdate` statements are
promoted lexically after the turn so the next session can recall a supported
fact without waiting for daemon curation. Questions containing those words are
not declarations. Raw observations are never embedded.

Source selection distinguishes a request from an answer. The optional prjct
mapping keeps failures, verifications and explicitly declared statements, not
arbitrary `user_input` prompts. Optional mapping definitions can classify other
record formats, but an adapter is eligible only when it is explicitly bound to
the same project owner. Ordinary recall suppresses non-user instructions without
keying behavior to a publisher namespace, as well as unanswered legacy team
threads; an explicit namespace lookup can still inspect project-local legacy
material. Existing owner journals are never erased.

Discovered artifacts retain their full bounded content (up to 512,000 bytes)
rather than an 8,000-character preview that could omit the answer. Indexed
source-document journal events have a 2 MiB serialized bound, consistent with
the document contract; non-document events keep their 64 KiB bound. Compact
mode enforces its smaller whole-authority capacity before mutation and promotes
when necessary. Record bodies remain bounded too. After updating a preset, sync
re-ingests changed bodies idempotently. Previously excluded answers become
available without changing their imported provenance.

## Consolidation and garbage collection

Mechanical token overlap only proposes consolidation candidates; it never
changes standing. Multi-window daemon jobs run a bounded orient/gather/synthesize/
prune pass under the existing job lease and daily call/token budget. The real
synthesis request receives the complete living-context shape (goal, constraints,
done, in-progress, blocked, decisions, evidence references and next steps), not
a session dump. Standing changes still require validated publication.

Retention value combines evidence, judgment type, actual positive/negative use,
age, and standing. Novelty alone is not value. Supported user/native decisions,
corrections, constraints, and preferences are protected. GC marks active document
roots, removes only unreferenced or low-value hot projections and stale vectors,
and records removed keys in a `gc.compacted` event. Facts and evidence remain in
compact history or the indexed append-only log.

`/memory rebuild` is an operator action protected by an advisory per-project
lock. Compact mode replays its in-database history and regenerates derived
structures without another authority file. Indexed mode applies the verified
JSONL history idempotently and reindexes active documents. Neither path starts a
daemon or reads another project's state.

## Quality gates

The gold suite is 113 documents and 42 queries. Documents are grouped into
topic clusters so that most of a cluster is a *distractor* sharing the target's
vocabulary, and several queries carry more than one correct answer. It covers
paraphrases, a bilingual query, temporal policy, failures and constraints.

The MiniLM diagnostic scores `candidateNoExpansion` — the system given only the
user's query — against BM25, the former feature-hash semantic leg, and old-style
RRF. MiniLM/hybrid retrieval is not described as better unless an authorized,
reproducible run reaches 1.2× BM25 nDCG@10 with no Recall@10 or MRR regression.
Until then BM25 is the supported quality baseline and the dense leg is an
optional candidate source, not a quality claim.

Scoring the no-expansion run is deliberate. Query expansions in the fixture are
written by hand, so a gate that scores the run *with* them measures how closely
the fixture author paraphrased the answer. An earlier version of this suite did
exactly that and reported nDCG@10 = 1.0000; its expansions repeated the target
document nearly verbatim, and the system scored 0.8216 without them against a
0.8623 bar. The current expansions restate the *question*, never the answer, and
are worth about +0.007 — which is roughly what an honest expansion is worth on a
corpus this size.

Historical evaluation fixtures may partition a corpus to compare fusion math,
but the production public path accepts only engines with the same project owner.
Mixed-project, team and shared engine sets are rejected before retrieval; no
quality score can authorize cross-project reads.

`npm run eval:real -- --cases /private/cases.json` builds a private snapshot of
explicitly authorized project-local sources and its cached local encoder. It
never opens an original authority or uses a remote embedding provider. Cases
specify project id, query, expected document ids and optional required excerpt
text, or require abstention.
The script checks lexical, hybrid and automatic-injection budgets, full embedding
coverage, and a second idempotent sync. Reports and source data stay outside the
repository. See [real-data evaluation](real-data-evaluation.md) for reuse and cleanup.

The suite is a diagnostic promotion gate, not proof of broad retrieval quality.
Forty-two queries over 113 documents is small; it must grow with observed
production failures. The session-learning suite separately checks four
correction → daemon fact → new-session recall paths and keeps N1–N4 abstention.

The benchmark drives the real ingest path (`MemoryEngine.index` and
`indexAll`) over ~1.05 KB documents whose word frequencies follow a Zipf-like
curve over ~5,000 terms, with a deterministic stand-in encoder that gives every
chunk a distinct vector. It reports single-document and batched ingest
separately, KNN latency, whole-query latency, and bytes per chunk. Encoder
inference is deliberately near-free so the throughput figure is the cost of the
storage path; a real local encoder is orders of magnitude slower and would
dominate, so model time is measured separately and never folded in.

Vocabulary shape is part of the measurement, not incidental to it. Lexical
search costs what it costs because a term that appears in most chunks forces
bm25 to score most of the index, so a corpus drawn uniformly from a small
vocabulary reports a worst case that no real corpus produces: the same 100,000
documents built from 60 words put whole-query latency at 568 ms against 103 ms
for the Zipf-like corpus.

Two earlier versions of this benchmark measured something other than the system.
The first bypassed the engine, wrote through projection methods nothing else
used, and stored the identical one-hot vector for all 100,000 chunks against
22-byte documents. The second drove the real path but kept the 60-word
vocabulary described above.

## Boundaries

Local disks on macOS and Linux are supported. Network filesystems, cross-machine
replication, native Windows, unbounded binary ingestion, autonomous community
summarization, and automatic truth decisions are not. The optional reranker is
not an exception: it scores how well a passage answers a query, never whether
what the passage says is true.
