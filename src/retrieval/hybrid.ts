import type { ScopeKind, SourceDocument } from '../contracts/documents.ts';
import { factIsValidAt } from '../contracts/memory.ts';
import type { LexicalHit, StoredFact, VectorHit } from '../storage/projection.ts';
import type { ProjectionPort } from '../storage/ports.ts';
import type { VectorIndex } from '../vector/vector-index.ts';
import { relevantKeys, relevanceTerms } from './relevance.ts';

export type MemoryQuery = Readonly<{
  queries: readonly string[];
  scopeId: string;
  asOf?: string;
  namespaces?: readonly string[];
  kinds?: readonly string[];
  limit?: number;
  maxBytes?: number;
  dense?: boolean;
  scoreThreshold?: number;
  signal?: AbortSignal;
}>;

export type MemoryHit = Readonly<{
  id: string;
  chunkId: string;
  scopeId?: string;
  scopeKind?: ScopeKind;
  statement: string;
  excerptTruncated?: boolean;
  contextChunkIds?: readonly string[];
  title?: string;
  uri?: string;
  namespace: string;
  source: string;
  kind: string;
  score: number;
  standing?: string;
  observedAt?: string;
  expiredAt?: string;
  validAt?: string;
  invalidAt?: string;
  provenance: string;
  evidenceIds: readonly string[];
  reason: readonly string[];
}>;

export type HybridSearchResult = Readonly<{
  status: 'ok' | 'partial' | 'abstained';
  items: readonly MemoryHit[];
  gaps: readonly string[];
  omitted: number;
}>;

export type SearchLegs = Readonly<{
  queries: readonly string[];
  exact: readonly (readonly LexicalHit[])[];
  lexical: readonly (readonly LexicalHit[])[];
  dense: readonly (readonly VectorHit[])[];
  denseGaps: readonly string[];
  denseModel?: string;
}>;

export type RankedCandidate = Readonly<{
  item: MemoryHit;
  tokenSet: Set<string>;
  fact?: StoredFact;
}>;

const tokens = (text: string): Set<string> => new Set(text.toLocaleLowerCase().match(/[\p{L}\p{N}_./:-]{2,}/gu) ?? []);
const exactContent = (item: MemoryHit): string => item.statement.normalize('NFC').replace(/\s+/gu, ' ').trim();
const GRAPH_ITEM_SCORE = 0.01;
const trustWeight = (trust: SourceDocument['trust']): number => ({ host: 1, user: 0.95, imported: 0.8, agent: 0.65 })[trust];

export const addRanking = (scores: Map<string, number>, reasons: Map<string, string[]>, ids: readonly string[], label: string, weight: number, qualities?: readonly number[]): void => {
  ids.forEach((id, index) => {
    scores.set(id, (scores.get(id) ?? 0) + weight * (qualities?.[index] ?? 1) / (60 + index + 1));
    reasons.set(id, [...(reasons.get(id) ?? []), label]);
  });
};

export const clipItems = (items: readonly MemoryHit[], maxBytes: number): { items: MemoryHit[]; omitted: number; truncated: number } => {
  const fit = items.reduce<{ items: MemoryHit[]; bytes: number; truncated: number }>((state, item) => {
    const size = (value: MemoryHit): number => Buffer.byteLength(JSON.stringify(value), 'utf8') + 1;
    if (state.bytes + size(item) <= maxBytes) return { ...state, items: [...state.items, item], bytes: state.bytes + size(item) };
    const chars = Array.from(item.statement);
    const shortened = (end: number): MemoryHit => ({ ...item, statement: `${chars.slice(0, end).join('')}…`, excerptTruncated: true });
    // Keep attribution intact. If even the envelope does not fit, try the next
    // result rather than returning an empty answer because the first was large.
    if (state.bytes + size(shortened(0)) > maxBytes) return state;
    const range = { low: 0, high: chars.length };
    while (range.low < range.high) {
      const mid = Math.ceil((range.low + range.high) / 2);
      if (state.bytes + size(shortened(mid)) <= maxBytes) range.low = mid;
      else range.high = mid - 1;
    }
    const clipped = shortened(range.low);
    return { items: [...state.items, clipped], bytes: state.bytes + size(clipped), truncated: state.truncated + 1 };
  }, { items: [], bytes: 2, truncated: 0 });
  return { items: fit.items, omitted: items.length - fit.items.length, truncated: fit.truncated };
};

/**
 * Per-leg candidate lists for one scope. Dense distance and BM25 scores are
 * left intact so a later merge can re-rank with a comparable signal; this
 * function does not fuse.
 */
export const collectLegs = async (
  projection: ProjectionPort,
  vector: VectorIndex,
  request: MemoryQuery,
): Promise<SearchLegs> => {
  const queries = [...new Set(request.queries.map(query => query.trim()).filter(Boolean))].slice(0, 4);
  const limit = Math.max(1, Math.min(50, request.limit ?? 12));
  const candidateLimit = Math.max(30, limit * 8);
  const asOf = request.asOf !== undefined ? Date.parse(request.asOf) : Date.now();
  if (!Number.isFinite(asOf)) throw new Error('asOf must be ISO-8601.');
  // Eligibility depends only on the chunk, the request and asOf, all fixed for
  // this call. The queries overlap and each budget round re-reads the last
  // round's hits, so a chunk is judged once instead of up to eight times.
  const judged = new Map<string, boolean>();
  const eligibleOf = (ids: readonly string[]): ReadonlySet<string> => {
    const unknown = ids.filter(id => !judged.has(id));
    const passed = new Set(unknown.length ? candidatesFromScores(projection, request, new Map(unknown.map(id => [id, 0])), new Map(), 0, asOf, false)
      .map(candidate => candidate.item.chunkId) : []);
    for (const id of unknown) judged.set(id, passed.has(id));
    return new Set(ids.filter(id => judged.get(id)));
  };
  const collect = async (query: string, budget: number): Promise<{
    exact: LexicalHit[]; lexical: LexicalHit[]; dense: VectorHit[]; gaps: string[];
  }> => {
    request.signal?.throwIfAborted();
    const exact = projection.exactSearch(query, budget);
    const lexical = projection.lexicalSearch(query, budget);
    const gaps: string[] = [];
    const dense = request.dense === false ? [] : await vector.search({ text: query, limit: budget, signal: request.signal }).catch(error => {
      gaps.push(`Dense retrieval unavailable: ${error instanceof Error ? error.message : String(error)}`);
      return [];
    });
    const ids = [...new Set([...exact, ...lexical, ...dense].map(hit => hit.chunkId))];
    const eligible = eligibleOf(ids);
    const needsMore = [exact, lexical, dense].some(hits => hits.length >= budget
      && hits.filter(hit => eligible.has(hit.chunkId)).length < candidateLimit);
    if (needsMore && budget < 1000) return collect(query, Math.min(1000, budget * 2));
    if (needsMore) gaps.push('Candidate budget exhausted while filtering; eligible matches may remain.');
    return { exact: exact.filter(hit => eligible.has(hit.chunkId)), lexical: lexical.filter(hit => eligible.has(hit.chunkId)),
      dense: dense.filter(hit => eligible.has(hit.chunkId)), gaps };
  };
  const results = await Promise.all(queries.map(query => collect(query, candidateLimit)));
  return { queries, exact: results.map(result => result.exact), lexical: results.map(result => result.lexical),
    dense: results.map(result => result.dense), denseGaps: [...results.flatMap(result => result.gaps), ...projection.sourceGaps()],
    denseModel: `${vector.provider.model}:${[...new Set(results.flatMap(result => result.dense[0]?.dimensions ?? []))].sort().join(',')}` };

};

export const candidatesFromScores = (
  projection: ProjectionPort,
  request: MemoryQuery,
  scores: ReadonlyMap<string, number>,
  reasons: ReadonlyMap<string, readonly string[]>,
  threshold: number,
  asOf: number,
  priors = true,
): RankedCandidate[] => {
  const chunks = projection.retrievalChunks([...scores.keys()]);
  return chunks.flatMap(chunk => {
    const document = chunk.document;
    if (document.scopeId !== request.scopeId) return [];
    // Compatibility for projections ingested by old or third-party presets.
    // A non-user instruction is an observed prompt, not a durable answer,
    // regardless of which publisher named its namespace. Explicit namespace
    // inspection can still retrieve it. Curated memory remains eligible.
    if (!request.namespaces?.length) {
      if (document.namespace !== 'memory' && document.kind === 'instruction' && document.trust !== 'user') return [];
      if (document.namespace === 'pi-team.journal' && (document.metadata.outcome === 'interrupted'
        || document.kind === 'thread' && !/^(Delivered|Result|Replies):\s*\S/mu.test(document.text))) return [];
    }
    if (request.namespaces?.length && !request.namespaces.includes(document.namespace)) return [];
    if (request.kinds?.length && !request.kinds.includes(document.kind)) return [];
    const fact = document.namespace === 'memory' ? projection.getFact(document.externalId) : undefined;
    if (fact ? !factIsValidAt(fact, asOf) : !(Date.parse(document.validFrom ?? document.observedAt) <= asOf
      && asOf < (document.validTo ? Date.parse(document.validTo) : Infinity))) return [];
    const base = scores.get(chunk.id) ?? 0;
    const utility = fact ? Math.max(-0.25, Math.min(0.25, fact.usefulness * 0.025)) : 0;
    const evidenceBoost = fact?.evidence.some(evidence => ['native_observation', 'declared'].includes(evidence.provenance)) ? 0.08 : 0;
    const confidence = fact?.confidence ?? trustWeight(document.trust);
    const score = priors ? base * (0.65 + 0.35 * confidence) + utility + evidenceBoost : base;
    if (score < threshold) return [];
    const item: MemoryHit = {
      id: fact?.id ?? document.externalId, chunkId: chunk.id, statement: chunk.text,
      scopeId: document.scopeId, scopeKind: document.scopeKind,
      ...(document.title ? { title: document.title } : {}), ...(document.uri ? { uri: document.uri } : {}),
      namespace: document.namespace, source: document.source, kind: document.kind, score,
      observedAt: document.observedAt, ...(fact ? { standing: fact.standing } : {}),
      ...((fact?.validAt ?? document.validFrom) ? { validAt: fact?.validAt ?? document.validFrom } : {}),
      ...((fact?.invalidAt ?? document.validTo) ? { invalidAt: fact?.invalidAt ?? document.validTo } : {}),
      ...(fact?.expiredAt ? { expiredAt: fact.expiredAt } : {}), provenance: fact
        ? (fact.evidence.some(evidence => evidence.provenance === 'native_observation') ? 'native_observation'
          : fact.evidence.some(evidence => evidence.provenance === 'declared') ? 'declared' : 'agent_report')
        : document.trust,
      evidenceIds: fact?.evidence.map(evidence => evidence.id) ?? [], reason: reasons.get(chunk.id) ?? [],
    };
    return [{ item, tokenSet: tokens(chunk.text), fact }];
  }).sort((a, b) => b.item.score - a.item.score || a.item.id.localeCompare(b.item.id));
};

export const presentCandidates = (
  ranked: readonly RankedCandidate[],
  options: Readonly<{
    limit: number;
    maxBytes: number;
    threshold: number;
    asOf: number;
    gaps: readonly string[];
    neighbors: (factIds: readonly string[], limit: number) => readonly StoredFact[];
    sourceCap?: boolean;
    identity?: (item: MemoryHit) => string;
    expand?: (selected: readonly RankedCandidate[]) => readonly MemoryHit[];
  }>,
): HybridSearchResult => {
  const { limit, threshold, asOf } = options;
  const identity = options.identity ?? ((item: MemoryHit): string => JSON.stringify([item.scopeId, item.namespace, item.id]));
  // Source capping only makes sense when there is more than one source to
  // balance. Applied unconditionally it silently truncates every result set in a
  // single-source scope to limit/3 items, which costs recall and buys nothing.
  const sourceCount = new Set(ranked.map(candidate => candidate.item.source)).size;
  const perSourceCap = options.sourceCap !== false && sourceCount > 1 ? Math.max(2, Math.ceil(limit / 3)) : limit;
  const diverse = ranked.reduce<RankedCandidate[]>((selected, candidate) => {
    const duplicate = selected.some(prior => identity(prior.item) === identity(candidate.item)
      || exactContent(prior.item) === exactContent(candidate.item));
    if (selected.length >= limit || duplicate) return selected;
    const sameSource = selected.filter(prior => prior.item.source === candidate.item.source).length;
    return sameSource >= perSourceCap ? selected : [...selected, candidate];
  }, []);
  // Graph neighbours carry a fixed low score, so they have to clear the same
  // threshold as everything else. Before, they were merged after the filter and
  // a caller asking for high-confidence hits got them anyway.
  const factIds = diverse.flatMap(candidate => candidate.fact?.id ?? []);
  const graph = GRAPH_ITEM_SCORE < threshold ? []
    : options.neighbors(factIds, Math.min(6, limit)).filter(fact => factIsValidAt(fact, asOf)).slice(0, Math.min(6, limit));
  const graphItems: MemoryHit[] = graph.map((fact: StoredFact) => ({
    id: fact.id, chunkId: `graph:${fact.id}`, statement: fact.statement, namespace: 'memory', source: 'graph',
    scopeId: fact.scopeId, scopeKind: fact.scopeId === 'shared' ? 'shared' : fact.scopeId.startsWith('p_') ? 'project' : 'team',
    kind: fact.kind, score: GRAPH_ITEM_SCORE, standing: fact.standing, observedAt: fact.recordedAt,
    ...(fact.expiredAt ? { expiredAt: fact.expiredAt } : {}), ...(fact.validAt ? { validAt: fact.validAt } : {}),
    ...(fact.invalidAt ? { invalidAt: fact.invalidAt } : {}), provenance: fact.evidence.some(item => item.provenance === 'native_observation')
      ? 'native_observation' : fact.evidence.some(item => item.provenance === 'declared') ? 'declared' : 'agent_report',
    evidenceIds: fact.evidence.map(item => item.id), reason: ['temporal-graph'],
  }));
  // Keyed by id with the RANKED item winning. Building the map the other way
  // round let a graph neighbour that also surfaced through ranking replace a
  // properly scored hit with its fixed-score stub.
  const selectedRankedIds = new Set(diverse.map(candidate => identity(candidate.item)));
  const expanded = options.expand?.(diverse) ?? graphItems;
  const unique = [...diverse.map(candidate => candidate.item),
    ...expanded.filter(item => !selectedRankedIds.has(identity(item)))];
  const bounded = unique.slice(0, limit);
  const clipped = clipItems(bounded, options.maxBytes);
  const diversityOmitted = [...new Set(ranked.map(candidate => identity(candidate.item)))]
    .filter(id => !selectedRankedIds.has(id)).length;
  const omitted = diversityOmitted + (unique.length - bounded.length) + clipped.omitted;
  const gaps = [...new Set(options.gaps)];
  // 'partial' means the answer is degraded — a retrieval leg failed, or the byte
  // budget cut results the caller asked for. Having more matches than `limit` is
  // ordinary and still reported through `omitted`; folding it into the status
  // made 'ok' unreachable for any query that matched more than it returned.
  const degraded = gaps.length > 0 || clipped.omitted > 0 || clipped.truncated > 0;
  return { status: clipped.items.length ? (degraded ? 'partial' : 'ok') : 'abstained',
    items: clipped.items, gaps: clipped.items.length ? gaps : [...gaps, 'No active memory matched the requested scope and time.'], omitted };
};

export const hybridSearch = async (projection: ProjectionPort, vector: VectorIndex, request: MemoryQuery): Promise<HybridSearchResult> => {
  const queries = [...new Set(request.queries.map(query => query.trim()).filter(Boolean))].slice(0, 4);
  if (!queries.length) return { status: 'abstained', items: [], gaps: ['No retrieval query was provided.'], omitted: 0 };
  const limit = Math.max(1, Math.min(50, request.limit ?? 12));
  const asOf = request.asOf !== undefined ? Date.parse(request.asOf) : Date.now();
  if (!Number.isFinite(asOf)) throw new Error('asOf must be ISO-8601.');
  const legs = await collectLegs(projection, vector, { ...request, asOf: new Date(asOf).toISOString() });
  const scores = new Map<string, number>();
  const reasons = new Map<string, string[]>();
  const statistics = projection.lexicalStatistics(relevanceTerms(queries));
  legs.queries.forEach((query, index) => {
    const ids = [...new Set([...legs.exact[index] ?? [], ...legs.lexical[index] ?? [], ...legs.dense[index] ?? []].map(hit => hit.chunkId))];
    const chunks = projection.retrievalChunks(ids);
    const accepted = new Set(relevantKeys(query, chunks.map(chunk => ({ key: chunk.id,
      text: `${chunk.document.title ?? ''}\n${chunk.text}` })), statistics,
    new Map((legs.dense[index] ?? []).map(hit => [hit.chunkId, hit.similarity]))));
    for (const chunk of chunks) {
      if (query === chunk.id || query === chunk.document.externalId || query === chunk.document.uri) {
        accepted.add(chunk.id);
      }
    }
    addRanking(scores, reasons, (legs.exact[index] ?? []).filter(hit => accepted.has(hit.chunkId)).map(hit => hit.chunkId), 'exact', 2.5);
    addRanking(scores, reasons, (legs.lexical[index] ?? []).filter(hit => accepted.has(hit.chunkId)).map(hit => hit.chunkId), 'bm25', 1);
    addRanking(scores, reasons, (legs.dense[index] ?? []).filter(hit => accepted.has(hit.chunkId)).map(hit => hit.chunkId), 'dense', 1.25);
  });
  const threshold = Math.max(0, Math.min(1, request.scoreThreshold ?? 0));
  const ranked = candidatesFromScores(projection, request, scores, reasons, threshold, asOf);
  return presentCandidates(ranked, {
    limit, maxBytes: Math.max(512, Math.min(32_768, request.maxBytes ?? 4096)),
    threshold, asOf, gaps: ranked.length ? legs.denseGaps : [...legs.denseGaps, 'Insufficient evidence: no candidate meets the default relevance gate.'],
    neighbors: (factIds, hop) => request.namespaces?.length && !request.namespaces.includes('memory') ? []
      : projection.graphNeighbors(factIds, hop).filter(fact => fact.scopeId === request.scopeId
        && (!request.kinds?.length || request.kinds.includes(fact.kind))),
  });
};
