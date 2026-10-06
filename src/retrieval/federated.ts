import type { MemoryEngine } from '../engine.ts';
import { factIsValidAt } from '../contracts/memory.ts';
import {
  addRanking, candidatesFromScores, presentCandidates,
  type HybridSearchResult, type MemoryHit, type MemoryQuery, type RankedCandidate,
} from './hybrid.ts';
import { lexicalTerms, queryWords, rankLexically, words } from './lexical.ts';
import { relevantKeys, relevanceTerms } from './relevance.ts';

export type FederatedQuery = Omit<MemoryQuery, 'scopeId'>;
// Relevance floors, not probabilities. A nearest neighbour always exists even
// when a scope knows nothing about the question. Real unrelated-query controls
// reached cosine 0.30; keep a margin and require substantial lexical coverage.
export const MIN_DENSE_SIMILARITY = 0.35;
export const MIN_LEXICAL_QUALITY = 0.2;
const keyOf = (item: MemoryHit): string => JSON.stringify([item.scopeKind, item.scopeId, item.namespace, item.id]);
const chunkKey = (item: MemoryHit): string => JSON.stringify([keyOf(item), item.chunkId]);

/** Candidate fusion: one lexical corpus, real cosine, no source quotas or scope priors. */
export const federatedSearch = async (engines: readonly MemoryEngine[], request: FederatedQuery): Promise<HybridSearchResult> => {
  if (!engines.length) return { status: 'abstained', items: [], gaps: ['No memory scope is open.'], omitted: 0 };
  if (engines.some(engine => engine.scopeKind !== 'project')) throw new Error('Memory search is project-local.');
  if (new Set(engines.map(engine => engine.scopeId)).size > 1) throw new Error('Mixed-project search is not allowed.');
  const queries = [...new Set(request.queries.map(query => query.trim()).filter(Boolean))].slice(0, 4);
  if (!queries.length) return { status: 'abstained', items: [], gaps: ['No retrieval query was provided.'], omitted: 0 };
  const limit = Math.max(1, Math.min(50, request.limit ?? 12));
  const maxBytes = Math.max(512, Math.min(32_768, request.maxBytes ?? 4096));
  const threshold = Math.max(0, Math.min(1, request.scoreThreshold ?? 0));
  const asOf = request.asOf !== undefined ? Date.parse(request.asOf) : Date.now();
  if (!Number.isFinite(asOf)) throw new Error('asOf must be ISO-8601.');
  const uniqueEngines = engines.filter((engine, index) => engines.findIndex(other =>
    other.scopeId === engine.scopeId && other.scopeKind === engine.scopeKind) === index);
  const searched = await Promise.all(uniqueEngines.map(async engine => {
    try {
      const legs = await engine.candidates({ ...request, asOf: new Date(asOf).toISOString(), queries, limit });
      const ids = [...new Set([...legs.exact.flat(), ...legs.lexical.flat(), ...legs.dense.flat()].map(hit => hit.chunkId))];
      // Filter eligibility before global ranking. Confidence/provenance is not
      // relevance: an observed shell error is not an answer to an unrelated task.
      const candidates = ids.length ? candidatesFromScores(engine.projection,
        { ...request, scopeId: engine.scopeId }, new Map(ids.map(id => [id, 0])), new Map(), 0, asOf, false) : [];
      const statistics = engine.projection?.lexicalStatistics(relevanceTerms(queries));
      return { engine, legs, candidates, statistics, error: undefined };
    } catch (error) {
      return { engine, legs: undefined, candidates: [], statistics: undefined, error: error instanceof Error ? error.message : String(error) };
    }
  }));
  const gaps = searched.flatMap(entry => entry.error
    ? [`Scope ${entry.engine.scopeKind}/${entry.engine.scopeId} is unavailable: ${entry.error}`]
    : (entry.legs?.denseGaps ?? []).map(gap => `${entry.engine.scopeKind}/${entry.engine.scopeId}: ${gap}`));
  const candidates = searched.flatMap(entry => entry.candidates);
  const lexical = rankLexically(candidates.map(candidate => ({ key: chunkKey(candidate.item),
    title: candidate.item.title, statement: candidate.item.statement })), queries, searched.flatMap(entry => entry.statistics ?? []));
  const denseModels = new Set(searched.flatMap(entry => entry.legs?.dense.flat().length ? [entry.legs.denseModel] : []));
  const compatible = denseModels.size <= 1;
  if (!compatible) gaps.push('Dense federation requires one embedding model; using lexical retrieval for incompatible scopes.');
  const scores = new Map<string, number>();
  const reasons = new Map<string, string[]>();
  queries.forEach((query, index) => {
    // An explicitly named identifier is not interchangeable with another
    // project's PRD. When the corpus contains that identifier, require it in
    // this query's candidates; otherwise preserve semantic-only fallback.
    const identifiers = lexicalTerms(query).filter(term => /[-_/]/u.test(term)
      && candidates.some(candidate => `${candidate.item.title ?? ''}\n${candidate.item.uri ?? ''}`.toLowerCase().includes(term)));
    const anchored = new Set(candidates.filter(candidate => identifiers.every(identifier =>
      `${candidate.item.title ?? ''}\n${candidate.item.statement}`.toLowerCase().includes(identifier))).map(candidate => chunkKey(candidate.item)));
    const accept = (key: string): boolean => !identifiers.length || !anchored.size || anchored.has(key);
    const lex = [...lexical[index]!].filter(([key]) => accept(key)).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const dense = searched.flatMap(entry => {
      const byChunk = new Map(entry.candidates.map(candidate => [candidate.item.chunkId, candidate]));
      return (compatible ? entry.legs?.dense[index] ?? [] : []).flatMap(hit => {
        const candidate = byChunk.get(hit.chunkId);
        return candidate ? [{ key: chunkKey(candidate.item), similarity: hit.similarity }] : [];
      });
    }).filter(hit => hit.similarity > 0 && accept(hit.key)).sort((a, b) => b.similarity - a.similarity || a.key.localeCompare(b.key));
    // Cosine is measured on stored vectors, NOT 1/(1+int8 L2). Only compatible
    // embedding spaces participate; sharing a numeric range is not calibration.
    // Admission is query-level: without a strong signal, abstain. Once there
    // is evidence, retain weaker candidates for the active agent to inspect;
    // applying the floor to every item needlessly loses multi-answer recall.
    const terms = queryWords(query);
    const completeLexicalMatch = terms.length > 0 && candidates.some(candidate => {
      if (!accept(chunkKey(candidate.item))) return false;
      const present = new Set(words(`${candidate.item.title ?? ''}\n${candidate.item.statement}`));
      return terms.every(term => present.has(term));
    });
    const supported = completeLexicalMatch || lex.some(([, quality]) => quality >= MIN_LEXICAL_QUALITY)
      || dense.some(hit => hit.similarity >= MIN_DENSE_SIMILARITY);
    const statistics = searched.find(entry => entry.statistics)?.statistics;
    const admitted = statistics ? relevantKeys(query, candidates.map(candidate => ({ key: chunkKey(candidate.item),
      text: `${candidate.item.title ?? ''}\n${candidate.item.statement}` })), statistics,
    new Map(dense.map(hit => [hit.key, hit.similarity]))) : new Set<string>();
    const relevantLex = lex.filter(([key]) => admitted.has(key));
    const relevantDense = dense.filter(hit => admitted.has(hit.key));
    if (supported || relevantLex.length) {
      addRanking(scores, reasons, relevantLex.map(([key]) => key), 'global-bm25', 1, relevantLex.map(([, score]) => score));
      addRanking(scores, reasons, relevantDense.map(hit => hit.key), 'cosine', 1.25, relevantDense.map(hit => hit.similarity));
    }
    for (const candidate of candidates) {
      const item = candidate.item;
      const identifier = query === item.id || query === item.chunkId || query === item.uri;
      const literal = (/^".*"$/u.test(query) || lexicalTerms(query).length === 1)
        && `${item.title ?? ''}\n${item.statement}`.toLowerCase().includes(query.replace(/^"|"$/gu, '').toLowerCase());
      if (identifier || literal) {
        const key = chunkKey(item);
        scores.set(key, (scores.get(key) ?? 0) + 2.5 / 61);
        reasons.set(key, [...(reasons.get(key) ?? []), identifier ? 'identifier' : 'literal']);
      }
    }
  });
  const ranked = candidates.flatMap(candidate => {
    const key = chunkKey(candidate.item);
    const score = scores.get(key) ?? 0;
    return score > 0 && score >= threshold ? [{ ...candidate,
      item: { ...candidate.item, score, reason: reasons.get(key) ?? [] } }] : [];
  }).sort((a, b) => b.item.score - a.item.score || chunkKey(a.item).localeCompare(chunkKey(b.item)));
  // External ids are publisher-local. Do not lose another scope/namespace's
  // document just because it uses the same id (README, policy, etc.).
  const best = ranked.filter((candidate, index) => ranked.findIndex(other => keyOf(other.item) === keyOf(candidate.item)) === index);
  const expanded = best.map(candidate => {
    const owner = uniqueEngines.find(engine => engine.scopeId === candidate.item.scopeId && engine.scopeKind === candidate.item.scopeKind);
    try {
      const window = owner?.projection.chunkWindow(candidate.item.chunkId);
      return window && window.chunkIds.length > 1 ? { ...candidate, tokenSet: new Set(words(window.text)),
        item: { ...candidate.item, statement: window.text, contextChunkIds: window.chunkIds } }
        : { ...candidate, tokenSet: new Set(words(candidate.item.statement)) };
    } catch (error) {
      gaps.push(`Context continuation unavailable for ${candidate.item.id}: ${String(error)}`);
      return candidate;
    }
  });
  return presentCandidates(expanded, {
    limit, maxBytes, threshold, asOf, gaps: expanded.length ? gaps : [...gaps, 'Insufficient evidence: no candidate meets the default relevance gate.'], sourceCap: false, identity: keyOf,
    neighbors: () => [],
    expand: selected => searched.flatMap(entry => {
      const seeds = selected.filter(candidate => candidate.item.scopeId === entry.engine.scopeId
        && candidate.item.scopeKind === entry.engine.scopeKind).flatMap(candidate => candidate.fact?.id ?? []);
      if (!seeds.length || threshold > 0.01 || request.namespaces?.length && !request.namespaces.includes('memory')) return [];
      try {
        return entry.engine.projection.graphNeighbors(seeds, Math.min(6, limit)).filter(fact =>
          factIsValidAt(fact, asOf) && (!request.kinds?.length || request.kinds.includes(fact.kind))).map(fact => ({
          id: fact.id, chunkId: `graph:${fact.id}`, scopeId: entry.engine.scopeId, scopeKind: entry.engine.scopeKind,
          statement: fact.statement, namespace: 'memory', source: 'graph', kind: fact.kind, score: 0.01,
          standing: fact.standing, observedAt: fact.recordedAt, expiredAt: fact.expiredAt, validAt: fact.validAt, invalidAt: fact.invalidAt,
          provenance: fact.evidence.some(e => e.provenance === 'native_observation') ? 'native_observation'
            : fact.evidence.some(e => e.provenance === 'declared') ? 'declared' : 'agent_report',
          evidenceIds: fact.evidence.map(e => e.id), reason: ['temporal-graph'],
        }));
      } catch (error) {
        gaps.push(`Scope ${entry.engine.scopeKind}/${entry.engine.scopeId} graph unavailable: ${String(error)}`);
        return [];
      }
    }),
  });
};
