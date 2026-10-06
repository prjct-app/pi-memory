import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { MemoryEngine } from '../src/engine.ts';
import { federatedSearch } from '../src/retrieval/federated.ts';
import { TransformerEmbeddingProvider } from '../src/vector/providers.ts';
import { sha256 } from '../src/workspace/project-identity.ts';
import { retrievalMetrics } from '../tests/eval/metrics.ts';

type Doc = { type: 'document'; id: string; kind: string; text: string; trust: 'host' | 'user' | 'agent' | 'imported'; validTo?: string };
type Case = { type: 'query'; query: string; expansions?: string[]; positives: string[] };
type Metrics = { queries: number; recallAt10: number; mrr: number; ndcgAt10: number };

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const suite = resolve(arg('--suite') ?? 'tests/fixtures/retrieval-gold.jsonl');
const rows = (await readFile(suite, 'utf8')).split('\n').filter(Boolean).map(line => JSON.parse(line) as Doc | Case);
const docs = rows.filter((row): row is Doc => row.type === 'document');
const cases = rows.filter((row): row is Case => row.type === 'query');

const hash = (text: string): number[] => {
  const vector = Array.from({ length: 256 }, () => 0);
  const normalized = ` ${text.toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ')} `;
  for (const size of [3, 4, 5]) for (let i = 0; i <= normalized.length - size; i++) {
    const gram = normalized.slice(i, i + size);
    const digest = Number.parseInt(sha256(gram).slice(0, 8), 16);
    vector[digest % vector.length] += digest & 1 ? 1 : -1;
  }
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map(value => value / norm);
};
const cosine = (a: number[], b: number[]): number => a.reduce((sum, value, index) => sum + value * (b[index] ?? 0), 0);
const rrf = (lists: string[][]): string[] => {
  const scores = new Map<string, number>();
  for (const list of lists) list.forEach((id, index) => scores.set(id, (scores.get(id) ?? 0) + 1 / (61 + index)));
  return [...scores].sort((a, b) => b[1] - a[1]).map(([id]) => id);
};
const metrics = (rankings: string[][]): Metrics => {
  const measured = retrievalMetrics(rankings, cases.map(item => new Set(item.positives)), 10);
  return { queries: measured.queries, recallAt10: measured.recallAtK, mrr: measured.mrr, ndcgAt10: measured.ndcgAtK };
};

const root = await mkdtemp(join(tmpdir(), 'pi-memory-eval-'));
const provider = new TransformerEmbeddingProvider();
const engine = new MemoryEngine({ root, scopeId: 'p_eval', sessionId: 'eval', provider });
const scopes = [engine];
try {
  for (const doc of docs) await engine.index({ namespace: 'gold', externalId: doc.id, scopeId: 'p_eval', scopeKind: 'project',
    source: 'gold', kind: doc.kind, title: doc.id, text: doc.text, version: sha256(doc.text), contentHash: sha256(doc.text),
    observedAt: '2026-01-01T00:00:00.000Z', ...(doc.validTo ? { validTo: doc.validTo } : {}), trust: doc.trust, metadata: {} });
  // Project-isolated eval: the gold corpus lives only in p_eval.
  const docVectors = new Map(docs.map(doc => [doc.id, hash(doc.text)]));
  const lexical = cases.map(item => engine.projection.lexicalSearch(item.query, 100).map(hit => engine.projection.chunks([hit.chunkId])[0]?.document.externalId).filter((id): id is string => !!id));
  const hashing = cases.map(item => [...docVectors].map(([id, vector]) => ({ id, score: cosine(hash(item.query), vector) })).sort((a, b) => b.score - a.score).map(row => row.id));
  const fused = lexical.map((ranking, index) => rrf([ranking, hashing[index]!.slice(0, 10)]));
  const candidateNoExpansion = [] as string[][];
  const candidate = [] as string[][];
  const federated = [] as string[][];
  const perScopeRrf = [] as string[][];
  for (const item of cases) {
    candidateNoExpansion.push((await engine.search({ queries: [item.query], dense: true, limit: 10, maxBytes: 16384 })).items.map(hit => hit.id));
    candidate.push((await engine.search({ queries: [item.query, ...(item.expansions ?? [])], dense: true, limit: 10, maxBytes: 16384 })).items.map(hit => hit.id));
    federated.push((await federatedSearch(scopes, { queries: [item.query], dense: true, limit: 10, maxBytes: 16384 })).items.map(hit => hit.id));
    // Frozen reference strategy: merge independently ranked scopes using the
    // former project/team/shared priors. Same corpus partition, no expansions.
    const local = await Promise.all(scopes.map(async (scope, index) =>
      (await scope.search({ queries: [item.query], dense: true, limit: 10, maxBytes: 32768 })).items
        .map(hit => ({ id: hit.id, score: hit.score * [1, 0.9, 0.85][index]! }))));
    perScopeRrf.push(local.flat().sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, 10).map(hit => hit.id));
  }
  const report = { bm25: metrics(lexical), hashing: metrics(hashing), fused: metrics(fused),
    candidateNoExpansion: metrics(candidateNoExpansion), candidate: metrics(candidate),
    perScopeRrf: metrics(perScopeRrf), federatedNoExpansion: metrics(federated),
 };
  const worstCases = cases.map((item, index) => ({ query: item.query, positives: item.positives,
    rank: candidateNoExpansion[index]!.findIndex(id => item.positives.includes(id)) + 1,
    rankWithExpansions: candidate[index]!.findIndex(id => item.positives.includes(id)) + 1,
    fusedRank: fused[index]!.findIndex(id => item.positives.includes(id)) + 1,
    top: candidateNoExpansion[index]!.slice(0, 3) })).filter(item => item.rank !== 1);

  // The gate measures candidateNoExpansion, not candidate. Expansions are
  // written into the fixture by hand; scoring the system on them measures how
  // well the fixture author paraphrased the answer, not how well retrieval
  // works. Each baseline metric is the best any baseline achieved on that
  // metric, so a baseline cannot hide a strong recall behind a weak nDCG.
  const baselines = [report.bm25, report.hashing, report.fused];
  const best = { ndcgAt10: Math.max(...baselines.map(item => item.ndcgAt10)),
    recallAt10: Math.max(...baselines.map(item => item.recallAt10)), mrr: Math.max(...baselines.map(item => item.mrr)) };
  const measured = report.candidateNoExpansion;
  const passed = measured.ndcgAt10 >= best.ndcgAt10 * 1.2
    && measured.recallAt10 >= best.recallAt10 && measured.mrr >= best.mrr;
  const federatedPassed = report.federatedNoExpansion.ndcgAt10 >= best.ndcgAt10 * 1.2
    && report.federatedNoExpansion.recallAt10 >= best.recallAt10 && report.federatedNoExpansion.mrr >= best.mrr
    && report.federatedNoExpansion.ndcgAt10 >= report.perScopeRrf.ndcgAt10
    && report.federatedNoExpansion.recallAt10 >= report.perScopeRrf.recallAt10
    && report.federatedNoExpansion.mrr >= report.perScopeRrf.mrr;

  console.log(JSON.stringify({ suite, corpus: docs.length, queries: cases.length, ...report, worstCases,
    federatedGate: { passed: federatedPassed, scopes: scopes.length, measures: 'federatedNoExpansion' },
    gate: { passed, measures: 'candidateNoExpansion', best, requiredNdcgAt10: Number((best.ndcgAt10 * 1.2).toFixed(4)),
      actualNdcgAt10: Number(measured.ndcgAt10.toFixed(4)),
      expansionLift: Number((report.candidate.ndcgAt10 - measured.ndcgAt10).toFixed(4)) } }, null, 2));
  if (!passed || !federatedPassed) process.exitCode = 1;
} finally {
  await engine.dispose();
  await rm(root, { recursive: true, force: true });
}
