import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { MemoryEngine, hostEvidence } from '../src/engine.ts';
import type { EmbeddingProvider } from '../src/vector/providers.ts';
import { TestEmbeddingProvider } from './helpers.ts';
import { sha256 } from '../src/workspace/project-identity.ts';

class RecoveringEmbeddingProvider implements EmbeddingProvider {
  readonly model = 'recovering-v1';
  readonly isLocal = true;
  available = false;
  async embed(texts: readonly string[]): Promise<number[][]> {
    if (!this.available) throw new Error('offline');
    return texts.map(() => [1, 0, 0, 0, 0, 0, 0, 0]);
  }
}

test('records evidence-backed temporal facts and retrieves them through dense synonyms', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-engine-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  const evidence = hostEvidence({ excerpt: 'migration test succeeded', sessionId: 's1' });
  const stored = await engine.recordFact({ kind: 'decision', statement: 'Use SQLite for durable project storage',
    subject: 'project storage', predicate: 'uses', object: 'SQLite', entities: [], evidence: [evidence], episodeIds: [],
    confidence: 0.95, validAt: '2026-01-01T00:00:00.000Z', tags: { area: 'storage' } });
  assert.equal(stored.fact.standing, 'supported');
  const recalled = await engine.search({ queries: ['database persistence'], dense: true, maxBytes: 4096 });
  assert.equal(recalled.items[0]?.id, stored.fact.id);
  assert.ok(recalled.items[0]?.reason.includes('dense'));
});

test('a superseded fact is deleted: gone now and gone from history', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-engine-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  const old = await engine.recordFact({ kind: 'decision', statement: 'Deploy releases on Friday', entities: [], evidence: [], episodeIds: [],
    confidence: 0.6, validAt: '2025-01-01T00:00:00.000Z', invalidAt: '2099-01-01T00:00:00.000Z', tags: {} });
  await engine.resolveFact(old.fact.id, 'superseded', 'Schedule changed');
  assert.equal((await engine.search({ queries: ['deploy releases'], dense: false })).items.length, 0);
  const historical = await engine.search({ queries: ['deploy releases'], dense: false, asOf: '2025-06-01T00:00:00.000Z' });
  assert.equal(historical.items.length, 0, 'deleting is deleting: no history is kept');
  assert.equal(engine.projection.getFact(old.fact.id), undefined);
});

test('dense outages leave lexical chunks available for an explicit backfill', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-engine-'));
  const provider = new RecoveringEmbeddingProvider();
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  const indexed = await engine.index({ namespace: 'docs', externalId: 'offline', scopeId: 'p_test', scopeKind: 'project',
    source: 'test', kind: 'procedure', text: 'Recover the dense index after an outage.', version: 'v1',
    contentHash: '0'.repeat(64), observedAt: '2026-01-01T00:00:00.000Z', trust: 'host', metadata: {} });
  assert.deepEqual(indexed, { chunks: 1, embedded: 0, dense: false });
  provider.available = true;
  assert.equal(await engine.vector.backfill(), 1);
  assert.equal(await engine.vector.backfill(), 0);
  assert.equal((await engine.search({ queries: ['recover'], dense: true })).items[0]?.id, 'offline');
});

test('authority commit is lexically searchable without vector upsert', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-lex-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  const fact = engine.composeFact({
    kind: 'decision', statement: 'Use north-datastore-alpha for durable project storage',
    entities: [], evidence: [], episodeIds: [], confidence: 0.9, tags: {},
  });
  engine.authorityTransaction(() => engine.commitAuthority({ type: 'fact.recorded', fact }));
  assert.equal(engine.projection.getFact(fact.id)?.statement, fact.statement);
  assert.ok(engine.projection.chunkCount({ namespace: 'memory', externalId: fact.id }) >= 1);
  const found = await engine.search({ queries: ['north-datastore-alpha'], dense: false });
  assert.equal(found.items[0]?.id, fact.id);
});

test('rebuild is locked and still reconstructs the projection', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-rebuild-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  await engine.recordFact({ kind: 'decision', statement: 'Use SQLite for memory', entities: [], evidence: [], episodeIds: [], confidence: 0.8, tags: {} });
  await writeFile(join(root, 'rebuild.lock'), '{}');
  await assert.rejects(engine.rebuild(), /already running/);
  await rm(join(root, 'rebuild.lock'), { force: true });
  const rebuilt = await engine.rebuild();
  assert.equal(rebuilt.documents, 1);
});

test('indexed replay repairs an event committed before its projection and remains idempotent', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-engine-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider: new TestEmbeddingProvider(), storage: 'indexed' });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  await engine.journal.append({ type: 'document.deleted', namespace: 'docs', externalId: 'missing', reason: 'crash-window' });
  assert.equal((await engine.replay()).events, 1);
  assert.equal((await engine.replay()).events, 0);
});

test('compact journal append and projection application are one durable operation', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-compact-engine-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  await engine.journal.append({ type: 'document.deleted', namespace: 'docs', externalId: 'missing', reason: 'atomic' });
  assert.equal((await engine.replay()).events, 0);
  assert.equal((await engine.journal.readAll()).length, 1);
});

test('capacity promotion atomically preserves history, jobs, checkpoints, vectors and retrieval', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-promotion-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 'compact', provider: new TestEmbeddingProvider() });
  t.after(() => rm(root, { recursive: true, force: true }));
  engine.projection.upsertOperationalCheckpoint('p_test', 'handoff', '{"goal":"preserve"}', 1);
  const identity = { adapter: 'docs', documentKey: 'docs\u0000one', namespace: 'docs', externalId: 'one', revision: 'r1',
    contentHash: sha256('r1'), kind: 'document', observedAt: '2026-01-01T00:00:00.000Z', trust: 'host' as const, metadata: {} };
  engine.curation.upsertFingerprint(identity, 1);
  engine.curation.writeTopic({ id: 'topic_preserve', scopeId: 'p_test', title: 'Preserve', summary: 'Promotion state', factIds: [] }, 0, 1);
  engine.curation.setCoverage(identity.documentKey, 10, 20, 1);
  engine.curation.saveWindowDraft(identity.documentKey, identity.revision, 0, [], ['keep qualification'], 'draft');
  engine.curation.linkFact('mem_future', identity);
  engine.curation.recordSpend({ calls: 2, inputTokens: 3, outputTokens: 4, embeddingCalls: 5 }, Date.parse('2026-01-01T00:00:00.000Z'));
  engine.curation.enqueue({ id: 'job_batch', scopeId: 'p_test', adapter: 'docs', documentKey: identity.documentKey,
    action: 'analyze', inputRevision: 'r1', contentHash: identity.contentHash });
  const claimed = engine.curation.claim('owner', 60_000)!;
  const accepted = engine.curation.acceptBatch(claimed, identity, 'topic_preserve', 1, { factIds: [] }, 'owner')!;
  engine.curation.finish(claimed.id, 'owner', 'no_change', undefined, Date.now(), true);
  engine.curation.enqueue({ id: 'job_preserve', scopeId: 'p_test', adapter: 'docs', documentKey: identity.documentKey,
    action: 'review', inputRevision: 'r1', contentHash: identity.contentHash }, 3);
  const observer = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 'observer', provider: new TestEmbeddingProvider() });
  assert.equal(observer.storageMode, 'compact');
  for (const index of Array.from({ length: 40 }, (_, value) => value)) {
    const entropy = Array.from({ length: 40 }, (_, value) => sha256(`${index}:${value}`)).join(' ');
    await engine.recordFact({ id: `mem_promote_${index}`, kind: 'fact', statement: `Promotion fact ${index}: ${entropy}`,
      entities: [], evidence: [], episodeIds: [], confidence: 0.8, tags: {} });
    if (engine.storageMode === 'indexed') break;
  }
  assert.equal(engine.storageMode, 'indexed');
  assert.equal(engine.curation.getJob('job_preserve')?.status, 'pending');
  assert.equal(engine.curation.fingerprint(identity.documentKey)?.revision, 'r1');
  assert.equal(engine.curation.watermark('docs', identity.documentKey)?.outcome, 'no_change');
  assert.equal(engine.curation.topic('topic_preserve')?.summary, 'Promotion state');
  assert.deepEqual(engine.curation.coverage(identity.documentKey), { offset: 10, total: 20 });
  assert.equal(engine.curation.windowDrafts(identity.documentKey, 'r1')[0]?.qualifications[0], 'keep qualification');
  assert.deepEqual(engine.curation.dependents(identity.documentKey), ['mem_future']);
  assert.equal(engine.curation.acceptedBatch('job_batch')?.id, accepted);
  assert.equal(engine.curation.spend(Date.parse('2026-01-01T00:00:00.000Z')).embeddingCalls, 5);
  assert.equal(engine.projection.operationalCheckpoint('p_test', 'handoff'), '{"goal":"preserve"}');
  assert.ok(engine.projection.stats().facts > 0);
  assert.ok(engine.projection.stats().vectors > 0);
  assert.equal((await engine.search({ queries: ['Promotion fact 0'], dense: false })).items[0]?.id, 'mem_promote_0');
  const history = await engine.journal.readAll();
  assert.ok(history.some(event => event.payload.type === 'fact.recorded' && event.payload.fact.id === 'mem_promote_0'));
  await observer.recordFact({ id: 'mem_observer', kind: 'fact', statement: 'An already-open client reroutes after promotion.',
    entities: [], evidence: [], episodeIds: [], confidence: 0.8, tags: {} });
  assert.equal(observer.storageMode, 'indexed');
  assert.equal(engine.projection.getFact('mem_observer')?.statement, 'An already-open client reroutes after promotion.');
  const reopened = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 'indexed', provider: new TestEmbeddingProvider() });
  assert.equal(reopened.storageMode, 'indexed');
  assert.equal(reopened.curation.getJob('job_preserve')?.status, 'pending');
  assert.equal(reopened.curation.acceptedBatch('job_batch')?.id, accepted);
  assert.equal(reopened.curation.watermark('docs', identity.documentKey)?.revision, 'r1');
  assert.equal(reopened.curation.windowDrafts(identity.documentKey, 'r1')[0]?.summary, 'draft');
  assert.equal(reopened.projection.operationalCheckpoint('p_test', 'handoff'), '{"goal":"preserve"}');
  await reopened.dispose();
  await observer.dispose();
  await engine.dispose();
});

test('a search embeds its queries in one encoder pass, and a query already embedded is not embedded again', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-engine-'));
  const inner = new TestEmbeddingProvider();
  const calls: string[][] = [];
  const provider: EmbeddingProvider = {
    model: inner.model, isLocal: true,
    embed: async (texts, options) => { if (options?.inputType === 'query') calls.push([...texts]); return inner.embed(texts); },
  };
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  await engine.recordFact({ kind: 'decision', statement: 'Use SQLite for durable project storage', entities: [], evidence: [],
    episodeIds: [], confidence: 0.9, validAt: '2026-01-01T00:00:00.000Z', tags: {} });
  const first = await engine.search({ queries: ['database persistence', 'storage engine', 'durable data'], dense: true, maxBytes: 4096 });
  assert.equal(calls.length, 1, `one pass for every query of the search: ${JSON.stringify(calls)}`);
  assert.deepEqual([...calls[0]!].sort(), ['database persistence', 'durable data', 'storage engine']);
  const again = await engine.search({ queries: ['database persistence', 'storage engine', 'durable data'], dense: true, maxBytes: 4096 });
  assert.equal(calls.length, 1, 'the same queries again cost no encoder pass');
  assert.deepEqual(again.items.map(item => item.id), first.items.map(item => item.id), 'and find the same memory');
});
