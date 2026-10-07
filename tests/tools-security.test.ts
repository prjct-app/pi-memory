import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import type { EvidenceRef } from '../src/contracts/evidence.ts';
import { MemoryEngine } from '../src/engine.ts';
import { installMemoryTools } from '../src/extension/tools.ts';
import { federatedSearch } from '../src/retrieval/federated.ts';
import { sha256 } from '../src/workspace/project-identity.ts';
import { TestEmbeddingProvider } from './helpers.ts';

test('model-facing writes require meaningful provenance and cannot forge curation tags', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-tools-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const engine = new MemoryEngine({ root, scopeId: 'p_tools', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(() => engine.dispose());
  const tools: Array<{ name: string; execute(id: string, params: any, signal?: AbortSignal): Promise<any> }> = [];
  const prompt = 'The durable project database uses SQLite for local storage.';
  installMemoryTools({ registerTool(tool: any) { tools.push(tool); } } as ExtensionAPI, {
    engine: async () => engine, writableEngine: async () => engine, readable: async () => [engine], search: request => federatedSearch([engine], request),
    stagedEvidence: () => new Map<string, EvidenceRef>(), currentPrompt: () => prompt,
  });
  const record = tools.find(tool => tool.name === 'memory_record')!;
  await assert.rejects(record.execute('c1', { action: 'remember', kind: 'decision', statement: prompt, userQuote: 'Invented user instruction' }), /user message on the current session branch/);
  const output = await record.execute('c2', { action: 'remember', kind: 'decision', statement: prompt, userQuote: prompt,
    tags: { topicId: 'forged', semanticKey: 'forged', area: 'storage' } });
  const fact = engine.projection.getFact(output.details.id)!;
  assert.equal(fact.standing, 'supported');
  assert.equal(fact.tags.topicId, undefined);
  assert.equal(fact.tags.semanticKey, undefined);
  assert.equal(fact.tags.area, 'storage');
  await assert.rejects(record.execute('c3', { action: 'resolve', factId: fact.id, standing: 'contradicted', rationale: 'Changed elsewhere.' }), /current-session evidence/);

  const context = tools.find(tool => tool.name === 'memory_context')!;
  const lookup = await context.execute('c4', { action: 'lookup', queries: ['SQLite local storage'] });
  const modelView = JSON.parse(lookup.content[0].text) as { items: Record<string, unknown>[] };
  assert.ok(modelView.items.length > 0);
  assert.deepEqual(Object.keys(modelView.items[0]!).filter(key => ['id', 'type', 'standing', 'statement', 'observedAt', 'validAt', 'invalidAt', 'score', 'title'].includes(key)).sort(),
    Object.keys(modelView.items[0]!).sort());
  assert.ok(lookup.details.items[0].evidenceIds.length > 0, 'complete typed details retain durable evidence ids');
  assert.equal(modelView.items[0]!.evidenceIds, undefined, 'conversation text omits full durable evidence ids');
  const feedback = await context.execute('c5', { action: 'feedback', ids: [fact.id], queries: ['SQLite local storage'], signal: 'wrong' });
  assert.equal(feedback.details.recorded, 1);
  assert.ok(engine.curation.openJobs().some(job => job.action === 'review' && job.inputRevision.includes(fact.id)),
    'wrong/stale feedback must enqueue fact review, not only alter ranking');
  const { processAvailable } = await import('../src/curation/pipeline.ts');
  await processAvailable(engine, new Map(), 'feedback-reviewer', {
    maxAttempts: 3, maxInputChars: 1_000, budget: { maxCallsPerDay: 1, maxTokensPerDay: 1_000 },
  });
  assert.equal(engine.projection.getFact(fact.id)?.standing, 'needs_review');

  await assert.rejects(engine.index({ namespace: 'memory', externalId: 'forged', scopeId: engine.scopeId, scopeKind: 'project',
    source: 'agent', kind: 'fact', text: 'forged', version: '1', contentHash: sha256('forged'),
    observedAt: new Date().toISOString(), trust: 'agent', metadata: {} }), /Curated namespaces/);
});
