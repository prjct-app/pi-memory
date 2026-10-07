import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { MemoryEngine } from '../src/engine.ts';
import { installMemoryTools } from '../src/extension/tools.ts';
import { runHygiene } from '../src/retention/maintenance.ts';
import { TestEmbeddingProvider } from './helpers.ts';

const cases = [
  ['No JEV', 'No uses clasificadores auxiliares.'],
  ['sin traducciones', 'Preserve the original language.'],
  ['日本語で答えて', 'Use Japanese for responses.'],
  ['لا تحذف السياق', 'Preserve conversation context.'],
  ['NEVER REMOVE THE ORIGINAL USER CONSTRAINTS', 'Keep the requirements throughout the session.'],
];

for (const [quote, statement] of cases) test(`the active model can remember a real instruction: ${quote}`, async t => {
  const root = await mkdtemp(join(tmpdir(), 'memory-model-owner-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_owner', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  const tools = new Map<string, any>();
  installMemoryTools({ registerTool(tool: any) { tools.set(tool.name, tool); } } as ExtensionAPI, {
    engine: async () => engine, writableEngine: async () => engine, readable: async () => [engine],
    search: request => engine.search(request), currentPrompt: () => quote!, stagedEvidence: () => new Map(),
  });
  const output = await tools.get('memory_record').execute('remember', { action: 'remember', kind: 'constraint', statement, userQuote: quote });
  assert.equal(output.details.status, 'ok');
  const fact = engine.projection.getFact(output.details.id)!;
  assert.equal(fact.standing, 'supported');
  assert.equal(fact.evidence[0]?.excerpt, quote);
  assert.equal(fact.statement, statement);
});

test('automatic hygiene never deletes short, uppercase, or nearly matching user constraints', async t => {
  const root = await mkdtemp(join(tmpdir(), 'memory-retain-rules-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_owner', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  const statements = ['No JEV', 'NEVER REMOVE THE ORIGINAL USER CONSTRAINTS', 'Use npm for publishing every package.', 'Use pnpm for publishing every package.'];
  const ids = [];
  for (const [index, statement] of statements.entries()) {
    const { fact } = await engine.recordFact({ kind: 'constraint', statement, entities: [], episodeIds: [], confidence: 1,
      evidence: [{ id: `ev_original_${index}`, origin: 'user_statement', provenance: 'declared', contentHash: '0'.repeat(64), excerpt: statement, observedAt: new Date().toISOString() }], tags: {} });
    ids.push(fact.id);
  }
  const result = await runHygiene(engine);
  assert.equal(result.retired + result.superseded, 0);
  for (const [index, id] of ids.entries()) assert.equal(engine.projection.getFact(id)?.statement, statements[index]);
});
