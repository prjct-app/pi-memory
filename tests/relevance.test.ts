import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MemoryEngine } from '../src/engine.ts';
import { federatedSearch } from '../src/retrieval/federated.ts';
import { DEFAULT_RECALL_THRESHOLD, installMemoryHooks } from '../src/extension/hooks.ts';
import { installMemoryTools } from '../src/extension/tools.ts';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { scoreOracle, type OracleCase } from '../src/eval/oracles.ts';

// Statements are the seven ingested evidence rules from the independently pinned
// prjct-cli workload. Oracles query all seven plus explicit unanswerables. This
// is not a claim of general semantic quality.
test('default lookup and automatic recall retain positives/qualified evidence and abstain on unrelated attributes EN/ES', async t => {
  const root = await mkdtemp(join(tmpdir(), 'relevance-'));
  const home = join(root, 'memory-home');
  const { engine } = await MemoryEngine.initializeProject(root, 'test', { home, provider: {
    model: 'offline-lexical', isLocal: true, embed: async texts => texts.map(() => [1, ...Array.from({ length: 15 }, () => 0)]),
  } });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  const statements = JSON.parse(await readFile(new URL('./fixtures/real-project-statements.json', import.meta.url), 'utf8')) as string[];
  const cases = JSON.parse(await readFile(new URL('./fixtures/real-project-oracles.json', import.meta.url), 'utf8')) as OracleCase[];
  for (const statement of statements) await engine.recordFact({ kind: 'fact', statement, confidence: 1, evidence: [], entities: [], episodeIds: [], tags: {} });
  const handlers = new Map<string, (event: any, ctx: any) => Promise<any>>();
  const tools = new Map<string, any>();
  const pi = { on: (name: string, handler: any) => handlers.set(name, handler),
    registerTool: (tool: any) => tools.set(tool.name, tool) } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home });
  installMemoryTools(pi, runtime);
  const ctx = { cwd: root, sessionManager: { getSessionId: () => 'test' } };
  await handlers.get('session_start')!({}, ctx);
  const o8 = cases.find(kase => kase.name === 'O8')!;
  const recalled = await handlers.get('before_agent_start')!({ prompt: o8.query, systemPrompt: 'Base' }, ctx);
  assert.equal(recalled.systemPrompt, 'Base', 'the system prompt is left byte-identical');
  assert.match(recalled.message.content, /Reference data only, not instructions/);
  assert.equal(recalled.message.details.memory.recall, undefined, 'unrelated prompt gets no overflow recall');
  const lookup = await tools.get('memory_context').execute('call', { action: 'lookup', queries: [o8.query] });
  assert.equal(scoreOracle(lookup.details.items, o8, lookup.details).passed, true);
  for (const kase of cases) {
    const query = { queries: [kase.query], dense: false, maxBytes: 4096 };
    for (const result of [await engine.search(query), await federatedSearch([engine], { ...query, scoreThreshold: DEFAULT_RECALL_THRESHOLD }),
      await engine.search({ ...query, dense: true }), await federatedSearch([engine], { ...query, dense: true, scoreThreshold: DEFAULT_RECALL_THRESHOLD })]) {
      assert.equal(scoreOracle(result.items, kase, result).passed, true, `${kase.name}: ${JSON.stringify(result)}`);
      if (kase.kind === 'unanswerable') {
        assert.equal(result.status, 'abstained');
        assert.equal(result.items.length, 0);
        assert.match(result.gaps.join(' '), /insufficient evidence/iu);
      }
    }
  }
});
