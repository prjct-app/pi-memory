import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { installMemoryHooks } from '../src/extension/hooks.ts';
import { judgeAutoCaptures } from '../src/retention/intake.ts';

const scripted = (scores: Record<string, number>) => {
  const asked: { state: Record<string, unknown>; questions: Readonly<Record<string, string>> }[] = [];
  const ask = async (state: Record<string, unknown>, questions: Readonly<Record<string, string>>) => {
    asked.push({ state, questions });
    const statements = state.statements as Record<string, string>;
    return new Map(Object.keys(questions).map(key => {
      const text = statements[key.replace(/_durable$/, '')] ?? '';
      const hit = Object.entries(scores).find(([needle]) => text.includes(needle));
      return [key, hit ? hit[1] : 0] as const;
    }));
  };
  return { ask, asked };
};

test('raw output is refused without asking; Jev keeps lessons and drops the state of one run in one request', async () => {
  const jev = scripted({ 'pnpm install': 0.9, '4 of 76': 0.1 });
  const verdicts = await judgeAutoCaptures([
    'bash: ENOENT: no such file or directory, open src/route.ts',
    'Installing with npm breaks the lockfile; this workspace must use pnpm install.',
    'The test run failed 4 of 76 tests across 2 files.',
  ], jev.ask);
  assert.deepEqual(verdicts.map(verdict => verdict.keep), [false, true, false]);
  assert.equal(verdicts[0]?.reason, 'raw tool output');
  assert.equal(jev.asked.length, 1, 'one request for the batch');
  assert.equal(Object.keys(jev.asked[0]!.questions).length, 2, 'raw output never reaches Jev');
});

test('without Jev, or when it fails, nothing auto-derived is stored', async () => {
  assert.deepEqual((await judgeAutoCaptures(['A real lesson about pnpm install and the lockfile.'], undefined)).map(v => v.keep), [false]);
  const failing = async () => { throw new Error('timeout'); };
  assert.deepEqual((await judgeAutoCaptures(['A real lesson about pnpm install and the lockfile.'], failing)).map(v => v.reason), ['Jev unavailable']);
});

test('automatic failure observations do not ask Jev or promote unreviewed rules', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-memory-intake-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const jev = scripted({ 'Vitest 4': 0.1, 'pnpm': 0.92 });
  const handlers = new Map<string, (event: any, context: any) => Promise<any>>();
  const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home, rerank: { rerank: { ask: jev.ask } as never } });
  const ctx = { cwd, sessionManager: { getSessionId: () => 'intake' }, getContextUsage: () => ({ tokens: 0 }) } as any;
  await handlers.get('session_start')!({}, ctx);
  await runtime.initialize();
  const fail = async (id: string, text: string) => handlers.get('tool_result')!({ toolName: 'migrate', toolCallId: id, isError: true,
    content: [{ type: 'text', text }] }, ctx);
  await fail('a', 'Error: Vitest 4 run failed because ChatSessionCache.validateSession returned false for an ongoing session in the cache test');
  await fail('b', 'Error: npm install failed with ERESOLVE because this workspace requires pnpm; running npm corrupts the lockfile');
  await handlers.get('turn_end')!({}, ctx);
  await runtime.flushed();
  const engine = await runtime.engine();
  const failures = engine.projection.activeFacts(engine.scopeId, 20).filter(fact => fact.kind === 'failure').map(fact => fact.statement);
  assert.ok(!failures.some(text => /Vitest 4/.test(text)), `one run's state was stored: ${failures.join(' | ')}`);
  assert.equal(jev.asked.length, 0, 'automatic hooks do not call a classifier');
  assert.deepEqual(failures, [], 'the active model owns promotion to durable memory');
});
