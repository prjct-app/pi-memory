import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { installMemoryHooks } from '../src/extension/hooks.ts';

test('automatic failure observations do not promote unreviewed rules', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-memory-intake-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const handlers = new Map<string, (event: any, context: any) => Promise<any>>();
  const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home});
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
  assert.deepEqual(failures, [], 'the active model owns promotion to durable memory');
});
