import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createPanel } from '@prjct.app/pi-tui-kit';
import { memoryPanelSpec, type MemoryOps, type MemorySnapshot } from '../src/extension/memory-panel.ts';

const theme: any = { fg: (_tone: string, text: string) => text, bold: (text: string) => text };
const tick = () => new Promise(resolve => setImmediate(resolve));
const ready: MemorySnapshot = {
  initialized: true, ready: true, project: 'p_1', scope: 'project/p_1',
  stats: { documents: 4, chunks: 9, vectors: 9, facts: 12, events: 30, bytes: 2 * 1024 * 1024 },
  curation: { fingerprints: 4, pending: 2, claimed: 1, failed: 0, blocked: 0, published: 8, spend: {} as any },
  sources: [
    { adapter: 'pi-session', due: true, reason: '3 new session events', last: { adapter: 'pi-session', lastAt: new Date(Date.now() - 3_600_000).toISOString(), at: {} as any, discovered: 5, indexed: 2, ok: true } },
    { adapter: 'prjct-observations', due: false, reason: 'synced recently', last: { adapter: 'prjct-observations', lastAt: new Date().toISOString(), at: {} as any, discovered: 1, indexed: 0, ok: false, detail: 'prjct not found' } },
  ],
};

function open(snapshot: MemorySnapshot, overrides: Partial<MemoryOps> = {}) {
  const calls: string[] = [];
  const ops: MemoryOps = {
    load: async () => snapshot,
    init: async () => { calls.push('init'); return 'Initialized · project p_1'; },
    sync: async adapter => { calls.push(`sync ${adapter ?? 'all'}`); return `Synced ${adapter ?? 'all sources'} · 5 seen · 2 new · 0 queued`; },
    gc: async () => 'GC · removed 0',
    checkpointWal: async () => 'WAL checkpoint · ok',
    rebuild: async () => { calls.push('rebuild'); return 'Rebuilt · ok'; },
    ...overrides,
  };
  const panel = createPanel(memoryPanelSpec(ops, snapshot), { terminal: { columns: 120, rows: 30 }, requestRender() {} } as any, theme, () => undefined);
  return {
    calls,
    screen: () => panel.render(120).join('\n'),
    press: async (...keys: string[]) => { for (const key of keys) { panel.handleInput!(key); await tick(); await tick(); } },
  };
}

test('the store and each source are rows, with the facts beside them', () => {
  const view = open(ready).screen();
  assert.match(view, /Memory {2}project\/p_1 · 2 sources · 1 due/);
  assert.match(view, /● store\s+12 facts · 3 queued · 2\.0MiB/);
  assert.match(view, /○ pi-session\s+due/);
  assert.match(view, /✕ prjct-observations\s+sync failed/);
  assert.match(view, /curation\s+3 queued · 8 published/);
});

test('sync acts on the selected source and leaves a trace', async () => {
  const view = open(ready);
  await view.press('\x1b[B', 's');
  assert.deepEqual(view.calls, ['sync pi-session']);
  assert.match(view.screen(), /History\s*\n.*now {2}Synced pi-session/);
});

test('rebuild asks again; an uninitialized project offers only initialize', async () => {
  const view = open(ready);
  await view.press('R');
  assert.deepEqual(view.calls, []);
  await view.press('R');
  assert.deepEqual(view.calls, ['rebuild']);
  const fresh = open({ initialized: false, ready: false, sources: [] });
  assert.match(fresh.screen(), /! memory not initialized/);
  assert.match(fresh.screen(), /i Initialize · \? keys/);
});

