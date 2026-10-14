import assert from 'node:assert/strict';
import { appendFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { installMemory } from '../src/index.ts';
import { backfill, indexFile } from '../src/history/indexer.ts';
import { projectRoot } from '../src/history/project.ts';
import { boundary, runHistory } from '../src/history/query.ts';
import { openHistoryStore } from '../src/history/store.ts';

const line = (value: unknown): string => `${JSON.stringify(value)}\n`;
const header = (id: string, cwd: string, at = '2026-10-01T10:00:00.000Z') => line({ type: 'session', version: 3, id, timestamp: at, cwd });
const user = (id: string, text: string, at: string) => line({ type: 'message', id, parentId: null, timestamp: at, message: { role: 'user', content: text } });
const assistant = (id: string, text: string, at: string) => line({ type: 'message', id, parentId: null, timestamp: at,
  message: { role: 'assistant', content: [{ type: 'thinking', thinking: '', thinkingSignature: 'ENCRYPTED' }, { type: 'text', text },
    { type: 'toolCall', id: 'c1', name: 'bash', arguments: { command: 'git status' } }] } });
const toolResult = (id: string, text: string, at: string) => line({ type: 'message', id, parentId: null, timestamp: at,
  message: { role: 'toolResult', toolCallId: 'c1', toolName: 'bash', content: [{ type: 'text', text }] } });
const compaction = (id: string, summary: string, at: string, tokensBefore = 232_000) =>
  line({ type: 'compaction', id, parentId: null, timestamp: at, summary, firstKeptEntryId: 'x', tokensBefore });

const fixture = async (t: { after: (fn: () => Promise<void>) => void }) => {
  const root = await mkdtemp(join(tmpdir(), 'pi-history-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const home = join(root, 'home');
  const sessions = join(root, 'sessions');
  const cwd = { be: join(home, 'Apps/prjct/prjct-be'), fe: join(home, 'Apps/prjct/prjct-fe'), lead: join(home, 'Apps/prjct'),
    other: join(home, 'Apps/fluenty'), apps: join(home, 'Apps') };
  const journal = async (folder: string, name: string, body: string): Promise<string> => {
    await mkdir(join(sessions, folder), { recursive: true });
    const path = join(sessions, folder, name);
    await writeFile(path, body);
    return path;
  };
  const store = openHistoryStore(join(root, 'memory', 'history.sqlite'));
  t.after(async () => store.close());
  return { root, home, sessions, cwd, journal, store };
};

test('indexes complete lines only, incrementally, and re-indexes a journal that was rewritten', async t => {
  const f = await fixture(t);
  const path = await f.journal('be', 's1.jsonl', header('s1', f.cwd.be) + user('u1', 'migrar el widget a la API', '2026-10-01T10:01:00.000Z'));
  assert.deepEqual(await indexFile(f.store, path), { added: 1, bytes: Buffer.byteLength(header('s1', f.cwd.be) + user('u1', 'migrar el widget a la API', '2026-10-01T10:01:00.000Z')) });
  const partial = assistant('a1', 'Listo, el widget usa la API nueva.', '2026-10-01T10:02:00.000Z');
  await appendFile(path, partial.slice(0, 40));
  assert.equal((await indexFile(f.store, path)).added, 0, 'a line Pi is still writing waits');
  await appendFile(path, partial.slice(40));
  assert.equal((await indexFile(f.store, path)).added, 1);
  assert.equal((await indexFile(f.store, path)).added, 0, 'idempotent');
  assert.equal(f.store.status().entries, 2);
  await writeFile(path, header('s1', f.cwd.be) + user('u9', 'otro contenido', '2026-10-01T11:00:00.000Z'));
  assert.equal((await indexFile(f.store, path)).added, 1, 'a shorter journal is indexed again from the start');
  assert.equal(f.store.status().entries, 1);
  assert.equal(f.store.search('widget', {}, 10).total, 0, 'old terms are gone with the old rows');
});

test('reasoning is not indexed; tool calls and results are; secrets never enter the index or the output', async t => {
  const f = await fixture(t);
  const path = await f.journal('be', 's1.jsonl', header('s1', f.cwd.be)
    + assistant('a1', 'Reviso el repo', '2026-10-01T10:01:00.000Z')
    + toolResult('t1', 'export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nOn branch main', '2026-10-01T10:02:00.000Z'));
  await indexFile(f.store, path);
  const context = { sessionId: 's1', cwd: f.cwd.be, home: f.home };
  assert.equal(f.store.search('ENCRYPTED', {}, 10).total, 0);
  assert.equal(f.store.search('AKIAIOSFODNN7EXAMPLE', {}, 10).total, 0, 'the secret is not a search term');
  const found = await runHistory(f.store, { action: 'search', query: 'branch main' }, context) as any;
  assert.equal(found.entries[0].entryId, 't1');
  assert.equal(found.entries[0].name, 'bash');
  const read = await runHistory(f.store, { action: 'read', entryId: 't1' }, context) as any;
  assert.doesNotMatch(read.text, /AKIAIOSFODNN7EXAMPLE/u);
  assert.match(read.text, /On branch main/u);
  assert.equal((await runHistory(f.store, { action: 'search', query: 'git status' }, context) as any).entries[0].entryId, 'a1', 'tool call arguments are searchable');
});

test('the default scope is the project: sibling checkouts and the parent folder, not other projects', async t => {
  const f = await fixture(t);
  await f.journal('be', 'be.jsonl', header('be', f.cwd.be) + user('b1', 'regla OTP: nunca imprimir códigos', '2026-10-02T10:00:00.000Z'));
  await f.journal('fe', 'fe.jsonl', header('fe', f.cwd.fe) + user('f1', 'pantalla de códigos OTP', '2026-10-03T10:00:00.000Z'));
  await f.journal('lead', 'lead.jsonl', header('lead', f.cwd.lead) + user('l1', 'el lead decide el OTP', '2026-10-04T10:00:00.000Z'));
  await f.journal('other', 'other.jsonl', header('other', f.cwd.other) + user('o1', 'OTP de otro cliente', '2026-10-05T10:00:00.000Z'));
  await f.journal('apps', 'apps.jsonl', header('apps', f.cwd.apps) + user('x1', 'OTP en la carpeta Apps', '2026-10-05T11:00:00.000Z'));
  assert.equal((await backfill(f.store, f.sessions)).added, 5);
  assert.equal(projectRoot(f.cwd.be, f.store.cwds(), f.home), f.cwd.lead);
  assert.equal(projectRoot(f.cwd.apps, f.store.cwds(), f.home), f.cwd.apps, 'a direct child of home is its own project only');
  assert.equal(projectRoot('/private/tmp/worktree', f.store.cwds(), f.home), '/private/tmp/worktree');
  const context = { sessionId: 'be', cwd: f.cwd.be, home: f.home };
  const project = await runHistory(f.store, { action: 'search', query: 'otp' }, context) as any;
  assert.deepEqual(project.entries.map((entry: any) => entry.entryId).sort(), ['b1', 'f1', 'l1']);
  assert.equal(project.project, f.cwd.lead);
  const session = await runHistory(f.store, { action: 'search', query: 'otp', scope: 'session' }, context) as any;
  assert.deepEqual(session.entries.map((entry: any) => entry.entryId), ['b1']);
  assert.equal(session.entries[0].currentSession, true);
  const global = await runHistory(f.store, { action: 'search', query: 'otp', scope: 'global' }, context) as any;
  assert.equal(global.totalMatches, 5);
  assert.equal((await runHistory(f.store, { action: 'search', query: 'codigos' }, context) as any).totalMatches, 2, 'accents do not matter');
});

test('dates filter by the local calendar day; newer entries rank first among equally relevant ones', async t => {
  const f = await fixture(t);
  const day = (d: number, h = 12) => new Date(2026, 9, d, h).toISOString();
  await f.journal('be', 's.jsonl', header('s', f.cwd.be) + user('d1', 'deploy railway', day(1)) + user('d5', 'deploy railway', day(5))
    + user('d9', 'deploy railway', day(9)));
  await backfill(f.store, f.sessions);
  const context = { sessionId: 's', cwd: f.cwd.be, home: f.home };
  const ranked = await runHistory(f.store, { action: 'search', query: 'deploy railway' }, context, Date.parse(day(10))) as any;
  assert.deepEqual(ranked.entries.map((entry: any) => entry.entryId), ['d9', 'd5', 'd1']);
  const windowed = await runHistory(f.store, { action: 'search', query: 'deploy', since: '2026-10-05', until: '2026-10-05' }, context) as any;
  assert.deepEqual(windowed.entries.map((entry: any) => entry.entryId), ['d5']);
  assert.equal(boundary('2026-10-05', true), new Date(2026, 9, 5, 23, 59, 59, 999).toISOString());
  await assert.rejects(runHistory(f.store, { action: 'search', query: 'x', since: 'ayer' }, context), /since must be a date/u);
  const listed = await runHistory(f.store, { action: 'search', since: '2026-10-09' }, context) as any;
  assert.deepEqual(listed.entries.map((entry: any) => entry.entryId), ['d9'], 'no query lists entries newest first');
});

test('terms are matched all together first, then any of them, and the result says so', async t => {
  const f = await fixture(t);
  await f.journal('be', 's.jsonl', header('s', f.cwd.be) + user('u1', 'volumeBackup falló en staging', '2026-10-01T10:00:00.000Z'));
  await backfill(f.store, f.sessions);
  const context = { sessionId: 's', cwd: f.cwd.be, home: f.home };
  const all = await runHistory(f.store, { action: 'search', query: 'volumeBackup staging' }, context) as any;
  assert.equal(all.match, 'all');
  const any = await runHistory(f.store, { action: 'search', query: 'volumeBackup producción' }, context) as any;
  assert.equal(any.match, 'any');
  assert.match(any.hint, /some of them/u);
  const operators = await runHistory(f.store, { action: 'search', query: 'variableDelete( OR NEAR "x' }, context) as any;
  assert.equal(operators.status, 'empty', 'FTS syntax in a query is literal text, never an error');
});

test('summaries lists past compactions; read returns them whole, pages long text and resolves shared ids', async t => {
  const f = await fixture(t);
  const long = `## Goal\n${'detalle '.repeat(4_000)}`;
  await f.journal('be', 'a.jsonl', header('a', f.cwd.be) + compaction('k1', '## Goal\nmigrar Supabase', '2026-10-01T10:00:00.000Z', 200_000)
    + compaction('k2', long, '2026-10-02T10:00:00.000Z') + user('dup', 'mensaje en a', '2026-10-02T11:00:00.000Z'));
  await f.journal('fe', 'b.jsonl', header('b', f.cwd.fe) + user('dup', 'mensaje en b', '2026-10-03T11:00:00.000Z'));
  await backfill(f.store, f.sessions);
  const context = { sessionId: 'a', cwd: f.cwd.be, home: f.home };
  const summaries = await runHistory(f.store, { action: 'summaries' }, context) as any;
  assert.deepEqual(summaries.entries.map((entry: any) => [entry.entryId, entry.tokensBefore]), [['k2', 232_000], ['k1', 200_000]]);
  const first = await runHistory(f.store, { action: 'read', entryId: 'k2' }, context) as any;
  assert.equal(first.text.length, 16_000);
  assert.equal(first.totalCharacters, long.length);
  const second = await runHistory(f.store, { action: 'read', entryId: 'k2', offset: first.nextOffset }, context) as any;
  const third = await runHistory(f.store, { action: 'read', entryId: 'k2', offset: second.nextOffset }, context) as any;
  assert.equal(first.text + second.text + third.text, long, 'pages follow nextOffset to the end');
  assert.equal(third.nextOffset, null);
  assert.equal((await runHistory(f.store, { action: 'read', entryId: 'dup' }, context) as any).text, 'mensaje en a', 'the current session wins');
  const elsewhere = await runHistory(f.store, { action: 'read', entryId: 'dup' }, { ...context, sessionId: 'z' }) as any;
  assert.equal(elsewhere.status, 'ambiguous');
  assert.equal((await runHistory(f.store, { action: 'read', entryId: 'dup', sessionId: 'b' }, context) as any).text, 'mensaje en b');
  assert.equal((await runHistory(f.store, { action: 'read', entryId: 'nope' }, context) as any).status, 'not_found');
});

test('one process scans other journals at a time; the lease expires', async t => {
  const f = await fixture(t);
  assert.equal(f.store.lease('a', 1_000, 0), true);
  assert.equal(f.store.lease('b', 1_000, 500), false);
  assert.equal(f.store.lease('a', 1_000, 600), true, 'the holder renews');
  assert.equal(f.store.lease('b', 1_000, 2_000), true, 'an expired lease is taken over');
  f.store.release('b');
  assert.equal(f.store.lease('c', 1_000, 2_100), true);
});

test('the extension adds one tool and one command, and nothing to the context', async t => {
  const f = await fixture(t);
  const path = await f.journal('be', 'live.jsonl', header('live', f.cwd.be) + user('u1', 'no despliegues sin autorización', '2026-10-01T10:00:00.000Z'));
  const registered: { tools: any[]; commands: string[]; events: string[] } = { tools: [], commands: [], events: [] };
  const pi = {
    on(event: string) { registered.events.push(event); },
    registerTool(tool: any) { registered.tools.push(tool); },
    registerCommand(name: string) { registered.commands.push(name); },
  } as unknown as ExtensionAPI;
  installMemory(pi, { home: join(f.root, 'ext'), sessionsRoot: f.sessions, backfillDelayMs: 60_000 });
  assert.deepEqual(registered.tools.map(tool => tool.name), ['memory_context']);
  assert.deepEqual(registered.commands, ['memory']);
  for (const event of ['context', 'before_agent_start', 'before_provider_request', 'agent_end', 'turn_end']) {
    assert.equal(registered.events.includes(event), false, `${event} is not hooked`);
  }
  assert.ok(registered.tools[0].description.length < 200, 'the description stays short; it rides every request');
  const ctx = { cwd: f.cwd.be, sessionManager: { getSessionFile: () => path, getSessionId: () => 'live', getCwd: () => f.cwd.be } };
  const result = await registered.tools[0].execute('call', { action: 'search', query: 'autorización' }, undefined, undefined, ctx);
  assert.equal(result.details.entries[0].entryId, 'u1', 'the current journal is indexed before the query');
  await assert.rejects(registered.tools[0].execute('call', { action: 'read' }, undefined, undefined, ctx), /read needs entryId/u);
});
