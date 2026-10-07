import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readdir, readFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { installMemoryHooks } from '../src/extension/hooks.ts';
import { registerKnownSources } from '../src/sources/install.ts';
import { piSessionSource } from '../src/sources/presets.ts';
import { SourceRegistry } from '../src/sources/registry.ts';
import {
  appendSessionObservation, appendSessionObservations, clipSessionSummary, declaredMemoryQuote, sessionLogRoot,
  sessionObservationWorthy,
} from '../src/sources/session-log.ts';
import { MemoryEngine } from '../src/engine.ts';
import { TestEmbeddingProvider } from './helpers.ts';

const ctx = (cwd: string, sessionId = 's1', mode = 'tui') => ({
  cwd, mode, sessionManager: { getSessionId: () => sessionId }, getContextUsage: () => ({ tokens: 0 }),
});

test('session log keeps failures and corrections, not routine successes or secrets', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-log-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  assert.equal(sessionObservationWorthy({ kind: 'failure', tool: 'bash', outcome: 'failed', summary: 'bash failed: missing module' }), false);
  assert.equal(sessionObservationWorthy({ kind: 'failure', tool: 'bash', outcome: 'failed',
    summary: 'bash failed because the cache key omitted inode and size, leaving stale generated output' }), true);
  assert.equal(sessionObservationWorthy({ kind: 'failure', tool: 'bash', outcome: 'failed',
    summary: 'bash failed error TS2688: Cannot find type definition file for node. Command exited with code 2' }), true,
    'compiler errors remain learnable even when the host appends an exit code');
  assert.equal(sessionObservationWorthy({ kind: 'failure', tool: 'bash', outcome: 'failed',
    summary: 'bash failed Command exited with code 1' }), false);
  assert.equal(sessionObservationWorthy({ kind: 'instruction', tool: 'user_input', outcome: 'stated', summary: 'Always use pnpm in this repo, never npm.' }), true);
  assert.equal(sessionObservationWorthy({ kind: 'instruction', tool: 'user_input', outcome: 'stated',
    summary: 'Recuerda usar Zod para validar los límites de la API.' }), true);
  assert.equal(declaredMemoryQuote('Recuerda usar Zod para validar los límites de la API.'),
    'Recuerda usar Zod para validar los límites de la API.');
  assert.equal(declaredMemoryQuote('¿Puedes recordar qué librería valida la API?'), undefined,
    'questions must not become declarations');
  assert.equal(declaredMemoryQuote('¿Qué sigue?\nRecuerda usar Zod para validar los límites de la API.'),
    'Recuerda usar Zod para validar los límites de la API.', 'a separate explicit declaration remains eligible');
  assert.equal(declaredMemoryQuote('Remember to use API_TOKEN=super-secret-value for deployments.'), undefined,
    'secret-bearing declarations must not become facts');
  assert.equal(sessionObservationWorthy({ kind: 'instruction', tool: 'user_input', outcome: 'stated', summary: 'please list the files' }), false);
  assert.equal(sessionObservationWorthy({ kind: 'failure', tool: 'memory_record', outcome: 'failed', summary: 'memory_record failed: nope' }), false);
  assert.match(clipSessionSummary('token sk-abcdefghijklmnopqrstuvwxyz012345'), /REDACTED/);
  assert.equal(await appendSessionObservation({
    projectId: 'p_test', home, record: {
      id: 'obs_ok', kind: 'instruction', tool: 'user_input', outcome: 'stated',
      summary: 'list files in src', observedAt: '2026-01-01T00:00:00.000Z', provenance: 'declared', sessionId: 's1',
    },
  }), false);
  assert.equal(await appendSessionObservation({
    projectId: 'p_test', home, record: {
      id: 'obs_fail', kind: 'failure', tool: 'bash', outcome: 'failed',
      summary: 'bash failed because auth refresh raced while using gh auth token ghp_abcdefghijklmnopqrstuvwxyz012345',
      observedAt: '2026-01-01T00:00:00.000Z', provenance: 'native_observation', sessionId: 's1',
    },
  }), true);
  const docs = await piSessionSource({ home, scope: { kind: 'project', id: 'p_test' } }).scan();
  assert.equal(docs.length, 1);
  assert.equal(docs[0]?.kind, 'failure');
  assert.equal(docs[0]?.trust, 'host');
  assert.doesNotMatch(docs[0]?.text ?? '', /ghp_/);
});

test('failed tools and learnable prompts persist for daemon scan without copying bodies', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-hooks-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const handlers = new Map<string, (event: any, ctx: any) => Promise<any>>();
  const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home });
  await handlers.get('session_start')!({}, ctx(cwd));
  await runtime.initialize();
  await handlers.get('before_agent_start')!(
    { prompt: 'Always use pnpm in this repo, never npm.', systemPrompt: 'Base' }, ctx(cwd));
  await handlers.get('tool_result')!(
    { toolName: 'bash', toolCallId: 'c1', isError: true, content: [{ type: 'text', text: 'cache remained stale because the key omitted inode and size' }] },
    ctx(cwd));
  await handlers.get('tool_result')!(
    { toolName: 'bash', toolCallId: 'c2', isError: false, content: [{ type: 'text', text: 'ok' }] },
    ctx(cwd));
  await handlers.get('turn_end')!({}, ctx(cwd));
  await runtime.flushed();
  const engine = await MemoryEngine.forProject(cwd, 's1', { home, provider: new TestEmbeddingProvider() });
  t.after(() => engine.dispose());
  const root = sessionLogRoot(engine.scopeId, home);
  const files = await (async () => {
    for (const _ of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) {
      const names = await readdir(root).catch(() => [] as string[]);
      if (names.some(name => name.endsWith('.jsonl'))) return names;
      await new Promise(resolve => setTimeout(resolve, 40));
    }
    return readdir(root).catch(() => [] as string[]);
  })();
  assert.ok(files.some(name => name.endsWith('.jsonl')), `expected session jsonl, got ${files.join(',')}`);
  const body = await readFile(join(sessionLogRoot(engine.scopeId, home), files.find(name => name.endsWith('.jsonl'))!), 'utf8');
  assert.match(body, /Always use pnpm/);
  assert.match(body, /cache remained stale because the key omitted inode and size/);
  assert.doesNotMatch(body, /missing script/);
  assert.doesNotMatch(body, /"outcome":"succeeded"/);
  const registry = new SourceRegistry();
  // Sessions are ingested as one digest per session once idle; no wait in a test.
  await registerKnownSources(registry, engine.scopeId, { home, sessionSettleMs: 0 });
  const synced = await registry.sync(async () => engine, 'pi-session-digest');
  assert.ok(synced.queued >= 1);
  assert.ok([...engine.projection.eachActiveDocument()].every(document => document.namespace === 'memory'),
    'source sync must not project raw pi.session observations');
});

test('one turn flush deduplicates repeated failures by semantic key and summary hash', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-dedup-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const summary = 'bash failed because the cache key omitted inode and size, leaving stale generated output';
  const records = Array.from({ length: 40 }, (_value, index) => ({
    id: `ignored-${index}`, kind: 'failure' as const, tool: 'bash', outcome: 'failed' as const, summary,
    observedAt: '2026-01-01T00:00:00.000Z', provenance: 'native_observation' as const, sessionId: `s${index}`,
  }));
  assert.equal(await appendSessionObservations({ projectId: 'p_test', home, records }), 1);
  const source = piSessionSource({ home, scope: { kind: 'project', id: 'p_test' } });
  const documents = await source.scan();
  assert.equal(documents.length, 1);
  assert.match(documents[0]?.metadata.semanticKey ?? '', /^pi-session\.failure\.bash\./u);
  assert.match(documents[0]?.metadata.summaryHash ?? '', /^[0-9a-f]{64}$/u);
  const engine = await MemoryEngine.forScope('project', 'p_test', 'dedup', { home, provider: new TestEmbeddingProvider() });
  t.after(() => engine.dispose());
  const registry = new SourceRegistry();
  registry.register(source);
  assert.equal((await registry.sync(async () => engine, 'pi-session')).queued, 1);
  await appendSessionObservations({ projectId: 'p_test', home, records: [{ ...records[0]!,
    observedAt: '2026-01-02T00:00:00.000Z', sessionId: 'later-session' }] });
  assert.equal((await registry.sync(async () => engine, 'pi-session')).queued, 0,
    'a later timestamp for the same semanticKey + summary hash must not requeue analysis');
});

test('a declared correction is supported and recalled by the next session before agent start', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-correction-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const handlers = new Map<string, (event: any, context: any) => Promise<any>>();
  const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home });
  await handlers.get('session_start')!({}, ctx(cwd, 'correction-session'));
  await runtime.initialize();
  await handlers.get('before_agent_start')!({ prompt: 'Never use npm; use pnpm for this repository.', systemPrompt: 'Base' }, ctx(cwd, 'correction-session'));
  await handlers.get('turn_end')!({}, ctx(cwd, 'correction-session'));
  await runtime.flushed();
  const first = await runtime.engine();
  const correction = first.projection.activeFacts(first.scopeId, 20).find(fact => fact.kind === 'correction');
  assert.equal(correction?.standing, 'supported');
  assert.equal(correction?.evidence[0]?.provenance, 'declared');
  assert.equal(first.projection.stats().vectors, 0, 'interactive corrections stay lexical and never embed raw session text');
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'correction-session'));
  await handlers.get('session_start')!({}, ctx(cwd, 'new-session'));
  const recalled = await handlers.get('before_agent_start')!({
    prompt: 'Should this repository use npm or pnpm?', systemPrompt: 'Base',
  }, ctx(cwd, 'new-session'));
  assert.match(recalled.message.content, /<project_memory trust="untrusted">[\s\S]*Never use npm; use pnpm/);
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'new-session'));
});

test('a task brief from another agent (pi -p, rpc, a subagent child) never becomes a correction', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-brief-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const brief = 'You are a QA gate subagent. NO installs (never npm/pnpm install), no servers, no restarts. Never use npm; use pnpm for this repository.';
  const runs: [string, string, Record<string, string>][] = [['print', 'brief-print', {}], ['rpc', 'brief-rpc', {}], ['tui', 'brief-child', { PI_SUBAGENTS_CHILD: '1' }]];
  for (const [mode, sessionId, env] of runs) {
    const saved = { ...process.env };
    Object.assign(process.env, env);
    t.after(() => { process.env = saved; });
    const handlers = new Map<string, (event: any, context: any) => Promise<any>>();
    const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
    const runtime = installMemoryHooks(pi, { home });
    await handlers.get('session_start')!({}, ctx(cwd, sessionId, mode));
    await runtime.initialize();
    await handlers.get('before_agent_start')!({ prompt: brief, systemPrompt: 'Base' }, ctx(cwd, sessionId, mode));
    await handlers.get('turn_end')!({}, ctx(cwd, sessionId, mode));
    await runtime.flushed();
    const engine = await runtime.engine();
    assert.equal(engine.projection.activeFacts(engine.scopeId, 20).some(fact => fact.kind === 'correction'), false, `${mode} ${sessionId}`);
    await handlers.get('session_shutdown')!({}, ctx(cwd, sessionId, mode));
    process.env = saved;
  }
});

test('an explicit recuerda declaration is supported and recalled by the next session', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-remember-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const handlers = new Map<string, (event: any, context: any) => Promise<any>>();
  const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home });
  await handlers.get('session_start')!({}, ctx(cwd, 'remember-session'));
  await runtime.initialize();
  const declaration = 'Remember to use Zod for validating the API boundaries.';
  await handlers.get('before_agent_start')!({ prompt: declaration, systemPrompt: 'Base' }, ctx(cwd, 'remember-session'));
  await handlers.get('turn_end')!({}, ctx(cwd, 'remember-session'));
  await runtime.flushed();
  const first = await runtime.engine();
  const remembered = first.projection.activeFacts(first.scopeId, 20).find(fact => fact.statement === declaration);
  assert.equal(remembered?.kind, 'procedure');
  assert.equal(remembered?.standing, 'supported');
  assert.equal(remembered?.evidence[0]?.provenance, 'declared');
  assert.equal(first.projection.stats().vectors, 0, 'explicit declarations stay lexical and never embed raw prompts');
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'remember-session'));
  await handlers.get('session_start')!({}, ctx(cwd, 'new-session'));
  const recalled = await handlers.get('before_agent_start')!({
    prompt: 'Should we use Zod for validating the API boundaries?', systemPrompt: 'Base',
  }, ctx(cwd, 'new-session'));
  assert.match(recalled.message.content, /<project_memory trust="untrusted">[\s\S]*Remember to use Zod/);
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'new-session'));
});

test('a recuerda declaration retains its language across sessions without translation', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-remember-es-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const spoken = 'Recuerda usar Zod para validar los límites de la API.';
  const handlers = new Map<string, (event: any, context: any) => Promise<any>>();
  const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home });
  await handlers.get('session_start')!({}, ctx(cwd, 'remember-es'));
  await runtime.initialize();
  await handlers.get('before_agent_start')!({ prompt: spoken, systemPrompt: 'Base' }, ctx(cwd, 'remember-es'));
  await handlers.get('turn_end')!({}, ctx(cwd, 'remember-es'));
  await runtime.flushed();
  const project = await runtime.engine();
  const fact = project.projection.activeFacts(project.scopeId, 20).find(item => item.statement === spoken);
  assert.ok(fact, 'the declaration was stored verbatim');
  assert.equal(fact.standing, 'supported');
  assert.equal(fact.evidence[0]?.excerpt, spoken, 'the Spanish words remain the evidence');
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'remember-es'));
  await handlers.get('session_start')!({}, ctx(cwd, 'new-es'));
  const recalled = await handlers.get('before_agent_start')!({
    prompt: '¿Cómo validar los límites de la API con Zod?', systemPrompt: 'Base',
  }, ctx(cwd, 'new-es'));
  assert.ok(recalled.message.content.includes(spoken));
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'new-es'));
});

test('tool observations alone never invent durable memories', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-compiler-'));
  const cwd = join(home, 'work');
  await mkdir(cwd, { recursive: true });
  t.after(() => rm(home, { recursive: true, force: true }));
  const handlers = new Map<string, (event: any, context: any) => Promise<any>>();
  const pi = { on(name: string, handler: any) { handlers.set(name, handler); } } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home });
  await handlers.get('session_start')!({}, ctx(cwd, 'fail-session'));
  await runtime.initialize();
  await handlers.get('tool_result')!({
    toolName: 'bash', toolCallId: 'tsc1', isError: true,
    content: [{ type: 'text', text: 'error TS2688: Cannot find type definition file for node. Command exited with code 2' }],
  }, ctx(cwd, 'fail-session'));
  await handlers.get('turn_end')!({}, ctx(cwd, 'fail-session'));
  await runtime.flushed();
  const first = await runtime.engine();
  assert.equal(first.projection.activeFacts(first.scopeId, 20).find(fact => fact.kind === 'failure'), undefined,
    'the active model decides which observations become facts');
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'fail-session'));
  await handlers.get('session_start')!({}, ctx(cwd, 'next-session'));
  const recalled = await handlers.get('before_agent_start')!({
    prompt: 'error TS2688 Cannot find type definition file for node', systemPrompt: 'Base',
  }, ctx(cwd, 'next-session'));
  assert.doesNotMatch(recalled.message.content, /error TS2688/, 'unrecorded observations do not become durable facts');
  await handlers.get('session_shutdown')!({}, ctx(cwd, 'next-session'));
});

test('pi-session provenance is host-owned and its source cannot be symlinked to another project', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-session-isolation-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  await assert.rejects(appendSessionObservation({ projectId: 'p_a', home, record: {
    id: 'forged', kind: 'failure', tool: 'bash', outcome: 'failed',
    summary: 'bash failed because a stale cache key omitted the file inode and size',
    observedAt: '2026-01-01T00:00:00.000Z', provenance: 'declared', sessionId: 'agent',
  } }), /host-native provenance/);
  const other = sessionLogRoot('p_b', home);
  await mkdir(other, { recursive: true });
  await mkdir(join(home, 'p_a'), { recursive: true });
  await symlink(other, sessionLogRoot('p_a', home), 'dir');
  await assert.rejects(piSessionSource({ home, scope: { kind: 'project', id: 'p_a' } }).scan(), /symbolic link/);
});
