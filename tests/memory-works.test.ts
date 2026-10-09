import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { gitRoot, installMemoryHooks } from '../src/extension/hooks.ts';
import { installMemoryTools } from '../src/extension/tools.ts';
import { sessionFailureStatement } from '../src/sources/session-log.ts';
import { resolveMemoryProject } from '../src/workspace/memory-registry.ts';
import type { Translator } from '../src/curation/translator.ts';
import { TestEmbeddingProvider } from './helpers.ts';

process.env.PI_MEMORY_OFFLINE = '1';

type Handler = (event: any, ctx: any) => any;

const harness = async (t: { after(fn: () => unknown): void }, options: { git: boolean; translator?: Translator }) => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-works-'));
  const repo = join(root, 'repo');
  const cwd = join(repo, 'packages', 'app');
  await mkdir(cwd, { recursive: true });
  if (options.git) await mkdir(join(repo, '.git'));
  const home = join(root, 'home');
  const handlers = new Map<string, Handler>();
  const tools = new Map<string, any>();
  const sent: any[] = [];
  const pi = {
    on(name: string, handler: Handler) { handlers.set(name, handler); },
    registerTool(tool: any) { tools.set(tool.name, tool); },
    sendMessage(message: any, sendOptions: any) { sent.push({ message, options: sendOptions }); },
  } as unknown as ExtensionAPI;
  const runtime = installMemoryHooks(pi, { home, embeddingProvider: new TestEmbeddingProvider(),
    ...(options.translator ? { translator: options.translator } : {}) });
  installMemoryTools(pi, runtime);
  const ctx = { cwd, mode: 'tui', sessionManager: { getSessionId: () => 'works-session' }, getContextUsage: () => undefined };
  await handlers.get('session_start')!({}, ctx);
  t.after(async () => { await handlers.get('session_shutdown')!({}, ctx); await rm(root, { recursive: true, force: true }); });
  return { root, repo, cwd, home, handlers, tools, runtime, ctx, sent };
};

test('a subdirectory resolves to its repository root, and home is never a repository', async () => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-git-'));
  await mkdir(join(root, '.git'));
  await mkdir(join(root, 'a', 'b'), { recursive: true });
  assert.equal(gitRoot(join(root, 'a', 'b')), root);
  assert.equal(gitRoot(tmpdir()), undefined);
  await rm(root, { recursive: true, force: true });
});

test('memory_record in an uninitialized git repository initializes it at the root and records', async t => {
  const h = await harness(t, { git: true });
  const lookupBefore = await h.tools.get('memory_context').execute('c0', { action: 'lookup', queries: ['package manager'] });
  assert.equal(lookupBefore.details.status, 'abstained');
  assert.match(lookupBefore.details.gaps.join(' '), /No memory has been recorded/);
  assert.equal(await resolveMemoryProject(h.repo, h.home), undefined, 'reading never initializes');

  const recorded = await h.tools.get('memory_record').execute('c1', {
    action: 'remember', kind: 'decision', statement: 'The workspace uses pnpm for local installs of every extension repository.',
  });
  assert.equal(recorded.details.status, 'ok');
  const binding = await resolveMemoryProject(h.repo, h.home);
  assert.ok(binding, 'the repository root, not the subdirectory, is bound');
  const found = await h.tools.get('memory_context').execute('c2', { action: 'lookup', queries: ['pnpm local installs'] });
  assert.equal(found.details.status, 'ok');
  assert.match(JSON.stringify(found.details.items), /pnpm/);
});

test('outside a git repository a record explains why and nothing is created', async t => {
  const h = await harness(t, { git: false });
  await assert.rejects(h.tools.get('memory_record').execute('c1', {
    action: 'remember', kind: 'decision', statement: 'Nothing should be written for a directory that is not a repository.',
  }), /git repository/);
  assert.equal(await resolveMemoryProject(h.cwd, h.home), undefined);
});

test('an agent-recorded memory is in the next prompt\'s snapshot without loading an encoder', async t => {
  const h = await harness(t, { git: true });
  const embedded = { calls: 0 };
  const engine0 = await h.tools.get('memory_record').execute('c1', {
    action: 'remember', kind: 'decision', statement: 'Compiled Pi builds live in the agent builds directory outside repositories.',
  });
  assert.equal(engine0.details.status, 'ok');
  const engine = await h.runtime.engine();
  const original = (engine as any).vector.provider.embed.bind((engine as any).vector.provider);
  (engine as any).vector.provider.embed = async (...args: any[]) => { embedded.calls += 1; return original(...args); };
  const result = await h.handlers.get('before_agent_start')!({ prompt: '¿dónde quedan los artefactos compilados?', systemPrompt: 'base' }, h.ctx);
  assert.match(result.message.content, /<project_memory trust="untrusted">[\s\S]*- decision \(unconfirmed\): Compiled Pi builds live/);
  assert.equal(result.message.details.memory.recall, undefined, 'the snapshot already carries the whole memory');
  assert.equal(embedded.calls, 0, 'no encoder for memory that fits the digest');
  const again = await h.handlers.get('before_agent_start')!({ prompt: 'otra pregunta', systemPrompt: 'base' }, h.ctx);
  assert.equal(again.systemPrompt, result.systemPrompt, 'the digest is stable across prompts');
});

test('a snapshot still in context is not resent; after compaction removes it, it is', async t => {
  const h = await harness(t, { git: true });
  await h.tools.get('memory_record').execute('c1', {
    action: 'remember', kind: 'decision', statement: 'Compiled Pi builds live in the agent builds directory outside repositories.',
  });
  const transcript: any[] = [];
  const ctx = { ...h.ctx, sessionManager: { ...h.ctx.sessionManager, buildSessionProjection: () => ({ messages: transcript }) } };
  const first = await h.handlers.get('before_agent_start')!({ prompt: 'where do builds go?', systemPrompt: 'base' }, ctx);
  assert.equal(first.systemPrompt, undefined, 'a returned prompt would be forced for this turn only');
  assert.match(first.message.content, /<memory_snapshot[\s\S]*Compiled Pi builds live/);
  transcript.push({ role: 'user', content: 'where do builds go?' }, { role: 'custom', ...first.message, timestamp: 1 });
  const second = await h.handlers.get('before_agent_start')!({ prompt: 'otra pregunta', systemPrompt: 'base' }, ctx);
  assert.equal(second, undefined, 'the same snapshot is already in context');
  transcript.splice(0, transcript.length, { role: 'compactionSummary', summary: 'earlier work', timestamp: 2 });
  const third = await h.handlers.get('before_agent_start')!({ prompt: 'y ahora?', systemPrompt: 'base' }, ctx);
  assert.deepEqual(third.message.content, first.message.content, 'resent once compaction removed it');
});

test('a remember declaration initializes repository memory; a tool failure alone does not', async t => {
  const failing = await harness(t, { git: true });
  await failing.handlers.get('tool_result')!({ toolName: 'bash', toolCallId: 'f1', isError: true,
    content: [{ type: 'text', text: 'Error: EACCES: permission denied, open /etc/hosts' }] }, failing.ctx);
  await failing.handlers.get('turn_end')!({}, failing.ctx);
  await failing.runtime.flushed();
  assert.equal(await resolveMemoryProject(failing.repo, failing.home), undefined);

  const declaring = await harness(t, { git: true });
  await declaring.handlers.get('before_agent_start')!({ prompt: 'remember that we never publish releases without warning', systemPrompt: 'base' }, declaring.ctx);
  await declaring.handlers.get('turn_end')!({}, declaring.ctx);
  await declaring.runtime.flushed();
  assert.ok(await resolveMemoryProject(declaring.repo, declaring.home));
  const found = await declaring.tools.get('memory_context').execute('c', { action: 'lookup', queries: ['publish releases'] });
  assert.match(JSON.stringify(found.details.items), /never publish releases/);
});

test('a declaration keeps the original language even with a legacy translator configured', async t => {
  const spoken = 'recuerda que nunca publicamos releases sin avisar';
  const declaring = await harness(t, { git: true, translator: { provider: 'unused', model: 'unused', toEnglish: async () => { throw new Error('Do not translate user instructions'); } } });
  await declaring.handlers.get('before_agent_start')!({ prompt: spoken, systemPrompt: 'base' }, declaring.ctx);
  await declaring.handlers.get('turn_end')!({}, declaring.ctx);
  await declaring.runtime.flushed();
  const engine = await declaring.runtime.engine();
  const fact = engine.projection.activeFacts(engine.scopeId, 20).find(item => item.statement === spoken);
  assert.ok(fact);
  assert.equal(fact.evidence[0]?.provenance, 'declared');
  assert.equal(fact.evidence[0]?.excerpt, spoken);
});

test('an offline declaration is stored without needing a translation model', async t => {
  const declaring = await harness(t, { git: true });
  const spoken = 'recuerda que nunca publicamos releases sin avisar';
  await declaring.handlers.get('before_agent_start')!({ prompt: spoken, systemPrompt: 'base' }, declaring.ctx);
  await declaring.handlers.get('turn_end')!({}, declaring.ctx);
  await declaring.runtime.flushed();
  const engine = await declaring.runtime.engine();
  assert.ok(engine.projection.activeFacts(engine.scopeId, 20).some(fact => fact.statement === spoken));
});

for (const statement of [
  'El daemon nunca se inicia solo, hay que configurarlo antes de que sincronice.',
  'Le daemon ne demarre jamais tout seul, il faut le configurer avant.',
  'The daemon never starts on its own; configure it before it can sync.',
]) test(`memory_record preserves authored text offline: ${statement}`, async t => {
  const h = await harness(t, { git: true });
  const recorded = await h.tools.get('memory_record').execute('original', { action: 'remember', kind: 'decision', statement });
  assert.equal(recorded.details.status, 'ok');
  const engine = await h.runtime.engine();
  assert.ok(engine.projection.activeFacts(engine.scopeId, 20).some(fact => fact.statement === statement));
});

test('a Spanish conversation still produces an English memory with the original words as evidence', async t => {
  const h = await harness(t, { git: true });
  const spoken = 'recuerda que nunca publicamos releases sin avisar al equipo';
  await h.handlers.get('before_agent_start')!({ prompt: spoken, systemPrompt: 'base' }, h.ctx);
  const recorded = await h.tools.get('memory_record').execute('mix', {
    action: 'remember', kind: 'constraint',
    statement: 'Never publish a release without warning the team first.',
    userQuote: spoken,
  });
  assert.equal(recorded.details.status, 'ok');
  const engine = await h.runtime.engine();
  const fact = engine.projection.activeFacts(engine.scopeId, 20)
    .find(item => /Never publish a release/u.test(item.statement));
  assert.ok(fact, 'the English statement is what gets stored and re-injected');
  // Evidence is the record of what was actually said. Translating it would
  // make the provenance a lie, so it stays exactly as spoken.
  assert.equal(fact.evidence.some(item => item.excerpt === spoken), true);
});

test('invented user quotations are refused even when their words resemble the statement', async t => {
  const h = await harness(t, { git: true });
  await h.handlers.get('before_agent_start')!({ prompt: 'Never publish before review.', systemPrompt: 'base' }, h.ctx);
  await assert.rejects(h.tools.get('memory_record').execute('wrong', {
    action: 'remember', kind: 'constraint', statement: 'Never publish without reviewing the team changes.',
    userQuote: 'Never publish without reviewing the team changes.',
  }), /user message on the current session branch/u);
});

test('failure statements keep the diagnosis and drop red tests and runner framing', () => {
  const tap = 'bash failed\nTAP version 13\n# Subtest: layout\nnot ok 3 - layout\n  error: |-\n  code: ERR_ASSERTION\n  duration_ms: 27.1';
  assert.equal(sessionFailureStatement(tap), '');
  const env = 'bash failed\n> npm run build\nnpm error code EACCES\nnpm error syscall mkdir\nnpm error path /usr/local/lib/node_modules\n    at Object.mkdir (node:fs:1)\nnpm error code EACCES';
  const statement = sessionFailureStatement(env);
  assert.equal(statement, 'bash: npm error code EACCES · npm error syscall mkdir · npm error path /usr/local/lib/node_modules');
  assert.equal(sessionFailureStatement('bash failed\nall good here'), '');
});

test('memory larger than the digest gets vectors in the background and per-prompt recall for the rest', async t => {
  const h = await harness(t, { git: true });
  for (const index of Array.from({ length: 40 }, (_, i) => i)) {
    await h.tools.get('memory_record').execute(`c${index}`, {
      action: 'remember', kind: 'fact', userQuote: undefined,
      statement: `Service ${index} runbook: restart the worker pool, drain the queue, then verify health checks for region ${index} before paging.`,
    });
  }
  await h.tools.get('memory_record').execute('late', {
    action: 'remember', kind: 'failure', statement: 'Deploys to the staging cluster fail when the kubeconfig context points at production.',
  });
  const engine = await h.runtime.engine();
  const deadline = Date.now() + 5_000;
  while (engine.projection.stats().vectors === 0 && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 20));
  assert.ok(engine.projection.stats().vectors > 0, 'vectors once memory exceeds the digest');
  const result = await h.handlers.get('before_agent_start')!({ prompt: 'why do deploys to the staging cluster fail?', systemPrompt: 'base' }, h.ctx);
  assert.doesNotMatch(result.message.content, /Service \d+ runbook/, 'facts are not in the core snapshot');
  assert.equal(result.systemPrompt, undefined, 'failures never ride in the system prompt');
  assert.match(result.message?.content ?? '', /<retained_memory[\s\S]*kubeconfig context points at production/);
});

test('memory tools are hidden outside repositories and restored inside one', async t => {
  const { installMemory } = await import('../src/index.ts');
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-visibility-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const repo = join(root, 'repo');
  await mkdir(join(repo, '.git'), { recursive: true });
  const plain = join(root, 'plain');
  await mkdir(plain);
  const handlers = new Map<string, Handler[]>();
  const state = { active: ['read', 'bash', 'memory_context', 'memory_record'] };
  const pi = {
    on(name: string, handler: Handler) { handlers.set(name, [...handlers.get(name) ?? [], handler]); },
    registerTool() {}, registerCommand() {},
    getActiveTools: () => state.active, setActiveTools: (next: string[]) => { state.active = next; },
  } as unknown as ExtensionAPI;
  installMemory(pi, { home: join(root, 'home') });
  const start = async (cwd: string) => {
    for (const handler of handlers.get('session_start') ?? []) await handler({}, { cwd, sessionManager: { getSessionId: () => `s-${cwd}` } });
    await new Promise(resolve => setTimeout(resolve, 50));
  };
  await start(plain);
  assert.deepEqual(state.active, ['read', 'bash']);
  await start(repo);
  assert.deepEqual(state.active, ['read', 'bash', 'memory_context', 'memory_record']);
  for (const handler of handlers.get('session_shutdown') ?? []) await handler({}, {});
});

test('recall searches the question inside an instruction-wrapped prompt', async () => {
  const { recallQueries } = await import('../src/extension/hooks.ts');
  assert.deepEqual(recallQueries('¿Qué gestor de paquetes usamos en este repo? Responde en una línea, sin leer archivos.'), [
    '¿Qué gestor de paquetes usamos en este repo? Responde en una línea, sin leer archivos.',
    '¿Qué gestor de paquetes usamos en este repo?',
    'Responde en una línea, sin leer archivos.',
  ]);
  assert.equal(recallQueries('a\nb\nc\nd\ne\nfive longer sentences here\nsix longer sentences here\nseven longer sentences here\neight longer ones').length, 4);
});

test('a close and distinctive dense match passes the relevance gate for a long natural question', async () => {
  const { relevantKeys } = await import('../src/retrieval/relevance.ts');
  const statistics = { documents: 3, frequencies: new Map<string, number>() } as never;
  const candidates = [
    { key: 'builds', text: 'Compiled Pi extension builds live in the agent builds directory, outside repositories.' },
    { key: 'pnpm', text: 'The workspace uses pnpm for installs.' },
    { key: 'theme', text: 'The editor theme is dark.' },
  ];
  const query = '¿Dónde viven los artefactos compilados de las extensiones? Responde en una línea.';
  assert.deepEqual([...relevantKeys(query, candidates, statistics, new Map([['builds', 0.69], ['pnpm', 0.04], ['theme', 0.01]]))], ['builds']);
  assert.deepEqual([...relevantKeys(query, candidates, statistics, new Map([['builds', 0.69], ['pnpm', 0.62], ['theme', 0.61]]))], [],
    'an encoder that scores everything alike proves nothing');
  assert.deepEqual([...relevantKeys(query, candidates, statistics, new Map([['builds', 0.69]]))], [], 'one scored candidate cannot show discrimination');
});

test('the core snapshot carries rules only, and a new fact does not resend it', async t => {
  const h = await harness(t, { git: true });
  await h.tools.get('memory_record').execute('r1', {
    action: 'remember', kind: 'constraint', statement: 'Open every pull request against the develop branch, never main.',
  });
  const first = await h.handlers.get('before_agent_start')!({ prompt: 'ship it', systemPrompt: 'base' }, h.ctx);
  assert.match(first.message.content, /constraint[^\n]*develop branch/);
  await h.tools.get('memory_record').execute('f1', {
    action: 'remember', kind: 'fact', statement: 'The staging database snapshot is refreshed every Monday at noon.',
  });
  const second = await h.handlers.get('before_agent_start')!({ prompt: 'ship it again', systemPrompt: 'base' }, h.ctx);
  assert.doesNotMatch(second.message.content, /<project_memory[^]*staging database/, 'facts never ride in the core');
  assert.equal(second.message.details.memory.revision, first.message.details.memory.revision,
    'the revision follows the core, so uniqueMemory drops the repeated snapshot');
});

test('after a compaction the core goes back in without waiting for a typed prompt', async t => {
  const h = await harness(t, { git: true });
  await h.tools.get('memory_record').execute('r1', {
    action: 'remember', kind: 'procedure', statement: 'After merging, label the Linear ticket ready to verify instead of moving it to Done.',
  });
  await h.handlers.get('session_compact')!({}, h.ctx);
  assert.equal(h.sent.length, 1);
  assert.equal(h.sent[0].message.customType, 'pi-memory-recall');
  assert.match(h.sent[0].message.content, /ready to verify/);
  assert.equal(h.sent[0].options.triggerTurn, false, 'restoring memory never starts a turn');
});

test('user quotations include steering and earlier branch messages but never assistant or tool text', async t => {
  const h = await harness(t, { git: true });
  const branch: any[] = [{ type: 'message', message: { role: 'user', content: 'No JEV' } }];
  Object.assign(h.ctx.sessionManager, { getBranch: () => branch });
  await h.handlers.get('before_agent_start')!({ prompt: 'Implement memory', systemPrompt: 'base' }, h.ctx);
  branch.push({ type: 'message', message: { role: 'user', content: [{ type: 'text', text: 'Sin traducciones' }] } });
  branch.push({ type: 'message', message: { role: 'assistant', content: [{ type: 'text', text: 'Invented instruction' }] } });
  for (const quote of ['No JEV', 'Sin traducciones']) {
    const result = await h.tools.get('memory_record').execute(quote, { action: 'remember', kind: 'constraint', statement: quote, userQuote: quote });
    assert.equal(result.details.standing, 'supported');
  }
  await assert.rejects(h.tools.get('memory_record').execute('forged', { action: 'remember', kind: 'constraint', statement: 'Fake rule', userQuote: 'Invented instruction' }), /user message/);
  const ctx2 = { ...h.ctx, sessionManager: { getSessionId: () => 'second-session', getBranch: () => [] } };
  await h.handlers.get('session_shutdown')!({}, h.ctx);
  await h.handlers.get('session_start')!({}, ctx2);
  const restored = await h.runtime.engine();
  assert.ok(restored.projection.activeFacts(restored.scopeId).some(fact => fact.statement === 'No JEV'));
  await assert.rejects(h.tools.get('memory_record').execute('old-quote', { action: 'remember', statement: 'Different statement', userQuote: 'No JEV' }), /user message/);
});

test('similar declarations are retained until the active model resolves them', async t => {
  const h = await harness(t, { git: true });
  await h.handlers.get('before_agent_start')!({ prompt: 'remember that we never bulk-generate a content calendar before one representative article passes review', systemPrompt: 'base' }, h.ctx);
  await h.handlers.get('turn_end')!({}, h.ctx);
  await h.runtime.flushed();
  await h.handlers.get('before_agent_start')!({ prompt: 'remember that we do not bulk-generate the content calendar until one representative article has passed review', systemPrompt: 'base' }, h.ctx);
  await h.handlers.get('turn_end')!({}, h.ctx);
  await h.runtime.flushed();
  const engine = await h.runtime.engine();
  const stored = engine.projection.activeFacts(engine.scopeId, 50).filter(fact => /content calendar/.test(fact.statement));
  assert.equal(stored.length, 2);
});
