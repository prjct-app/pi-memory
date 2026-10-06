import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { AssistantMessage, ToolResultMessage } from '@earendil-works/pi-ai';
import { assertCheckpoint, readCheckpoint, writeCheckpoint } from '../src/handoff/checkpoint.ts';
import { createHandoffController } from '../src/handoff/hooks.ts';
import { budgetForModel, DEFAULT_HANDOFF_BUDGET, selectHandoffMessages } from '../src/handoff/select.ts';
import { groupTurns, type HandoffMessage } from '../src/handoff/turns.ts';
import { MemoryEngine } from '../src/engine.ts';
import { TestEmbeddingProvider } from './helpers.ts';

const user = (text: string): HandoffMessage => ({ role: 'user', content: [{ type: 'text', text }] });
const assistant = (text: string, ids: string[] = []): HandoffMessage => ({
  role: 'assistant', content: [{ type: 'text', text }], toolCalls: ids.map(id => ({ id })),
});
const tool = (id: string, text: string): HandoffMessage => ({ role: 'toolResult', toolCallId: id, content: [{ type: 'text', text }] });

test('grouping keeps tool calls with matching results', () => {
  const messages = [user('one'), assistant('call', ['t1']), tool('t1', 'ok'), user('two'), assistant('done')];
  const turns = groupTurns(messages);
  assert.equal(turns.length, 2);
  assert.deepEqual(turns[0]!.messages.map(message => message.role), ['user', 'assistant', 'toolResult']);
});

test('real Pi multi-call tool loops require a bijection and remain atomic out of order', () => {
  const usage = { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const call: AssistantMessage = {
    role: 'assistant', content: [
      { type: 'toolCall', id: 'call_a', name: 'read', arguments: { path: 'README.md' } },
      { type: 'toolCall', id: 'call_b', name: 'read', arguments: { path: 'package.json' } },
    ],
    api: 'anthropic-messages', provider: 'anthropic', model: 'offline', usage, stopReason: 'toolUse', timestamp: 1,
  };
  const resultA: ToolResultMessage = {
    role: 'toolResult', toolCallId: 'call_a', toolName: 'read', content: [{ type: 'text', text: 'A' }], isError: false, timestamp: 3,
  };
  const resultB: ToolResultMessage = {
    role: 'toolResult', toolCallId: 'call_b', toolName: 'read', content: [{ type: 'text', text: 'B' }], isError: false, timestamp: 2,
  };
  // A missing result is synthesized exactly as Pi transports it; duplicates still refuse.
  const missing = selectHandoffMessages([user('inspect'), call, resultA], undefined);
  assert.equal(missing.ok, true);
  if (missing.ok) {
    assert.deepEqual(missing.messages.slice(0, 3), [user('inspect'), call, resultA]);
    assert.deepEqual(missing.messages[3], { role: 'toolResult', toolCallId: 'call_b', toolName: 'read',
      content: [{ type: 'text', text: 'No result provided' }], isError: true, timestamp: 1 });
  }
  assert.equal(selectHandoffMessages([user('inspect'), call, resultA, resultA, resultB], undefined).ok, false);
  const selected = selectHandoffMessages(
    [user('discard old'), assistant('old'), user('inspect'), call, resultB, resultA], undefined,
    { maxTokens: 8_000, maxBytes: 32_768, maxMessages: 4, toolSchemaReserveTokens: 0 },
  );
  assert.equal(selected.ok, true);
  if (selected.ok) assert.deepEqual(selected.messages, [user('inspect'), call, resultB, resultA]);
});

test('budget omits older complete turns and keeps the latest user turn', () => {
  const messages = [
    user('old requirement A'), assistant('old'),
    user('latest requirement B with evidence ev_1'), assistant('work', ['t2']), tool('t2', 'result'),
  ];
  const selected = selectHandoffMessages(messages, undefined, { maxTokens: 8_000, maxBytes: 32_768, maxMessages: 3, toolSchemaReserveTokens: 10 });
  assert.equal(selected.ok, true);
  if (selected.ok) {
    const texts = JSON.stringify(selected.messages);
    assert.match(texts, /latest requirement B/);
    assert.equal(texts.includes('old requirement A'), false);
    assert.match(texts, /t2/);
    assert.ok(selected.postTokens <= 80);
  }
});

test('latest Pi summary is the deterministic fallback unless an explicit checkpoint is newer', () => {
  const compaction: HandoffMessage = { role: 'compactionSummary', summary: 'Earlier verified decisions', timestamp: 100 };
  const branch: HandoffMessage = { role: 'branchSummary', summary: 'Returned branch result', timestamp: 200 };
  const messages = [compaction, user('old turn'), assistant('old answer'), branch, user('current turn'), assistant('current answer')];
  const budget = { maxTokens: 8_000, maxBytes: 32_768, maxMessages: 3, toolSchemaReserveTokens: 0 };
  const fallback = selectHandoffMessages(messages, undefined, budget);
  assert.equal(fallback.ok, true);
  if (fallback.ok) assert.deepEqual(fallback.messages, [branch, user('current turn'), assistant('current answer')]);
  const checkpoint = {
    projectId: 'p_test', sessionId: 's1', goal: 'New explicit goal', constraints: ['new constraint'], done: [],
    inProgress: [], blocked: [], decisions: [], evidenceRefs: ['ev_new'], nextSteps: [], updatedAt: new Date(300).toISOString(),
  } as const;
  const explicit = selectHandoffMessages(messages, checkpoint, budget);
  assert.equal(explicit.ok, true);
  if (explicit.ok) {
    assert.match(JSON.stringify(explicit.messages), /New explicit goal/);
    assert.equal(explicit.messages.some(message => message.role === 'branchSummary'), false);
  }
  const newerSummary = selectHandoffMessages(messages, { ...checkpoint, updatedAt: new Date(150).toISOString() }, budget);
  assert.equal(newerSummary.ok, true);
  if (newerSummary.ok) assert.equal(newerSummary.messages[0]?.role, 'branchSummary');
});

test('an errored assistant tool call no longer blocks every later request', () => {
  // Real session: a WebSocket error left a partial write call with no result.
  const failed = { ...assistant('partial write', ['lost']), stopReason: 'error' } as HandoffMessage;
  const later = [user('continue'), assistant('read', ['ok']), tool('ok', 'fine')];
  const selected = selectHandoffMessages([user('write it'), failed, ...later], undefined);
  assert.equal(selected.ok, true);
  if (selected.ok) {
    assert.equal(selected.messages.includes(failed), false, 'Pi skips errored assistant messages');
    assert.deepEqual(selected.messages.slice(-3), later);
  }
  const orphanResult = selectHandoffMessages([user('go'), tool('unknown', 'stray')], undefined);
  assert.equal(orphanResult.ok, true);
  if (orphanResult.ok) assert.deepEqual(orphanResult.messages, [user('go')]);
});

test('duplicate tool results are still refused', () => {
  const selected = selectHandoffMessages([user('go'), assistant('call', ['x']), tool('x', 'a'), tool('x', 'a')], undefined);
  assert.equal(selected.ok, false);
});

test('a growing current turn drops older complete tool rounds instead of aborting', () => {
  const messages = [
    user('keep the operator request'),
    assistant('first call', ['old-1']), tool('old-1', 'large old result '.repeat(200)),
    assistant('second call', ['old-2']), tool('old-2', 'another old result '.repeat(200)),
    assistant('latest call', ['latest']), tool('latest', 'latest evidence'),
  ];
  const selected = selectHandoffMessages(messages, undefined,
    { maxTokens: 8_000, maxBytes: 32_768, maxMessages: 3, toolSchemaReserveTokens: 0 });
  assert.equal(selected.ok, true);
  if (selected.ok) {
    assert.deepEqual(selected.messages, [user('keep the operator request'), assistant('latest call', ['latest']), tool('latest', 'latest evidence')]);
    assert.match(selected.reason, /2 older tool round/);
  }
});

test('an oversized newest tool result is truncated in place instead of aborting', () => {
  const overhead = { systemTokens: 3_402, systemBytes: 13_610, toolSchemaTokens: 3_401, toolSchemaBytes: 13_604 };
  const request = user('continue the refactor');
  const messages = [
    request,
    assistant('read', ['big']), tool('big', `HEAD-MARK ${'y'.repeat(100_000)} TAIL-MARK`),
  ];
  const selected = selectHandoffMessages(messages, assertCheckpoint({ projectId: 'p', sessionId: 's', goal: 'finish the refactor' }), DEFAULT_HANDOFF_BUDGET, overhead);
  assert.equal(selected.ok, true);
  if (selected.ok) {
    assert.ok(selected.postTokens <= DEFAULT_HANDOFF_BUDGET.maxTokens);
    assert.ok(selected.postBytes <= DEFAULT_HANDOFF_BUDGET.maxBytes);
    assert.equal(selected.truncatedFields, 1);
    assert.equal(selected.messages.length, 4);
    assert.equal(selected.messages[1], request);
    const result = selected.messages[3]!;
    assert.equal(result.role, 'toolResult');
    assert.equal(result.toolCallId, 'big');
    const text = (result.content as { text: string }[])[0]!.text;
    assert.match(text, /^HEAD-MARK/);
    assert.match(text, /TAIL-MARK$/);
    assert.match(text, /chars omitted to fit the context budget/);
    // Truncation should use the available room, not collapse to the minimum.
    assert.ok(selected.postTokens > DEFAULT_HANDOFF_BUDGET.maxTokens * 0.9);
  }
});

test('truncation drops oversized images and caps large tool-call arguments before touching the request', () => {
  const request = user('write the generated file');
  const call = {
    role: 'assistant',
    content: [{ type: 'toolCall', id: 'w', name: 'write', arguments: { path: 'out.ts', content: 'z'.repeat(80_000) } }],
  } as HandoffMessage;
  const shot = { role: 'toolResult', toolCallId: 'w',
    content: [{ type: 'image', data: 'i'.repeat(200_000), mimeType: 'image/png' }, { type: 'text', text: 'written' }] } as HandoffMessage;
  const selected = selectHandoffMessages([request, call, shot], undefined, DEFAULT_HANDOFF_BUDGET,
    { systemTokens: 3_402, systemBytes: 13_610, toolSchemaTokens: 3_401, toolSchemaBytes: 13_604 });
  assert.equal(selected.ok, true);
  if (selected.ok) {
    assert.equal(selected.messages[0], request);
    const [, keptCall, keptShot] = selected.messages;
    const args = (keptCall!.content as { arguments: { path: string; content: string } }[])[0]!.arguments;
    assert.equal(args.path, 'out.ts');
    assert.ok(args.content.length < 80_000);
    assert.deepEqual((keptShot!.content as { type: string }[]).map(block => block.type), ['text', 'text']);
    assert.ok(selected.postBytes <= DEFAULT_HANDOFF_BUDGET.maxBytes);
    assert.ok(selected.postTokens <= DEFAULT_HANDOFF_BUDGET.maxTokens);
  }
});

test('the budget follows the active model context window instead of a fixed 16k cap', () => {
  assert.deepEqual(budgetForModel(undefined), DEFAULT_HANDOFF_BUDGET);
  const small = budgetForModel({ contextWindow: 8_000, maxTokens: 4_000 });
  assert.equal(small.maxTokens, 6_000, 'a usable small model must never receive the 16k fallback');
  assert.ok(small.maxBytes < DEFAULT_HANDOFF_BUDGET.maxBytes);
  assert.equal(small.maxMessages, Number.MAX_SAFE_INTEGER, 'the token budget, not the number of turns, bounds context');
  for (const contextWindow of [0, -1, NaN, Infinity, 0.5]) assert.deepEqual(budgetForModel({ contextWindow }), DEFAULT_HANDOFF_BUDGET);
  const sol = budgetForModel({ contextWindow: 272_000, maxTokens: 128_000 });
  assert.equal(sol.maxTokens, 272_000 - 32_768);
  assert.ok(sol.maxBytes >= sol.maxTokens * 4);
  assert.ok(sol.maxMessages > DEFAULT_HANDOFF_BUDGET.maxMessages);
  // Two ~55KB reads in one round: aborted under 16k, whole under a 272k model budget.
  const doc = (word: string) => Array.from({ length: 700 }, (_, index) => index === 350
    ? `line 0351: THE CODE WORD IS ${word}` : `line ${index}: lorem ipsum dolor sit amet consectetur adipiscing elit sed do`).join('\n');
  const messages = [user('read both'), assistant('read', ['a', 'b']), tool('a', doc('PAPAYA-7731')), tool('b', doc('MANGO-2290'))];
  const overhead = { systemTokens: 3_402, systemBytes: 13_610, toolSchemaTokens: 3_401, toolSchemaBytes: 13_604 };
  const selected = selectHandoffMessages(messages, undefined, sol, overhead);
  assert.equal(selected.ok, true);
  if (selected.ok) {
    assert.equal(selected.truncatedFields, 0);
    assert.deepEqual(selected.messages, messages);
  }
});

test('an unconfigured controller derives its ceiling from ctx.model', async () => {
  const controller = createHandoffController({ engine: async () => undefined });
  const big = tool('r', 'q'.repeat(200_000));
  const messages = [user('read the large document'), assistant('read', ['r']), big];
  const ctx = (model: object) => ({
    cwd: '/transient-model-budget', model: { provider: 'offline', id: 'm', ...model }, getSystemPrompt: () => 'policy',
    abort: () => { throw new Error('must not abort'); }, ui: { notify: () => undefined },
    sessionManager: { getSessionId: () => 's-model-budget' },
  }) as never;
  const result = await controller.safeContext(messages, ctx({ contextWindow: 272_000, maxTokens: 128_000 }));
  assert.equal(result.messages.at(-1), big);
});

test('a long session keeps early constraints and short tool results while they fit the model', () => {
  const messages = [user('Keep the existing API contract: tenant IDs are strings.'),
    ...Array.from({ length: 600 }, (_, index) => [
      assistant('read', [`read-${index}`]), tool(`read-${index}`, `Verified module ${index}; keep its public interface.`),
    ]).flat(), user('Continue implementing the original request.')];
  const selected = selectHandoffMessages(messages, undefined, budgetForModel({ contextWindow: 272_000, maxTokens: 128_000 }));
  assert.equal(selected.ok, true);
  if (selected.ok) {
    assert.deepEqual(selected.messages, messages);
    assert.equal(selected.truncatedFields, 0);
  }
});

test('the newest tool round remains atomic and refuses when even truncation cannot fit', () => {
  const messages = [user('keep the request'), assistant('latest call', ['latest']), tool('latest', 'x'.repeat(20_000))];
  const selected = selectHandoffMessages(messages, undefined,
    { maxTokens: 100, maxBytes: 500, maxMessages: 3, toolSchemaReserveTokens: 20 });
  assert.equal(selected.ok, false);
  if (!selected.ok) {
    assert.match(selected.instruction, /newest complete tool round/i);
    assert.doesNotMatch(selected.instruction, /messages 20_000/);
  }
});

test('serialized message bytes include JSON array framing', () => {
  const messages = [user('one'), assistant('two')];
  const selected = selectHandoffMessages(messages, undefined,
    { maxTokens: 8_000, maxBytes: 32_768, maxMessages: 4, toolSchemaReserveTokens: 0 });
  assert.equal(selected.ok, true);
  if (selected.ok) assert.equal(selected.postBytes, Buffer.byteLength(JSON.stringify(selected.messages), 'utf8'));
});

test('the default byte ceiling admits signed reasoning payloads that fit the token ceiling', () => {
  const signed = {
    ...assistant('latest signed call', ['signed']),
    content: [{ type: 'thinking', thinking: 'brief', thinkingSignature: 's'.repeat(150_000) }],
  } as HandoffMessage;
  const selected = selectHandoffMessages([user('continue'), signed, tool('signed', 'ok')], undefined,
    DEFAULT_HANDOFF_BUDGET, { systemTokens: 3_402, systemBytes: 13_610, toolSchemaTokens: 3_401, toolSchemaBytes: 13_604 });
  assert.equal(selected.ok, true);
  if (selected.ok) {
    assert.ok(selected.postTokens <= DEFAULT_HANDOFF_BUDGET.maxTokens);
    assert.ok(selected.postBytes > 170_000);
    assert.ok(selected.postBytes <= DEFAULT_HANDOFF_BUDGET.maxBytes);
  }
});

test('no checkpoint and oversized current turn fails closed', () => {
  const huge = user('x'.repeat(20_000));
  const selected = selectHandoffMessages([huge], undefined, { maxTokens: 50, maxBytes: 100, maxMessages: 4, toolSchemaReserveTokens: 20 });
  assert.equal(selected.ok, false);
  if (!selected.ok) assert.match(selected.instruction, /current request/i);
});

test('checkpoints are private project/session rows excluded from recall and semantic stats', async t => {
  const root = await mkdtemp(join(tmpdir(), 'handoff-cp-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  // This assertion inspects indexed rows directly; compact checkpoint behavior
  // is covered through the public API in the compact runtime suite.
  const a = new MemoryEngine({ root: join(root, 'a'), scopeId: 'p_aaa', sessionId: 's1', provider: new TestEmbeddingProvider(), storage: 'indexed' });
  const b = new MemoryEngine({ root: join(root, 'b'), scopeId: 'p_bbb', sessionId: 's1', provider: new TestEmbeddingProvider(), storage: 'indexed' });
  t.after(async () => { await a.dispose(); await b.dispose(); });
  const before = a.projection.stats();
  const checkpoint = {
    projectId: 'p_aaa', sessionId: 'sess-a', goal: 'Ship isolation', constraints: ['no shared db'],
    done: [], inProgress: ['handoff'], blocked: [], decisions: ['sqlite'], evidenceRefs: ['ev_1'], nextSteps: ['review'],
    updatedAt: new Date().toISOString(),
  } as const;
  await writeCheckpoint(a, checkpoint);
  assert.deepEqual(readCheckpoint(a, 'sess-a'), checkpoint);
  const revised = { ...checkpoint, goal: 'Ship isolated handoff', updatedAt: new Date(Date.parse(checkpoint.updatedAt) + 1).toISOString() };
  await writeCheckpoint(a, revised);
  assert.deepEqual(readCheckpoint(a, 'sess-a'), revised);
  await assert.rejects(() => writeCheckpoint(a, checkpoint), /stale/);
  assert.deepEqual(readCheckpoint(a, 'sess-a'), revised);
  assert.equal(readCheckpoint(b, 'sess-a'), undefined);
  assert.throws(() => a.projection.operationalCheckpoint('p_bbb', 'sess-a'), /does not own/);
  assert.throws(() => a.projection.upsertOperationalCheckpoint('p_bbb', 'sess-x', '{}', Date.now()), /does not own/);
  const foreign = a.projection.db.prepare("SELECT COUNT(*) AS n FROM operational_checkpoints WHERE project_id='p_bbb'").get() as { n: number };
  assert.equal(foreign.n, 0);
  assert.equal(a.projection.documentByKey({ namespace: 'memory.checkpoint', externalId: 'session:sess-a' }), undefined);
  const after = a.projection.stats();
  assert.deepEqual({ documents: after.documents, chunks: after.chunks, vectors: after.vectors, facts: after.facts, events: after.events },
    { documents: before.documents, chunks: before.chunks, vectors: before.vectors, facts: before.facts, events: before.events });
  const recalled = await a.search({ queries: ['Ship isolation no shared db'], dense: false });
  assert.equal(recalled.items.length, 0);
  await assert.rejects(() => writeCheckpoint(a, { ...checkpoint, projectId: 'p_bbb' }), /projectId/);
  await assert.rejects(() => writeCheckpoint(a, { ...checkpoint, goal: 'x'.repeat(4_001) }), /4000 bytes/);
});

test('an explicitly bounded handoff fails closed when its engine fails', async () => {
  const controller = createHandoffController({
    engine: async () => { throw new Error('boom'); }, budget: DEFAULT_HANDOFF_BUDGET,
  });
  const original: HandoffMessage[] = [user('secret oversized original')];
  controller.activate('p_fault', '/fault/project', 's1', 'test fault');
  const notices: string[] = [];
  const aborted = { n: 0 };
  const result = await controller.safeContext(original, {
    cwd: '/fault/project',
    model: { provider: 'offline', id: 'b' },
    getSystemPrompt: () => 'short system prompt',
    abort: () => { aborted.n += 1; },
    ui: { notify: (message: string) => { notices.push(message); } },
    sessionManager: { getSessionId: () => 's1' },
  } as never);
  assert.equal(result.messages.length, 1);
  assert.equal(JSON.stringify(result.messages).includes('secret oversized original'), false);
  assert.equal(aborted.n, 1);
  assert.ok(notices.some(item => item.includes('fail-open') || item.includes('Handoff fault')));
});

test('model_select engine failure fails closed for an explicitly bounded context', async () => {
  const handlers = new Map<string, (event: any, ctx: any) => unknown>();
  const pi = { on(name: string, handler: (event: any, ctx: any) => unknown) { handlers.set(name, handler); } };
  const controller = createHandoffController({ budget: DEFAULT_HANDOFF_BUDGET, engine: async () => { throw new Error('owner unavailable'); } });
  const { installHandoffHooks } = await import('../src/handoff/hooks.ts');
  installHandoffHooks(pi as never, controller, async () => { throw new Error('owner unavailable'); });
  const aborted = { n: 0 };
  const ctx = {
    cwd: '/offline/project', model: { provider: 'offline', id: 'b' }, getSystemPrompt: () => 'short system prompt',
    abort: () => { aborted.n += 1; }, ui: { notify: () => undefined },
    sessionManager: { getSessionId: () => 's-failed' },
  };
  await handlers.get('model_select')!({
    model: { provider: 'offline', id: 'b' }, previousModel: { provider: 'offline', id: 'a' }, source: 'set',
  }, ctx);
  const result = await handlers.get('context')!({ messages: [user('private original')] }, ctx) as { messages: HandoffMessage[] };
  assert.equal(JSON.stringify(result.messages).includes('private original'), false);
  assert.match(JSON.stringify(result.messages), /failed safely/);
  assert.equal(aborted.n, 1);
});

test('context is bounded before a switch and stays bounded when model_select is missed', async t => {
  const root = await mkdtemp(join(tmpdir(), 'handoff-observe-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const engine = new MemoryEngine({ root, scopeId: 'p_observe', sessionId: 's-observe', provider: new TestEmbeddingProvider() });
  t.after(() => engine.dispose());
  const controller = createHandoffController({
    engine: async () => engine,
    budget: { maxTokens: 8_000, maxBytes: 2_000, maxMessages: 2, toolSchemaReserveTokens: 10 },
  });
  const ctx = {
    cwd: root, model: { provider: 'offline', id: 'a' }, getSystemPrompt: () => 'system', abort: () => undefined,
    ui: { notify: () => undefined }, sessionManager: { getSessionId: () => 's-observe' },
  };
  const history = [user('old history'), assistant('old answer'), user('current instruction')];
  assert.deepEqual((await controller.safeContext(history, ctx as never)).messages, [user('current instruction')]);
  ctx.model = { provider: 'offline', id: 'b' };
  const bounded = await controller.safeContext(history, ctx as never);
  assert.deepEqual(bounded.messages, [user('current instruction')]);
  assert.equal(controller.get(engine.scopeId, 's-observe')?.active, true);
});

test('measured active tool definitions replace the fixed fallback reserve and can refuse safely', async t => {
  const root = await mkdtemp(join(tmpdir(), 'handoff-overhead-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const engine = new MemoryEngine({ root, scopeId: 'p_overhead', sessionId: 's-overhead', provider: new TestEmbeddingProvider() });
  t.after(() => engine.dispose());
  const controller = createHandoffController({
    engine: async () => engine,
    budget: { maxTokens: 100, maxBytes: 1_000, maxMessages: 2, toolSchemaReserveTokens: 1 },
    toolOverhead: () => ({ toolSchemaTokens: 150, toolSchemaBytes: 600 }),
  });
  controller.activate(engine.scopeId, root, 's-overhead', 'switched');
  const aborted = { n: 0 };
  const result = await controller.safeContext([user('latest')], {
    cwd: root, model: { provider: 'offline', id: 'b' }, getSystemPrompt: () => 'short system policy',
    abort: () => { aborted.n += 1; }, ui: { notify: () => undefined },
    sessionManager: { getSessionId: () => 's-overhead' },
  } as never);
  assert.deepEqual(result.messages, []);
  assert.equal(aborted.n, 1);
});

test('public context hook output is the bounded provider payload across A→B→A', async t => {
  const root = await mkdtemp(join(tmpdir(), 'handoff-ab-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const engine = new MemoryEngine({ root, scopeId: 'p_handoff', sessionId: 's-ab', provider: new TestEmbeddingProvider() });
  t.after(() => engine.dispose());
  const handlers = new Map<string, (event: any, ctx: any) => unknown>();
  const pi = { on(name: string, handler: (event: any, ctx: any) => unknown) { handlers.set(name, handler); } };
  const controller = createHandoffController({ engine: async () => engine, budget: { maxTokens: 8_000, maxBytes: 2_000, maxMessages: 4, toolSchemaReserveTokens: 20 } });
  const { installHandoffHooks } = await import('../src/handoff/hooks.ts');
  installHandoffHooks(pi as never, controller, async () => engine);
  const modelA = { provider: 'offline', id: 'a' };
  const modelB = { provider: 'offline', id: 'b' };
  const ctx = {
    cwd: root,
    model: modelB,
    getSystemPrompt: () => 'short system prompt',
    abort: () => undefined,
    ui: { notify: () => undefined },
    sessionManager: { getSessionId: () => 's-ab' },
  };
  await handlers.get('model_select')!({ model: modelB, previousModel: modelA, source: 'set' }, ctx);
  const history = [
    user('ancient dump '.repeat(40)), assistant('old'),
    user('keep this latest requirement'), assistant('call', ['t9']), tool('t9', 'ok'),
  ];
  const first = await handlers.get('context')!({ messages: history }, ctx) as { messages: HandoffMessage[] };
  ctx.model = modelA;
  await handlers.get('model_select')!({ model: modelA, previousModel: modelB, source: 'cycle' }, ctx);
  const second = await handlers.get('context')!({ messages: history }, ctx) as { messages: HandoffMessage[] };
  const outgoingProviderPayloads = [first.messages, second.messages];
  for (const payload of outgoingProviderPayloads) {
    const sent = JSON.stringify(payload);
    assert.match(sent, /keep this latest requirement/);
    assert.equal(sent.includes('ancient dump'), false);
  }
  assert.deepEqual(first.messages, second.messages);
  assert.equal(handlers.get('before_provider_request')!({ payload: { secret: 'do-not-log' } }, ctx), undefined);
});

test('A to B to A stays bounded across context calls', () => {
  const messages = [
    user('first'), assistant('a'),
    user('second'), assistant('b'),
    user('third latest'), assistant('c', ['z']), tool('z', 'ok'),
  ];
  const first = selectHandoffMessages(messages, undefined, { maxTokens: 120, maxBytes: 2_000, maxMessages: 8, toolSchemaReserveTokens: 10 });
  const again = selectHandoffMessages(messages, undefined, { maxTokens: 120, maxBytes: 2_000, maxMessages: 8, toolSchemaReserveTokens: 10 });
  assert.equal(first.ok, true);
  assert.equal(again.ok, true);
  if (first.ok && again.ok) {
    assert.match(JSON.stringify(first.messages), /third latest/);
    assert.deepEqual(first.messages, again.messages);
  }
});


test('default handoff preserves every turn and opaque reasoning despite memory faults', async () => {
  const controller = createHandoffController({ engine: async () => { throw new Error('memory unavailable'); } });
  const messages: HandoffMessage[] = Array.from({ length: 500 }, (_, i) => user(`Steering ${i}: ${'x'.repeat(1000)}`));
  messages.push({ role: 'assistant', content: [{ type: 'thinking', thinking: 'retained', thinkingSignature: 'opaque' }] } as never);
  const ctx = { cwd: '/project', sessionManager: { getSessionId: () => 'long' }, abort: () => { throw new Error('must not abort'); } } as never;
  const result = await controller.safeContext(messages, ctx);
  assert.equal(result.messages, messages);
  assert.equal(result.unchanged, true);
});
