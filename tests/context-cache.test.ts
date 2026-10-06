import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createContextWindow } from '../src/handoff/window.ts';
import { createHandoffController, installHandoffHooks } from '../src/handoff/hooks.ts';
import type { HandoffMessage } from '../src/handoff/turns.ts';

const user = (content: string): HandoffMessage => ({ role: 'user', content });
const answer = (content: string): HandoffMessage => ({ role: 'assistant', content: [{ type: 'text', text: content }] });
const budget = { maxTokens: 8_000, maxBytes: 32_768, maxMessages: 8, toolSchemaReserveTokens: 0 };
const overhead = { systemTokens: 0, systemBytes: 0 };
const context = () => ({
  cwd: '/transient', model: { provider: 'offline', id: 'a' }, getSystemPrompt: () => 'stable policy',
  abort: () => undefined, ui: { notify: () => undefined }, sessionManager: { getSessionId: () => 's' },
});

test('an identity window exposes the original messages to callers but the context hook returns no replacement', async () => {
  const handlers = new Map<string, (event: any, ctx: any) => any>();
  const engine = async () => undefined;
  const controller = createHandoffController({ engine, budget });
  installHandoffHooks({ on: (name: string, fn: any) => handlers.set(name, fn) } as never, controller, engine);
  const messages = [user('NEW_CONTEXT')];
  const selected = createContextWindow()(messages, undefined, budget, overhead);
  assert.ok(selected.ok);
  assert.equal(selected.unchanged, true);
  assert.equal(selected.messages[0], messages[0]);
  const direct = await controller.safeContext(messages, context() as never);
  assert.equal(direct.unchanged, true);
  assert.equal(direct.messages, messages, 'identity calls do not clone the message list');
  assert.deepEqual(await handlers.get('context')!({ messages }, context()), {});
});

test('the context hook returns replacement messages after eviction', async () => {
  const handlers = new Map<string, (event: any, ctx: any) => any>();
  const engine = async () => undefined;
  const controller = createHandoffController({ engine, budget });
  installHandoffHooks({ on: (name: string, fn: any) => handlers.set(name, fn) } as never, controller, engine);
  const messages = Array.from({ length: 8 }, (_, n) => [user(`user ${n}`), answer(`answer ${n}`)]).flat();
  const result = await handlers.get('context')!({ messages }, context());
  assert.deepEqual(result, { messages: messages.slice(-6) });
  assert.deepEqual(await handlers.get('context')!({ messages }, context()), result, 'watermark eviction is still a replacement');
});

test('the context hook returns replacement messages after default observation masking', async () => {
  const handlers = new Map<string, (event: any, ctx: any) => any>();
  const engine = async () => undefined;
  const controller = createHandoffController({ engine, budget: { ...budget, maxTokens: 100_000, maxBytes: 500_000, maxMessages: 20 } });
  installHandoffHooks({ on: (name: string, fn: any) => handlers.set(name, fn) } as never, controller, engine);
  const messages = [user('inspect'), ...Array.from({ length: 9 }, (_, n) => [
    { role: 'assistant', content: [{ type: 'toolCall', id: `r${n}`, name: 'read', arguments: { path: `src/${n}.ts` } }] },
    { role: 'toolResult', toolCallId: `r${n}`, toolName: 'read', content: [{ type: 'text', text: 'large output\n'.repeat(n === 0 ? 8_000 : 200) }] },
  ]).flat()];
  const result = await handlers.get('context')!({ messages }, context());
  assert.equal(result.messages.length, messages.length);
  assert.match(JSON.stringify(result.messages[2]), /elided stale read output/);
  messages.forEach((message, index) => { if (index !== 2) assert.equal(result.messages[index], message); });
  assert.deepEqual(await handlers.get('context')!({ messages }, context()), result, 'an unchanged frontier still masks');
});

test('batch pruning leaves an append-only prefix and never resurrects discarded turns', () => {
  const select = createContextWindow();
  const original = Array.from({ length: 8 }, (_, n) => [user(`user ${n}`), answer(`answer ${n}`)]).flat();
  const first = select(original, undefined, budget, overhead);
  assert.ok(first.ok);
  assert.equal(first.messages.length, 6); // 75% watermark, not full to the limit
  assert.equal(first.unchanged, undefined);
  const same = select(structuredClone(original), undefined, budget, overhead);
  assert.ok(same.ok);
  assert.deepEqual(same.messages, first.messages);
  const grown = [...original, user('next requirement')];
  const next = select(grown, undefined, budget, overhead);
  assert.ok(next.ok);
  assert.deepEqual(next.messages.slice(0, first.messages.length), first.messages);
  assert.equal(next.messages.at(-1)?.content, 'next requirement');
  // A larger budget / cheaper overhead must not re-introduce the evicted prefix.
  const larger = select(grown, undefined, { ...budget, maxMessages: 100 }, overhead);
  assert.ok(larger.ok);
  assert.deepEqual(larger.messages, next.messages);
  assert.equal(larger.unchanged, undefined, 'a retained watermark is not the full host history');
});

test('history replacement resets the watermark and retains the new summary', () => {
  const select = createContextWindow();
  select(Array.from({ length: 10 }, (_, n) => user(`old ${n}`)), undefined, budget, overhead);
  const summary: HandoffMessage = { role: 'compactionSummary', summary: 'MANUAL_COMPACTION', timestamp: 5 };
  const replacement = [summary, user('new branch requirement')];
  const next = select(replacement, undefined, budget, overhead);
  assert.ok(next.ok);
  assert.deepEqual(next.messages, replacement);
  assert.equal(next.unchanged, true, 'an existing host summary is not a synthetic checkpoint');
});

test('a synthetic checkpoint is a replacement even when every host message fits', () => {
  const messages = [user('continue')];
  const selected = createContextWindow()(messages, {
    projectId: 'p', sessionId: 's', goal: 'Finish the task', constraints: [], done: [], inProgress: [],
    blocked: [], decisions: [], evidenceRefs: [], nextSteps: [], updatedAt: '2026-01-01T00:00:00.000Z',
  }, budget, overhead);
  assert.ok(selected.ok);
  assert.equal(selected.omittedTurns, 0);
  assert.equal(selected.truncatedFields, 0);
  assert.equal(selected.messages.length, messages.length + 1);
  assert.equal(selected.unchanged, undefined);
});

test('truncation is a replacement even when message count and order are preserved', () => {
  const messages = [user('inspect'), answer('x'.repeat(100_000))];
  const selected = createContextWindow()(messages, undefined, budget, overhead);
  assert.ok(selected.ok);
  assert.ok(selected.truncatedFields > 0);
  assert.equal(selected.omittedTurns, 0);
  assert.equal(selected.messages.length, messages.length);
  assert.equal(selected.unchanged, undefined);
});

test('exact recall dedup preserves the first cached copy and re-emits after eviction', () => {
  const select = createContextWindow();
  const recall = { role: 'custom', customType: 'pi-memory-recall', content: 'same retained fact', timestamp: 1 };
  const initial = [user('first'), recall, answer('done')];
  const first = select(initial, undefined, budget, overhead);
  assert.ok(first.ok);
  const again = [...initial, user('second'), { ...recall, timestamp: 2 }];
  const second = select(again, undefined, budget, overhead);
  assert.ok(second.ok);
  assert.deepEqual(second.messages, [...first.messages, user('second')]);
  assert.equal(second.unchanged, undefined, 'recall dedup dropped a host message');
  const evicted = select([user('fresh after compact'), { ...recall, timestamp: 3 }], undefined, budget, overhead);
  assert.ok(evicted.ok);
  assert.equal(evicted.messages.length, 2);
});

test('recall survives the request that evicts its first copy, not just the next request', () => {
  const window = createContextWindow();
  const recall = { role: 'custom', customType: 'pi-memory-recall', content: 'needed fact', timestamp: 1 };
  const initial = [user('first'), recall, answer('done')];
  assert.ok(window(initial, undefined, budget, overhead).ok);
  const history = [...initial, ...Array.from({ length: 4 }, (_, n) => [user(`next ${n}`), answer('done')]).flat(),
    user('need fact again'), { ...recall, timestamp: 2 }];
  const result = window(history, undefined, budget, overhead);
  assert.ok(result.ok);
  assert.ok(result.messages.some(message => message.content === recall.content), 'evicting the original must recover the new copy immediately');
});

test('distinct recall is retained without modifying previously cached messages', () => {
  const select = createContextWindow();
  const history = [user('one'), { role: 'custom', customType: 'pi-memory-recall', content: 'fact A' }, answer('done')];
  const first = select(history, undefined, budget, overhead);
  assert.ok(first.ok);
  const next = select([...history, user('two'), { role: 'custom', customType: 'pi-memory-recall', content: 'fact B' }], undefined, budget, overhead);
  assert.ok(next.ok);
  assert.deepEqual(next.messages.slice(0, first.messages.length), first.messages);
  assert.match(JSON.stringify(next.messages), /fact B/);
  assert.equal(next.unchanged, true, 'distinct recall keeps the host history intact');
});

test('a mandatory oversized current turn is refused, not silently clipped or sent in full', async () => {
  const controller = createHandoffController({ engine: async () => undefined, budget });
  const result = await controller.safeContext([user('PRIVATE_OVERSIZED ' + 'x'.repeat(100_000))], context() as never);
  assert.match(JSON.stringify(result.messages), /failed safely/);
  assert.doesNotMatch(JSON.stringify(result.messages), /PRIVATE_OVERSIZED/);
  assert.equal(result.unchanged, undefined);
});

test('concurrent context waits for pending ownership and cannot replay original history', async () => {
  const deferred: { reject?: (error: Error) => void } = {};
  const controller = createHandoffController({ engine: () => new Promise((_resolve, reject) => { deferred.reject = reject; }), budget });
  const ctx = context();
  const activation = controller.prepare(ctx as never, 'switch');
  const bounded = controller.safeContext([user('PRIVATE_HISTORY')], ctx as never);
  await Promise.resolve();
  assert.ok(deferred.reject);
  deferred.reject(new Error('invalid owner'));
  await activation;
  const result = await bounded;
  assert.doesNotMatch(JSON.stringify(result.messages), /PRIVATE_HISTORY/);
  assert.match(JSON.stringify(result.messages), /failed safely/);
});

test('compaction belongs to Pi: automatic, overflow and manual compaction are never cancelled', () => {
  const handlers = new Map<string, (event: any, ctx: any) => any>();
  const engine = async () => undefined;
  const controller = createHandoffController({ engine, budget });
  installHandoffHooks({ on: (name: string, fn: any) => handlers.set(name, fn) } as never, controller, engine);
  assert.equal(handlers.has('session_before_compact'), false);
});

test('a session clear after activation settles cannot transport an older context into the same session key', async () => {
  const deferred: { resolve?: (value: undefined) => void } = {};
  const engine = { current: undefined as any };
  const controller = createHandoffController({
    engine: () => engine.current ? Promise.resolve(engine.current) : new Promise(resolve => { deferred.resolve = resolve; }), budget,
  });
  const ctx = context();
  const activating = controller.prepare(ctx as never, 'old switch');
  const clearing = activating.then(() => {
    controller.clear();
    engine.current = { scopeId: 'p_new', projection: { operationalCheckpoint: () => undefined } };
    controller.activate('p_new', ctx.cwd, 's', 'new session with same key');
  });
  const oldContext = controller.safeContext([user('OLD_PRIVATE_CONTEXT')], ctx as never);
  await Promise.resolve();
  deferred.resolve!(undefined);
  await clearing;
  const oldResult = await oldContext;
  assert.match(JSON.stringify(oldResult.messages), /failed safely/);
  assert.doesNotMatch(JSON.stringify(oldResult.messages), /OLD_PRIVATE_CONTEXT/);
  const current = await controller.safeContext([user('NEW_CONTEXT')], ctx as never);
  assert.deepEqual(current.messages, [user('NEW_CONTEXT')]);
});

test('refusal itself respects tiny message budgets and does not add an oversized SAFE message', async () => {
  const controller = createHandoffController({ engine: async () => undefined,
    budget: { maxTokens: 10, maxBytes: 30, maxMessages: 1, toolSchemaReserveTokens: 0 } });
  const result = await controller.safeContext([user('PRIVATE_LARGE ' + 'x'.repeat(1000))], context() as never);
  assert.deepEqual(result.messages, []);
});

test('a compaction summary stays at the head on every later append-only request', () => {
  const select = createContextWindow();
  const summary: HandoffMessage = { role: 'compactionSummary', summary: 'MANUAL_COMPACTION', timestamp: 5 };
  const history = [summary, user('first'), answer('done')];
  const first = select(history, undefined, budget, overhead);
  assert.ok(first.ok);
  const next = select([...structuredClone(history), user('second')], undefined, budget, overhead);
  assert.ok(next.ok);
  assert.equal(next.messages[0]?.role, 'compactionSummary', 'the floor must not filter out the summary');
  assert.deepEqual(next.messages.slice(0, first.messages.length), first.messages, 'append-only prefix is preserved');
});

test('UI-only details do not count against the byte budget', () => {
  const select = createContextWindow({ enabled: false, keepRounds: 8, minTokens: 300, advanceTokens: 24_000 });
  const rounds = Array.from({ length: 3 }, (_, n) => [
    { role: 'assistant', content: [{ type: 'toolCall', id: `j${n}`, name: 'agent_jobs', arguments: {} }] },
    { role: 'toolResult', toolCallId: `j${n}`, toolName: 'agent_jobs', content: [{ type: 'text', text: 'running' }],
      details: { log: 'x'.repeat(20_000) } },
  ] as HandoffMessage[]).flat();
  const messages = [user('poll the job'), ...rounds];
  const selected = select(messages, undefined, { ...budget, maxMessages: 20 }, overhead);
  assert.ok(selected.ok);
  assert.equal(selected.unchanged, true, 'details are never sent, so no tool round may be evicted for them');
});

test('a history batch flushes pending observation masking in the same request', () => {
  const select = createContextWindow({ enabled: true, keepRounds: 1, minTokens: 10, advanceTokens: 1_000_000 },
    { enabled: true, targetTokens: 100, advanceTokens: 100 });
  const round = (id: string): HandoffMessage[] => [
    { role: 'assistant', content: [{ type: 'toolCall', id, name: 'read', arguments: { path: `${id}.ts` } }] },
    { role: 'toolResult', toolCallId: id, toolName: 'read', content: [{ type: 'text', text: 'source line\n'.repeat(400) }] },
  ] as HandoffMessage[];
  const messages = [user('inspect'), ...round('a'), ...round('b'), answer('found it'),
    user('now fix it'), ...round('c'), ...round('d'), ...round('e')];
  const big = { maxTokens: 1_000_000, maxBytes: 10_000_000, maxMessages: 100, toolSchemaReserveTokens: 0 };
  const selected = select(messages, undefined, big, overhead);
  assert.ok(selected.ok);
  // Masking alone would never advance (1M batch); the history batch triggers it,
  // so every stale output is rewritten in this one request instead of two.
  assert.equal(selected.observations?.masked, 4, 'stale outputs are masked with the same prefix rewrite');
  const next = select([...structuredClone(messages), ...round('f')], undefined, big, overhead);
  assert.ok(next.ok);
  assert.deepEqual(next.messages.slice(0, selected.messages.length - 2).map(m => JSON.stringify(m)),
    selected.messages.slice(0, selected.messages.length - 2).map(m => JSON.stringify(m)), 'no second rewrite of the retained prefix');
});
