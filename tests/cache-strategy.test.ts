import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createContextWindow } from '../src/handoff/window.ts';
import { budgetForModel, estimateHandoffTokens } from '../src/handoff/select.ts';
import { renderMemoryEnvelope, uniqueMemory } from '../src/handoff/memory-envelope.ts';
import { retainHistory } from '../src/handoff/history.ts';
import { maskObservations } from '../src/handoff/observations.ts';
import { turnIsComplete, type HandoffMessage } from '../src/handoff/turns.ts';

const budget = budgetForModel({ contextWindow: 500_000, maxTokens: 500_000 });
const overhead = { systemTokens: 0, systemBytes: 0, toolSchemaTokens: 0 };
const noMask = { enabled: false, keepRounds: 8, minTokens: 300, advanceTokens: 24_000 };
const envelope = (revision: string, recall?: string): HandoffMessage => {
  const memory = { version: 1 as const, revision, snapshot: revision === 'empty' ? 'No eligible facts' : `DIGEST_${revision}`, ...(recall ? { recall } : {}) };
  return { role: 'custom', ...{ customType: 'pi-memory-recall', details: { memory } }, content: renderMemoryEnvelope(memory) };
};
const round = (n: number, size = 8000): HandoffMessage[] => [
  { role: 'assistant', content: [{ type: 'thinking', thinking: 'signed', thinkingSignature: `sig${n}` },
    { type: 'toolCall', id: `r${n}`, name: 'read', arguments: { path: `file${n}` } }] },
  { role: 'toolResult', toolCallId: `r${n}`, content: [{ type: 'text', text: 'x'.repeat(size) }] },
];
const user = (content: string): HandoffMessage => ({ role: 'user', content });

const economicPolicy = { enabled: true, targetTokens: 1, advanceTokens: 1 };
test('retired history stays serialized identically when observation output shrinks', () => {
  const source = [user('old'),
    { role: 'assistant', content: [{ type: 'toolCall', id: 'r0', name: 'read', arguments: {} }] },
    round(0, 20_000)[1]!, user('current')];
  const retired = retainHistory(source, 0, economicPolicy);
  assert.equal(retired.frontier, 3);
  const masked = maskObservations(source, 3, { enabled: true, keepRounds: 0, minTokens: 1, advanceTokens: 1 });
  assert.equal(masked.masked, 1);
  const maskedCost = masked.messages.slice(1, 3).reduce((sum, message) => sum + estimateHandoffTokens(message), 0);
  assert.ok(maskedCost <= estimateHandoffTokens(retired.messages[1]!), 'real observation stub removes retirement savings');
  assert.deepEqual(retainHistory(masked.messages, retired.frontier, economicPolicy), retired);
});

test('economic frontier accounts for negative intervals and never advances without net savings', () => {
  const small = [user('old'), ...round(0, 0), user('current')];
  assert.equal(retainHistory(small, 0, economicPolicy).frontier, 0);
  const history = [user('old'), ...round(0, 0), ...round(1, 20_000), user('current')];
  const retired = retainHistory(history, 0, economicPolicy);
  assert.equal(retired.frontier, 5);
  assert.equal(retired.messages.length, 4, 'negative interval crossed by frontier must also retire');
  const partial = retainHistory(history, 3, { enabled: true, targetTokens: 1_000_000, advanceTokens: 1 });
  const historicalTokens = partial.messages.slice(0, -1).reduce((sum, message) => sum + estimateHandoffTokens(message), 0);
  const advanced = retainHistory(history, 3, { enabled: true, targetTokens: historicalTokens - 1, advanceTokens: 1 });
  assert.equal(advanced.frontier, 5, 'covered negative savings count toward batch threshold');
});

test('opaque, mixed and duplicate compatibility tool calls never retire', () => {
  const calls: HandoffMessage[] = [
    { role: 'assistant', toolCalls: [{ id: 'r0' }], content: [{ type: 'thinking', thinking: 'opaque' }] },
    { ...round(0)[0]!, toolCalls: [{ id: 'r0' }] },
    { ...round(0)[0]!, toolCalls: [{ id: 'r0' }, { id: 'r0' }] },
    { role: 'assistant', content: [
      { type: 'toolCall', id: 'r0', name: 'read', arguments: {} },
      { type: 'toolCall', id: 'r0', name: 'read', arguments: {} },
    ] },
  ];
  for (const call of calls) {
    const history = [user('old'), call, round(0, 20_000)[1]!, user('new')];
    assert.deepEqual(retainHistory(history, 0, economicPolicy).messages, history);
  }
});

test('retained envelopes append recall without repeating digest; snapshots revoke and A-B-A is ordered', () => {
  const a = envelope('A', 'recall 1');
  const input = [a, envelope('A', 'recall 2'), envelope('A', 'recall 2'), envelope('B'), envelope('A'), envelope('empty')];
  const result = uniqueMemory(input);
  assert.equal(result[0], a);
  assert.equal(result[1]?.content, 'recall 2');
  assert.equal(result.length, 5);
  assert.match(String(result[2]?.content), /DIGEST_B/);
  assert.match(String(result[3]?.content), /DIGEST_A/);
  assert.match(String(result[4]?.content), /replaces earlier rule snapshots/);
  assert.match(String(result[4]?.content), /No eligible facts/);
  assert.deepEqual(uniqueMemory([input[1]!]), [input[1]], 'evicted digest must be delivered again');
});

test('envelope survives the same selection that evicts its first copy', () => {
  const window = createContextWindow(noMask);
  const initial = [user('first'), envelope('A', 'recall')];
  const small = { ...budget, maxMessages: 6 };
  window(initial, undefined, small, overhead);
  const history = [...initial, ...Array.from({ length: 8 }, (_, n) => user(`next${n}`)), envelope('A', 'recall')];
  const result = window(history, undefined, small, overhead);
  assert.ok(result.ok);
  assert.match(JSON.stringify(result.messages), /DIGEST_A/);
  assert.match(JSON.stringify(result.messages), /recall/);
});

test('same-size source replacement resets watermark; identical clones retain it', () => {
  const window = createContextWindow(noMask);
  const original = Array.from({ length: 12 }, (_, n) => ({ ...user(`old ${n}`), timestamp: n }));
  const small = { ...budget, maxMessages: 6 };
  const first = window(original, undefined, small, overhead);
  assert.ok(first.ok);
  const clone = window(structuredClone(original), undefined, budget, overhead);
  assert.ok(clone.ok);
  assert.deepEqual(clone.messages, first.messages);
  const changed = structuredClone(original);
  changed[0] = { ...changed[0]!, content: 'new 0' };
  const next = window(changed, undefined, budget, overhead);
  assert.ok(next.ok);
  assert.deepEqual(next.messages, changed, 'equal role/time/id/size cannot prove unchanged history');
});

test('economic batches preserve all requirements, conclusions, checkpoints, pairing and oversized current read', () => {
  const policy = { enabled: true, targetTokens: 3000, advanceTokens: 4000 };
  const history = [user('ORIGINAL OPERATOR CONSTRAINT'), ...round(0),
    { role: 'assistant', content: [{ type: 'text', text: 'Verified conclusion' }] }, user('second'), ...round(1),
    user('current'), ...round(2, 120_000)];
  const checkpoint = { role: 'compactionSummary', summary: 'CONTINUITY', timestamp: 1 };
  const window = createContextWindow(noMask, policy);
  const first = window([checkpoint, ...history], undefined, budget, overhead);
  assert.ok(first.ok);
  assert.ok(first.messages.includes(history.at(-1)!));
  assert.ok(first.messages.includes(checkpoint));
  assert.match(JSON.stringify(first.messages), /ORIGINAL OPERATOR CONSTRAINT/);
  assert.match(JSON.stringify(first.messages), /Verified conclusion/);
  assert.equal(first.truncatedFields, 0);
  assert.ok(turnIsComplete({ messages: first.messages, toolCallIds: [] }));
  const expanded = [...history.slice(0, -3), user('another'), ...round(3, 24_000), ...history.slice(-3)];
  const selected = createContextWindow(noMask, policy)([checkpoint, ...expanded], undefined, budget, overhead);
  assert.ok(selected.ok);
  assert.match(JSON.stringify(selected.messages), /Retired completed read-only/);
  assert.equal(selected.truncatedFields, 0);
  assert.deepEqual(selected.messages.slice(-3), history.slice(-3));
  assert.ok(turnIsComplete({ messages: selected.messages, toolCallIds: [] }));
});

test('economic boundary stays fixed during new current tool rounds and idle; pending and side effects never retire', () => {
  const policy = { enabled: true, targetTokens: 1000, advanceTokens: 1000 };
  const history = [user('requirements'), ...round(0, 20_000), user('current'), ...round(1)];
  const window = createContextWindow(noMask, policy);
  const first = window(history, undefined, budget, overhead);
  assert.ok(first.ok);
  const next = window([...structuredClone(history), ...round(2)], undefined, budget, overhead);
  assert.ok(next.ok);
  assert.deepEqual(next.messages.slice(0, first.messages.length), first.messages);
  assert.match(JSON.stringify(first.messages), /Retired completed read-only/);
  const sideEffects = [user('old'), { role: 'assistant', content: [{ type: 'toolCall', id: 'w', name: 'write', arguments: {} }] },
    { role: 'toolResult', toolCallId: 'w', content: 'x'.repeat(40_000) }, user('new')];
  assert.deepEqual(retainHistory(sideEffects, 0, policy).messages, sideEffects);
  const pending = [user('old'), round(0)[0]!, user('new')];
  assert.deepEqual(retainHistory(pending, 0, policy).messages, pending);
  // The existing hard selector may normalize pending flow; the economic
  // transform itself must never remove either the call or current request.
});
