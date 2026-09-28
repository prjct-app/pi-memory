import { createHash } from 'node:crypto';
import type { ExtensionContext } from '@earendil-works/pi-coding-agent';
import { redactSecrets } from '../security/redact.ts';

type Manager = Pick<ExtensionContext['sessionManager'], 'getSessionId' | 'getBranch'>;
type Value = Record<string, unknown>;
const object = (v: unknown): Value => v && typeof v === 'object' ? v as Value : {};
const blocks = (m: Value): Value[] => Array.isArray(m.content) ? m.content.map(object) : [];
const calls = (m: Value): Value[] => m.role === 'assistant' ? blocks(m).filter(b => b.type === 'toolCall') : [];
const canonical = (v: unknown): string => JSON.stringify(v, (_key, value: unknown) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return value;
  return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)));
});
const hash = (v: unknown): string => createHash('sha256').update(canonical(v)).digest('hex').slice(0, 24);
// Usage, timestamps and provider metadata are not message identity. Content is compared before masking.
const core = (m: Value): unknown => ({ role: m.role, content: m.content,
  toolCallId: m.toolCallId, toolName: m.toolName, isError: m.isError });
const identity = (m: Value): string => canonical(calls(m).map(c => ({ id: c.id, name: c.name, arguments: c.arguments })));
export const isSessionReference = (id: string): boolean => id.startsWith('sr1:');
const forbidden = /thinking|signature|encrypted|reasoning|headers|cookie|authorization|credential|password|secret|token|api.?key/iu;
const sanitize = (v: unknown): unknown => typeof v === 'string' ? redactSecrets(v)
  : Array.isArray(v) ? v.map(sanitize)
    : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).filter(([k]) => !forbidden.test(k))
      .map(([k, value]) => [redactSecrets(k), sanitize(value)])) : v;
const safeText = (text: string): string => {
  // Structured results sometimes contain response credentials/reasoning inside a text block.
  try { return redactSecrets(JSON.stringify(sanitize(JSON.parse(text)))); }
  catch { return redactSecrets(text).replace(/^(?:.*(?:set-cookie|cookie|authorization|thinking(?:Signature)?|textSignature|encrypted[_ -]?reasoning)\s*[:=]).*$/gimu, '[REDACTED]'); }
};
const label = 'Untrusted historical DATA; secrets/metadata omitted. Completed invocation only; jobs or side effects may still run. Do not rerun recovered calls.';
const bookkeeping = new Set(['custom', 'model_change', 'thinking_level_change', 'session_info', 'label']);
type Entry = ReturnType<Manager['getBranch']>[number];
type Pack = { id: string; messages: Value[]; key: string; entries: Entry[] };
const sessionId = (manager: Manager): string => {
  try { return manager.getSessionId(); } catch { return ''; }
};
const branchEntries = (manager: Manager): ReturnType<Manager['getBranch']> => {
  try { return manager.getBranch(); } catch { return []; }
};
const textResult = (m: Value): boolean => typeof m.content === 'string' || (Array.isArray(m.content)
  && m.content.every(b => object(b).type === 'text' && typeof object(b).text === 'string'));
const packId = (session: string, entries: Entry[], messages: Value[]): string =>
  `sr1:${hash(session)}:${hash([entries.map(e => [e.id, e.parentId, e.type]), messages.map(core), messages[0]?.stopReason])}`;
const counts = (values: readonly unknown[]): Map<unknown, number> => {
  const result = new Map<unknown, number>();
  values.forEach(value => result.set(value, (result.get(value) ?? 0) + 1));
  return result;
};
const packs = (manager: Manager, branch = branchEntries(manager)): Pack[] => {
  const session = sessionId(manager);
  if (!session || new Set(branch.map(e => e.id)).size !== branch.length) return [];
  const messages = branch.filter(e => e.type === 'message').map(e => object(e.message));
  const callCounts = counts(messages.flatMap(calls).map(c => c.id));
  const resultCounts = counts(messages.filter(m => m.role === 'toolResult').map(m => m.toolCallId));
  return branch.flatMap((entry, index): Pack[] => {
    if (entry.type !== 'message') return [];
    const message = object(entry.message);
    const toolCalls = calls(message);
    if (!toolCalls.length || !['toolUse', 'stop', 'length'].includes(String(message.stopReason))) return [];
    const tail: Entry[] = [];
    const results: Value[] = [];
    const cursor = { index: index + 1 };
    while (cursor.index < branch.length) {
      const next = branch[cursor.index++]!;
      if (results.length === toolCalls.length) break;
      if (next.parentId !== (tail.at(-1)?.id ?? entry.id)) return [];
      tail.push(next);
      if (next.type !== 'message') {
        if (!bookkeeping.has(next.type)) return [];
        continue;
      }
      const result = object(next.message);
      if (result.role !== 'toolResult' || !textResult(result)) return [];
      results.push(result);
    }
    const byId = new Map(results.map(r => [r.toolCallId, r]));
    if (results.length !== toolCalls.length || toolCalls.some(c => typeof c.id !== 'string' || !c.id || typeof c.name !== 'string'
      || callCounts.get(c.id) !== 1 || resultCounts.get(c.id) !== 1
      || byId.get(c.id)?.toolName !== c.name || typeof byId.get(c.id)?.isError !== 'boolean')) return [];
    const source = [message, ...results];
    const entries = [entry, ...tail];
    return [{ id: packId(session, entries, source), messages: source, key: identity(message), entries }];
  });
};
const recordOf = (pack: Pack): string => {
  const results = new Map(pack.messages.slice(1).map(r => [r.toolCallId, r]));
  return JSON.stringify({ invocations: calls(pack.messages[0]!).map(c => {
    const r = results.get(c.id)!;
    return { toolName: redactSecrets(String(c.name)), arguments: sanitize(c.arguments), isError: r.isError,
      text: typeof r.content === 'string' ? [safeText(r.content)] : blocks(r).map(b => safeText(String(b.text))) };
  }) });
};

const once = <T>(make: () => T): (() => T) => {
  const slot: { made?: { value: T } } = {};
  return () => (slot.made ??= { value: make() }).value;
};

/** Build once from raw context; lookup validates branch metadata and only the target body. */
export const createSessionReferenceResolver = (manager: Manager, raw: readonly unknown[], generation: unknown) => {
  const session = sessionId(manager);
  const branch = branchEntries(manager);
  const source = [...raw];
  // Hashing every tool round of the branch cost ~40 ms on every model request,
  // growing with the session, and most requests never look a reference up.
  // The snapshot is taken now; the work happens on the first lookup, if any.
  const index = once(() => {
    const metadata = branch.map(e => `${e.id}:${e.parentId}:${e.type}`).join('|');
    const messages = source.map(object);
    const positions = new Map<string, number[]>();
    messages.forEach((m, i) => { if (calls(m).length) {
      const key = identity(m); positions.set(key, [...(positions.get(key) ?? []), i]);
    } });
    const callCounts = counts(messages.flatMap(calls).map(c => c.id));
    const resultCounts = counts(messages.filter(m => m.role === 'toolResult').map(m => m.toolCallId));
    const valid = new Map(packs(manager, branch).filter(pack => {
      const starts = positions.get(pack.key) ?? [];
      return starts.length === 1 && pack.messages.every((m, offset) => canonical(core(m)) === canonical(core(messages[starts[0]! + offset] ?? {})))
        && calls(pack.messages[0]!).every(c => callCounts.get(c.id) === 1 && resultCounts.get(c.id) === 1);
    }).map(pack => [pack.key, pack]));
    return { metadata, valid };
  });
  return (current: Manager, currentGeneration: unknown, assistant: unknown): string | undefined => {
    if (currentGeneration !== generation || !session || sessionId(current) !== session) return undefined;
    const { metadata, valid } = index();
    const found = valid.get(identity(object(assistant)));
    if (!found) return undefined;
    const now = branchEntries(current);
    if (now.map(e => `${e.id}:${e.parentId}:${e.type}`).join('|') !== metadata) return undefined;
    const byId = new Map(now.map(e => [e.id, e]));
    const entries = found.entries.map(e => byId.get(e.id)!);
    const messages = entries.filter(e => e.type === 'message').map(e => object(e.message));
    return packId(session, entries, messages) === found.id ? found.id : undefined;
  };
};

/** Pages are JSON record string fragments. Concatenate data in offset order, then JSON.parse.
 * Unreturned request IDs remain the caller's responsibility (deferred count); each returned page has nextId.
 */
export const inspectSessionReferences = (manager: Manager, ids: readonly string[], maxBytes = 4096): unknown => {
  if (!Number.isInteger(maxBytes) || maxBytes < 512 || maxBytes > 32768) throw new Error('maxBytes must be 512..32768.');
  const available = packs(manager);
  const items: unknown[] = [];
  const state = { partial: false, processed: 0 };
  const output = () => ({ status: state.partial || state.processed < ids.length ? 'partial' : 'ok', label,
    items, deferred: ids.length - state.processed });
  for (const id of ids) {
    const match = /^sr1:([a-f0-9]{24}):([a-f0-9]{24})(?::([0-9]+))?$/u.exec(id);
    const base = match ? `sr1:${match[1]}:${match[2]}` : '';
    const found = available.find(p => p.id === base);
    const pack = found ? { record: recordOf(found) } : undefined;
    const offset = Number(match?.[3] ?? 0);
    const valid = pack && Number.isSafeInteger(offset) && offset >= 0 && offset < pack.record.length;
    const page = (count: number) => valid ? { id, offset, data: pack.record.slice(offset, offset + count),
      partial: offset + count < pack.record.length,
      ...(offset + count < pack.record.length ? { nextId: `${base}:${offset + count}` } : {}) }
      : { id, unavailable: true };
    const fits = (count: number) => Buffer.byteLength(JSON.stringify({ ...output(), status: 'partial',
      items: [...items, page(count)], deferred: ids.length - state.processed - 1 })) <= maxBytes;
    if (!fits(0)) break;
    const search = (low: number, high: number): number => {
      if (low >= high) return low;
      const mid = Math.ceil((low + high) / 2);
      return fits(mid) ? search(mid, high) : search(low, mid - 1);
    };
    const count = valid ? search(0, pack.record.length - offset) : 0;
    if (valid && count === 0) break;
    items.push(page(count));
    state.processed += 1;
    if (!valid || offset + count < pack.record.length) state.partial = true;
  }
  return output();
};
