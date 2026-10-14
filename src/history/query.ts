import { open } from 'node:fs/promises';
import { entryOf, type EntryKind } from './entry.ts';
import { redactAll } from './indexer.ts';
import { projectRoot } from './project.ts';
import type { EntryHit, Filters, HistoryStore } from './store.ts';

/**
 * memory_context: search, read and list the session history. Text always
 * comes from the journal line the index points at, redacted on the way out.
 */

export type Scope = 'session' | 'project' | 'global';

export type HistoryRequest = Readonly<{
  action: 'search' | 'read' | 'summaries';
  query?: string;
  entryId?: string;
  sessionId?: string;
  since?: string;
  until?: string;
  scope?: Scope;
  kind?: EntryKind;
  role?: string;
  offset?: number;
}>;

export type HistoryContext = Readonly<{ sessionId: string; cwd: string; home?: string; indexing?: Readonly<{ done: number; files: number }> }>;

export const PAGE = 20;
export const READ_CHARS = 16_000;
const EXCERPT_CHARS = 320;
const CANDIDATES = 400;
const DAY_MS = 86_400_000;

/** A calendar day (YYYY-MM-DD) is the user's local day; anything else must parse as a date. */
export const boundary = (value: string | undefined, end: boolean): string | undefined => {
  if (!value) return undefined;
  const day = /^(\d{4})-(\d{2})-(\d{2})$/u.exec(value.trim());
  const date = day
    ? new Date(Number(day[1]), Number(day[2]) - 1, Number(day[3]), end ? 23 : 0, end ? 59 : 0, end ? 59 : 0, end ? 999 : 0)
    : new Date(value);
  if (Number.isNaN(date.getTime())) throw new Error(`${end ? 'until' : 'since'} must be a date such as 2026-10-14 or an ISO timestamp.`);
  return date.toISOString();
};

/** The original text of an indexed entry, read from its journal line. */
export const entryText = async (hit: Pick<EntryHit, 'path' | 'offset' | 'length'>): Promise<string | undefined> => {
  const handle = await open(hit.path, 'r').catch(() => undefined);
  if (!handle) return undefined;
  try {
    const buffer = Buffer.alloc(hit.length);
    await handle.read(buffer, 0, hit.length, hit.offset);
    const entry = entryOf(JSON.parse(buffer.toString('utf8')) as Record<string, unknown>);
    return entry ? redactAll(entry.text) : undefined;
  } catch {
    return undefined;
  } finally {
    await handle.close();
  }
};

const fold = (text: string): string => text.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();

export const excerpt = (text: string, query?: string): string => {
  const terms = (query?.match(/[\p{L}\p{N}]+/gu) ?? []).map(fold);
  const folded = fold(text);
  const at = terms.map(term => folded.indexOf(term)).filter(index => index >= 0).sort((a, b) => a - b)[0] ?? 0;
  const start = Math.max(0, at - 80);
  const window = text.slice(start, start + EXCERPT_CHARS).replace(/\s+/gu, ' ').trim();
  return `${start > 0 ? '…' : ''}${window}${start + EXCERPT_CHARS < text.length ? '…' : ''}`;
};

const filtersFor = (store: HistoryStore, request: HistoryRequest, context: HistoryContext, kinds?: readonly EntryKind[]) => {
  const scope = request.scope ?? 'project';
  const root = scope === 'project' ? projectRoot(context.cwd, store.cwds(), context.home) : undefined;
  const filters: Filters = {
    ...(scope === 'session' ? { sessionId: request.sessionId ?? context.sessionId } : request.sessionId ? { sessionId: request.sessionId } : {}),
    ...(root ? { under: root } : {}),
    since: boundary(request.since, false),
    until: boundary(request.until, true),
    ...(kinds ? { kinds } : request.kind ? { kinds: [request.kind] } : {}),
    ...(request.role ? { role: request.role } : {}),
  };
  return { filters, scope: { scope, ...(root ? { project: root } : {}) } };
};

const view = async (hit: EntryHit, context: HistoryContext, query?: string) => {
  const text = await entryText(hit);
  return {
    entryId: hit.entryId, sessionId: hit.sessionId, timestamp: hit.timestamp, kind: hit.kind, role: hit.role,
    ...(hit.name ? { name: hit.name } : {}), ...(hit.tokensBefore ? { tokensBefore: hit.tokensBefore } : {}),
    ...(hit.sessionId === context.sessionId ? { currentSession: true } : { cwd: hit.cwd }),
    characters: hit.chars, excerpt: text === undefined ? '[journal line unavailable]' : excerpt(text, query),
  };
};

/** bm25 decides relevance; among relevant entries, newer ones rank higher (half weight after a month). */
const rerank = (hits: readonly EntryHit[], now: number): EntryHit[] => [...hits]
  .map(hit => ({ hit, value: -hit.score / (1 + Math.max(0, now - Date.parse(hit.timestamp)) / (30 * DAY_MS)) }))
  .sort((a, b) => b.value - a.value).map(item => item.hit);

export const runHistory = async (store: HistoryStore, request: HistoryRequest, context: HistoryContext, now = Date.now()) => {
  const offset = Math.max(0, Math.floor(request.offset ?? 0));
  const indexing = context.indexing && context.indexing.done < context.indexing.files
    ? { indexing: `${context.indexing.done} of ${context.indexing.files} older journals indexed so far; results may be incomplete.` } : {};

  if (request.action === 'read') {
    if (!request.entryId) throw new Error('read needs entryId (from search or summaries).');
    const found = store.find(request.entryId, request.sessionId);
    const hit = found.find(item => item.sessionId === context.sessionId) ?? (found.length === 1 ? found[0] : undefined);
    if (!hit) {
      if (!found.length) return { status: 'not_found', entryId: request.entryId, ...indexing };
      return { status: 'ambiguous', entryId: request.entryId, hint: 'Pass sessionId to choose one.',
        candidates: found.map(item => ({ sessionId: item.sessionId, timestamp: item.timestamp, kind: item.kind, cwd: item.cwd })) };
    }
    const text = await entryText(hit);
    if (text === undefined) return { status: 'unavailable', entryId: hit.entryId, sessionId: hit.sessionId, hint: 'The session journal no longer has this line.' };
    const page = text.slice(offset, offset + READ_CHARS);
    const next = offset + page.length;
    return { status: 'ok', entryId: hit.entryId, sessionId: hit.sessionId, timestamp: hit.timestamp, kind: hit.kind, role: hit.role,
      ...(hit.name ? { name: hit.name } : {}), ...(hit.tokensBefore ? { tokensBefore: hit.tokensBefore } : {}),
      totalCharacters: text.length, text: page, nextOffset: next < text.length ? next : null };
  }

  if (request.action === 'summaries') {
    const { filters, scope } = filtersFor(store, request, context, ['compaction', 'branch_summary']);
    const { hits, total } = store.list(filters, PAGE, offset);
    return { status: hits.length ? 'ok' : 'empty', ...scope, totalMatches: total, ...indexing,
      entries: await Promise.all(hits.map(hit => view(hit, context))),
      nextOffset: offset + hits.length < total ? offset + hits.length : null };
  }

  const { filters, scope } = filtersFor(store, request, context);
  const query = request.query?.trim() ?? '';
  if (!query) {
    const { hits, total } = store.list(filters, PAGE, offset);
    return { status: hits.length ? 'ok' : 'empty', ...scope, totalMatches: total, ...indexing,
      entries: await Promise.all(hits.map(hit => view(hit, context))),
      nextOffset: offset + hits.length < total ? offset + hits.length : null };
  }
  const { hits, total, match } = store.search(query, filters, Math.max(CANDIDATES, offset + PAGE));
  const page = rerank(hits, now).slice(offset, offset + PAGE);
  const ranked = Math.min(total, hits.length);
  return { status: page.length ? 'ok' : 'empty', ...scope, match, totalMatches: total, ...indexing,
    ...(match === 'any' ? { hint: 'No entry has every term; these have some of them.' } : {}),
    entries: await Promise.all(page.map(hit => view(hit, context, query))),
    nextOffset: offset + page.length < ranked ? offset + page.length : null };
};
