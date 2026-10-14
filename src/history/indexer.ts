import { open, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { redactSecrets } from '../security/redact.ts';
import { entryOf, headerOf, type SessionHeader } from './entry.ts';
import type { HistoryStore, IndexedRow } from './store.ts';

/**
 * Incremental indexing of session journals, by byte offset. Only complete
 * lines are read, so a line Pi is still writing is picked up next time. Work
 * is cut into time slices with a yield between them, so indexing a large
 * journal never holds the terminal for long.
 */

const CHUNK_BYTES = 1024 * 1024;
/** Work per slice before yielding to the event loop. */
const SLICE_MS = 12;
const REDACT_SLICE = 1024 * 1024;
const NEWLINE = 0x0a;

const pause = (): Promise<void> => new Promise(resolve => setImmediate(resolve));

/** redactSecrets bounds its input; long tool output is redacted in line-aligned slices. */
export const redactAll = (text: string): string => {
  if (text.length <= REDACT_SLICE) return redactSecrets(text);
  const cut = text.lastIndexOf('\n', REDACT_SLICE);
  const at = cut > 0 ? cut + 1 : REDACT_SLICE;
  return redactSecrets(text.slice(0, at)) + redactAll(text.slice(at));
};

const parse = (line: string): Record<string, unknown> | undefined => {
  try {
    const value = JSON.parse(line) as unknown;
    return value && typeof value === 'object' ? value as Record<string, unknown> : undefined;
  } catch {
    return undefined;
  }
};

export type IndexResult = Readonly<{ added: number; bytes: number }>;

export const indexFile = async (store: HistoryStore, path: string, signal?: AbortSignal): Promise<IndexResult> => {
  const size = (await stat(path).catch(() => undefined))?.size;
  if (size === undefined) return { added: 0, bytes: 0 };
  const known = store.file(path);
  if (size < known.bytes) store.reset(known.id);
  const start = size < known.bytes ? 0 : known.bytes;
  if (size === start) return { added: 0, bytes: start };
  const handle = await open(path, 'r');
  try {
    const state = { position: start, pending: Buffer.alloc(0), pendingAt: start, added: 0, header: undefined as SessionHeader | undefined };
    while (state.position < size && !signal?.aborted) {
      const chunk = Buffer.alloc(Math.min(CHUNK_BYTES, size - state.position));
      const { bytesRead } = await handle.read(chunk, 0, chunk.length, state.position);
      if (!bytesRead) break;
      state.position += bytesRead;
      const buffer = state.pending.length ? Buffer.concat([state.pending, chunk.subarray(0, bytesRead)]) : chunk.subarray(0, bytesRead);
      const last = buffer.lastIndexOf(NEWLINE);
      if (last < 0) { state.pending = Buffer.from(buffer); continue; }
      const slice = { rows: [] as IndexedRow[], started: performance.now() };
      const cursor = { at: 0 };
      while (cursor.at <= last) {
        const end = buffer.indexOf(NEWLINE, cursor.at);
        const line = parse(buffer.toString('utf8', cursor.at, end));
        const offset = state.pendingAt + cursor.at;
        if (line) {
          state.header = state.header ?? headerOf(line);
          const entry = entryOf(line);
          if (entry) slice.rows.push({ ...entry, text: redactAll(entry.text), chars: entry.text.length, offset, length: end - cursor.at });
        }
        cursor.at = end + 1;
        if (performance.now() - slice.started >= SLICE_MS && cursor.at <= last) {
          state.added += store.append(known.id, state.header, slice.rows, state.pendingAt + cursor.at);
          slice.rows = [];
          await pause();
          slice.started = performance.now();
        }
      }
      state.added += store.append(known.id, state.header, slice.rows, state.pendingAt + last + 1);
      state.pending = Buffer.from(buffer.subarray(last + 1));
      state.pendingAt += last + 1;
      await pause();
    }
    return { added: state.added, bytes: state.pendingAt };
  } finally {
    await handle.close();
  }
};

/** Session journals under Pi's sessions root: one folder per cwd, plus any loose files. */
export const journals = async (root: string): Promise<{ path: string; size: number; mtimeMs: number }[]> => {
  const top = await readdir(root, { withFileTypes: true }).catch(() => []);
  const paths = (await Promise.all(top.map(async entry => {
    const path = join(root, entry.name);
    if (entry.isFile()) return entry.name.endsWith('.jsonl') ? [path] : [];
    if (!entry.isDirectory()) return [];
    const inner = await readdir(path).catch(() => [] as string[]);
    return inner.filter(name => name.endsWith('.jsonl')).map(name => join(path, name));
  }))).flat();
  const stats = await Promise.all(paths.map(async path => {
    const info = await stat(path).catch(() => undefined);
    return info?.isFile() ? [{ path, size: info.size, mtimeMs: info.mtimeMs }] : [];
  }));
  return stats.flat().sort((a, b) => b.mtimeMs - a.mtimeMs);
};

export type BackfillProgress = Readonly<{ files: number; done: number; added: number }>;

/** Indexes every journal that grew since it was last seen, newest first. */
export const backfill = async (
  store: HistoryStore, root: string,
  options: { signal?: AbortSignal; skip?: string; onFile?: (progress: BackfillProgress) => boolean | void } = {},
): Promise<BackfillProgress> => {
  const files = (await journals(root)).filter(file => file.path !== options.skip && file.size > store.file(file.path).bytes);
  const progress = { files: files.length, done: 0, added: 0 };
  for (const file of files) {
    if (options.signal?.aborted) break;
    const result = await indexFile(store, file.path, options.signal);
    progress.done += 1;
    progress.added += result.added;
    if (options.onFile?.({ ...progress }) === false) break;
  }
  return { ...progress };
};
