import { rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, sep } from 'node:path';
import type { HistoryStore } from './store.ts';

/**
 * Mechanical garbage collection of session journals: no model judges value.
 * A journal is junk when it never compacted and ran in a temporary folder
 * (benchmarks, probes), received no instruction at all (no user message and
 * no team or extension message: a teammate driven only by team messages is
 * real work), or got two replies or fewer. Recent
 * journals are never candidates: another terminal may still be writing them.
 * Deletion is a hard delete of the journal and its index rows, and only of the
 * exact files a preview listed, unchanged since.
 */

export type JunkReason = 'temporary folder' | 'no instructions' | 'two replies or fewer';
export type Candidate = Readonly<{ path: string; cwd: string | null; startedAt: string | null; bytes: number; mtimeMs: number; reason: JunkReason }>;

export const QUIET_MS = 24 * 60 * 60 * 1000;
const TEMPORARY = [...new Set([tmpdir(), '/tmp', '/private/tmp', '/private/var/folders', '/var/folders'].map(path => resolve(path)))];

const temporary = (cwd: string | null): boolean => Boolean(cwd) && TEMPORARY.some(root => resolve(cwd!) === root || resolve(cwd!).startsWith(`${root}${sep}`));

export const junkReason = (row: { cwd: string | null; users: number; messages: number; replies: number; summaries: number }): JunkReason | undefined => {
  // A session long enough to compact holds real work, wherever it ran.
  if (row.summaries) return undefined;
  if (temporary(row.cwd)) return 'temporary folder';
  if (!row.users && !row.messages) return 'no instructions';
  if (row.replies <= 2) return 'two replies or fewer';
  return undefined;
};

export const gcCandidates = async (store: HistoryStore, keep: readonly string[], now = Date.now()): Promise<Candidate[]> => {
  const kept = new Set(keep.map(path => resolve(path)));
  const rows = store.journalStats().flatMap(row => {
    const reason = kept.has(resolve(row.path)) ? undefined : junkReason(row);
    return reason ? [{ row, reason }] : [];
  });
  const found = await Promise.all(rows.map(async ({ row, reason }) => {
    const info = await stat(row.path).catch(() => undefined);
    if (!info?.isFile() || now - info.mtimeMs < QUIET_MS) return [];
    return [{ path: row.path, cwd: row.cwd, startedAt: row.startedAt, bytes: info.size, mtimeMs: info.mtimeMs, reason }];
  }));
  return found.flat().sort((a, b) => b.bytes - a.bytes);
};

/** Deletes the previewed journals that are still exactly as previewed. */
export const collect = async (store: HistoryStore, candidates: readonly Candidate[]) => {
  const outcome = { deleted: 0, bytes: 0, skipped: 0 };
  for (const candidate of candidates) {
    const info = await stat(candidate.path).catch(() => undefined);
    if (info && (info.size !== candidate.bytes || info.mtimeMs !== candidate.mtimeMs)) { outcome.skipped += 1; continue; }
    if (info) await rm(candidate.path, { force: true });
    store.forget(candidate.path);
    outcome.deleted += 1;
    outcome.bytes += candidate.bytes;
  }
  return outcome;
};

export const describeCandidates = (candidates: readonly Candidate[], examples = 8): string => {
  const total = candidates.reduce((sum, item) => sum + item.bytes, 0);
  const reasons = Object.entries(candidates.reduce<Record<string, number>>((counts, item) => ({ ...counts, [item.reason]: (counts[item.reason] ?? 0) + 1 }), {}))
    .map(([reason, count]) => `${count} ${reason}`).join(', ');
  const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;
  return [
    `${candidates.length} junk sessions, ${mb(total)}${reasons ? ` (${reasons})` : ''}.`,
    ...candidates.slice(0, examples).map(item => `  ${mb(item.bytes)}  ${item.startedAt?.slice(0, 10) ?? '?'}  ${item.reason}  ${item.cwd ?? item.path}`),
    ...(candidates.length > examples ? [`  … ${candidates.length - examples} more`] : []),
  ].join('\n');
};
