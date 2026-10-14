import { randomUUID } from 'node:crypto';
import { rm } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { StringEnum, Type } from '@earendil-works/pi-ai';
import { getAgentDir, type ExtensionAPI, type ExtensionContext } from '@earendil-works/pi-coding-agent';
import { ENTRY_KINDS } from './history/entry.ts';
import { collect, describeCandidates, gcCandidates, type Candidate } from './history/gc.ts';
import { backfill, indexFile, type BackfillProgress } from './history/indexer.ts';
import { runHistory, type HistoryRequest } from './history/query.ts';
import { openHistoryStore, type HistoryStore } from './history/store.ts';

/**
 * Session-history memory. Pi keeps every session journal; this indexes them
 * so the agent can search what was compacted away or happened in an earlier
 * session, read any entry by id, and list past compaction summaries.
 *
 * Nothing reaches the model unless it calls memory_context: no injected
 * recall, no rewritten context, no retired turns. The journals stay the only
 * source of truth; the index is derived and can be rebuilt at any time.
 */

export type MemoryExtensionOptions = Readonly<{
  /** Where history.sqlite lives (default: PI_MEMORY_HOME, PRJCT_HOME or ~/.prjct, then pi-memory/). */
  home?: string;
  /** Pi's sessions root (default: <agent dir>/sessions). */
  sessionsRoot?: string;
  /** Delay before older journals are indexed, so startup never waits on it. */
  backfillDelayMs?: number;
}>;

export const memoryHome = (override?: string): string => resolve(override ?? process.env.PI_MEMORY_HOME ?? process.env.PRJCT_HOME ?? join(homedir(), '.prjct'));
export const historyDatabasePath = (home?: string): string => join(memoryHome(home), 'pi-memory', 'history.sqlite');

const LEASE_MS = 120_000;
const USAGE = 'Usage: /memory [status] | index | rebuild | gc | gc confirm';

const parameters = Type.Object({
  action: StringEnum(['search', 'read', 'summaries'] as const),
  query: Type.Optional(Type.String()),
  entryId: Type.Optional(Type.String()),
  sessionId: Type.Optional(Type.String()),
  since: Type.Optional(Type.String()),
  until: Type.Optional(Type.String()),
  scope: Type.Optional(StringEnum(['project', 'session', 'global'] as const)),
  kind: Type.Optional(StringEnum(ENTRY_KINDS)),
  offset: Type.Optional(Type.Integer({ minimum: 0 })),
});

export const installMemory = (pi: ExtensionAPI, options: MemoryExtensionOptions = {}): void => {
  const owner = `${process.pid}-${randomUUID()}`;
  const R = {
    store: undefined as HistoryStore | undefined,
    file: undefined as string | undefined,
    sessionId: '',
    cwd: process.cwd(),
    current: undefined as Promise<unknown> | undefined,
    again: false,
    backfill: undefined as AbortController | undefined,
    progress: undefined as BackfillProgress | undefined,
    timer: undefined as NodeJS.Timeout | undefined,
    /** The exact journals the last /memory gc listed; confirm deletes only these. */
    preview: undefined as Candidate[] | undefined,
  };
  const dbPath = historyDatabasePath(options.home);
  const sessionsRoot = (): string => options.sessionsRoot ?? join(getAgentDir(), 'sessions');
  const store = (): HistoryStore => (R.store ??= openHistoryStore(dbPath));

  const remember = (ctx: ExtensionContext): void => {
    R.file = ctx.sessionManager.getSessionFile();
    R.sessionId = ctx.sessionManager.getSessionId();
    R.cwd = ctx.sessionManager.getCwd?.() ?? ctx.cwd;
  };

  /** Indexes the current journal; overlapping requests fold into one more pass. */
  const indexCurrent = (): Promise<unknown> => {
    if (R.current) { R.again = true; return R.current; }
    const run = async (): Promise<void> => {
      do {
        R.again = false;
        if (R.file) await indexFile(store(), R.file).catch(() => undefined);
      } while (R.again);
    };
    R.current = run().finally(() => { R.current = undefined; });
    return R.current;
  };

  const runBackfill = async (): Promise<BackfillProgress | undefined> => {
    if (R.backfill) return R.progress;
    const history = store();
    if (!history.lease(owner, LEASE_MS)) return undefined;
    const controller = new AbortController();
    R.backfill = controller;
    try {
      R.progress = await backfill(history, sessionsRoot(), {
        signal: controller.signal, ...(R.file ? { skip: R.file } : {}),
        onFile: progress => { R.progress = progress; return history.lease(owner, LEASE_MS); },
      });
      return R.progress;
    } finally {
      R.backfill = undefined;
      try { history.release(owner); } catch { /* closed during shutdown */ }
    }
  };

  const stop = (): void => {
    if (R.timer) clearTimeout(R.timer);
    R.timer = undefined;
    R.backfill?.abort();
  };

  pi.on('session_start', async (_event, ctx) => {
    remember(ctx);
    stop();
    R.timer = setTimeout(() => {
      R.timer = undefined;
      void indexCurrent().then(runBackfill).catch(() => undefined);
    }, options.backfillDelayMs ?? 3_000);
    R.timer.unref?.();
  });
  const follow = async (_event: unknown, ctx: ExtensionContext): Promise<void> => { remember(ctx); void indexCurrent(); };
  pi.on('message_end', follow);
  pi.on('session_compact', follow);
  pi.on('session_tree', follow);
  pi.on('session_shutdown', async () => {
    stop();
    await R.current?.catch(() => undefined);
    R.store?.close();
    R.store = undefined;
  });

  pi.registerTool({
    name: 'memory_context',
    label: 'Memory',
    description: 'Search and read session history, including what compaction removed and earlier sessions of this project. '
      + 'Actions: search, read, summaries.',
    parameters,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      remember(ctx);
      await indexCurrent();
      const request = params as HistoryRequest;
      if (request.action === 'read' && !request.entryId) {
        throw new Error('read needs entryId from search or summaries. search: query (words, any order), since/until (YYYY-MM-DD or ISO), '
          + 'scope project (default) | session | global, kind, offset. summaries: since/until, scope, sessionId.');
      }
      const result = await runHistory(store(), request, {
        sessionId: R.sessionId, cwd: R.cwd,
        ...(R.backfill && R.progress ? { indexing: { done: R.progress.done, files: R.progress.files } } : {}),
      });
      return { content: [{ type: 'text' as const, text: JSON.stringify(result) }], details: result };
    },
  });

  pi.registerCommand('memory', {
    description: 'Session history index: status, index, rebuild, gc',
    getArgumentCompletions: prefix => ['status', 'index', 'rebuild', 'gc', 'gc confirm'].filter(value => value.startsWith(prefix.trim()))
      .map(value => ({ value, label: value })),
    handler: async (args, ctx) => {
      remember(ctx);
      const action = args.trim() || 'status';
      const say = (text: string, level: 'info' | 'warning' = 'info'): void => {
        if (ctx.hasUI) ctx.ui.notify(text, level); else process.stdout.write(`${text}\n`);
      };
      const describe = (): string => {
        const status = store().status();
        const running = R.backfill && R.progress ? ` · indexing ${R.progress.done}/${R.progress.files}` : '';
        return `Session history: ${status.sessions} sessions, ${status.entries.toLocaleString('en-US')} entries, `
          + `${status.summaries} summaries, ${(status.indexedBytes / 1e6).toFixed(0)} MB of journals${running}`;
      };
      if (action === 'status') return say(describe());
      if (action === 'index') {
        await indexCurrent();
        const progress = await runBackfill();
        return say(progress ? `${describe()} · +${progress.added.toLocaleString('en-US')} entries` : 'Another Pi process is indexing the history.', progress ? 'info' : 'warning');
      }
      if (action === 'rebuild') {
        stop();
        await R.current?.catch(() => undefined);
        R.store?.close();
        R.store = undefined;
        await Promise.all(['', '-wal', '-shm'].map(suffix => rm(`${dbPath}${suffix}`, { force: true })));
        await indexCurrent();
        const progress = await runBackfill();
        return say(progress ? `Rebuilt. ${describe()}` : 'Rebuilt the current session; another Pi process holds the history scan.');
      }
      if (action === 'gc') {
        await indexCurrent();
        const progress = await runBackfill();
        if (!progress) return say('Another Pi process is indexing the history; run /memory gc when it finishes.', 'warning');
        R.preview = await gcCandidates(store(), R.file ? [R.file] : []);
        return say(R.preview.length ? `${describeCandidates(R.preview)}\nDelete them for good with /memory gc confirm.` : 'No junk sessions.');
      }
      if (action === 'gc confirm') {
        if (!R.preview) return say('Run /memory gc first to see what would be deleted.', 'warning');
        const outcome = await collect(store(), R.preview);
        R.preview = undefined;
        return say(`Deleted ${outcome.deleted} sessions (${(outcome.bytes / 1e6).toFixed(1)} MB)`
          + `${outcome.skipped ? `; ${outcome.skipped} changed since the preview and were kept` : ''}.`);
      }
      return say(USAGE, 'warning');
    },
  });
};

export default function memory(pi: ExtensionAPI): void {
  installMemory(pi);
}
