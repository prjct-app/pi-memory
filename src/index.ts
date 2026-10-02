import { existsSync } from 'node:fs';
import { join } from 'node:path';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import type { AutocompleteItem } from '@earendil-works/pi-tui';
import { checkpointAndEnqueueLegacy } from './curation/migrate.ts';
import { loadDaemonConfig } from './daemon/config.ts';
import type { MemoryEngine } from './engine.ts';
import { installMemoryHooks } from './extension/hooks.ts';
import type { HandoffBudget } from './handoff/select.ts';
import type { ObservationPolicy } from './handoff/observations.ts';
import type { HistoryPolicy } from './handoff/history.ts';
import type { OutputCapPolicy } from './handoff/caps.ts';
import { installMemoryTools } from './extension/tools.ts';
import { createChildView, publishChildView } from './extension/child-view.ts';
import { contextLine, contextPrune, queueContextPrune } from './extension/context-prune.ts';
import { planSweep, runSweep } from './retention/sweep.ts';
import { runGc } from './retention/gc.ts';
import { registerKnownSources, scopedEngines, type SourceInstallOptions } from './sources/install.ts';
import { SourceRegistry, type SourceSyncResult } from './sources/registry.ts';
import { dueAdapters, PI_SESSION_SYNC_POLICY, type SyncPolicy } from './sources/schedule.ts';
import { SESSION_ADAPTER_ID } from './sources/session-log.ts';
import { memoryDatabasePath, memoryHomeFor, resolveLegacyProject, sha256 } from './workspace/project-identity.ts';
import { resolveMemoryProject } from './workspace/memory-registry.ts';
import {
  errorModel, panelDismissed, presentMemoryPanel, resultModel, sourcesModel, statusModel, syncModel,
} from './extension/panel.ts';
import { memoryPanelSpec, type MemoryOps } from './extension/memory-panel.ts';
import { brand, openPanel, repairToolArgs } from '@prjct.app/pi-tui-kit';
import { ensureEvaluator, evaluatorLine } from './extension/evaluator.ts';

export type MemoryExtensionOptions = Readonly<{
  home?: string; recallThreshold?: number;
  /** Explicit bound for messages retained after a real model switch. */
  handoff?: HandoffBudget;
  /** Stale tool-output masking in the request context; `{ enabled: false }` turns it off. */
  observations?: Partial<ObservationPolicy>;
  /** Soft completed-history target; `{ enabled: false }` disables economic retirement. */
  history?: Partial<HistoryPolicy>;
  /** Source caps for bash/grep/find output; `{ enabled: false }` turns them off. */
  outputCaps?: Partial<OutputCapPolicy>;
  /** Thresholds that make a source due; `{ enabled: false }` turns it off. */
  sync?: SyncPolicy;
  /** Optional publisher adapters. pi-session remains the only default source. */
  sources?: Omit<SourceInstallOptions, 'home'>;
}>;

const USAGE = 'Usage: /memory init | setup | status | sources | sync [adapter] | index {json} | replay | rebuild | gc | prune | purge [confirm] | checkpoint-wal | migrate-curated | checkpoint {json}';
const ACTIONS = new Set(['init', 'setup', 'status', 'sources', 'sync', 'index', 'replay', 'rebuild', 'gc', 'prune', 'purge', 'checkpoint-wal', 'migrate-curated', 'checkpoint']);
const ACTION_COMPLETIONS: readonly AutocompleteItem[] = [
  { value: 'init', label: 'init', description: 'Initialize memory for this checkout' },
  { value: 'setup', label: 'setup', description: 'Set or rotate the optional TypeSafe evaluator key' },
  { value: 'status', label: 'status', description: 'Show project memory status' },
  { value: 'sources', label: 'sources', description: 'Show source adapters and sync state' },
  { value: 'sync', label: 'sync', description: 'Scan all sources now, or choose an adapter' },
  { value: 'index', label: 'index', description: 'Index one JSON source document' },
  { value: 'checkpoint', label: 'checkpoint', description: 'Save an operational checkpoint from JSON' },
  { value: 'replay', label: 'replay', description: 'Replay durable memory history' },
  { value: 'rebuild', label: 'rebuild', description: 'Rebuild the searchable projection' },
  { value: 'gc', label: 'gc', description: 'Run bounded memory garbage collection' },
  { value: 'prune', label: 'prune', description: 'Remove superseded memory from this session\'s context now' },
  { value: 'purge', label: 'purge', description: 'Show what dead or junk memory would be deleted for good; purge confirm deletes it' },
  { value: 'checkpoint-wal', label: 'checkpoint-wal', description: 'Checkpoint the SQLite write-ahead log' },
  { value: 'migrate-curated', label: 'migrate-curated', description: 'Queue legacy documents for curation' },
];

const argumentCompletions = (prefix: string, adapterIds: readonly string[]): AutocompleteItem[] | null => {
  const sync = /^sync\s+([^\s]*)$/u.exec(prefix);
  if (sync) {
    const adapterPrefix = sync[1] ?? '';
    const matching = adapterIds.filter(id => id.startsWith(adapterPrefix)).map(id => ({
      value: `sync ${id}`, label: id, description: brand(`scan only the ${id} source`),
    }));
    return matching.length ? matching : null;
  }
  if (/\s/u.test(prefix)) return null;
  const matching = ACTION_COMPLETIONS.filter(item => item.value.startsWith(prefix))
    .map(item => ({ ...item, description: brand((item.description ?? item.value).replace(/^./u, first => first.toLowerCase())) }));
  return matching.length ? matching : null;
};

/**
 * Scans publisher sources, records fingerprints and enqueues analysis jobs.
 * It does not copy raw source bodies into the memory journal. The standalone
 * daemon publishes curated knowledge. Every adapter must declare this project's
 * owner; team, shared and foreign-project routing is rejected.
 */
const syncSources = async (registry: SourceRegistry, sessionId: string, project: MemoryEngine,
  target: string | undefined, options: MemoryExtensionOptions): Promise<SourceSyncResult[]> => {
  const engines = scopedEngines(sessionId, project, options.home === undefined ? {} : { home: options.home });
  try {
    // An explicit /memory sync always runs, whatever the watermarks say.
    return target
      ? [await registry.sync(engines.resolve, target, undefined, project.projection)]
      : await registry.syncAll(engines.resolve, undefined, project.projection);
  } finally {
    await engines.dispose();
  }
};

export const installMemory = (pi: ExtensionAPI, options: MemoryExtensionOptions = {}): void => {
  repairToolArgs(pi);
  const registry = new SourceRegistry();
  const registered = { done: false };
  const running = { now: false };
  const lastFault = { message: undefined as string | undefined };

  const sources = async (engine: MemoryEngine): Promise<SourceRegistry> => {
    if (registered.done) return registry;
    registered.done = true;
    await registerKnownSources(registry, engine.scopeId, {
      ...options.sources,
      ...(options.home === undefined ? {} : { home: options.home }),
    });
    return registry;
  };

  /**
   * Sync is not automatic in the sense of running on a schedule or at every
   * start. Each turn adds to a watermark table, and only when work since the
   * last run crosses a threshold does a source get re-read — in the background,
   * so the turn is never waiting on it, and never twice at once.
   */
  // Automatic sync only enqueues analysis for the daemon; without an analysis
  // provider those jobs never run, so there is nothing to enqueue.
  const analysis: { configured?: Promise<boolean> } = {};
  const analysisConfigured = (): Promise<boolean> => analysis.configured ??= loadDaemonConfig({ home: memoryHomeFor(options.home) })
    .then(config => Boolean(config.provider), () => false);

  const syncIfDue = async (project: MemoryEngine): Promise<void> => {
    if (options.sync?.enabled === false || running.now) return;
    if (!(await analysisConfigured())) return;
    const ready = await sources(project);
    const due = ready.list().filter(adapter => dueAdapters(project.projection, [adapter], adapter === SESSION_ADAPTER_ID
      ? { ...PI_SESSION_SYNC_POLICY, enabled: options.sync?.enabled ?? true }
      : options.sync ?? {})[0]?.due);
    if (!due.length) return;
    running.now = true;
    const engines = scopedEngines(project.journal.sessionId, project, options.home === undefined ? {} : { home: options.home });
    try {
      for (const adapter of due) {
        await ready.sync(engines.resolve, adapter, undefined, project.projection).catch(error => {
          lastFault.message = error instanceof Error ? error.message : String(error);
        });
      }
    } finally {
      await engines.dispose().catch(() => undefined);
      running.now = false;
    }
  };

  // Outside a git repository nothing can be recorded, so the tools only cost
  // schema tokens and failed calls. Restore exactly what was hidden.
  const MEMORY_TOOLS = ['memory_context', 'memory_record'];
  const hidden: { tools: readonly string[] } = { tools: [] };
  const syncToolVisibility = async (): Promise<void> => {
    if (typeof pi.getActiveTools !== 'function' || typeof pi.setActiveTools !== 'function') return;
    const where = await runtime.location().catch(() => undefined);
    const bound = where ? await resolveMemoryProject(where.path, memoryHomeFor(options.home)).catch(() => undefined) : undefined;
    const available = Boolean(where?.repository || bound);
    const active = pi.getActiveTools();
    const next = available ? [...new Set([...active, ...hidden.tools])] : active.filter(name => !MEMORY_TOOLS.includes(name));
    hidden.tools = available ? [] : MEMORY_TOOLS.filter(name => active.includes(name) || hidden.tools.includes(name));
    if (next.length !== active.length || next.some((name, index) => name !== active[index])) pi.setActiveTools(next);
  };

  const runtime = installMemoryHooks(pi, {
    ...(options.home === undefined ? {} : { home: options.home }),
    ...(options.recallThreshold === undefined ? {} : { recallThreshold: options.recallThreshold }),
    ...(options.handoff === undefined ? {} : { handoff: options.handoff }),
    ...(options.observations === undefined ? {} : { observations: options.observations }),
    ...(options.history === undefined ? {} : { history: options.history }),
    ...(options.outputCaps === undefined ? {} : { outputCaps: options.outputCaps }),
    // Deliberately not awaited by the hook: a source scan must never sit
    // between the user's prompt and the agent starting.
    onActivity: project => { void syncIfDue(project).catch(() => undefined); },
    onSessionStart: () => syncToolVisibility(),
  });
  installMemoryTools(pi, runtime);
  // Subagents and Team Experts start without extensions; their parent reads memory for them through this view.
  publishChildView(createChildView({ engine: () => runtime.engine(), search: runtime.search }));


  const adapterIds = [...new Set([
    SESSION_ADAPTER_ID,
    ...(options.sources?.prjct ? ['prjct-observations'] : []),
    ...(options.sources?.extra ?? []).map(adapter => adapter.id),
  ].filter(id => /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(id)))].sort();

  pi.registerCommand('memory', {
    description: brand('project memory: panel, init, sync, gc, prune, rebuild'),
    getArgumentCompletions: prefix => argumentCompletions(prefix, adapterIds),
    handler: async (args, ctx) => {
      const [first, second] = args.trim().split(/\s+/).filter(Boolean);
      if ((first === undefined || first === 'status' || first === 'sources') && second === undefined
        && ctx.mode === 'tui' && ctx.hasUI && typeof ctx.ui.custom === 'function') {
        const home = memoryHomeFor(options.home);
        const summary = (value: unknown): string => typeof value === 'object' && value
          ? Object.entries(value).filter(([, item]) => item === null || ['string', 'number', 'boolean'].includes(typeof item))
            .slice(0, 4).map(([key, item]) => `${key} ${String(item)}`).join(' · ')
          : String(value);
        const ops: MemoryOps = {
          load: async () => {
            const binding = await resolveMemoryProject((await runtime.location()).path, home);
            if (!binding) return { initialized: false, ready: false, legacy: Boolean(await resolveLegacyProject(ctx.cwd, home)), sources: [] };
            if (!existsSync(memoryDatabasePath(home, binding.projectId))) return { initialized: true, ready: false, project: binding.projectId, sources: [] };
            const engine = await runtime.engine();
            const ready = await sources(engine);
            return {
              initialized: true, ready: true, project: binding.projectId, scope: `${engine.scopeKind}/${engine.scopeId}`,
              stats: engine.projection.stats(), curation: engine.curation.stats(),
              sources: ready.list().map(adapter => ({
                ...dueAdapters(engine.projection, [adapter], adapter === SESSION_ADAPTER_ID
                  ? { ...PI_SESSION_SYNC_POLICY, enabled: options.sync?.enabled ?? true }
                  : options.sync ?? {})[0]!,
                last: engine.projection.syncState(adapter) ?? null,
              })),
              ...(lastFault.message ? { error: lastFault.message } : {}),
              ...(contextPrune() ? { context: contextLine(contextPrune()?.status()) } : {}),
            };
          },
          // The panel is itself a custom screen, so it reports the evaluator
          // state instead of opening the secret prompt inside it. /memory setup
          // is the interactive path.
          init: async () => {
            const done = await runtime.initialize();
            const evaluator = await ensureEvaluator({ ...ctx, mode: 'rpc' }, done.engine.root).catch(() => undefined);
            return `${done.created ? 'Initialized' : 'Already initialized'} · project ${done.binding.projectId}`
              + (evaluator ? ` · ${evaluatorLine(evaluator.resolved, evaluator.enabled)}` : '');
          },
          sync: async adapter => {
            const engine = await runtime.engine();
            const results = await syncSources(await sources(engine), ctx.sessionManager.getSessionId(), engine, adapter, options);
            const seen = results.reduce((sum, item) => sum + item.discovered, 0);
            const added = results.reduce((sum, item) => sum + item.indexed, 0);
            const queued = results.reduce((sum, item) => sum + item.queued, 0);
            const gaps = results.flatMap(item => item.gaps);
            return `Synced ${adapter ?? 'all sources'} · ${seen} seen · ${added} new · ${queued} queued${gaps.length ? ` · gap: ${gaps[0]}` : ''}`;
          },
          gc: async () => `GC · ${summary(await runGc(await runtime.engine()))}`,
          checkpointWal: async () => `WAL checkpoint · ${summary((await runtime.engine()).projection.checkpointWal())}`,
          rebuild: async () => `Rebuilt · ${summary(await (await runtime.engine()).rebuild())}`,
          ...(contextPrune() ? { prune: async () => queueContextPrune(ctx) } : {}),
        };
        try {
          await openPanel(ctx, memoryPanelSpec(ops, await ops.load()));
        } catch (error) {
          if (!panelDismissed(error)) throw error;
        }
        return;
      }
      const show = (model: Parameters<typeof presentMemoryPanel>[1]): Promise<void> => presentMemoryPanel(ctx,
        (args.trim().split(/\s+/)[0] || 'status') === 'status'
          ? { ...model, rows: [{ id: 'cache-policy', cells: ['cache policy session-references-v2'] }, ...model.rows] } : model);
      try {
      const [action = 'status', target] = args.trim().split(/\s+/).filter(Boolean);
      if (!ACTIONS.has(action)) throw new Error(USAGE);
      const home = memoryHomeFor(options.home);
      if (action === 'init') {
        if (target) throw new Error('Usage: /memory init');
        const initialized = await runtime.initialize();
        // The evaluator is optional, so it is offered after memory already
        // exists and never decides whether initialization succeeded. Declining
        // the prompt leaves a working project, not a failed command.
        const evaluator = await ensureEvaluator(ctx, initialized.engine.root)
          .catch(() => undefined);
        await show(resultModel('memory · init', [
          `status ${initialized.created ? 'initialized' : 'already initialized'}`,
          `project ${initialized.binding.projectId}`,
          `checkout ${initialized.binding.checkoutId}`,
          `source ${initialized.binding.source}`,
          `location ${initialized.binding.location}`,
          ...(evaluator ? [evaluatorLine(evaluator.resolved, evaluator.enabled)] : []),
        ]));
        return;
      }
      if (action === 'prune') {
        if (target) throw new Error('Usage: /memory prune');
        // Context, not the store: it needs no initialized project.
        await show(resultModel('memory · prune', [queueContextPrune(ctx)]));
        return;
      }
      if (action === 'purge') {
        if (target && target !== 'confirm') throw new Error('Usage: /memory purge [confirm]');
        const project = await runtime.engine();
        const rerank = (await runtime.reranker?.())?.rerank;
        const ask = rerank?.ask ? (state: Record<string, unknown>, questions: Readonly<Record<string, string>>, signal?: AbortSignal) =>
          rerank.ask!(state, questions, signal ? { signal } : {}) : undefined;
        const plan = await planSweep(project, ask);
        const examples = plan.junk.slice(0, 5).map(item => `junk  ${item.statement.slice(0, 90)}`);
        if (target !== 'confirm') {
          await show(resultModel('memory · purge (preview)', [
            `dead ${plan.dead.length} superseded or contradicted facts`,
            `junk ${plan.junk.length} live failures that are one run's state${plan.judged ? '' : ' (raw output only: no Jev key)'}`,
            ...examples,
            'run /memory purge confirm to back up and delete them for good',
          ]));
          return;
        }
        const done = await runSweep(project, plan, join(home, 'backups', `purge-${new Date().toISOString().slice(0, 10)}`));
        await show(resultModel('memory · purge', [
          `deleted ${done.facts} facts · ${done.documents} documents · ${done.evidence} evidence · ${done.events} journal events`,
          ...(done.deferred ? [`deferred ${done.deferred} journal streams another session is writing; the next GC finishes them`] : []),
          `backup ${done.backup ?? 'none needed'}`,
        ]));
        return;
      }
      if (action === 'setup') {
        if (target) throw new Error('Usage: /memory setup');
        // The key is global but the switch that uses it is per project, so
        // there has to be a project. Say that plainly instead of surfacing a
        // raw "not initialized" through the host's extension error channel.
        const project = await runtime.engine().catch(error => {
          if (error instanceof Error && /not initialized/iu.test(error.message)) return undefined;
          throw error;
        });
        if (!project) {
          await show(resultModel('memory · setup', [
            'status not initialized',
            'run /memory init first — it offers the evaluator key as part of setup',
          ]));
          return;
        }
        const evaluator = await ensureEvaluator(ctx, project.root, { force: true });
        await show(resultModel('memory · setup', [
          evaluatorLine(evaluator.resolved, evaluator.enabled),
          ...(evaluator.prompted ? [] : ['no terminal UI · set TYPESAFE_API_KEY or run /memory setup in a TUI session']),
        ]));
        return;
      }
      if (action === 'status') {
        const binding = await resolveMemoryProject((await runtime.location()).path, home);
        if (!binding) {
          const legacy = await resolveLegacyProject(ctx.cwd, home);
          await show(resultModel('memory · status', [
            'initialized no', 'ready no', `legacy ${legacy ? 'available' : 'none'}`, 'run /memory init',
          ]));
          return;
        }
        if (!existsSync(memoryDatabasePath(home, binding.projectId))) {
          await show(resultModel('memory · status', [
            'initialized yes', 'ready no', `project ${binding.projectId}`, 'run /memory init to repair',
          ]));
          return;
        }
        const opened = await runtime.engine();
        await show(statusModel({
          scope: `${opened.scopeKind}/${opened.scopeId}`, stats: opened.projection.stats(), curation: opened.curation.stats(),
          ...(lastFault.message ? { error: lastFault.message } : {}),
        }));
        return;
      }
      const engine = await runtime.engine();
      const scope = `${engine.scopeKind}/${engine.scopeId}`;
      if (action === 'sources') {
        const ready = await sources(engine);
        await show(sourcesModel({
          scope,
          adapters: ready.list().map(adapter => ({
            ...dueAdapters(engine.projection, [adapter], adapter === SESSION_ADAPTER_ID
              ? { ...PI_SESSION_SYNC_POLICY, enabled: options.sync?.enabled ?? true }
              : options.sync ?? {})[0]!,
            last: engine.projection.syncState(adapter) ?? null,
          })),
          ...(lastFault.message ? { error: lastFault.message } : {}),
        }));
        return;
      }
      if (action === 'sync') {
        const registryReady = await sources(engine);
        if (target && !registryReady.get(target)) throw new Error(`Unknown source adapter: ${target}. Try /memory sources.`);
        const results = await syncSources(registryReady, ctx.sessionManager.getSessionId(), engine, target, options);
        await show(syncModel(results, lastFault.message));
        return;
      }
      if (action === 'index') {
        const raw = args.trim().slice('index'.length).trim();
        if (!raw) throw new Error('Usage: /memory index {"namespace":"project.docs","externalId":"id","text":"..."}');
        const parsed: unknown = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('/memory index requires a JSON object.');
        const row = parsed as { namespace?: unknown; externalId?: unknown; text?: unknown; title?: unknown; uri?: unknown; source?: unknown; kind?: unknown; metadata?: unknown };
        if (typeof row.namespace !== 'string' || typeof row.externalId !== 'string' || typeof row.text !== 'string' || !row.text.trim()) {
          throw new Error('/memory index requires namespace, externalId, and non-empty text strings.');
        }
        const metadata = row.metadata && typeof row.metadata === 'object' && !Array.isArray(row.metadata)
          ? Object.fromEntries(Object.entries(row.metadata).flatMap(([key, value]) => typeof value === 'string' ? [[key, value]] : [])) : {};
        const indexed = await engine.index({ namespace: row.namespace, externalId: row.externalId, scopeId: engine.scopeId,
          scopeKind: engine.scopeKind, source: typeof row.source === 'string' ? row.source : 'operator-indexed',
          kind: typeof row.kind === 'string' ? row.kind : 'document', ...(typeof row.title === 'string' ? { title: row.title } : {}),
          text: row.text, ...(typeof row.uri === 'string' ? { uri: row.uri } : {}), version: sha256(row.text), contentHash: sha256(row.text),
          observedAt: new Date().toISOString(), trust: 'user', metadata });
        await show(resultModel('memory · index', [`chunks ${indexed.chunks}`, `dense ${String(indexed.dense)}`]));
        return;
      }
      if (action === 'migrate-curated') {
        const migrated = await checkpointAndEnqueueLegacy(engine);
        await show(resultModel('memory · migrate', Object.entries(migrated).filter(([, value]) =>
          value === null || ['string', 'number', 'boolean'].includes(typeof value)).slice(0, 8).map(([key, value]) => `${key} ${String(value)}`)));
        return;
      }
      if (action === 'checkpoint') {
        const raw = args.trim().slice('checkpoint'.length).trim();
        if (!raw) throw new Error('Usage: /memory checkpoint {"goal":"...","constraints":[],"done":[],"inProgress":[],"blocked":[],"decisions":[],"evidenceRefs":[],"nextSteps":[]}');
        const parsed: unknown = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)
          || typeof (parsed as { goal?: unknown }).goal !== 'string' || !(parsed as { goal: string }).goal.trim()) {
          throw new Error('/memory checkpoint requires a non-empty goal.');
        }
        const saved = await runtime.handoff.persist(engine, ctx.sessionManager.getSessionId(),
          parsed as Parameters<typeof runtime.handoff.persist>[2]);
        await show(resultModel('memory · checkpoint', [`ok ${String(Boolean(saved))}`]));
        return;
      }
      const report = action === 'replay' ? await engine.replay(false)
        : action === 'rebuild' ? await engine.rebuild()
        : action === 'gc' ? await runGc(engine)
        : action === 'checkpoint-wal' ? engine.projection.checkpointWal()
        : undefined;
      if (report === undefined) throw new Error(USAGE);
      await show(resultModel(`memory · ${action}`, typeof report === 'object' && report
        ? Object.entries(report).filter(([, value]) => value === null || ['string', 'number', 'boolean'].includes(typeof value))
          .slice(0, 8).map(([key, value]) => `${key} ${String(value)}`)
        : [String(report)]));
      } catch (error) {
        if (panelDismissed(error)) return;
        const message = error instanceof Error ? error.message : String(error);
        if (!message.startsWith('Usage:')) lastFault.message = message;
        await presentMemoryPanel(ctx, errorModel(message)).catch(() => undefined);
        throw error;
      }
    },
  });
};

export default function memoryExtension(pi: ExtensionAPI): void {
  installMemory(pi);
}

export { hostEvidence, MemoryEngine } from './engine.ts';
