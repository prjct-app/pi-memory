import { join } from 'node:path';
import { existsSync } from 'node:fs';
import { tryCreateSdkAnalyzer } from '../curation/analyzer.ts';
import { processAvailable } from '../curation/pipeline.ts';
import { jobIdFor } from '../curation/store.ts';
import { CurationBlockError, type Analyzer } from '../curation/types.ts';
import { MemoryEngine } from '../engine.ts';
import { registerKnownSources } from '../sources/install.ts';
import { SourceRegistry, type SourceAdapter } from '../sources/registry.ts';
import { memoryDatabasePath, trustedProjectIds } from '../workspace/project-identity.ts';
import { memoryProjectBindings, registeredMemoryProjectIds } from '../workspace/memory-registry.ts';
import { daemonStateDir, type DaemonConfig } from './config.ts';
import { GlobalBudgetLedger } from './budget.ts';
import { applyMaintenance, planMaintenance, runHygiene, type MaintenanceOptions } from '../retention/maintenance.ts';
import { backfillProject, createSdkCurator, curateProject, readUserMessages, sessionsDirFor, type Curator } from '../curation/curator.ts';
import { getAgentDir } from '@earendil-works/pi-coding-agent';

export type CycleReport = Readonly<{
  scopes: number;
  queued: number;
  processed: number;
  modelCalls: number;
  embeddingCalls: number;
  inputTokens: number;
  outputTokens: number;
  blocked?: string;
  failedScopes?: number;
  /** Standing changes written by the maintenance pass. */
  superseded?: number;
  reviewed?: number;
  removedDocuments?: number;
  /** Rules the curator retired, folded or rewrote. */
  curated?: number;
}>;

export type CycleOptions = Readonly<{
  config: DaemonConfig;
  owner: string;
  analyzer?: Analyzer;
  block?: CurationBlockError;
  engines?: readonly MemoryEngine[];
  extraAdapters?: readonly SourceAdapter[];
  signal?: AbortSignal;
  /**
   * Revisiting stored facts is what makes memory living rather than a capture
   * log. `false` keeps a cycle to ingestion only.
   */
  maintenance?: MaintenanceOptions | boolean;
  /** Judges the rule set as a whole with the session's own model. */
  curator?: Curator;
}>;

export const discoverProjectIds = async (home: string): Promise<string[]> =>
  [...new Set([...await registeredMemoryProjectIds(home), ...await trustedProjectIds(home)])]
    .filter(projectId => existsSync(memoryDatabasePath(home, projectId))).sort();

export const registryFor = async (engine: MemoryEngine, home: string): Promise<SourceRegistry> => {
  const registry = new SourceRegistry();
  if (engine.scopeKind !== 'project') return registry;
  await registerKnownSources(registry, engine.scopeId, { home });
  return registry;
};

const enqueueDueReviews = (engine: MemoryEngine, adapterId: string, now: number, maxReviewAgeMs: number): number => {
  const added = { n: 0 };
  const open = new Set(engine.curation.openJobs().map(job => `${job.action}:${job.documentKey}`));
  for (const identity of engine.curation.adapterFingerprints(adapterId)) {
    const expired = Boolean(identity.validTo && Date.parse(identity.validTo) <= now);
    const aged = (engine.curation.fingerprintAgeMs(identity.documentKey, now) ?? 0) > maxReviewAgeMs;
    if (!expired && !aged) continue;
    const action = expired ? 'review' as const : 'analyze' as const;
    if (open.has(`${action}:${identity.documentKey}`)) continue;
    const reason = expired ? 'expired' : 'age';
    engine.curation.enqueue({
      id: jobIdFor(engine.scopeId, action, identity.documentKey, `${identity.revision}:${reason}`),
      scopeId: engine.scopeId, adapter: adapterId, documentKey: identity.documentKey,
      action, inputRevision: identity.revision, contentHash: identity.contentHash,
    });
    added.n += 1;
  }
  return added.n;
};

/** Curation uses the configured session model through Pi's public SDK. */
export const resolveCurator = async (config: DaemonConfig): Promise<Curator | undefined> => {
  if (!config.provider || !config.model) return undefined;
  return createSdkCurator({ provider: config.provider, model: config.model }).catch(() => undefined);
};

const maintainProject = async (engine: MemoryEngine, options: CycleOptions, totals: {
  superseded: number; reviewed: number; removedDocuments: number;
}): Promise<void> => {
  // Opt-in: un ciclo normal ingiere, no reescribe lo ya guardado.
  if (options.maintenance === false) return;
  if (!options.maintenance && !options.config.maintenance) return;
  const settings = options.maintenance === true || !options.maintenance ? {} : options.maintenance;
  try {
    const plan = await planMaintenance(engine, {
      embed: texts => engine.embeddings.embed(texts, { inputType: 'passage' }),
      ...settings,
    });
    const applied = await applyMaintenance(engine, plan, {
      ...(settings.maxActions ? { maxActions: settings.maxActions } : {}),
    });
    totals.superseded += applied.superseded + applied.retired;
    totals.reviewed += applied.reviewed;
    totals.removedDocuments += applied.removed;
  } catch (error) {
    if (options.signal?.aborted) throw error;
  }
};

const processProject = async (engine: MemoryEngine, options: CycleOptions, totals: {
  queued: number; processed: number; modelCalls: number; embeddingCalls: number; inputTokens: number; outputTokens: number;
}, globalBudget: GlobalBudgetLedger): Promise<void> => {
  if (engine.scopeKind !== 'project') throw new Error('Memory opens only a project-owned database.');
  options.signal?.throwIfAborted();
  const registry = await registryFor(engine, options.config.home);
  for (const extra of options.extraAdapters ?? []) {
    if (!registry.get(extra.id)) registry.register(extra);
  }
  const adapters = new Map<string, SourceAdapter>();
  for (const id of registry.list()) {
    const adapter = registry.get(id);
    if (!adapter) continue;
    if (adapter.scope.kind !== 'project' || adapter.scope.id !== engine.scopeId) {
      throw new Error(`Daemon adapter ${id} is outside the active project authority.`);
    }
    adapters.set(id, adapter);
    const result = await registry.sync(async () => engine, id, options.signal, engine.projection);
    totals.queued += result.queued + enqueueDueReviews(engine, id, Date.now(), options.config.maxReviewAgeMs);
  }
  const processed = await processAvailable(engine, adapters, options.owner, {
    ...(options.analyzer ? { analyzer: options.analyzer } : {}),
    ...(options.block ? { block: options.block } : {}),
    maxAttempts: options.config.maxAttempts,
    maxInputChars: options.config.maxInputChars,
    budget: { maxCallsPerDay: options.config.maxCallsPerDay, maxTokensPerDay: options.config.maxTokensPerDay },
    globalBudget,
    leaseMs: options.config.leaseMs,
    deadlineMs: options.config.jobDeadlineMs,
  }, options.signal);
  for (const result of processed) {
    totals.processed += 1;
    totals.modelCalls += result.modelCalls;
    totals.embeddingCalls += result.embeddingCalls;
    totals.inputTokens += result.inputTokens;
    totals.outputTokens += result.outputTokens;
  }
};

export const runCycle = async (options: CycleOptions): Promise<CycleReport> => {
  const totals = { queued: 0, processed: 0, modelCalls: 0, embeddingCalls: 0, inputTokens: 0, outputTokens: 0 };
  const upkeep = { superseded: 0, reviewed: 0, removedDocuments: 0, curated: 0 };
  const scoped = { n: 0, failed: 0 };
  const globalBudget = new GlobalBudgetLedger(options.config.home);
  const processOne = async (engine: MemoryEngine): Promise<void> => {
    try {
      // A session-close run only curates. A full run learns from sessions it has
      // not read yet through the configured Pi model; the old path spent one
      // generative call per session document (31 min for 9 projects) and
      // re-extracted facts it already had on every cycle.
      if (!options.config.projectId) {
        const location = (await memoryProjectBindings(options.config.home).catch(() => []))
          .find(binding => binding.projectId === engine.scopeId)?.location;
        if (options.curator && location) {
          const learned = await backfillProject(engine, options.curator, join(daemonStateDir(options.config.home), 'curation'),
            sessionsDirFor(getAgentDir(), location), options.signal).catch(() => undefined);
          if (learned) upkeep.curated += learned.added + learned.retired + learned.superseded + learned.rewritten;
        } else {
          await processProject(engine, options, totals, globalBudget);
        }
        await maintainProject(engine, options, upkeep);
      }
      await runHygiene(engine).catch(() => undefined);
      if (options.curator) {
        const { sessionFile, projectId } = options.config;
        const said = sessionFile && projectId === engine.scopeId ? await readUserMessages(sessionFile) : [];
        const curated = await curateProject(engine, options.curator, join(daemonStateDir(options.config.home), 'curation'), options.signal,
          said, sessionFile ? sessionFile.split('/').at(-1) : undefined).catch(() => undefined);
        if (curated) upkeep.curated += curated.added + curated.retired + curated.superseded + curated.rewritten;
      }
      scoped.n += 1;
    } catch (error) {
      if (options.signal?.aborted) throw error;
      scoped.failed += 1;
    }
  };
  try {
    if (options.engines) {
      for (const engine of options.engines) await processOne(engine);
    } else {
      // A session-close run touches only the project that session worked in.
      const ids = options.config.projectId ? [options.config.projectId] : await discoverProjectIds(options.config.home);
      for (const id of ids) {
        const engine = await MemoryEngine.forScope('project', id, options.owner, { home: options.config.home }).catch(() => undefined);
        if (!engine) { scoped.failed += 1; continue; }
        try { await processOne(engine); } finally { await engine.dispose().catch(() => undefined); }
      }
    }
  } finally {
    globalBudget.close();
  }
  return {
    scopes: scoped.n, queued: totals.queued, processed: totals.processed, modelCalls: totals.modelCalls,
    embeddingCalls: totals.embeddingCalls, inputTokens: totals.inputTokens, outputTokens: totals.outputTokens,
    ...(upkeep.superseded ? { superseded: upkeep.superseded } : {}),
    ...(upkeep.reviewed ? { reviewed: upkeep.reviewed } : {}),
    ...(upkeep.removedDocuments ? { removedDocuments: upkeep.removedDocuments } : {}),
    ...(upkeep.curated ? { curated: upkeep.curated } : {}),
    ...(scoped.failed ? { failedScopes: scoped.failed } : {}), ...(options.block ? { blocked: options.block.code } : {}),
  };
};

export const resolveAnalyzer = async (config: DaemonConfig): Promise<{ analyzer?: Analyzer; block?: CurationBlockError }> =>
  tryCreateSdkAnalyzer({
    ...(config.provider ? { provider: config.provider } : {}),
    ...(config.model ? { model: config.model } : {}),
    maxOutputChars: config.maxOutputChars,
  });

export const sleep = (ms: number, signal?: AbortSignal): Promise<void> => new Promise((resolve, reject) => {
  const timer = setTimeout(() => resolve(), ms);
  const onAbort = (): void => {
    clearTimeout(timer);
    reject(signal?.reason ?? new Error('aborted'));
  };
  if (signal?.aborted) {
    clearTimeout(timer);
    reject(signal.reason ?? new Error('aborted'));
    return;
  }
  signal?.addEventListener('abort', onAbort, { once: true });
});

export const runLoop = async (options: CycleOptions, once: boolean): Promise<CycleReport> => {
  const cycle = { last: { scopes: 0, queued: 0, processed: 0, modelCalls: 0, embeddingCalls: 0, inputTokens: 0, outputTokens: 0 } as CycleReport };
  const tick = async (): Promise<CycleReport> => {
    const resolved = options.analyzer || options.block
      ? { ...(options.analyzer ? { analyzer: options.analyzer } : {}), ...(options.block ? { block: options.block } : {}) }
      : await resolveAnalyzer(options.config);
    // The curator uses the model the session ran on, through the same subscription.
    const curator = options.curator ?? await resolveCurator(options.config);
    cycle.last = await runCycle({ ...options, ...resolved, ...(curator ? { curator } : {}) });
    return cycle.last;
  };
  if (once) return tick();
  const loop = async (): Promise<CycleReport> => {
    await tick();
    if (options.signal?.aborted) return cycle.last;
    await sleep(options.config.intervalMs, options.signal).catch(() => undefined);
    if (options.signal?.aborted) return cycle.last;
    return loop();
  };
  return loop();
};
