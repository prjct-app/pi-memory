import { existsSync, realpathSync } from 'node:fs';
import { mkdir, realpath, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import { contentText } from '@earendil-works/pi-ai';
import type { EvidenceRef } from '../contracts/evidence.ts';
import { factIsValidAt, type MemoryKind } from '../contracts/memory.ts';
import { hostEvidence, MemoryEngine } from '../engine.ts';
import type { EmbeddingProvider } from '../vector/providers.ts';
import { CORE_DIGEST_BYTES, CORE_KINDS, memoryDigest, memoryStatusLine } from './digest.ts';
import { setMode } from '@prjct.app/pi-tui-kit';
import { createHandoffController, installHandoffHooks } from '../handoff/hooks.ts';
import type { HandoffBudget } from '../handoff/select.ts';
import type { ObservationPolicy } from '../handoff/observations.ts';
import type { HistoryPolicy } from '../handoff/history.ts';
import { capToolOutput, DEFAULT_OUTPUT_CAP_POLICY, isCappedTool, type OutputCapPolicy } from '../handoff/caps.ts';
import { federatedSearch } from '../retrieval/federated.ts';
import type { Translator } from '../curation/translator.ts';
import { loadDaemonConfig } from '../daemon/config.ts';
import { spawnCurationRun } from '../daemon/lifecycle.ts';
import { admitCapture } from '../retention/capture-gate.ts';
import { redactSecrets } from '../security/redact.ts';
import {
  appendSessionObservations, clipSessionSummary, declaredCorrectionQuote, declaredMemoryQuote,
  sessionObservationId, sessionObservationIdentity, sessionObservationWorthy,
  type SessionObservation,
} from '../sources/session-log.ts';
import { memoryHomeFor, sha256 } from '../workspace/project-identity.ts';
import { resolveMemoryProject } from '../workspace/memory-registry.ts';

export type MemorySession = Readonly<{
  generation: object;
  engine?: Promise<MemoryEngine>;
  /** Set only after ownership validation succeeds, never for a pending open. */
  authorityId?: string;
  readable?: Promise<readonly MemoryEngine[]>;
  ctx?: ExtensionContext;
  prompt: string;
  evidence: ReadonlyMap<string, EvidenceRef>;
  /** Last context size seen, to turn a running total into a per-turn delta. */
  contextTokens: number;
  /** One durable JSONL append is performed when the turn settles. */
  observations: readonly SessionObservation[];
  /** Exact current-prompt corrections and explicit remember declarations awaiting promotion. */
  declarations: readonly Readonly<{
    quote: string; observedAt: string; sessionId: string; kind: Extract<MemoryKind, 'correction' | 'procedure'>;
  }>[];
}>;

const textContent = (content: readonly unknown[]): string => content.flatMap(part => {
  if (!part || typeof part !== 'object') return [];
  const text = (part as { text?: unknown }).text;
  return typeof text === 'string' ? [text] : [];
}).join('\n');

/**
 * Only the person at an interactive terminal states corrections. A prompt in
 * print, json or rpc mode, or in a subagent child, was written by another
 * agent: `pi -p '<task brief>'` filed every imperative line of the brief
 * ("NO installs", "halt execution") as a project-wide correction.
 */
export const fromPerson = (ctx: Pick<ExtensionContext, 'mode'>): boolean =>
  ctx.mode === 'tui' && process.env.PI_SUBAGENTS_CHILD !== '1';

const clip = (text: string, max = 2048): string => text.length <= max ? text : `${text.slice(0, max)}…`;
const evidenceHandle = (evidence: ReadonlyMap<string, EvidenceRef>, id: string): string => {
  const digest = sha256(id);
  const available = [8, 12, 16, 24, 32, 48, 64].map(length => `e_${digest.slice(0, length)}`)
    .find(handle => evidence.get(handle)?.id === id || !evidence.has(handle));
  if (!available) throw new Error('Unable to allocate a unique session evidence handle.');
  return available;
};
/**
 * A repository is one memory project wherever Pi was started inside it. The
 * home directory never counts as a repository root, even with a dotfiles .git.
 */
export const gitRoot = (directory: string): string | undefined => {
  if (directory === homedir()) return undefined;
  if (existsSync(join(directory, '.git'))) return directory;
  const parent = dirname(directory);
  return parent === directory ? undefined : gitRoot(parent);
};

/**
 * A prompt usually wraps its question in instructions ("…? Answer in one
 * line, without reading files."). Searching the whole prompt dilutes both
 * lexical coverage and the embedding, so its sentences are searched too.
 */
export const recallQueries = (prompt: string): string[] => {
  const sentences = prompt.split(/(?<=[?!.。])\s+|\n+/u).map(part => part.trim()).filter(part => part.length >= 12);
  return [...new Set([prompt.trim(), ...sentences])].filter(Boolean).slice(0, 4);
};

const NOT_INITIALIZED = /not initialized/iu;
/** Facts considered for the digest; far more than fit its byte budget. */
const DIGEST_SCAN_LIMIT = 500;
export const isNotInitialized = (error: unknown): boolean => error instanceof Error && NOT_INITIALIZED.test(error.message);

export type MemorySearch = (
  request: Parameters<typeof federatedSearch>[1],
) => ReturnType<typeof federatedSearch>;

// The shared evidence-coverage gate runs before ranking for both lookup and
// automatic recall. Do not confuse a rank-fusion score with answerability:
// an additional positive floor discarded supported Spanish/qualified matches.
export const DEFAULT_RECALL_THRESHOLD = 0;

export const installMemoryHooks = (pi: ExtensionAPI, options: {
  home?: string;
  /** Legacy: automatic recall was removed; memory_context lookups carry their own threshold. */
  recallThreshold?: number;
  handoff?: HandoffBudget;
  observations?: Partial<ObservationPolicy>;
  history?: Partial<HistoryPolicy>;
  outputCaps?: Partial<OutputCapPolicy>;
  /** Called after each turn is counted, so the caller can sync when due. */
  onActivity?: (project: MemoryEngine) => Promise<void> | void;
  /** Encoder override (tests, custom deployments); defaults to the project's configured provider. */
  embeddingProvider?: EmbeddingProvider;
  /** Called once the new session's context is in place; never awaited. */
  onSessionStart?: () => Promise<void> | void;
  /** Translator override (tests, custom deployments); defaults to the active session model. */
  /** Legacy injection retained for callers; storage preserves original text. */
  translator?: Translator;
} = {}) => {
  const engineOptions = (): { home?: string; provider?: EmbeddingProvider } => ({
    ...(options.home === undefined ? {} : { home: options.home }),
    ...(options.embeddingProvider ? { provider: options.embeddingProvider } : {}),
  });
  // Off unless a caller opts in: Pi's own tools already truncate at 50 KiB / 2000 lines
  // and keep the full output on disk. A second, tighter cut here hid evidence the model
  // had asked for (a test log's middle, grep matches past 12k chars).
  const outputCaps: OutputCapPolicy = { ...DEFAULT_OUTPUT_CAP_POLICY, enabled: false, ...options.outputCaps };
  const onActivity = options.onActivity;
  const slot: { current: MemorySession } = { current: {
    generation: {}, prompt: '', evidence: new Map(), contextTokens: 0, observations: [], declarations: [],
  } };
  const get = (): MemorySession => slot.current;
  const set = (update: Partial<MemorySession>): MemorySession => (slot.current = { ...slot.current, ...update });
  /**
   * The checkout memory binds to: an existing binding for the exact directory
   * wins (explicit /memory init there), otherwise the enclosing git repository.
   */
  const location = async (cwd: string): Promise<Readonly<{ path: string; repository: boolean }>> => {
    const exact = await realpath(resolve(cwd)).catch(() => resolve(cwd));
    if (await resolveMemoryProject(exact, memoryHomeFor(options.home))) return { path: exact, repository: gitRoot(exact) !== undefined };
    const root = gitRoot(exact);
    return root ? { path: root, repository: true } : { path: exact, repository: false };
  };

  /** Where a new binding is created: the enclosing repository, else the directory itself. */
  const repositoryPath = (cwd: string): string => {
    const exact = (() => { try { return realpathSync(resolve(cwd)); } catch { return resolve(cwd); } })();
    return gitRoot(exact) ?? exact;
  };

  const engine = async (): Promise<MemoryEngine> => {
    const current = get();
    if (current.engine) {
      const project = await current.engine;
      if (!current.ctx || get().generation !== current.generation) throw new Error('Memory session changed while opening its authority.');
      const binding = await resolveMemoryProject((await location(current.ctx.cwd)).path, memoryHomeFor(options.home));
      if (get().generation !== current.generation) throw new Error('Memory session changed while validating its authority.');
      if (!binding) throw new Error('Memory project binding disappeared after the authority was opened.');
      if (project.scopeId !== binding.projectId) throw new Error('Memory project binding changed after the authority was opened.');
      return project;
    }
    if (!current.ctx) throw new Error('Memory session has not started.');
    const opening = location(current.ctx.cwd).then(where => MemoryEngine.forInitializedProject(where.path,
      current.ctx!.sessionManager.getSessionId(), engineOptions()));
    const slot: { pending?: Promise<MemoryEngine> } = {};
    const pending = opening.then(project => {
      if (get().engine === slot.pending) set({ authorityId: project.scopeId });
      return project;
    }).catch(error => {
      if (get().engine === slot.pending) set({ engine: undefined, readable: undefined });
      throw error;
    });
    slot.pending = pending;
    set({ engine: pending });
    return pending;
  };

  const initialize = async (): ReturnType<typeof MemoryEngine.initializeProject> => {
    const current = get();
    if (!current.ctx) throw new Error('Memory session has not started.');
    if (current.engine) {
      const opened = await current.engine;
      if (get().generation !== current.generation) throw new Error('Memory session changed during initialization.');
      const binding = await resolveMemoryProject((await location(current.ctx.cwd)).path, memoryHomeFor(options.home));
      if (!binding) throw new Error('The open memory authority has no project binding.');
      if (get().generation !== current.generation) throw new Error('Memory session changed during initialization.');
      if (opened.scopeId !== binding.projectId) throw new Error('Memory project binding changed after the authority was opened.');
      return { engine: opened, binding, created: false };
    }
    // Synchronous on purpose: a concurrent initialize or shutdown must see this pending engine.
    const task = MemoryEngine.initializeProject(repositoryPath(current.ctx.cwd), current.ctx.sessionManager.getSessionId(),
      engineOptions());
    const pending = task.then(initialized => initialized.engine);
    void pending.catch(() => undefined);
    set({ engine: pending, readable: undefined });
    try {
      const initialized = await task;
      if (get().generation !== current.generation) throw new Error('Memory session changed during initialization.');
      set({ engine: Promise.resolve(initialized.engine), authorityId: initialized.engine.scopeId, readable: Promise.resolve([initialized.engine]) });
      return initialized;
    } catch (error) {
      if (get().engine === pending) set({ engine: undefined, readable: undefined });
      throw error;
    }
  };

  /**
   * The engine a write needs. Inside a git repository the first thing worth
   * remembering initializes the project; outside one there is nowhere to bind.
   */
  const writableEngine = async (): Promise<MemoryEngine> => {
    try { return await engine(); }
    catch (error) {
      const current = get();
      if (!isNotInitialized(error) || !current.ctx) throw error;
      if (!(await location(current.ctx.cwd)).repository) {
        throw new Error('Memory is available inside a git repository; this directory is not one.');
      }
      return (await initialize()).engine;
    }
  };

  const handoffEngine = async (): Promise<MemoryEngine | undefined> => {
    const current = get();
    if (!current.ctx) throw new Error('Memory session has not started.');
    const binding = await resolveMemoryProject((await location(current.ctx.cwd)).path, memoryHomeFor(options.home));
    if (get().generation !== current.generation) throw new Error('Memory session changed while resolving handoff ownership.');
    if (!binding) {
      if (get().authorityId) throw new Error('Memory project binding disappeared after the authority was opened.');
      return undefined;
    }
    if (get().authorityId && get().authorityId !== binding.projectId) throw new Error('Memory project binding changed after the authority was opened.');
    const project = await engine();
    const verified = await resolveMemoryProject((await location(current.ctx.cwd)).path, memoryHomeFor(options.home));
    if (get().generation !== current.generation) throw new Error('Memory session changed while opening the handoff authority.');
    if (!verified) throw new Error('Memory project binding disappeared after the authority was opened.');
    if (project.scopeId !== verified.projectId || project.scopeId !== binding.projectId) throw new Error('Memory project binding changed after the authority was opened.');
    return project;
  };

  /** The active project's memory only. Team/shared databases are not readable. */
  const readable = async (): Promise<readonly MemoryEngine[]> => {
    const pending = engine().then(project => [project] as const);
    set({ readable: pending });
    return pending;
  };

  const queueSessionObservation = (record: SessionObservation): void => {
    const current = get();
    set({ observations: [...current.observations, record].slice(-64) });
  };

  const promoteDeclaredFacts = async (project: MemoryEngine,
    declarations: MemorySession['declarations']): Promise<void> => {
    for (const declaration of declarations) {
      // A declared rule is already authoritative text; preserve it verbatim.
      const statement = declaration.quote;
      // Only identical authored text is a duplicate; similar wording may change a constraint.
      if (project.projection.activeFacts(project.scopeId, DIGEST_SCAN_LIMIT).some(fact => fact.statement === statement && fact.kind === declaration.kind)) continue;
      const observationKind = declaration.kind === 'correction' ? 'correction' : 'instruction';
      // Identity stays keyed on what was actually said.
      const identity = sessionObservationIdentity(observationKind, 'user_input', declaration.quote);
      const id = `mem_${sha256(`declared:${project.scopeId}:${identity.semanticKey}:${identity.summaryHash}`).slice(0, 32)}`;
      if (project.projection.getFact(id)) continue;
      await project.recordFact({
        id, kind: declaration.kind, statement, confidence: 1,
        entities: [], episodeIds: [], validAt: declaration.observedAt,
        evidence: [{
          id: `ev_${sha256(`declared:${project.scopeId}:${identity.summaryHash}`).slice(0, 24)}`,
          origin: 'user_statement', provenance: 'declared', contentHash: sha256(declaration.quote),
          excerpt: declaration.quote, observedAt: declaration.observedAt,
          actorId: declaration.sessionId, sessionId: declaration.sessionId,
        }],
        tags: { semanticKey: identity.semanticKey, summaryHash: identity.summaryHash, source: 'pi-session' },
      }, undefined, { dense: false }).catch(error => {
        if (!(error instanceof Error) || !/already exists/u.test(error.message)) throw error;
      });
    }
  };


  /**
   * Embeds what was written without blocking the turn. The first run loads
   * the local encoder (~1.3s once); later memories embed in milliseconds.
   */
  const backfill: { running?: Promise<void> } = {};
  const scheduleBackfill = (project: MemoryEngine): void => {
    // Memory that fits the prompt digest never needs vectors, so the encoder
    // (~500MB resident, ~1.3s to load) is only loaded for larger memories.
    if (backfill.running || memoryDigest(project.projection.activeFacts(project.scopeId, DIGEST_SCAN_LIMIT)).complete) return;
    backfill.running = project.backfillVectors().then(() => undefined, () => undefined)
      .finally(() => { backfill.running = undefined; });
  };

  const flushSessionObservations = async (): Promise<void> => {
    const pending = get();
    if (!pending.observations.length && !pending.declarations.length) return;
    set({ observations: [], declarations: [] });
    // An explicit declaration is worth initializing the repository's memory;
    // tool failures alone are only kept where memory already exists.
    const project = await (pending.declarations.length ? writableEngine() : engine()).catch(error => {
      if (isNotInitialized(error) || /git repository/u.test(String((error as Error)?.message))) return undefined;
      throw error;
    });
    if (!project) return;
    if (get().generation !== pending.generation) throw new Error('Memory session changed while flushing observations.');
    try {
      await appendSessionObservations({ projectId: project.scopeId, records: pending.observations,
        ...(options.home === undefined ? {} : { home: options.home }) });
      await promoteDeclaredFacts(project, pending.declarations);
      scheduleBackfill(project);
    } catch (error) {
      if (get().generation !== pending.generation) throw error;
      set({ observations: [...pending.observations, ...get().observations].slice(-64),
        declarations: [...pending.declarations, ...get().declarations].slice(-16) });
      throw error;
    }
  };

  // Keep storage and indexing off the send path. Shutdown awaits ordered flushes.
  const queue: { tail: Promise<void>; inFlight: number } = { tail: Promise.resolve(), inFlight: 0 };
  const inBackground = (task: () => Promise<void>): Promise<void> => {
    queue.inFlight += 1;
    queue.tail = queue.tail.then(task).catch(() => undefined)
      .finally(() => { queue.inFlight -= 1; });
    return queue.tail;
  };
  const flushInBackground = (): Promise<void> => inBackground(flushSessionObservations);

  const search: MemorySearch = async request => federatedSearch(await readable(), request);

  const handoff = createHandoffController({ engine: handoffEngine, toolOverhead: () => {
    try {
      const active = new Set(pi.getActiveTools());
      const definitions = pi.getAllTools().filter(tool => active.has(tool.name)).map(tool => ({
        name: tool.name, description: tool.description, parameters: tool.parameters,
        ...(tool.promptGuidelines?.length ? { promptGuidelines: tool.promptGuidelines } : {}),
      }));
      const toolSchemaBytes = Buffer.byteLength(JSON.stringify(definitions), 'utf8');
      return { toolSchemaBytes, toolSchemaTokens: Math.ceil(toolSchemaBytes / 4) };
    } catch { return {}; }
  }, ...(options.handoff === undefined ? {} : { budget: options.handoff }),
  // Off unless a caller opts in. Measured 2026-09-26 on four real sessions: the model
  // received 29-60% of its history, and 64-93% of the elided outputs belonged to the
  // task in progress. self-compact is the one context limit; this layer only keeps the
  // hard window fallback in selectHandoffMessages.
  observations: options.observations ?? { enabled: false },
  history: options.history ?? { enabled: false },
  });

  /**
   * Records what this turn cost. The host reports the size of the whole
   * context, not the growth, so the delta is taken here; a context that shrank
   * has been compacted, and its new size is the growth since.
   */
  const countTurn = async (ctx: ExtensionContext): Promise<void> => {
    const current = get();
    const project = await engine();
    if (get().generation !== current.generation) throw new Error('Memory session changed while counting activity.');
    const seen = ctx.getContextUsage?.()?.tokens ?? null;
    const previous = get().contextTokens;
    const grown = seen === null ? 0 : seen > previous ? seen - previous : seen;
    if (seen !== null) set({ contextTokens: seen });
    project.projection.recordActivity({ turns: 1, tokens: grown });
    await onActivity?.(project);
  };

  pi.on('session_start', async (_event, ctx) => {
    handoff.clear();
    if (ctx.model) handoff.observeModel(ctx.cwd, ctx.sessionManager.getSessionId(), ctx.model);
    const previous = get();
    set({ generation: {}, engine: undefined, authorityId: undefined, readable: undefined, ctx, prompt: '', evidence: new Map(), contextTokens: 0,
      observations: [], declarations: [] });
    void Promise.resolve(options.onSessionStart?.()).catch(() => undefined);
    if (previous.ctx) {
      const opened = await previous.readable?.catch(() => []) ?? [];
      const pending = previous.engine ? [await previous.engine.catch(() => undefined)] : [];
      const engines = [...new Set([...opened, ...pending].filter(memory => memory !== undefined))];
      for (const memory of engines) await memory.dispose().catch(() => undefined);
    }  });

  /**
   * Only hard rules count toward the status line; the rest is found with
   * memory_context when the model asks for it.
   */
  const coreSnapshot = (project: MemoryEngine) => {
    const facts = project.projection.activeFacts(project.scopeId, DIGEST_SCAN_LIMIT)
      .filter(fact => (fact.standing === 'supported' || fact.standing === 'needs_review') && factIsValidAt(fact, Date.now()));
    return { facts, digest: memoryDigest(facts, Date.now(), CORE_DIGEST_BYTES, CORE_KINDS) };
  };

  /**
   * Memory never enters the context on its own: no snapshot, no recall, no
   * message. The model reads it with memory_context when it decides to. This
   * hook only counts the turn, captures what the person declared, and updates
   * the status line. It never returns systemPrompt either: Pi takes any returned
   * prompt as a forced prompt for prompted turns only, which flipped the cache.
   */
  pi.on('before_agent_start', async (event, ctx) => {
    const current = get();
    const stale = (): boolean => get().generation !== current.generation;
    void flushInBackground();
    set({ ctx, prompt: event.prompt });
    await countTurn(ctx).catch(() => undefined);
    if (stale()) return undefined;
    const prompt = clipSessionSummary(event.prompt);
    const observedAt = new Date().toISOString();
    const correction = declaredCorrectionQuote(event.prompt);
    const remembered = correction ? undefined : declaredMemoryQuote(event.prompt);
    const quote = correction ?? remembered;
    const summary = quote ?? prompt;
    const kind = correction ? 'correction' as const : 'instruction' as const;
    if (fromPerson(ctx) && sessionObservationWorthy({ kind, tool: 'user_input', outcome: 'stated', summary })) {
      queueSessionObservation({
        id: sessionObservationId(ctx.sessionManager.getSessionId(), 'user_input', summary, kind),
        kind, tool: 'user_input', outcome: 'stated', summary, observedAt, provenance: 'declared',
        sessionId: ctx.sessionManager.getSessionId(),
      });
      if (quote) {
        const declaration: MemorySession['declarations'][number] = {
          quote, observedAt, sessionId: ctx.sessionManager.getSessionId(),
          kind: correction ? 'correction' : 'procedure',
        };
        set({ declarations: [...get().declarations, declaration].slice(-16) });
      }
    }
    const project = await engine().catch(() => undefined);
    if (stale() || !project) return undefined;
    const { facts, digest } = coreSnapshot(project);
    try { setMode(ctx, 'memory', memoryStatusLine(digest, facts.length, 0)); } catch { /* the status line must not cost the prompt */ }
    return undefined;
  });

  /**
   * Caps noisy bash/grep/find output at the source. The full text goes to a file
   * first; if that write fails the output is left untouched rather than lost.
   */
  const capOutput = async (toolName: string, toolCallId: string, content: readonly unknown[], details: unknown,
    raw: string): Promise<{ type: 'text'; text: string }[] | undefined> => {
    if (!isCappedTool(toolName) || content.some(part => (part as { type?: unknown })?.type !== 'text')) return undefined;
    try {
      const existing = (details as { fullOutputPath?: unknown } | undefined)?.fullOutputPath;
      const path = typeof existing === 'string' && existing
        ? existing
        : join(tmpdir(), 'pi-memory-tool-output', `${toolCallId.replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 120) || 'call'}.txt`);
      const text = capToolOutput(toolName, raw, path, outputCaps);
      if (text === undefined) return undefined;
      if (path !== existing) {
        await mkdir(join(tmpdir(), 'pi-memory-tool-output'), { recursive: true });
        await writeFile(path, raw, 'utf8');
      }
      return [{ type: 'text', text }];
    } catch {
      return undefined;
    }
  };

  pi.on('tool_result', async (event, ctx) => {
    if (event.toolName === 'memory_context' || event.toolName === 'memory_record') return;
    const raw = textContent(event.content as readonly unknown[]);
    const bounded = clip(`${event.toolName} ${event.isError ? 'failed' : 'succeeded'}\n${raw || '(no textual output)'}`, 2560);
    const excerpt = clip(redactSecrets(bounded));
    const evidence = hostEvidence({ excerpt, actorId: ctx.sessionManager.getSessionId(), sessionId: ctx.sessionManager.getSessionId(), toolCallId: event.toolCallId });
    const handle = evidenceHandle(get().evidence, evidence.id);
    const entries = [...get().evidence.entries(), [handle, evidence] as const].slice(-64);
    set({ evidence: new Map(entries) });
    if (event.isError) {
      queueSessionObservation({
        id: sessionObservationId(ctx.sessionManager.getSessionId(), event.toolName, excerpt, 'failure'),
        kind: 'failure', tool: event.toolName, outcome: 'failed', summary: excerpt,
        observedAt: new Date().toISOString(), provenance: 'native_observation',
        sessionId: ctx.sessionManager.getSessionId(),
      });
    }
    // Results are staged for memory_record, never tagged: a failure handle in
    // every failed result cost context for evidence the model almost never cited.
    const capped = await capOutput(event.toolName, event.toolCallId, event.content as readonly unknown[], event.details, raw);
    return capped ? { content: capped } : undefined;
  });

  pi.on('turn_end', () => { void flushInBackground(); });

  installHandoffHooks(pi, handoff, handoffEngine);

  pi.on('session_shutdown', async () => {
    const current = get();
    if (queue.inFlight) await queue.tail;
    await flushSessionObservations().catch(() => undefined);
    if (get().generation !== current.generation) return;
    const { engine: pending, readable: opened } = get();
    handoff.clear();
    set({ generation: {}, engine: undefined, authorityId: undefined, readable: undefined, ctx: undefined, evidence: new Map(), prompt: '', contextTokens: 0,
      observations: [], declarations: [] });
    // The project engine is one of the readable ones; dispose the set, not both.
    const engines = await opened?.catch(() => []) ?? (pending ? [await pending] : []);
    for (const memory of engines) await memory.dispose().catch(() => undefined);
    if (!engines.length && pending) await (await pending).dispose().catch(() => undefined);
    // Curation needs judgement, so it runs with the session's own model and
    // subscription, once, in a detached process after the session closes.
    const model = current.ctx?.model;
    const project = engines.find(memory => memory.scopeKind === 'project');
    const sessionFile = current.ctx?.sessionManager.getSessionFile?.();
    if (process.env.PI_MEMORY_CURATE_ON_CLOSE === '1' && project && model && current.ctx && fromPerson(current.ctx)) {
      await loadDaemonConfig({ ...(options.home === undefined ? {} : { home: options.home }), provider: model.provider, model: model.id,
        projectId: project.scopeId, ...(sessionFile ? { sessionFile } : {}) })
        .then(config => spawnCurationRun(config)).catch(() => undefined);
    }
  });

  return {
    engine, writableEngine, initialize, readable, search, handoff,
    location: async () => location(get().ctx?.cwd ?? process.cwd()),
    scheduleBackfill,
    /** Settles once every background observation flush has finished. */
    flushed: (): Promise<void> => queue.tail,
    stagedEvidence: () => get().evidence, currentPrompt: () => get().prompt,
    userPrompts: () => {
      const current = get();
      const branch = current.ctx?.sessionManager.getBranch?.();
      if (!branch) return [current.prompt];
      return branch.flatMap(entry => {
        if (entry.type !== 'message' || entry.message.role !== 'user') return [];
        const content = entry.message.content;
        return [contentText(content)];
      });
    },
  };
};
