import { existsSync, realpathSync } from 'node:fs';
import { mkdir, realpath, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { EvidenceRef } from '../contracts/evidence.ts';
import { factIsValidAt, type MemoryKind } from '../contracts/memory.ts';
import { renderMemoryEnvelope, type MemoryEnvelope } from '../handoff/memory-envelope.ts';
import { hostEvidence, MemoryEngine } from '../engine.ts';
import type { EmbeddingProvider } from '../vector/providers.ts';
import { CORE_DIGEST_BYTES, CORE_KINDS, memoryDigest } from './digest.ts';
import { createHandoffController, installHandoffHooks } from '../handoff/hooks.ts';
import type { HandoffBudget } from '../handoff/select.ts';
import type { ObservationPolicy } from '../handoff/observations.ts';
import type { HistoryPolicy } from '../handoff/history.ts';
import { capToolOutput, DEFAULT_OUTPUT_CAP_POLICY, isCappedTool, type OutputCapPolicy } from '../handoff/caps.ts';
import { federatedSearch } from '../retrieval/federated.ts';
import { createRerankProvider, readRerankConfig } from '../retrieval/rerank.ts';
import { isEnglish } from '../contracts/language.ts';
import { createSdkTranslator, type Translator } from '../curation/translator.ts';
import { nearDuplicateFact } from '../curation/similar.ts';
import { runHygiene } from '../retention/maintenance.ts';
import { loadDaemonConfig } from '../daemon/config.ts';
import { spawnCurationRun } from '../daemon/lifecycle.ts';
import { admitCapture } from '../retention/capture-gate.ts';
import { judgeAutoCaptures } from '../retention/intake.ts';
import { createTurnJudge } from '../handoff/turn-judge.ts';
import { redactSecrets } from '../security/redact.ts';
import {
  appendSessionObservations, clipSessionSummary, declaredCorrectionQuote, declaredMemoryQuote,
  sessionFailureStatement, sessionObservationId, sessionObservationIdentity, sessionObservationWorthy,
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
  reranker?: Promise<Parameters<typeof federatedSearch>[2]>;
  translator?: Promise<Translator | undefined>;
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

const retainedJson = (value: unknown): string => JSON.stringify(value)
  .replaceAll('<', '\\u003c').replaceAll('>', '\\u003e').replaceAll('\u2028', '\\u2028').replaceAll('\u2029', '\\u2029');

export type MemorySearch = (
  request: Parameters<typeof federatedSearch>[1],
  options?: Parameters<typeof federatedSearch>[2],
) => ReturnType<typeof federatedSearch>;

// The shared evidence-coverage gate runs before ranking for both lookup and
// automatic recall. Do not confuse a rank-fusion score with answerability:
// an additional positive floor discarded supported Spanish/qualified matches.
export const DEFAULT_RECALL_THRESHOLD = 0;

export const installMemoryHooks = (pi: ExtensionAPI, options: {
  home?: string; recallThreshold?: number;
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
  /** Reranker override (tests, custom deployments); defaults to the global TypeSafe credential. */
  rerank?: Parameters<typeof federatedSearch>[2];
  /** Translator override (tests, custom deployments); defaults to the active session model. */
  translator?: Translator;
} = {}) => {
  const recallThreshold = options.recallThreshold ?? DEFAULT_RECALL_THRESHOLD;
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
    if (current.engine) return current.engine;
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
    const current = get();
    if (current.readable) return current.readable;
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
      // This shortcut stores the user's words as the statement itself, so a
      // declaration in another language is translated first. The quote below is
      // kept exactly as spoken: it is the evidence, and evidence is never
      // rewritten. If no model is reachable the declaration stays a queued
      // observation for the daemon rather than entering memory untranslated.
      const statement = await englishStatement(declaration.quote);
      if (!statement) continue;
      // A restated rule is already in memory; a second wording only adds noise.
      if (nearDuplicateFact(statement, project.projection.activeFacts(project.scopeId, DIGEST_SCAN_LIMIT))) continue;
      const observationKind = declaration.kind === 'correction' ? 'correction' : 'instruction';
      // Identity stays keyed on what was actually said. Translation is not
      // deterministic, so keying it on the English text would let the same
      // declaration land twice with two different wordings.
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

  const promoteSessionFailures = async (project: MemoryEngine, observations: readonly SessionObservation[]): Promise<void> => {
    const existing = project.projection.activeFacts(project.scopeId, 200).map(item => ({ statement: item.statement, kind: item.kind }));
    const candidates: { record: SessionObservation; statement: string; id: string; identity: ReturnType<typeof sessionObservationIdentity> }[] = [];
    for (const record of observations) {
      if (record.kind !== 'failure' || !sessionObservationWorthy(record)) continue;
      const statement = sessionFailureStatement(record.summary);
      if (!statement) continue;
      const admission = admitCapture({ statement, kind: 'failure', existing });
      if (!admission.accept) continue;
      // Identity follows the diagnosis, so the same failure with other durations or ids is one memory.
      const identity = sessionObservationIdentity('failure', record.tool, statement);
      const id = `mem_${sha256(`failure:${project.scopeId}:${identity.semanticKey}:${identity.summaryHash}`).slice(0, 32)}`;
      if (project.projection.getFact(id) || candidates.some(item => item.id === id)) continue;
      candidates.push({ record, statement, id, identity });
      existing.push({ statement, kind: 'failure' });
    }
    if (!candidates.length) return;
    // Garbage is never stored: Jev decides, in one request, which of these is a lesson and which is one run's state.
    const provider = (await reranker().catch(() => undefined))?.rerank;
    const ask = provider?.ask ? (state: Record<string, unknown>, questions: Readonly<Record<string, string>>, signal?: AbortSignal) =>
      provider.ask!(state, questions, signal ? { signal } : {}) : undefined;
    const verdicts = await judgeAutoCaptures(candidates.map(item => item.statement), ask);
    for (const [index, { record, statement, id, identity }] of candidates.entries()) {
      if (!verdicts[index]?.keep) continue;
      await project.recordFact({
        // Stays `supported` on purpose: only a supported fact is eligible for the
        // automatic snapshot, and recalling a diagnosis the project already hit
        // is the whole point. Ageing out the ones nobody ever used again is the
        // maintenance pass's job, not the capture path's.
        id, kind: 'failure', statement, confidence: 0.8, standing: 'supported',
        entities: [], episodeIds: [], validAt: record.observedAt,
        evidence: [{
          id: `ev_${sha256(`failure:${project.scopeId}:${identity.summaryHash}`).slice(0, 24)}`,
          origin: 'host_observation', provenance: 'native_observation', contentHash: sha256(statement),
          excerpt: statement, observedAt: record.observedAt,
          actorId: record.sessionId, sessionId: record.sessionId,
        }],
        tags: { semanticKey: identity.semanticKey, summaryHash: identity.summaryHash, source: 'pi-session', capture: 'auto-derived' },
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

  const encoder: { project?: MemoryEngine; ready: boolean; warming?: Promise<void> } = { ready: false };
  /** Starts loading the encoder in the background; true once it can serve queries. */
  const encoderReady = (project: MemoryEngine): boolean => {
    if (encoder.project !== project) Object.assign(encoder, { project, ready: false, warming: undefined });
    if (!encoder.ready && !encoder.warming) {
      encoder.warming = project.warmEncoder().then(ready => { if (encoder.project === project) encoder.ready = ready; });
    }
    return encoder.ready;
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
      await promoteSessionFailures(project, pending.observations);
      scheduleBackfill(project);
    } catch (error) {
      if (get().generation !== pending.generation) throw error;
      set({ observations: [...pending.observations, ...get().observations].slice(-64),
        declarations: [...pending.declarations, ...get().declarations].slice(-16) });
      throw error;
    }
  };

  // Flushing can call a model (translating a declaration to English before it
  // is stored), so it never runs on the send/turn path. Flushes are chained so
  // they stay ordered; shutdown awaits the chain.
  const queue: { tail: Promise<void>; inFlight: number; tidied?: object } = { tail: Promise.resolve(), inFlight: 0 };
  const inBackground = (task: () => Promise<void>): Promise<void> => {
    queue.inFlight += 1;
    queue.tail = queue.tail.then(task).catch(() => undefined)
      .finally(() => { queue.inFlight -= 1; });
    return queue.tail;
  };
  /**
   * Once per session, after the first turn, retire what can never help and fold
   * repeats. It rides the same queue as the flush, so it never touches the
   * prompt path, and it only uses an engine the session already opened.
   */
  const tidy = async (): Promise<void> => {
    const { generation, engine: opened } = get();
    if (queue.tidied === generation || !opened) return;
    queue.tidied = generation;
    const project = await opened.catch(() => undefined);
    if (project && get().generation === generation) await runHygiene(project);
  };
  const flushInBackground = (): Promise<void> => inBackground(async () => {
    await flushSessionObservations();
    await tidy();
  });

  /**
   * Stored memory is English, so a statement that arrives in another language
   * is translated on the way in rather than refused. Resolved once per session
   * from the model already in use, including the negative answer: when no model
   * is reachable nothing is stored in the other language, which is the
   * invariant that matters most.
   */
  const translator = async (): Promise<Translator | undefined> => {
    const current = get().translator;
    if (current) return current;
    const pending = (async (): Promise<Translator | undefined> => {
      if (options.translator) return options.translator;
      const model = get().ctx?.model;
      if (!model) return undefined;
      return createSdkTranslator({ provider: model.provider, model: model.id }).catch(() => undefined);
    })();
    set({ translator: pending });
    return pending;
  };

  /** English as stored, or undefined when it could not be produced. */
  const englishStatement = async (text: string, signal?: AbortSignal): Promise<string | undefined> => {
    if (isEnglish(text)) return text;
    const translate = await translator();
    if (!translate) return undefined;
    const translated = await translate.toEnglish(text, signal).catch(() => undefined);
    return translated && isEnglish(translated) ? translated : undefined;
  };

  const search: MemorySearch = async (request, searchOptions) => federatedSearch(await readable(), request, searchOptions);

  /**
   * Resolved once and cached, including the negative answer. Reranking is
   * optional: no key, no provider, and retrieval keeps the fused order it has
   * always returned. The automatic per-turn hook never asks for this — only a
   * tool call the agent made on purpose does.
   */
  const reranker = async (): Promise<Parameters<typeof federatedSearch>[2]> => {
    const current = get().reranker;
    if (current) return current;
    const pending = (async (): Promise<Parameters<typeof federatedSearch>[2]> => {
      if (options.rerank) return options.rerank;
      try {
        const [{ resolveKey }, { openSecretStore }] = await Promise.all([
          import('@prjct.app/pi-tui-kit'), import('../security/credentials.ts')]);
        const resolved = await resolveKey(await openSecretStore());
        if (!resolved.key) return {};
        const project = await engine();
        const config = await readRerankConfig(project.root);
        // Con credencial resuelta, el reranking va activo salvo que el proyecto lo
        // haya apagado explicitamente: tener la clave es la senal de que se quiere.
        const provider = createRerankProvider({ ...config, enabled: config.enabled ?? true, apiKey: resolved.key });
        return provider ? { rerank: provider, ...(config.candidates ? { rerankCandidates: config.candidates } : {}) } : {};
      } catch {
        // A locked keyring, a missing native binding or an unreadable config
        // must not take retrieval down with it.
        return {};
      }
    })();
    set({ reranker: pending });
    return pending;
  };
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
  // Jev judges old turns in the background; its verdicts only apply when history retirement is on.
  ...(options.history?.enabled === false || options.history === undefined ? {} : { judge: createTurnJudge({
    ask: async () => {
      const provider = (await reranker().catch(() => undefined))?.rerank;
      return provider?.ask ? (state: Record<string, unknown>, questions: Readonly<Record<string, string>>, signal?: AbortSignal) =>
        provider.ask!(state, questions, signal ? { signal } : {}) : undefined;
    },
    facts: async () => {
      const project = await engine().catch(() => undefined);
      return project ? project.projection.activeFacts(project.scopeId, 60).map(fact => fact.statement) : [];
    },
  }) }) });

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
   * Only hard rules ride in every snapshot; the rest is recalled when the prompt
   * makes it relevant. The revision follows the core alone, so a new preference
   * or failure does not resend the snapshot. The full digest still decides
   * whether memory is small enough to skip the encoder (~500MB resident).
   */
  const coreSnapshot = (project: MemoryEngine) => {
    const facts = project.projection.activeFacts(project.scopeId, DIGEST_SCAN_LIMIT)
      .filter(fact => (fact.standing === 'supported' || fact.standing === 'needs_review') && factIsValidAt(fact, Date.now()));
    const digest = memoryDigest(facts, Date.now(), CORE_DIGEST_BYTES, CORE_KINDS);
    return { facts, digest, small: memoryDigest(facts).complete, revision: sha256(digest.block ?? '') };
  };
  const memoryMessage = (revision: string, block: string | undefined, recall?: string, items = 0, omitted = 0) => {
    const memory: MemoryEnvelope = { version: 1, revision,
      snapshot: block ?? 'No eligible facts in this snapshot.', ...(recall ? { recall } : {}) };
    return { customType: 'pi-memory-recall', content: renderMemoryEnvelope(memory), display: false,
      details: { memory, items, omitted } };
  };

  /**
   * Turns that start without a typed prompt (a self-compact handoff, a subagent
   * result) never reach before_agent_start. After the window is rebuilt the core
   * goes back in right away, so those turns do not run without the rules.
   */
  const restoreCore = async (): Promise<void> => {
    const current = get();
    const project = await engine().catch(() => undefined);
    if (!project || get().generation !== current.generation) return;
    const { digest, revision } = coreSnapshot(project);
    if (!digest.block) return;
    pi.sendMessage(memoryMessage(revision, digest.block), { triggerTurn: false });
  };
  pi.on('session_compact', async () => { await restoreCore().catch(() => undefined); });
  pi.on('session_tree', async () => { await restoreCore().catch(() => undefined); });

  pi.on('before_agent_start', async (event, ctx) => {
    const current = get();
    const stale = (): boolean => get().generation !== current.generation;
    void flushInBackground();
    set({ ctx, prompt: event.prompt });
    await countTurn(ctx).catch(() => undefined);
    if (stale()) return { systemPrompt: event.systemPrompt };
    const prompt = clipSessionSummary(event.prompt);
    const observedAt = new Date().toISOString();
    const correction = declaredCorrectionQuote(event.prompt);
    const remembered = correction ? undefined : declaredMemoryQuote(event.prompt);
    const quote = correction ?? remembered;
    const summary = quote ?? prompt;
    const kind = correction ? 'correction' as const : 'instruction' as const;
    if (sessionObservationWorthy({ kind, tool: 'user_input', outcome: 'stated', summary })) {
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
    if (stale()) return { systemPrompt: event.systemPrompt };
    // No system prompt edits: a per-turn system prompt is dropped on automated
    // turns (subagent results, self-compact handoffs), which flipped the cached
    // prefix and re-billed the whole context. The snapshot itself says it is
    // reference data, not instructions.
    const base = event.systemPrompt;
    if (!project) return undefined;
    const { facts, digest, small, revision } = coreSnapshot(project);
    // L0 is policy only. Changing facts belong at the append-only message tail,
    // not in the early system prefix. Include empty snapshots to revoke memory.
    const systemPrompt = base;
    const deliver = (recall?: string, items = 0, omitted = 0) => ({ systemPrompt,
      message: memoryMessage(revision, digest.block, recall, items, omitted) });
    // The snapshot already carries all of memory: nothing to search, no encoder.
    if (facts.every(fact => digest.covered.has(fact.id))) return deliver();
    // Search the rest. Small memories stay lexical; for larger ones dense joins
    // once the encoder has loaded in the background; a prompt never waits for it.
    const dense = small ? false : encoderReady(project);
    const recalled = await search({ queries: recallQueries(event.prompt), limit: 8, maxBytes: 1500, dense, scoreThreshold: recallThreshold, namespaces: ['memory', 'memory.topic'] })
      .catch(() => undefined);
    if (stale()) return { systemPrompt: event.systemPrompt };
    const highConfidence = (recalled?.items ?? [])
      .filter(item => (item.standing === 'supported' || item.standing === 'needs_review') && !digest.covered.has(item.id)).slice(0, 4);
    const memoryBlock = highConfidence.length
      ? `<retained_memory trust="untrusted">\n${highConfidence.map(item => retainedJson({
        id: item.id, namespace: item.namespace, standing: item.standing, provenance: item.provenance,
        statement: item.statement, observedAt: item.observedAt, validAt: item.validAt, invalidAt: item.invalidAt,
      })).join('\n')}\n</retained_memory>`
      : undefined;
    // Deduplicate only after context selection; proposals are not deliveries.
    return deliver(memoryBlock, highConfidence.length, recalled?.omitted ?? 0);
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
    const capped = await capOutput(event.toolName, event.toolCallId, event.content as readonly unknown[], event.details, raw);
    // Every result is staged, but only a failure carries its handle: tagging all
    // results cost ~53k tokens in four days for a handle that was never cited.
    if (!capped && !event.isError) return undefined;
    return { content: [...(capped ?? event.content), ...(event.isError ? [{ type: 'text' as const, text: `[pi-memory evidence: ${handle}]` }] : [])] };
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
    if (project && model && process.env.PI_SUBAGENTS_CHILD !== '1') {
      await loadDaemonConfig({ ...(options.home === undefined ? {} : { home: options.home }), provider: model.provider, model: model.id,
        projectId: project.scopeId, ...(sessionFile ? { sessionFile } : {}) })
        .then(config => spawnCurationRun(config)).catch(() => undefined);
    }
  });

  return {
    engine, writableEngine, initialize, readable, search, reranker, englishStatement, handoff,
    location: async () => location(get().ctx?.cwd ?? process.cwd()),
    scheduleBackfill,
    /** Settles once every background observation flush has finished. */
    flushed: (): Promise<void> => queue.tail,
    stagedEvidence: () => get().evidence, currentPrompt: () => get().prompt,
  };
};
