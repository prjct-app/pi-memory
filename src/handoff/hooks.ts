import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { MemoryEngine } from '../engine.ts';
import { budgetForModel, DEFAULT_HANDOFF_BUDGET, estimateHandoffTokens, selectHandoffMessages, type HandoffBudget } from './select.ts';
import { assertCheckpoint, readCheckpoint, writeCheckpoint, type OperationalCheckpoint } from './checkpoint.ts';
import type { HandoffMessage } from './turns.ts';
import { createContextWindow } from './window.ts';
import { createSessionReferenceResolver } from './session-references.ts';
import { setMode } from '@prjct.app/pi-tui-kit';
import { DEFAULT_HISTORY_POLICY, type HistoryPolicy } from './history.ts';
import { DEFAULT_OBSERVATION_POLICY, type ObservationPolicy } from './observations.ts';

export type HandoffState = Readonly<{
  projectId: string;
  workspace: string;
  sessionId: string;
  active: boolean;
  notice?: string;
}>;

type ContextResult = Readonly<{
  messages: readonly HandoffMessage[];
  unchanged?: true;
}>;

type HandoffGate = Readonly<{
  workspace: string;
  sessionId: string;
  projectId?: string;
  failure?: string;
}>;

const keyOf = (projectId: string, sessionId: string): string => `${projectId}\u0000${sessionId}`;
const gateKeyOf = (workspace: string, sessionId: string): string => `${workspace}\u0000${sessionId}`;
const modelKeyOf = (model: Readonly<{ provider: string; id: string }>): string => `${model.provider}\u0000${model.id}`;

const SAFE: HandoffMessage = {
  role: 'user',
  content: [{ type: 'text', text: 'Model-switch handoff failed safely. Pi 0.85.1 context handlers cannot cancel the network request by throwing; this replacement context is the fail-safe. Reduce tool output or start a fresh session with a concise handoff. Manual /compact is an explicit paid alternative.' }],
};

export const createHandoffController = (options: {
  /** Undefined means transient context bounding without durable memory. */
  engine: () => Promise<MemoryEngine | undefined>;
  /** Explicit ceiling. Omitted means derived per request from the active model's context window. */
  budget?: HandoffBudget;
  toolOverhead?: () => { toolSchemaTokens?: number; toolSchemaBytes?: number };
  /** Stale tool-output masking; defaults to DEFAULT_OBSERVATION_POLICY. */
  observations?: Partial<ObservationPolicy>;
  /** Economic completed-history target, not a hard model/request limit. */
  history?: Partial<HistoryPolicy>;
} ) => {
  const observations: ObservationPolicy = { ...DEFAULT_OBSERVATION_POLICY, ...options.observations };
  const budget = options.budget ?? DEFAULT_HANDOFF_BUDGET;
  const budgetFor = (ctx: ExtensionContext): HandoffBudget => options.budget ?? budgetForModel(ctx.model);
  const slot: {
    states: ReadonlyMap<string, HandoffState>;
    gates: ReadonlyMap<string, HandoffGate>;
    models: ReadonlyMap<string, string>;
    pending: Map<string, Promise<void>>;
    windows: Map<string, ReturnType<typeof createContextWindow>>;
    generation: object;
  } = { states: new Map(), gates: new Map(), models: new Map(), pending: new Map(),
    windows: new Map(), generation: {} };
  const get = (projectId: string, sessionId: string): HandoffState | undefined => slot.states.get(keyOf(projectId, sessionId));
  const getGate = (workspace: string, sessionId: string): HandoffGate | undefined => slot.gates.get(gateKeyOf(workspace, sessionId));
  const put = (state: HandoffState): void => {
    slot.states = new Map([...slot.states, [keyOf(state.projectId, state.sessionId), state]]);
    slot.gates = new Map([...slot.gates, [gateKeyOf(state.workspace, state.sessionId), {
      workspace: state.workspace, sessionId: state.sessionId, projectId: state.projectId,
    }]]);
  };
  const clear = (): void => {
    slot.states = new Map();
    slot.gates = new Map();
    slot.models = new Map();
    slot.pending = new Map();
    slot.windows = new Map();
    slot.generation = {};
  };

  const observeModel = (workspace: string, sessionId: string, model: Readonly<{ provider: string; id: string }>): void => {
    slot.models = new Map([...slot.models, [gateKeyOf(workspace, sessionId), modelKeyOf(model)]]);
  };

  const activate = (projectId: string, workspace: string, sessionId: string, notice: string): void => {
    put({ projectId, workspace, sessionId, active: true, notice });
  };

  const failClosed = (workspace: string, sessionId: string, failure: string): void => {
    slot.gates = new Map([...slot.gates, [gateKeyOf(workspace, sessionId), { workspace, sessionId, failure }]]);
  };

  // Publish a pending gate synchronously. Concurrent context events must await
  // ownership resolution, never race through with the original history.
  const prepare = (ctx: ExtensionContext, notice: string): Promise<void> => {
    const sessionId = ctx.sessionManager.getSessionId();
    const key = gateKeyOf(ctx.cwd, sessionId);
    const existing = slot.pending.get(key);
    if (existing) return existing;
    const generation = slot.generation;
    const task = Promise.resolve().then(async () => {
      try {
        const project = await options.engine();
        if (generation !== slot.generation) throw new Error('Memory session changed during handoff activation.');
        const prior = getGate(ctx.cwd, sessionId);
        if (prior?.failure) return;
        if (prior?.projectId && project?.scopeId !== prior.projectId) throw new Error('Handoff project changed.');
        if (project) activate(project.scopeId, ctx.cwd, sessionId, notice);
        else slot.gates = new Map([...slot.gates, [key, { workspace: ctx.cwd, sessionId }]]);
      } catch (error) {
        if (generation !== slot.generation) throw error;
        failClosed(ctx.cwd, sessionId, `Handoff could not bind the project: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        if (generation === slot.generation) slot.pending.delete(key);
      }
    });
    slot.pending.set(key, task);
    return task;
  };

  const persist = async (engine: MemoryEngine, sessionId: string, draft: Omit<OperationalCheckpoint, 'projectId' | 'sessionId' | 'updatedAt'>): Promise<OperationalCheckpoint> =>
    writeCheckpoint(engine, {
      ...draft, projectId: engine.scopeId, sessionId, updatedAt: new Date().toISOString(),
    });

  const overheadFor = (ctx: ExtensionContext) => {
    const systemPrompt = ctx.getSystemPrompt();
    return {
      systemTokens: estimateHandoffTokens({ role: 'user', content: systemPrompt }),
      systemBytes: Buffer.byteLength(systemPrompt, 'utf8'),
      ...options.toolOverhead?.(),
    };
  };

  const refuse = (ctx: ExtensionContext, message: string): { messages: HandoffMessage[] } => {
    try { ctx.abort(); } catch { /* Best effort: extensions cannot guarantee zero network calls. */ }
    try { ctx.ui.notify(message, 'error'); } catch { /* UI must not restore original context. */ }
    try {
      const fallback = selectHandoffMessages([SAFE], undefined, budgetFor(ctx), overheadFor(ctx));
      if (fallback.ok) return { messages: [SAFE] };
    } catch { /* Invalid configuration or unavailable overhead: send no history. */ }
    // Even SAFE must fit. If fixed system/tools alone exceed the budget, no
    // message replacement can make the full request fit; never strip policies
    // or claim that best-effort abort is a guaranteed transport cancellation.
    return { messages: [] };
  };

  const boundContext = async (messages: readonly HandoffMessage[], ctx: ExtensionContext): Promise<ContextResult> => {
    const current = messages;
    const generation = slot.generation;
    const sessionId = ctx.sessionManager.getSessionId();
    const gate = getGate(ctx.cwd, sessionId);
    if (!gate) throw new Error('Context budget gate is unavailable.');
    if (gate.failure) throw new Error(gate.failure);
    const engine = await options.engine();
    if (generation !== slot.generation) throw new Error('Memory session changed during context selection.');
    if (gate.projectId && engine?.scopeId !== gate.projectId) throw new Error('Handoff project changed before context selection.');
    if (engine && !gate.projectId) activate(engine.scopeId, ctx.cwd, sessionId, 'Durable checkpoint authority available');
    const checkpoint = engine ? readCheckpoint(engine, sessionId) : undefined;
    const windowKey = gateKeyOf(ctx.cwd, sessionId);
    const window = slot.windows.get(windowKey) ?? createContextWindow(observations, { ...DEFAULT_HISTORY_POLICY, ...options.history });
    slot.windows.set(windowKey, window);
    const resolve = createSessionReferenceResolver(ctx.sessionManager, current, generation);
    const selected = window(current, checkpoint, budgetFor(ctx), overheadFor(ctx),
      assistant => resolve(ctx.sessionManager, slot.generation, assistant));
    if (!selected.ok) return refuse(ctx, selected.instruction);
    if (selected.judged) {
      const tokens = selected.judged.tokens;
      try { setMode(ctx, 'history', `history -${tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : tokens} retired · ${selected.judged.turns} turns`); }
      catch { /* the status line must not cost the context */ }
    }
    if (selected.unchanged) return { messages, unchanged: true };
    if (selected.observations?.masked) {
      try { ctx.ui.notify(
        `Context: elided ${selected.observations.masked} stale tool output(s), ~${selected.observations.maskedTokens} tokens. Recent rounds are intact; re-run a tool to see an elided output.`,
        'info',
      ); } catch { /* UI failure must not discard the valid bounded context. */ }
    }
    if (selected.omittedTurns > 0 || selected.truncatedFields > 0) {
      try { ctx.ui.notify(
        `Handoff ${selected.preTokens}→${selected.postTokens} tokens, ${selected.preBytes}→${selected.postBytes} bytes. ${selected.reason}`,
        'info',
      ); } catch { /* UI failure must not discard the valid bounded context. */ }
    }
    return { messages: [...selected.messages] };
  };

  const safeContext = async (messages: readonly HandoffMessage[], ctx: ExtensionContext): Promise<ContextResult> => {
    if (!options.budget && !options.history?.enabled && !options.observations?.enabled) return { messages, unchanged: true };
    const generation = slot.generation;
    try {
      const sessionId = ctx.sessionManager.getSessionId();
      if (!ctx.model) return refuse(ctx, 'Handoff refused: Pi did not expose the current model, so model-switch state cannot be verified.');
      const key = gateKeyOf(ctx.cwd, sessionId);
      const pending = slot.pending.get(key);
      if (pending) await pending;
      else if (!getGate(ctx.cwd, sessionId)) await prepare(ctx, 'Local context budget active');
      if (generation !== slot.generation) throw new Error('Memory session changed before context selection.');
      observeModel(ctx.cwd, sessionId, ctx.model);
      return await boundContext(messages, ctx);
    } catch (error) {
      return refuse(ctx, `Handoff fault: ${error instanceof Error ? error.message : String(error)}. Pi 0.85.1 swallows context-handler throws (fail-open); returning a known safe bounded context.`);
    }
  };

  return { activate, failClosed, observeModel, clear, persist, safeContext, get, getGate, prepare, budget };
};

export const installHandoffHooks = (pi: ExtensionAPI, controller: ReturnType<typeof createHandoffController>,
  _engine: () => Promise<MemoryEngine | undefined>): void => {
  pi.on('model_select', async (event, ctx) => {
    controller.observeModel(ctx.cwd, ctx.sessionManager.getSessionId(), event.model);
    if (!event.previousModel || modelKeyOf(event.model) === modelKeyOf(event.previousModel)) return;
    try {
      await controller.prepare(ctx, `Model switch ${event.source}: ${event.previousModel.provider}/${event.previousModel.id} → ${event.model.provider}/${event.model.id}`);
    } catch (error) {
      try { ctx.ui.notify(error instanceof Error ? error.message : String(error), 'error'); } catch { /* UI only. */ }
    }
  });

  pi.on('context', async (event, ctx) => {
    const result = await controller.safeContext(event.messages as HandoffMessage[], ctx);
    if (result.unchanged) return {};
    return { messages: result.messages as typeof event.messages };
  });

  // Leave provider-native cache keys, retention and routing untouched.
  pi.on('before_provider_request', () => undefined);
};

export { assertCheckpoint, readCheckpoint, writeCheckpoint };
