import { estimateTokens } from '@earendil-works/pi-coding-agent';
import type { OperationalCheckpoint } from './checkpoint.ts';
import { renderCheckpoint } from './checkpoint.ts';
import { shrinkToFit } from './shrink.ts';
import { normalizeToolFlow } from './normalize.ts';
import { groupTurns, toolCallIds, turnIsComplete, type HandoffMessage, type Turn } from './turns.ts';

export type HandoffBudget = Readonly<{
  maxTokens: number;
  maxBytes: number;
  maxMessages: number;
  /** Fallback reserve when a host does not expose active tool definitions. */
  toolSchemaReserveTokens: number;
}>;

export type HandoffOverhead = Readonly<{
  systemTokens: number;
  systemBytes: number;
  toolSchemaTokens?: number;
  toolSchemaBytes?: number;
}>;

export const DEFAULT_HANDOFF_BUDGET: HandoffBudget = {
  maxTokens: 16_000,
  maxBytes: 262_144,
  maxMessages: 48,
  toolSchemaReserveTokens: 1_500,
};

/**
 * Derive the ceiling from the active model instead of a fixed cost cap. A fixed
 * 16k budget applied to every request truncated or aborted ordinary document
 * reads on 272k+ models. Reserve room for the response, like Pi's own
 * compaction reserve, and fall back to the conservative default when the host
 * does not expose a usable context window.
 */
export const budgetForModel = (model: Readonly<{ contextWindow?: number; maxTokens?: number }> | undefined): HandoffBudget => {
  const window = model?.contextWindow;
  if (!window || !Number.isSafeInteger(window) || window < 1) return DEFAULT_HANDOFF_BUDGET;
  const output = model.maxTokens && Number.isSafeInteger(model.maxTokens) && model.maxTokens > 0 ? model.maxTokens : 16_384;
  const reserve = Math.min(Math.max(16_384, Math.min(output, 32_768)), Math.floor(window / 4));
  const maxTokens = window - reserve;
  return {
    maxTokens,
    // Signed reasoning and JSON escaping make bytes outgrow chars/4 estimates.
    maxBytes: Math.min(Number.MAX_SAFE_INTEGER, maxTokens * 16),
    // Short tool rounds must not evict context that still fits the model.
    maxMessages: Number.MAX_SAFE_INTEGER,
    toolSchemaReserveTokens: DEFAULT_HANDOFF_BUDGET.toolSchemaReserveTokens,
  };
};

const NO_OVERHEAD: HandoffOverhead = { systemTokens: 0, systemBytes: 0 };

export type HandoffResult = Readonly<{
  ok: true;
  messages: readonly HandoffMessage[];
  preTokens: number;
  postTokens: number;
  preMessageTokens: number;
  postMessageTokens: number;
  systemTokens: number;
  toolSchemaReserveTokens: number;
  toolSchemaBytes: number;
  preBytes: number;
  postBytes: number;
  systemBytes: number;
  omittedTurns: number;
  /** Oversized strings capped because the mandatory pack could not otherwise fit. */
  truncatedFields: number;
  reason: string;
}> | Readonly<{
  ok: false;
  instruction: string;
  preTokens: number;
  preBytes: number;
  systemTokens: number;
  toolSchemaReserveTokens: number;
  toolSchemaBytes: number;
}>;

// `details` is host/UI metadata that Pi never sends to the provider. Counting
// it priced a 53-token agent_jobs status at ~100KB, so the byte ceiling evicted
// a current-turn tool round on every call and re-billed the whole prompt.
const messageBytes = (message: HandoffMessage): number => {
  const { details: _details, ...sent } = message as HandoffMessage & { details?: unknown };
  return Buffer.byteLength(JSON.stringify(sent), 'utf8');
};

/**
 * Messages are immutable here, and selection prices the same objects many
 * times while it grows candidate packs. Re-serializing every candidate made the
 * context hook O(turns x history bytes): ~250ms per LLM call on a 15MB session.
 */
const measured = new WeakMap<object, { tokens?: number; bytes?: number }>();
const cachedBytes = (message: HandoffMessage): number => {
  const entry = measured.get(message) ?? {};
  if (entry.bytes === undefined) {
    entry.bytes = messageBytes(message);
    measured.set(message, entry);
  }
  return entry.bytes;
};
/** Equals Buffer.byteLength(JSON.stringify(messages)): brackets plus comma separators. */
const messagePackBytes = (messages: readonly HandoffMessage[]): number =>
  2 + Math.max(0, messages.length - 1) + messages.reduce((sum, message) => sum + cachedBytes(message), 0);

const assertBudget = (budget: HandoffBudget, overhead: HandoffOverhead): void => {
  const positive = [budget.maxTokens, budget.maxBytes, budget.maxMessages];
  if (positive.some(value => !Number.isSafeInteger(value) || value <= 0)) throw new Error('Handoff limits must be positive safe integers.');
  const reserves = [budget.toolSchemaReserveTokens, overhead.systemTokens, overhead.systemBytes,
    overhead.toolSchemaTokens ?? 0, overhead.toolSchemaBytes ?? 0];
  if (reserves.some(value => !Number.isSafeInteger(value) || value < 0)) throw new Error('Handoff overhead must be non-negative safe integers.');
};

export const estimateHandoffTokens = (message: HandoffMessage): number => {
  const entry = measured.get(message) ?? {};
  if (entry.tokens === undefined) {
    entry.tokens = uncachedTokens(message);
    measured.set(message, entry);
  }
  return entry.tokens;
};

const uncachedTokens = (message: HandoffMessage): number => {
  try {
    return estimateTokens(message as never);
  } catch {
    return Math.max(1, Math.ceil(messageBytes(message) / 4));
  }
};

const packCost = (messages: readonly HandoffMessage[], budget: HandoffBudget, overhead: HandoffOverhead): {
  tokens: number; messageTokens: number; bytes: number;
} => {
  const messageTokens = messages.reduce((sum, message) => sum + estimateHandoffTokens(message), 0);
  return {
    messageTokens,
    tokens: overhead.systemTokens + (overhead.toolSchemaTokens ?? budget.toolSchemaReserveTokens) + messageTokens,
    bytes: overhead.systemBytes + (overhead.toolSchemaBytes ?? 0) + messagePackBytes(messages),
  };
};

const checkpointMessage = (checkpoint: OperationalCheckpoint): HandoffMessage => ({
  role: 'user',
  content: [{ type: 'text', text: `Operational checkpoint (not a transcript):\n${renderCheckpoint(checkpoint)}` }],
});

// Pi 0.85.1 exposes applied compaction and branch summaries as AgentMessage
// roles. Only the newest is a fallback continuity checkpoint; a newer explicit
// checkpoint wins so stale summaries cannot displace current operator intent.
const isPiSummary = (message: HandoffMessage): boolean =>
  message.role === 'compactionSummary' || message.role === 'branchSummary';

const continuityPrefix = (
  messages: readonly HandoffMessage[], checkpoint: OperationalCheckpoint | undefined,
): readonly HandoffMessage[] => {
  const latestSummary = messages.filter(isPiSummary).at(-1);
  if (!latestSummary) return checkpoint ? [checkpointMessage(checkpoint)] : [];
  if (!checkpoint) return [latestSummary];
  const checkpointAt = Date.parse(checkpoint.updatedAt);
  const summaryAt = typeof latestSummary.timestamp === 'number' ? latestSummary.timestamp : Number.NaN;
  return Number.isFinite(checkpointAt) && Number.isFinite(summaryAt) && checkpointAt < summaryAt
    ? [latestSummary]
    : [checkpointMessage(checkpoint)];
};

const fits = (messages: readonly HandoffMessage[], budget: HandoffBudget, overhead: HandoffOverhead): boolean => {
  if (messages.length > budget.maxMessages) return false;
  const cost = packCost(messages, budget, overhead);
  return cost.tokens <= budget.maxTokens && cost.bytes <= budget.maxBytes;
};

type CurrentTurnFit = Readonly<{
  ok: true;
  turn: Turn;
  omittedToolRounds: number;
}> | Readonly<{
  ok: false;
  minimum: readonly HandoffMessage[];
}>;

/**
 * Keep the user request and the newest complete tool rounds. Older rounds in a
 * long-running agent turn are expendable once their effects are reflected in
 * later calls; retaining every round would eventually abort otherwise healthy
 * work. Tool call/result pairs remain atomic.
 */
const fitCurrentTurn = (turn: Turn, prefix: readonly HandoffMessage[], budget: HandoffBudget,
  overhead: HandoffOverhead): CurrentTurnFit => {
  if (fits([...prefix, ...turn.messages], budget, overhead)) {
    return { ok: true, turn, omittedToolRounds: 0 };
  }
  const starts = turn.messages.flatMap((message, index) => toolCallIds(message).length ? [index] : []);
  if (!starts.length) return { ok: false, minimum: turn.messages };
  const head = turn.messages.slice(0, starts[0]);
  const rounds = starts.map((start, index) => ({
    messages: turn.messages.slice(start, starts[index + 1] ?? turn.messages.length),
    toolCallIds: toolCallIds(turn.messages[start]!),
  } satisfies Turn));
  if (rounds.some(round => !turnIsComplete(round))) return { ok: false, minimum: turn.messages };
  const newest = rounds.at(-1)!;
  const minimum = [...head, ...newest.messages];
  if (!fits([...prefix, ...minimum], budget, overhead)) return { ok: false, minimum };
  const kept = { rounds: [newest] as Turn[] };
  for (const round of rounds.slice(0, -1).reverse()) {
    const candidate = [...prefix, ...head, ...[round, ...kept.rounds].flatMap(item => item.messages)];
    if (!fits(candidate, budget, overhead)) break;
    kept.rounds = [round, ...kept.rounds];
  }
  return {
    ok: true,
    turn: { messages: [...head, ...kept.rounds.flatMap(round => round.messages)], toolCallIds: turn.toolCallIds },
    omittedToolRounds: rounds.length - kept.rounds.length,
  };
};

export const selectHandoffMessages = (
  messages: readonly HandoffMessage[],
  checkpoint: OperationalCheckpoint | undefined,
  budget: HandoffBudget = DEFAULT_HANDOFF_BUDGET,
  overhead: HandoffOverhead = NO_OVERHEAD,
): HandoffResult => {
  assertBudget(budget, overhead);
  const retained = normalizeToolFlow(messages);
  const pre = packCost(retained, budget, overhead);
  const prefix = continuityPrefix(retained, checkpoint);
  const turns = groupTurns(retained.filter(message => !isPiSummary(message)));
  const toolTokens = overhead.toolSchemaTokens ?? budget.toolSchemaReserveTokens;
  const toolBytes = overhead.toolSchemaBytes ?? 0;
  const diagnostic = `Estimated tokens: system ${overhead.systemTokens} + active tools ${toolTokens} + messages ${pre.messageTokens} = ${pre.tokens}. Bytes: system ${overhead.systemBytes} + active tools ${toolBytes} + canonical messages ${pre.bytes - overhead.systemBytes - toolBytes} = ${pre.bytes}.`;
  const current = turns.at(-1);
  if (current && !turnIsComplete(current)) {
    return {
      ok: false, instruction: `Handoff refused: the current turn has an unmatched, duplicate, or orphaned tool call/result. ${diagnostic}`,
      preTokens: pre.tokens, preBytes: pre.bytes, systemTokens: overhead.systemTokens,
      toolSchemaReserveTokens: toolTokens, toolSchemaBytes: toolBytes,
    };
  }
  const currentFit = current ? fitCurrentTurn(current, prefix, budget, overhead) : undefined;
  const minimum = currentFit?.ok === false ? [...prefix, ...currentFit.minimum] : prefix;
  const continuityOf = (selected: readonly HandoffMessage[]): string =>
    selected[0]?.role === 'compactionSummary' || selected[0]?.role === 'branchSummary'
      ? 'latest Pi summary' : prefix.length ? 'explicit checkpoint' : 'no checkpoint';
  if (currentFit?.ok === false || (!current && !fits(minimum, budget, overhead))) {
    // Degrade before refusing: aborting a healthy agent loop because one tool
    // result is large is worse than showing the model a truncated copy of it.
    const shrunk = minimum.length <= budget.maxMessages
      ? shrinkToFit(minimum, pack => fits(pack, budget, overhead)) : undefined;
    if (shrunk) {
      const post = packCost(shrunk.messages, budget, overhead);
      const keptTurns = current ? 1 : 0;
      const omittedTurns = Math.max(0, turns.length - keptTurns);
      const rounds = (items: readonly HandoffMessage[]): number => items.filter(message => toolCallIds(message).length).length;
      const omittedToolRounds = current ? rounds(current.messages) - rounds(minimum) : 0;
      return {
        ok: true, messages: shrunk.messages, preTokens: pre.tokens, postTokens: post.tokens,
        preMessageTokens: pre.messageTokens, postMessageTokens: post.messageTokens,
        systemTokens: overhead.systemTokens, toolSchemaReserveTokens: toolTokens, toolSchemaBytes: toolBytes,
        preBytes: pre.bytes, postBytes: post.bytes, systemBytes: overhead.systemBytes,
        omittedTurns, truncatedFields: shrunk.truncatedFields,
        reason: `Kept ${continuityOf(shrunk.messages)} and ${keptTurns} complete turn(s); omitted ${omittedTurns} older turn(s) and ${omittedToolRounds} older tool round(s) from the current turn; truncated ${shrunk.truncatedFields} oversized field(s) to ${shrunk.keep} chars to fit the budget. Tokens: system ${overhead.systemTokens} + active tools ${toolTokens} + messages ${post.messageTokens} = ${post.tokens}. Bytes: system ${overhead.systemBytes} + active tools ${toolBytes} + canonical messages ${post.bytes - overhead.systemBytes - toolBytes} = ${post.bytes}.`,
      };
    }
    const required = packCost(minimum, budget, overhead);
    const requiredDiagnostic = `Required tokens: system ${overhead.systemTokens} + active tools ${toolTokens} + messages ${required.messageTokens} = ${required.tokens}. Bytes: system ${overhead.systemBytes} + active tools ${toolBytes} + canonical messages ${required.bytes - overhead.systemBytes - toolBytes} = ${required.bytes}.`;
    return {
      ok: false,
      instruction: `${prefix.length
        ? 'Handoff refused: continuity checkpoint plus the newest complete tool round and provider overhead exceed the token/byte/message budget. Reduce the latest tool output, update a smaller checkpoint, or start a new session.'
        : 'Handoff refused: the current request plus its newest complete tool round and provider overhead exceed the token/byte/message budget. Reduce the latest tool output or start a fresh session.'} ${requiredDiagnostic}`,
      preTokens: pre.tokens, preBytes: pre.bytes, systemTokens: overhead.systemTokens,
      toolSchemaReserveTokens: toolTokens, toolSchemaBytes: toolBytes,
    };
  }
  const omittedToolRounds = currentFit?.ok ? currentFit.omittedToolRounds : 0;
  const kept = { turns: currentFit?.ok ? [currentFit.turn] : [] as Turn[] };
  const older = current ? turns.slice(0, -1) : turns;
  for (const turn of [...older].reverse()) {
    if (!turnIsComplete(turn)) continue;
    const candidate = [...prefix, ...[turn, ...kept.turns].flatMap(item => item.messages)];
    if (!fits(candidate, budget, overhead)) break;
    kept.turns = [turn, ...kept.turns];
  }
  const selected = [...prefix, ...kept.turns.flatMap(turn => turn.messages)];
  const post = packCost(selected, budget, overhead);
  const continuity = continuityOf(prefix);
  return {
    ok: true, messages: selected, preTokens: pre.tokens, postTokens: post.tokens,
    preMessageTokens: pre.messageTokens, postMessageTokens: post.messageTokens,
    systemTokens: overhead.systemTokens, toolSchemaReserveTokens: toolTokens, toolSchemaBytes: toolBytes,
    preBytes: pre.bytes, postBytes: post.bytes, systemBytes: overhead.systemBytes,
    omittedTurns: Math.max(0, turns.length - kept.turns.length), truncatedFields: 0,
    reason: `Kept ${continuity} and ${kept.turns.length} complete turn(s); omitted ${Math.max(0, turns.length - kept.turns.length)} older turn(s) and ${omittedToolRounds} older tool round(s) from the current turn. Tokens: system ${overhead.systemTokens} + active tools ${toolTokens} + messages ${post.messageTokens} = ${post.tokens}. Bytes: system ${overhead.systemBytes} + active tools ${toolBytes} + canonical messages ${post.bytes - overhead.systemBytes - toolBytes} = ${post.bytes}.`,
  };
};
