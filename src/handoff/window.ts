import { createHash } from 'node:crypto';
import type { OperationalCheckpoint } from './checkpoint.ts';
import {
  selectHandoffMessages, type HandoffBudget, type HandoffOverhead, type HandoffResult,
} from './select.ts';
import {
  DEFAULT_OBSERVATION_POLICY, maskObservations, nextObservationFrontier, type ObservationPolicy,
} from './observations.ts';
import type { HandoffMessage } from './turns.ts';
import { uniqueMemory } from './memory-envelope.ts';
import { DEFAULT_HISTORY_POLICY, retainHistory, type HistoryPolicy, type ReferenceLookup, type TurnVerdicts } from './history.ts';

const digest = (value: unknown): string => createHash('sha256').update(JSON.stringify(value)).digest('hex');
/** Content identity, not role/time/size: equal-sized replacements must reset.
 * One serialization per source message per event, O(history bytes), rather
 * than serializing every growing candidate pack (quadratic). Pi deep-copies
 * context events, so an object-identity ledger cannot establish continuity.
 */
const fingerprintKey = (message: HandoffMessage): string => digest(message);
const prefixDigest = (keys: readonly string[], count: number): string =>
  createHash('sha256').update(keys.slice(0, count).join('\n')).digest('hex');
const isSummary = (message: HandoffMessage): boolean =>
  message.role === 'compactionSummary' || message.role === 'branchSummary';

export type WindowResult = HandoffResult & Readonly<{
  /** Present only when the full host history keeps its message identity and order. */
  unchanged?: true;
  /** Present when this call advanced the observation-masking frontier. */
  observations?: Readonly<{ masked: number; maskedTokens: number }>;
  /** Old turns the explicitly supplied reviewer judged no longer needed that are out of the context now. */
  judged?: Readonly<{ turns: number; tokens: number }>;
}>;

/** Session-local watermark, not a provider cache. No transcript or state is written to disk. */
export const createContextWindow = (policy: ObservationPolicy = DEFAULT_OBSERVATION_POLICY,
  history: HistoryPolicy = DEFAULT_HISTORY_POLICY) => {
  const state = { floor: 0, prefix: '', frontier: 0, frontierPrefix: '', historyFrontier: 0, historyPrefix: '' };
  return function selectWindow(messages: readonly HandoffMessage[], checkpoint: OperationalCheckpoint | undefined,
    budget: HandoffBudget, overhead: HandoffOverhead, reference?: ReferenceLookup, verdicts?: TurnVerdicts): WindowResult {
    // Compaction/tree edits replace the source history; only append-only histories
    // may reuse the watermark. Hashes work with Pi's deep-copied context events.
    const keys = messages.map(fingerprintKey);
    const floor = state.floor < messages.length
      && state.prefix === prefixDigest(keys, state.floor + 1) ? state.floor : 0;
    const priorFrontier = state.frontier <= messages.length
      && state.frontierPrefix === prefixDigest(keys, state.frontier) ? state.frontier : 0;
    const historyFrontier = state.historyFrontier <= messages.length
      && state.historyPrefix === prefixDigest(keys, state.historyFrontier) ? state.historyFrontier : 0;
    // Either policy advancing rewrites the prefix and re-bills everything after
    // it. Flush the other policy in the same request instead of paying a second
    // full miss a few turns later when its own batch threshold is reached.
    const natural = nextObservationFrontier(messages, priorFrontier, policy);
    const planned = retainHistory(maskObservations(messages, natural, policy).messages, historyFrontier, history,
      reference, natural > priorFrontier, verdicts);
    const frontier = planned.frontier > historyFrontier ? nextObservationFrontier(messages, priorFrontier, policy, true) : natural;
    state.frontier = frontier;
    state.frontierPrefix = prefixDigest(keys, frontier);
    // The masked view has the same indexes as the host history.
    const masking = maskObservations(messages, frontier, policy);
    const economic = frontier === natural ? planned : retainHistory(masking.messages, historyFrontier, history, reference, true, verdicts);
    state.historyFrontier = economic.frontier;
    state.historyPrefix = prefixDigest(keys, economic.frontier);
    // Economic replacement changes array indexes, but watermarks use source indexes.
    const view = masking.messages;
    const sourceIndexes = new Map(economic.messages.map((message, index) => [message, economic.sourceIndexes[index]!]));
    const economicView = economic.messages.filter(message => sourceIndexes.get(message)! >= floor);
    // The floor tracks the first retained non-summary message, so the summary
    // always sits below it; filtering it by the floor would drop it next turn.
    const summary = economic.messages.filter(isSummary).at(-1);
    const candidates = uniqueMemory([
      ...(summary ? [summary] : []), ...economicView.filter(message => !isSummary(message)),
    ], (replacement, source) => sourceIndexes.set(replacement, sourceIndexes.get(source)!));
    const selected = selectHandoffMessages(candidates, checkpoint, budget, overhead);
    if (!selected.ok) return selected;
    // Prune in batches, leaving growth room instead of shifting the cache prefix
    // on every next turn. The mandatory current turn always uses the hard limits.
    // A truncated pack is already at the hard limit; shrinking toward 75% would only cut more evidence.
    const compact = selected.omittedTurns > 0 && selected.truncatedFields === 0 ? selectHandoffMessages(candidates, checkpoint, {
      ...budget,
      maxTokens: Math.max(1, Math.floor(budget.maxTokens * 0.75)),
      maxBytes: Math.max(1, Math.floor(budget.maxBytes * 0.75)),
      maxMessages: Math.max(1, Math.floor(budget.maxMessages * 0.75)),
    }, overhead) : selected;
    const result = compact.ok ? compact : selected;
    const first = result.messages.find(message => !isSummary(message) && sourceIndexes.has(message));
    const nextFloor = first ? sourceIndexes.get(first)! : floor;
    state.floor = Math.max(floor, nextFloor);
    state.prefix = prefixDigest(keys, state.floor + 1);
    // Deduplication preceded selection. If selection evicted the first copy,
    // rerun from the advanced floor so a newer copy is available immediately.
    // The strict floor advance bounds retries by the number of input messages.
    const recallKeys = (items: readonly HandoffMessage[]) => new Set(items
      .filter(message => (message as HandoffMessage & { customType?: string }).customType === 'pi-memory-recall')
      .map(message => digest(message.content)));
    const retainedRecall = recallKeys(result.messages);
    if (state.floor > floor && [...recallKeys(view.slice(state.floor))].some(key => !retainedRecall.has(key))) {
      const recovered = selectWindow(messages, checkpoint, budget, overhead, reference, verdicts);
      return frontier > priorFrontier && recovered.ok
        ? { ...recovered, observations: { masked: masking.masked, maskedTokens: masking.maskedTokens } }
        : recovered;
    }
    const unchanged = masking.masked === 0 && result.omittedTurns === 0 && result.truncatedFields === 0
      && result.messages.length === messages.length
      && result.messages.every((message, index) => message === messages[index]);
    return {
      ...result,
      ...(economic.judged.turns ? { judged: economic.judged } : {}),
      ...(unchanged ? { unchanged: true as const } : {}),
      ...(frontier > priorFrontier ? { observations: { masked: masking.masked, maskedTokens: masking.maskedTokens } } : {}),
    };
  };
};
