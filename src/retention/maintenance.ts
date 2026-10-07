import type { MemoryEngine } from '../engine.ts';
import { semanticConsolidationCandidates, type SemanticConsolidationOptions } from './consolidation.ts';
import { planGc, runGc, type GcPlan } from './gc.ts';
import { assessValue, type ValueAssessment } from './value.ts';

/** Planning returns advisory candidates. Only explicit standing-change plans are applied. */

export type MaintenanceSupersession = Readonly<{ factId: string; replacementId: string; similarity: number }>;
export type MaintenanceReview = Readonly<{ factId: string; reason: string }>;

export type MaintenancePlan = Readonly<{
  supersede: readonly MaintenanceSupersession[];
  /** Stale auto-derived diagnoses: retired so collection can reach them. */
  retire: readonly MaintenanceReview[];
  review: readonly MaintenanceReview[];
  gc: GcPlan;
  assessments: readonly ValueAssessment[];
  /** Advisory candidates, never applied as standing changes. */
  suggestions?: readonly MaintenanceReview[];
}>;

export type MaintenanceResult = Readonly<{
  superseded: number; retired: number; reviewed: number; removed: number; retained: number; gaps: readonly string[];
}>;

/**
 * A verdict about a stored statement, supplied explicitly by the caller so the active model can
 * answer it without a second model: `instruction` is the probability that the
 * text tries to steer the reader rather than state something.
 */
export type FactVerdict = Readonly<{ instruction: number }>;
export type FactJudge = (facts: readonly Readonly<{ id: string; statement: string; kind: string }>[])
  => Promise<ReadonlyMap<string, FactVerdict>>;

export type MaintenanceOptions = Readonly<{
  /** Legacy option, ignored: age alone never retires knowledge. */
  failureTtlDays?: number;
  embed?: SemanticConsolidationOptions['embed'];
  /** Listing floor for consolidation candidates. */
  threshold?: number;
  /**
   * Listing floor for advisory duplicate candidates; never authorizes supersession.
   */
  supersedeThreshold?: number;
  judge?: FactJudge;
  /** Probability above which a statement is treated as an instruction, not knowledge. */
  instructionMax?: number;
  /** Nothing is written beyond this many standing changes in one pass. */
  maxActions?: number;
  now?: number;
}>;

export const DEFAULT_SUPERSEDE_THRESHOLD = 0.75;
export const DEFAULT_FAILURE_TTL_DAYS = 7;
/** Raw command output and missing-path reads are symptoms of one moment, not knowledge. */
export const RAW_FAILURE_TTL_DAYS = 0;
export const DEFAULT_INSTRUCTION_MAX = 0.7;
export const DEFAULT_MAX_ACTIONS = 50;

/** Only an open statement can change standing; a closed interval stays closed. */
const OPEN = new Set(['supported', 'candidate', 'needs_review']);

type StoredFact = ReturnType<MemoryEngine['projection']['activeFacts']>[number];

/** Compatibility entry point: automatic hygiene cannot decide which facts still matter. */
export const planHygiene = (_facts: readonly StoredFact[], _now = Date.now(),
  _options: Readonly<{ failureTtlDays?: number; rawFailureTtlDays?: number }> = {}): Pick<MaintenancePlan, 'supersede' | 'retire'> =>
  ({ supersede: [], retire: [] });

/** Kept for older callers; only an explicit, evidence-backed model decision changes standing. */
export const runHygiene = async (_engine: MemoryEngine, _now = Date.now()): Promise<MaintenanceResult> =>
  ({ superseded: 0, retired: 0, reviewed: 0, removed: 0, retained: 0, gaps: [] });

export const planMaintenance = async (engine: MemoryEngine,
  options: MaintenanceOptions = {}): Promise<MaintenancePlan> => {
  const now = options.now ?? Date.now();
  const facts = engine.projection.activeFacts(engine.scopeId);
  const byId = new Map(facts.map(fact => [fact.id, fact]));
  const suggestions = new Map<string, string>();

  if (options.embed) {
    const candidates = await semanticConsolidationCandidates(facts, {
      embed: options.embed, ...(options.threshold ? { threshold: options.threshold } : {}),
    });
    const floor = options.supersedeThreshold ?? DEFAULT_SUPERSEDE_THRESHOLD;
    for (const candidate of candidates) {
      const canonical = byId.get(candidate.canonicalId);
      const related = candidate.relatedIds.map(id => byId.get(id)).filter(Boolean);
      if (!canonical || related.length !== candidate.relatedIds.length) continue;
      // A contradiction is a decision about which statement is true. That is
      // never mechanical, so it is escalated instead of resolved.
      if (candidate.action === 'review-contradiction') {
        for (const fact of [canonical, ...related]) {
          if (OPEN.has(fact!.standing)) suggestions.set(fact!.id, `contradicts ${candidate.relatedIds.join(', ')}`);
        }
        continue;
      }
      if (candidate.similarity < floor) continue;
      for (const fact of related) {
        if (!OPEN.has(fact!.standing) || !OPEN.has(canonical.standing)) continue;
        suggestions.set(fact!.id, `possible duplicate of ${canonical.id} (similarity ${candidate.similarity.toFixed(2)}); requires model review`);
      }
    }
  }

  if (options.judge) {
    const open = facts.filter(fact => OPEN.has(fact.standing));
    if (open.length) {
      const verdicts = await options.judge(open.map(fact => ({ id: fact.id, statement: fact.statement, kind: fact.kind })));
      const max = options.instructionMax ?? DEFAULT_INSTRUCTION_MAX;
      for (const fact of open) {
        const verdict = verdicts.get(fact.id);
        if (verdict && verdict.instruction > max) {
          suggestions.set(fact.id, `reads as an instruction rather than knowledge (${verdict.instruction.toFixed(2)})`);
        }
      }
    }
  }

  return {
    supersede: [], retire: [], review: [],
    suggestions: [...suggestions].map(([factId, reason]) => ({ factId, reason })),
    gc: planGc(engine, now), assessments: facts.map(fact => assessValue(fact, now)),
  };
};

export const applyMaintenance = async (engine: MemoryEngine, plan: MaintenancePlan,
  options: Readonly<{ maxActions?: number; gc?: boolean; now?: number }> = {}): Promise<MaintenanceResult> => {
  const budget = { left: options.maxActions ?? DEFAULT_MAX_ACTIONS };
  const gaps: string[] = (plan.suggestions ?? []).map(item => `${item.factId}: ${item.reason}`);
  const count = { superseded: 0, retired: 0, reviewed: 0 };

  for (const item of plan.supersede) {
    if (budget.left <= 0) break;
    try {
      await engine.resolveFact(item.factId, 'superseded',
        `Maintenance: replaced by ${item.replacementId} (similarity ${item.similarity.toFixed(2)}).`, item.replacementId);
      count.superseded += 1;
      budget.left -= 1;
    } catch (error) {
      gaps.push(`supersede ${item.factId}: ${(error as Error).message}`);
    }
  }

  for (const item of plan.retire) {
    if (budget.left <= 0) break;
    try {
      await engine.resolveFact(item.factId, 'superseded', `Maintenance: ${item.reason}.`);
      count.retired += 1;
      budget.left -= 1;
    } catch (error) {
      gaps.push(`retire ${item.factId}: ${(error as Error).message}`);
    }
  }

  for (const item of plan.review) {
    if (budget.left <= 0) break;
    try {
      await engine.resolveFact(item.factId, 'needs_review', `Maintenance: ${item.reason}.`);
      count.reviewed += 1;
      budget.left -= 1;
    } catch (error) {
      gaps.push(`review ${item.factId}: ${(error as Error).message}`);
    }
  }

  // Collection is `runGc`'s job, and it already batches the journal entries and
  // compacts the projection. Re-planning here is deliberate: the standing
  // changes just written can make a document collectable that was protected
  // when the plan was drawn.
  const kept = { removed: 0, retained: plan.gc.retainedDocumentKeys.length };
  const collected = options.gc === false ? kept : await runGc(engine, options.now ?? Date.now()).catch((error: unknown) => {
    gaps.push(`gc: ${(error as Error).message}`);
    return kept;
  });

  return { ...count, removed: collected.removed, retained: collected.retained, gaps };
};
