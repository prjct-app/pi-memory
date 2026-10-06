import type { MemoryEngine } from '../engine.ts';
import { semanticConsolidationCandidates, type SemanticConsolidationOptions } from './consolidation.ts';
import { planGc, runGc, type GcPlan } from './gc.ts';
import { assessValue, type ValueAssessment } from './value.ts';
import { contentTermCount, nearDuplicateFact } from '../curation/similar.ts';

/**
 * The maintenance pass is what turns a capture log into living memory: it
 * revisits facts already stored instead of only ingesting new documents.
 *
 * Everything here is conservative on purpose. Knowledge is expensive to
 * recreate and cheap to keep, so the pass never deletes a fact: a duplicate is
 * superseded by the copy that replaces it, and anything else it distrusts is
 * moved to `needs_review` for a human or the agent to settle. Only documents
 * that `planGc` already considers expendable are removed.
 */

export type MaintenanceSupersession = Readonly<{ factId: string; replacementId: string; similarity: number }>;
export type MaintenanceReview = Readonly<{ factId: string; reason: string }>;

export type MaintenancePlan = Readonly<{
  supersede: readonly MaintenanceSupersession[];
  /** Stale auto-derived diagnoses: retired so collection can reach them. */
  retire: readonly MaintenanceReview[];
  review: readonly MaintenanceReview[];
  gc: GcPlan;
  assessments: readonly ValueAssessment[];
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
  /**
   * Days after which an auto-derived diagnosis nobody found useful again stops
   * being treated as project knowledge. A failure that keeps happening earns a
   * usefulness signal and is kept; a passing symptom ages out.
   */
  failureTtlDays?: number;
  embed?: SemanticConsolidationOptions['embed'];
  /** Listing floor for consolidation candidates. */
  threshold?: number;
  /**
   * Automatic supersession needs more confidence than merely showing a pair to
   * a reader, so it sits above the listing floor.
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

const BEHAVIOR = new Set(['correction', 'constraint', 'preference', 'decision', 'procedure']);
export const RAW_FAILURE = /^(bash|sh|zsh):|ENOENT|no (such )?file (or directory )?existed|did not exist at/iu;
const shouting = (text: string): boolean => {
  const letters = text.replace(/[^A-Za-z]/gu, '');
  return letters.length >= 20 && letters.replace(/[^A-Z]/gu, '').length / letters.length > 0.6;
};
// The copy that survives: confirmed over unconfirmed, the more complete wording, then the newer.
const better = (a: StoredFact, b: StoredFact): StoredFact => {
  const score = (fact: StoredFact) => [fact.standing === 'supported' ? 1 : 0, fact.statement.length, Date.parse(fact.recordedAt)];
  const [x, y] = [score(a), score(b)];
  const at = x.findIndex((value, index) => value !== y[index]);
  return at < 0 ? a : x[at]! > y[at]! ? a : b;
};

/**
 * Deterministic hygiene: no model, no encoder, safe to run on every session
 * start. It retires what can never help the agent and folds lexical repeats
 * into one statement. Nothing is deleted outright; each change is a standing
 * transition in the journal, so history survives.
 */
export const planHygiene = (facts: readonly StoredFact[], now = Date.now(),
  options: Readonly<{ failureTtlDays?: number; rawFailureTtlDays?: number }> = {}): Pick<MaintenancePlan, 'supersede' | 'retire'> => {
  const open = facts.filter(fact => OPEN.has(fact.standing));
  const retire = new Map<string, string>();
  const supersede: MaintenanceSupersession[] = [];
  const day = 86_400_000;
  const ttl = (options.failureTtlDays ?? DEFAULT_FAILURE_TTL_DAYS) * day;
  const rawTtl = (options.rawFailureTtlDays ?? RAW_FAILURE_TTL_DAYS) * day;
  for (const fact of open) {
    const age = now - Date.parse(fact.recordedAt);
    if (fact.kind === 'failure') {
      // A failure that came back and helped earns usefulness and is kept.
      if (fact.usefulness > 0 || fact.evidence[0]?.provenance !== 'native_observation') continue;
      const raw = RAW_FAILURE.test(fact.statement);
      if (age >= (raw ? rawTtl : ttl)) retire.set(fact.id, raw ? 'raw tool output, not project knowledge' : 'auto-captured failure never reused');
      continue;
    }
    if (!BEHAVIOR.has(fact.kind)) continue;
    if (shouting(fact.statement)) retire.set(fact.id, 'a verbatim outburst, not a rule');
    else if (contentTermCount(fact.statement) < 4) retire.set(fact.id, 'too short to mean anything out of context');
  }
  // Fold repeats among the behavior rules that survived; the more
  // complete wording is the one kept.
  const rules = open.filter(fact => BEHAVIOR.has(fact.kind) && !retire.has(fact.id));
  const gone = new Set<string>();
  for (const fact of rules) {
    if (gone.has(fact.id)) continue;
    const others = rules.filter(other => other.id !== fact.id && !gone.has(other.id));
    const twin = nearDuplicateFact(fact.statement, others);
    if (!twin) continue;
    const keep = better(fact, twin);
    const drop = keep === fact ? twin : fact;
    gone.add(drop.id);
    supersede.push({ factId: drop.id, replacementId: keep.id, similarity: 1 });
  }
  // A rule in another language is left for the curator to rewrite: retiring it
  // here would lose knowledge that exists in no other wording.
  // Point every folded copy at the statement that finally survives; if that one
  // is retired, the copy goes with it.
  const next = new Map(supersede.map(item => [item.factId, item.replacementId]));
  const survivor = (id: string): string => (next.has(id) ? survivor(next.get(id)!) : id);
  const folded: MaintenanceSupersession[] = [];
  for (const item of supersede) {
    const target = survivor(item.replacementId);
    if (retire.has(target)) retire.set(item.factId, retire.get(target)!);
    else folded.push({ ...item, replacementId: target });
  }
  return { supersede: folded, retire: [...retire].map(([factId, reason]) => ({ factId, reason })) };
};

/** Runs the hygiene rules and writes their standing changes. No encoder, no model. */
export const runHygiene = async (engine: MemoryEngine, now = Date.now()): Promise<MaintenanceResult> => {
  const plan = planHygiene(engine.projection.activeFacts(engine.scopeId, DIGEST_SCAN_LIMIT), now);
  if (!plan.supersede.length && !plan.retire.length) return { superseded: 0, retired: 0, reviewed: 0, removed: 0, retained: 0, gaps: [] };
  return applyMaintenance(engine, { ...plan, review: [], gc: planGc(engine, now), assessments: [] },
    { gc: false, maxActions: DIGEST_SCAN_LIMIT, now });
};
const DIGEST_SCAN_LIMIT = 500;

export const planMaintenance = async (engine: MemoryEngine,
  options: MaintenanceOptions = {}): Promise<MaintenancePlan> => {
  const now = options.now ?? Date.now();
  const facts = engine.projection.activeFacts(engine.scopeId);
  const byId = new Map(facts.map(fact => [fact.id, fact]));
  const supersede: MaintenanceSupersession[] = [];
  const review = new Map<string, string>();

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
          if (OPEN.has(fact!.standing)) review.set(fact!.id, `contradicts ${candidate.relatedIds.join(', ')}`);
        }
        continue;
      }
      if (candidate.similarity < floor) continue;
      for (const fact of related) {
        if (!OPEN.has(fact!.standing) || !OPEN.has(canonical.standing)) continue;
        supersede.push({ factId: fact!.id, replacementId: canonical.id, similarity: candidate.similarity });
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
          review.set(fact.id, `reads as an instruction rather than knowledge (${verdict.instruction.toFixed(2)})`);
        }
      }
    }
  }

  const hygiene = planHygiene(facts, now, options.failureTtlDays === undefined ? {} : { failureTtlDays: options.failureTtlDays });
  for (const item of hygiene.supersede) {
    if (!supersede.some(existing => existing.factId === item.factId)) supersede.push(item);
  }
  const retire = hygiene.retire.filter(item => !supersede.some(existing => existing.factId === item.factId));

  // A statement already being replaced does not also need a review row.
  for (const item of supersede) review.delete(item.factId);

  return {
    supersede, retire,
    review: [...review].filter(([factId]) => !retire.some(item => item.factId === factId))
      .map(([factId, reason]) => ({ factId, reason })),
    gc: planGc(engine, now), assessments: facts.map(fact => assessValue(fact, now)),
  };
};

export const applyMaintenance = async (engine: MemoryEngine, plan: MaintenancePlan,
  options: Readonly<{ maxActions?: number; gc?: boolean; now?: number }> = {}): Promise<MaintenanceResult> => {
  const budget = { left: options.maxActions ?? DEFAULT_MAX_ACTIONS };
  const gaps: string[] = [];
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
