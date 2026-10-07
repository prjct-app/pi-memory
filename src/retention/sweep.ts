import { cp, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import type { MemoryEngine } from '../engine.ts';

/**
 * One pass over what is already stored, for memory written before deleting
 * meant deleting: every fact left superseded or contradicted, with active facts preserved. Semantic judgments belong to the active model. Planning
 * reads only; running backs the store up first, then purges.
 */
export type SweepPlan = Readonly<{
  dead: readonly string[];
  junk: readonly Readonly<{ id: string; statement: string; reason: string }>[];
}>;

export const planSweep = async (engine: MemoryEngine): Promise<SweepPlan> => ({
  dead: engine.projection.deadFactIds(engine.scopeId),
  junk: [], // Active facts require a model decision; command-like text is still evidence.
});

export const runSweep = async (engine: MemoryEngine, plan: SweepPlan, backupRoot: string) => {
  const ids = [...plan.dead, ...plan.junk.map(item => item.id)];
  if (!ids.length) return { backup: undefined, facts: 0, documents: 0, evidence: 0, events: 0, deferred: 0 };
  const backup = join(backupRoot, engine.scopeId);
  await mkdir(backup, { recursive: true, mode: 0o700 });
  engine.projection.exportSnapshot(join(backup, 'memory.sqlite'));
  await cp(join(engine.root, 'events'), join(backup, 'events'), { recursive: true }).catch(error => {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  });
  return { backup, ...await engine.purgeFacts(ids) };
};
