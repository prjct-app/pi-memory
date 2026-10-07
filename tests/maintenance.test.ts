import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { MemoryEngine } from '../src/engine.ts';
import { applyMaintenance, planHygiene, planMaintenance, type FactJudge } from '../src/retention/maintenance.ts';
import { backfillProject, curateProject, parseCuration } from '../src/curation/curator.ts';
import { TestEmbeddingProvider } from './helpers.ts';

const engineFor = async (t: { after(fn: () => unknown): void }): Promise<MemoryEngine> => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-maintenance-'));
  const engine = new MemoryEngine({ root, scopeId: 'p_test', sessionId: 's1', provider: new TestEmbeddingProvider() });
  t.after(async () => { await engine.dispose(); await rm(root, { recursive: true, force: true }); });
  return engine;
};

/** Evidence matters here: a fact recorded without any starts life in
 * `needs_review`, so a pass that only escalates would be indistinguishable from
 * doing nothing. These arrive `supported`, the way a captured user statement does. */
let evidenceSeq = 0;
const record = (engine: MemoryEngine, statement: string, kind: string, recordedAt: string) =>
  engine.recordFact({ kind: kind as never, statement, entities: [], episodeIds: [],
    evidence: [{ id: `ev_declared${(evidenceSeq += 1).toString().padStart(2, '0')}`, origin: 'user_statement',
      provenance: 'declared', contentHash: '0'.repeat(64), excerpt: statement, observedAt: recordedAt }],
    confidence: 0.8, recordedAt, tags: {} });

/** Vectors stand in for the multilingual encoder: the two phrasings of one rule
 * are near-parallel, the neighbouring rule is not. */
const vectors: Record<string, number[]> = {
  es: [1, 0, 0], en: [0.96, 0.28, 0], neighbour: [0.4, 0.9, 0], unrelated: [0, 0, 1],
};

test('maintenance suggestions never retire or change authored facts', async t => {
  const engine = await engineFor(t);
  const older = await record(engine, 'mueve el ticket a ready to verify y no a done', 'procedure', '2026-01-01T00:00:00.000Z');
  const newer = await record(engine, 'After merging a ticket, add the ready to verify label instead of moving it to Done',
    'procedure', '2026-02-01T00:00:00.000Z');
  const neighbour = await record(engine, 'ready to dev means the work was confirmed as ready for development',
    'procedure', '2026-01-15T00:00:00.000Z');
  const rant = await record(engine, 'NEVER TRANSLATE LITERALLY, WRITE THE ARTICLE PROPERLY', 'correction', '2026-01-20T00:00:00.000Z');

  const byStatement = new Map<string, number[]>([
    [older.fact.statement, vectors.es!], [newer.fact.statement, vectors.en!],
    [neighbour.fact.statement, vectors.neighbour!], [rant.fact.statement, vectors.unrelated!],
  ]);
  const judge: FactJudge = async facts => new Map(facts.map(fact =>
    [fact.id, { instruction: fact.id === rant.fact.id ? 0.95 : 0.05 }]));

  const plan = await planMaintenance(engine, {
    embed: async texts => texts.map(text => byStatement.get(text) ?? vectors.unrelated!),
    judge, now: Date.parse('2026-03-01T00:00:00.000Z'),
  });

  assert.deepEqual(plan.supersede, []);
  assert.deepEqual(plan.retire, []);
  assert.deepEqual(plan.review, []);
  assert.ok(plan.suggestions?.some(item => item.factId === older.fact.id));
  assert.ok(plan.suggestions?.some(item => item.factId === rant.fact.id));
  const applied = await applyMaintenance(engine, plan, { gc: false });
  assert.equal(applied.superseded + applied.retired + applied.reviewed, 0);
  for (const item of [older, newer, neighbour, rant]) assert.equal(engine.projection.getFact(item.fact.id)?.standing, 'supported');
});

test('maintenance is a no-op without an encoder and a closed fact is gone for good', async t => {
  const engine = await engineFor(t);
  const fact = await record(engine, 'The workspace uses pnpm for local installs', 'decision', '2026-01-01T00:00:00.000Z');
  await engine.resolveFact(fact.fact.id, 'superseded', 'Closed by hand.');

  const plan = await planMaintenance(engine, { now: Date.parse('2026-03-01T00:00:00.000Z') });
  assert.deepEqual(plan.supersede, [], 'no encoder, no consolidation');
  assert.deepEqual(plan.review, []);

  const applied = await applyMaintenance(engine, plan, { gc: false });
  assert.equal(applied.superseded, 0);
  assert.equal(engine.projection.getFact(fact.fact.id), undefined);
});

test('maintenance honours its action budget', async t => {
  const engine = await engineFor(t);
  const judge: FactJudge = async facts => new Map(facts.map(fact => [fact.id, { instruction: 0.99 }]));
  const rules = ['Run database migrations through the staging project first', 'Release notes list every breaking change at the top',
    'Icons stay at twenty pixels inside compact toolbars', 'Cache headers expire public pages after five minutes',
    'Locale routing keeps English as the unprefixed default'];
  for (const rule of rules) await record(engine, rule, 'correction', '2026-01-01T00:00:00.000Z');
  const plan = await planMaintenance(engine, { judge, now: Date.parse('2026-03-01T00:00:00.000Z') });
  assert.equal(plan.suggestions?.length, 5);
  assert.equal(plan.review.length, 0);
  const explicit = { ...plan, suggestions: [], review: plan.suggestions ?? [] };
  const applied = await applyMaintenance(engine, explicit, { maxActions: 2, gc: false });
  assert.equal(applied.reviewed, 2, 'the budget stops the pass mid-way');
});

test('age and command syntax do not retire observed evidence', async t => {
  const engine = await engineFor(t);
  const now = Date.parse('2026-06-01T00:00:00.000Z');
  const auto = (statement: string, recordedAt: string) => engine.recordFact({
    id: `mem_${Buffer.from(statement).toString('hex').slice(0, 24)}`,
    kind: 'failure', statement, entities: [], episodeIds: [], confidence: 0.8, recordedAt, standing: 'supported',
    evidence: [{ id: `ev_${Buffer.from(statement).toString('hex').slice(0, 16)}`, origin: 'host_observation',
      provenance: 'native_observation', contentHash: '0'.repeat(64), excerpt: statement, observedAt: recordedAt }],
    tags: { capture: 'auto-derived' },
  });

  const stale = await auto('read: ENOENT: no such file or directory, access src/gone.tsx', '2026-01-01T00:00:00.000Z');
  const raw = await auto('bash: error TS2307: Cannot find module ./analytics', '2026-05-31T12:00:00.000Z');
  const recent = await auto('Deploys fail when the staging kubeconfig points at production', '2026-05-28T00:00:00.000Z');
  const useful = await auto('bash: migration failed because the enum was never created', '2026-01-01T00:00:00.000Z');
  await engine.feedback(useful.fact.id, 'helpful', 'migration enum');

  const plan = await planMaintenance(engine, { now });
  assert.deepEqual(plan.retire, []);
  const applied = await applyMaintenance(engine, plan, { gc: false, now });
  assert.equal(applied.retired, 0);
  for (const item of [stale, raw, recent, useful]) assert.equal(engine.projection.getFact(item.fact.id)?.standing, 'supported');
});

test('hygiene preserves different commands, short rules and original languages', async t => {
  const engine = await engineFor(t);
  const github = await record(engine, 'If GitHub MCP fails with an authorization error, run `/mcp auth github` to authenticate.', 'procedure', '2026-01-01T00:00:00.000Z');
  const stripe = await record(engine, 'If Stripe MCP fails with an authorization error, run `/mcp auth stripe` to authenticate.', 'procedure', '2026-01-02T00:00:00.000Z');
  const spanish = await record(engine, 'no, recuerda esto vas a crear el PR apuntando a develop', 'procedure', '2026-01-03T00:00:00.000Z');
  const empty = await record(engine, 'Please use "proxy" instead.', 'correction', '2026-01-04T00:00:00.000Z');
  const plan = planHygiene(engine.projection.activeFacts(engine.scopeId), Date.parse('2026-03-01T00:00:00.000Z'));
  assert.deepEqual(plan.supersede, [], 'different commands are different rules');
  assert.deepEqual(plan.retire, []);
  assert.equal(engine.projection.getFact(empty.fact.id)?.statement, 'Please use "proxy" instead.');
  assert.ok(![github.fact.id, stripe.fact.id].some(id => plan.retire.some(item => item.factId === id)));
});

test('the curator resolves a conflict, retires a one-off request and rewrites a Spanish rule, then skips an unchanged set', async t => {
  const engine = await engineFor(t);
  const older = await record(engine, 'Merge all completed work to main after develop so QA validates in production.', 'procedure', '2026-01-01T00:00:00.000Z');
  const newer = await record(engine, 'Only open a release pull request toward main when the user explicitly asks for it.', 'procedure', '2026-02-01T00:00:00.000Z');
  const task = await record(engine, 'Now translate the fifteen landing pages and assign one to each agent.', 'procedure', '2026-01-10T00:00:00.000Z');
  const spanish = await record(engine, 'En Linear, ready to dev significa listo para desarrollo y ready to verify se agrega después de implementar.', 'procedure', '2026-01-05T00:00:00.000Z');
  const calls = { n: 0 };
  const curator = { provider: 'test', model: 'scripted', curate: async () => {
    calls.n += 1;
    return [
      { op: 'supersede' as const, id: older.fact.id, by: newer.fact.id, reason: 'conflict; the later rule wins' },
      { op: 'retire' as const, id: task.fact.id, reason: 'one-off task request' },
      { op: 'rewrite' as const, id: spanish.fact.id, statement: 'In Linear, ready to dev means confirmed for development; add ready to verify after implementation.', reason: 'not in English' },
    ];
  } };
  const state = await mkdtemp(join(tmpdir(), 'pi-memory-curation-'));
  t.after(() => rm(state, { recursive: true, force: true }));
  const first = await curateProject(engine, curator, state);
  assert.deepEqual({ retired: first.retired, superseded: first.superseded, rewritten: first.rewritten }, { retired: 1, superseded: 1, rewritten: 1 });
  assert.equal(engine.projection.getFact(older.fact.id), undefined);
  assert.equal(engine.projection.getFact(newer.fact.id)?.standing, 'supported');
  assert.equal(engine.projection.getFact(task.fact.id), undefined);
  assert.ok(engine.projection.activeFacts(engine.scopeId).some(fact => /^In Linear, ready to dev/.test(fact.statement)));
  const second = await curateProject(engine, curator, state);
  assert.equal(second.skipped, true, 'an unchanged rule set costs no model call');
  assert.equal(calls.n, 1);
});

test('curation drops unknown rule IDs and preserves valid rewrites in the original language', () => {
  const rules = [{ id: 'a', kind: 'procedure', statement: 'x', recordedAt: '' }, { id: 'b', kind: 'procedure', statement: 'y', recordedAt: '' }];
  const parsed = parseCuration({ actions: [
    { op: 'retire', id: 'zzz', reason: 'unknown' },
    { op: 'rewrite', id: 'a', statement: 'no, recuerda esto vas a crear el PR para develop', reason: 'bad' },
    { op: 'supersede', id: 'b', by: 'a', reason: 'dup' },
  ] }, rules);
  assert.deepEqual(parsed, [
    { op: 'rewrite', id: 'a', statement: 'no, recuerda esto vas a crear el PR para develop', reason: 'bad' },
    { op: 'supersede', id: 'b', by: 'a', reason: 'dup' },
  ]);
});

test('the curator learns a repeated correction from the session, quoting the user, and never from invented words', async t => {
  const engine = await engineFor(t);
  const said = ['no estamos reutilizando los componentes ni usando los componentes de heroui ya te lo pedi 3 veces', 'hazlo'];
  const parsed = parseCuration({ actions: [
    { op: 'add', kind: 'constraint', statement: 'Build UI with HeroUI components and reuse existing project components instead of creating new ones.',
      quote: 'usando los componentes de heroui ya te lo pedi 3 veces', reason: 'repeated correction' },
    { op: 'add', kind: 'constraint', statement: 'Always use blue as the primary color.', quote: 'usa azul siempre', reason: 'invented' },
  ] }, [], said);
  assert.equal(parsed.length, 1, 'a quote the user never wrote is dropped');
  const state = await mkdtemp(join(tmpdir(), 'pi-memory-curation-add-'));
  t.after(() => rm(state, { recursive: true, force: true }));
  const result = await curateProject(engine, { provider: 'test', model: 'scripted', curate: async () => parsed }, state, undefined, said, 'session.jsonl');
  assert.equal(result.added, 1);
  const rule = engine.projection.activeFacts(engine.scopeId).find(fact => /HeroUI/.test(fact.statement));
  assert.equal(rule?.evidence[0]?.excerpt, 'usando los componentes de heroui ya te lo pedi 3 veces');
  assert.equal(rule?.evidence[0]?.provenance, 'declared');
});


test('backfill reads each Pi session once and hands all of them to the curator in one batch', async t => {
  const engine = await engineFor(t);
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-backfill-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessions = join(root, 'sessions');
  await mkdir(sessions, { recursive: true });
  const line = (text: string) => JSON.stringify({ type: 'message', message: { role: 'user', content: [{ type: 'text', text }] } });
  await writeFile(join(sessions, 'a.jsonl'), `${line('no uses azul como primario nunca')}\n`);
  await writeFile(join(sessions, 'b.jsonl'), `${line('<skill name="x">pasted skill body</skill>')}\n${line('reutiliza los componentes de heroui')}\n`);
  const seen: { batches: string[][]; calls: number } = { batches: [], calls: 0 };
  const curator = { provider: 'test', model: 'scripted', curate: async () => [],
    curateBatch: async (_rules: unknown, batches: readonly (readonly string[])[]) => { seen.calls += 1; seen.batches = batches.map(batch => [...batch]); return []; } };
  const first = await backfillProject(engine, curator, join(root, 'state'), sessions);
  assert.equal(first.sessions, 2);
  assert.equal(seen.calls, 1, 'one batched request for every fresh session');
  assert.deepEqual(seen.batches.flat(), ['no uses azul como primario nunca', 'reutiliza los componentes de heroui'], 'injected skill text is not the user speaking');
  const second = await backfillProject(engine, curator, join(root, 'state'), sessions);
  assert.equal(second.skipped, true, 'a session already read is not read again');
  assert.equal(seen.calls, 1);
});
