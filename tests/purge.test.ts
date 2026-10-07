import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm, utimes } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { MemoryEngine } from '../src/engine.ts';
import { withoutPurged } from '../src/retention/purge.ts';
import { TestEmbeddingProvider } from './helpers.ts';

type Storage = 'indexed' | 'compact';
const open = (home: string, storage: Storage, session = 's1') =>
  MemoryEngine.forScope('project', 'p_purge', session, { home, provider: new TestEmbeddingProvider(), storage });
const remember = (engine: MemoryEngine, kind: 'failure' | 'constraint', statement: string) =>
  engine.recordFact({ kind, statement, standing: 'supported', entities: [], evidence: [], episodeIds: [], confidence: 0.9, tags: {} }).then(recorded => recorded.fact);

const journalText = async (root: string): Promise<string> => {
  const days = await readdir(join(root, 'events')).catch(() => [] as string[]);
  const texts = await Promise.all(days.map(async day => Promise.all((await readdir(join(root, 'events', day)))
    .map(name => readFile(join(root, 'events', day, name), 'utf8')))));
  return texts.flat().join('');
};

for (const storage of ['indexed', 'compact'] as const) {
  test(`${storage}: a purged fact is gone from every table, search and the journal, and stays gone`, async t => {
    const home = await mkdtemp(join(tmpdir(), `pi-memory-purge-${storage}-`));
    t.after(() => rm(home, { recursive: true, force: true }));
    const engine = await open(home, storage);
    const junk = await remember(engine, 'failure', 'read ENOENT zebrafish route.ts failed once');
    const keep = await remember(engine, 'constraint', 'Tool schemas use typebox, never zod.');
    await engine.feedback(junk.id, 'used', 'zebrafish');
    await engine.resolveFact(junk.id, 'superseded', 'Raw tool output, not knowledge.', keep.id);
    assert.equal(engine.projection.getFact(junk.id), undefined, 'resolving a fact dead deletes it; it is no longer only flagged');

    const result = await engine.compactJournal();
    assert.equal(engine.projection.getFact(keep.id)?.statement, 'Tool schemas use typebox, never zod.');
    assert.equal((await engine.search({ queries: ['zebrafish ENOENT'], dense: false })).items.length, 0);
    if (storage === 'indexed') {
      const text = await journalText(engine.root);
      assert.doesNotMatch(text, /zebrafish/, 'not one byte of it left in the journal');
      assert.match(text, /typebox/);
      assert.equal(result.events, 2, 'the recording and the feedback; the resolution was never written');
    } else {
      assert.doesNotMatch(JSON.stringify(await engine.journal.readAll()), /zebrafish/);
    }

    // The chain still verifies, this writer keeps appending, and a replay brings nothing back.
    const later = await remember(engine, 'constraint', 'Run tests with npm test.');
    assert.ok((await engine.journal.readAll()).some(event => event.payload.type === 'fact.recorded' && event.payload.fact.id === later.id));
    await engine.replay();
    assert.equal(engine.projection.getFact(junk.id), undefined);
    await engine.dispose();
    const reopened = await open(home, storage, 's2');
    t.after(() => reopened.dispose());
    await reopened.replay();
    assert.equal(reopened.projection.getFact(junk.id), undefined);
    assert.equal(reopened.projection.getFact(later.id)?.statement, 'Run tests with npm test.');
  });
}

test('indexed: another writer\'s live stream is deferred, and its events cannot resurrect the fact', async t => {
  const home = await mkdtemp(join(tmpdir(), 'pi-memory-purge-live-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const other = await open(home, 'indexed', 'other');
  const junk = await remember(other, 'failure', 'transient quokka failure output');
  const mine = await open(home, 'indexed', 'mine');
  t.after(async () => { await other.dispose(); await mine.dispose(); });
  const result = await mine.purgeFacts([junk.id]);
  assert.equal(result.deferred, 1, 'the other session touched its file just now');
  assert.equal(mine.projection.getFact(junk.id), undefined);
  // The deferred event is still in that file; applying it again must not recreate the fact.
  const events = await mine.journal.readAll();
  const recorded = events.find(event => event.payload.type === 'fact.recorded' && event.payload.fact.id === junk.id)!;
  assert.ok(recorded, 'still on disk until the next pass');
  mine.projection.purgeFacts([], [recorded.id]);
  assert.equal(mine.projection.apply(recorded), false);
  assert.equal(mine.projection.getFact(junk.id), undefined);
  await other.recordFact({ kind: 'constraint', statement: 'The other writer keeps working.', standing: 'supported', entities: [], evidence: [], episodeIds: [], confidence: 0.9, tags: {} });
  // Once that file is quiet, the next pass removes it.
  const days = await readdir(join(mine.root, 'events'));
  for (const day of days) for (const name of await readdir(join(mine.root, 'events', day))) {
    await utimes(join(mine.root, 'events', day, name), new Date(0), new Date(0));
  }
  const again = await mine.purgeFacts([junk.id]);
  assert.equal(again.deferred, 0);
  assert.doesNotMatch(await journalText(mine.root), /quokka/);
});

test('a batch commit keeps its other facts when one of them is purged', () => {
  const fact = (id: string) => ({ id, scopeId: 'p', kind: 'fact', statement: id, entities: [], evidence: [], episodeIds: [],
    standing: 'supported', confidence: 1, recordedAt: '2026-01-01T00:00:00.000Z', tags: {} }) as never;
  const payload = { type: 'curation.batch.commit', batchId: 'b', sourceRevision: 'r', facts: [fact('a'), fact('b')], documents: [],
    resolves: [{ factId: 'a', standing: 'superseded', rationale: 'x' }] } as const;
  const cleaned = withoutPurged(payload as never, new Set(['a'])) as typeof payload;
  assert.deepEqual(cleaned.facts.map(item => (item as { id: string }).id), ['b']);
  assert.deepEqual(cleaned.resolves, []);
  assert.equal(withoutPurged(payload as never, new Set()), payload, 'nothing purged, nothing touched');
});

test('the sweep backs up and deletes closed facts while preserving all active evidence', async t => {
  const { planSweep, runSweep } = await import('../src/retention/sweep.ts');
  const home = await mkdtemp(join(tmpdir(), 'pi-memory-sweep-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const engine = await open(home, 'indexed');
  t.after(() => engine.dispose());
  // A fact left superseded the old way: written straight to the journal and projection.
  const legacy = await remember(engine, 'constraint', 'Old rule about platypus deploys.');
  engine.projection.apply(await engine.journal.append({ type: 'fact.resolved', factId: legacy.id, standing: 'superseded', rationale: 'old way' }));
  const run = await remember(engine, 'failure', 'The test run failed 4 of 76 tests in the wombat suite.');
  const lesson = await remember(engine, 'failure', 'Installing with npm corrupts the lockfile; use pnpm install.');
  const raw = await remember(engine, 'failure', 'bash: echidna: command not found');
  const plan = await planSweep(engine);
  assert.deepEqual(plan.dead, [legacy.id]);
  assert.deepEqual(plan.junk, []);
  assert.ok(engine.projection.getFact(run.id), 'semantic judgments remain with the active model');
  const done = await runSweep(engine, plan, join(home, 'backups'));
  assert.equal(done.facts, 1);
  assert.ok(done.backup && (await readdir(done.backup)).includes('memory.sqlite'));
  assert.equal(engine.projection.getFact(lesson.id)?.statement, 'Installing with npm corrupts the lockfile; use pnpm install.');
  assert.equal(engine.projection.getFact(legacy.id), undefined);
  assert.equal(engine.projection.getFact(raw.id)?.statement, raw.statement);
  assert.doesNotMatch(await journalText(engine.root), /platypus/);
  assert.match(await journalText(engine.root), /echidna/);
});
