import assert from 'node:assert/strict';
import { access, appendFile, mkdir, mkdtemp, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { collect, describeCandidates, gcCandidates } from '../src/history/gc.ts';
import { backfill } from '../src/history/indexer.ts';
import { openHistoryStore } from '../src/history/store.ts';

const line = (value: unknown): string => `${JSON.stringify(value)}\n`;
const header = (id: string, cwd: string) => line({ type: 'session', version: 3, id, timestamp: '2026-09-01T10:00:00.000Z', cwd });
const message = (id: string, role: string, text: string) => line({ type: 'message', id, parentId: null, timestamp: '2026-09-01T10:01:00.000Z',
  message: { role, content: role === 'user' ? text : [{ type: 'text', text }] } });
const replies = (count: number) => Array.from({ length: count }, (_, index) => message(`a${index}`, 'assistant', `respuesta ${index}`)).join('');
const exists = (path: string) => access(path).then(() => true, () => false);

test('gc lists only quiet junk journals and deletes exactly what it listed, index rows included', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-gc-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessions = join(root, 'sessions');
  const real = '/Users/someone/Apps/prjct/prjct-be';
  const old = new Date('2026-09-02T00:00:00.000Z');
  const journal = async (name: string, body: string, quiet = true) => {
    await mkdir(sessions, { recursive: true });
    const path = join(sessions, `${name}.jsonl`);
    await writeFile(path, body);
    if (quiet) await utimes(path, old, old);
    return path;
  };
  const paths = {
    bench: await journal('bench', header('bench', '/private/tmp/bench') + message('u', 'user', 'probe') + replies(5)),
    tmpWork: await journal('tmp-work', header('tmp-work', '/private/tmp/qa') + message('u', 'user', 'QA real')
      + line({ type: 'compaction', id: 'k', parentId: null, timestamp: '2026-09-01T11:00:00.000Z', summary: '## Goal', firstKeptEntryId: 'u', tokensBefore: 1 })),
    silent: await journal('silent', header('silent', real) + replies(4)),
    teammate: await journal('teammate', header('teammate', real) + line({ type: 'custom_message', id: 'm', parentId: null,
      timestamp: '2026-09-01T10:00:30.000Z', customType: 'team-message', content: 'lead: migra el widget' }) + replies(6)),
    trivial: await journal('trivial', header('trivial', real) + message('u', 'user', 'hola') + replies(2)),
    work: await journal('work', header('work', real) + message('u', 'user', 'migrar widget') + replies(3)),
    fresh: await journal('fresh', header('fresh', real) + replies(1), false),
    current: await journal('current', header('current', '/private/tmp/x') + replies(1)),
  };
  const store = openHistoryStore(join(root, 'history.sqlite'));
  t.after(async () => store.close());
  await backfill(store, sessions);
  const preview = await gcCandidates(store, [paths.current], Date.parse('2026-10-14T00:00:00.000Z'));
  assert.deepEqual(preview.map(item => [item.path, item.reason]).sort(), [
    [paths.bench, 'temporary folder'], [paths.silent, 'no instructions'], [paths.trivial, 'two replies or fewer'],
  ].sort(), 'real work, teammates driven by team messages, compacted temp sessions, recent and current journals are kept');
  assert.match(describeCandidates(preview), /^3 junk sessions, .* MB \(/u);

  await appendFile(paths.trivial, message('u2', 'user', 'sigo aquí'));
  const outcome = await collect(store, preview);
  assert.deepEqual({ deleted: outcome.deleted, skipped: outcome.skipped }, { deleted: 2, skipped: 1 }, 'a journal that changed since the preview is kept');
  assert.equal(await exists(paths.bench), false);
  assert.equal(await exists(paths.silent), false);
  assert.equal(await exists(paths.trivial), true);
  assert.equal(await exists(paths.work), true);
  assert.equal(store.search('probe', {}, 10).total, 0, 'deleted journals leave no search terms behind');
  assert.equal(store.journalStats().some(row => row.path === paths.bench), false);
});
