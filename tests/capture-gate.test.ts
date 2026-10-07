import assert from 'node:assert/strict';
import { test } from 'node:test';
import { admitCapture } from '../src/retention/capture-gate.ts';
import { sessionFailureStatement } from '../src/sources/session-log.ts';

test('capture gate rejects empty text and exact duplicates, leaving meaning to the model', () => {
  assert.equal(admitCapture({ statement: ' ', kind: 'learning', existing: [] }).accept, false);
  const existing = [{ statement: 'Use SQLite.', kind: 'decision' }];
  assert.equal(admitCapture({ statement: 'ok', kind: 'learning', existing }).accept, true);
  assert.equal(admitCapture({ statement: 'Use SQLite.', kind: 'decision', existing }).reason, 'duplicate');
  assert.equal(admitCapture({ statement: 'Do not use SQLite; Postgres is required.', kind: 'correction', existing }).accept, true);
});

test('a missing path is not project knowledge, a real diagnosis alongside one still is', () => {
  // Fue el 73% de una memoria real: rutas que no existian en un instante.
  assert.equal(sessionFailureStatement("read failed\nENOENT: no such file or directory, access '/repo/src/gone.tsx'"), '');
  assert.equal(sessionFailureStatement("bash failed\nENOENT: no such file or directory, open 'x'\nENOTDIR: not a directory, stat 'y'"), '');
  // Un diagnostico de verdad no se pierde por mencionar una ruta.
  assert.match(sessionFailureStatement("bash failed\nerror TS2307: Cannot find module './analytics' in src/app/page.tsx"),
    /TS2307/);
});
