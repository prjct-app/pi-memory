import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { KEYRING_ACCOUNT, saveKey, type SecretStore } from '@prjct.app/pi-tui-kit';
import { ensureEvaluator, evaluatorLine } from '../src/extension/evaluator.ts';
import { readRerankConfig, setRerankEnabled } from '../src/retrieval/rerank.ts';

const memoryStore = (): SecretStore => {
  const slot: { value: string | null } = { value: null };
  return {
    get: async account => account === KEYRING_ACCOUNT ? slot.value : null,
    set: async (_account, secret) => { slot.value = secret; },
    delete: async () => { slot.value = null; },
  };
};

const failingStore = (): SecretStore => ({
  get: async () => { throw new Error('Cannot access the OS keyring.'); },
  set: async () => { throw new Error('Cannot access the OS keyring.'); },
  delete: async () => { throw new Error('Cannot access the OS keyring.'); },
});

const tui = { mode: 'tui', hasUI: true, ui: {} } as never;
const rpc = { mode: 'rpc', hasUI: false, ui: {} } as never;

const counting = () => {
  const state = { calls: 0 };
  return { calls: () => state.calls, prompt: async () => { state.calls += 1; return undefined; } };
};

const root = async (name: string): Promise<string> => mkdtemp(join(tmpdir(), `pi-memory-${name}-`));

test('a shared key is not asked for again and does not enable reranking', async t => {
  const dir = await root('eval-have-key');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const store = memoryStore();
  await saveKey(store, 'typesafe-test-key-12345', true);
  const asked = counting();
  const found = await ensureEvaluator(tui, dir, { store, prompt: asked.prompt });
  assert.equal(asked.calls(), 0);
  assert.equal(found.prompted, false);
  assert.equal(found.resolved.state, 'usable');
  assert.equal(found.enabled, false);
  assert.equal((await readRerankConfig(dir)).enabled, false);
});

test('a broken or unreachable key is reported, not silently replaced', async t => {
  const dir = await root('eval-broken-key');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const asked = counting();
  const found = await ensureEvaluator(tui, dir, { store: failingStore(), prompt: asked.prompt });
  assert.equal(asked.calls(), 0);
  assert.equal(found.resolved.state, 'inaccessible');
  assert.equal(found.enabled, false);
});

test('declining the prompt still leaves a usable project with reranking off', async t => {
  const dir = await root('eval-declined');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const asked = counting();
  const found = await ensureEvaluator(tui, dir, { store: memoryStore(), prompt: asked.prompt });
  assert.equal(asked.calls(), 1);
  assert.equal(found.prompted, true);
  assert.equal(found.enabled, false);
  // Declinar no escribe nada: el proyecto queda sin opinion (`undefined`), y sin
  // credencial no se crea proveedor. La garantia es de comportamiento, no del flag.
  assert.notEqual((await readRerankConfig(dir)).enabled, true);
});

test('without a terminal UI nothing is asked and nothing is enabled', async t => {
  const dir = await root('eval-rpc');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const asked = counting();
  const found = await ensureEvaluator(rpc, dir, { store: memoryStore(), prompt: asked.prompt });
  assert.equal(asked.calls(), 0);
  assert.equal(found.resolved.state, 'missing');
  assert.equal(found.enabled, false);
});

test('setup asks even when a key is already stored, because rotating is the point', async t => {
  const dir = await root('eval-force');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const store = memoryStore();
  await saveKey(store, 'typesafe-test-key-12345', true);
  const asked = counting();
  await ensureEvaluator(tui, dir, { store, prompt: asked.prompt, force: true });
  assert.equal(asked.calls(), 1);
});

test('enabling the stage preserves the project configuration around it', async t => {
  const dir = await root('eval-config');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  await writeFile(join(dir, 'config.json'), JSON.stringify({ provider: 'openai-compatible', model: 'text-embed' }), 'utf8');
  await setRerankEnabled(dir, true);
  const written = JSON.parse(await readFile(join(dir, 'config.json'), 'utf8')) as Record<string, unknown>;
  assert.equal(written.provider, 'openai-compatible');
  assert.equal(written.model, 'text-embed');
  assert.deepEqual(written.rerank, { enabled: true });
});

test('the project configuration is never a place the key can live', async t => {
  const dir = await root('eval-nosecret');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const store = memoryStore();
  await saveKey(store, 'typesafe-test-key-12345', true);
  await ensureEvaluator(tui, dir, { store });
  const written = await readFile(join(dir, 'config.json'), 'utf8').catch(() => '');
  assert.equal(written.includes('typesafe-test-key-12345'), false);
  // The config type has no apiKey field at all, so the check is that nothing
  // put one there anyway.
  assert.equal('apiKey' in await readRerankConfig(dir), false);
});

test('the status line shows a fingerprint and never the key', async () => {
  const store = memoryStore();
  const saved = await saveKey(store, 'typesafe-test-key-12345', true);
  const line = evaluatorLine(saved, true);
  assert.equal(line.includes('typesafe-test-key-12345'), false);
  assert.ok(line.includes(saved.fingerprint ?? 'missing-fingerprint'));
  assert.ok(evaluatorLine({ state: 'missing', source: 'none', detail: 'none' }, false).includes('/memory setup'));
});
