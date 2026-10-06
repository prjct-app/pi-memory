import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readdir, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { CombinedAutocompleteProvider, type AutocompleteItem } from '@earendil-works/pi-tui';
import { installMemory } from '../src/index.ts';

type CommandDefinition = Readonly<{
  handler(args: string, ctx: any): Promise<void>;
  getArgumentCompletions?(prefix: string): AutocompleteItem[] | null | Promise<AutocompleteItem[] | null>;
}>;

test('memory command completes every action and configured sync adapter without opening storage', async t => {
  const rootDir = await mkdtemp(join(tmpdir(), 'pi-memory-completions-'));
  t.after(() => rm(rootDir, { recursive: true, force: true }));
  const home = join(rootDir, 'home');
  const commands = new Map<string, CommandDefinition>();
  const pi = {
    on() {}, registerTool() {},
    registerCommand(name: string, definition: CommandDefinition) { commands.set(name, definition); },
  } as unknown as ExtensionAPI;
  installMemory(pi, { home, sources: {
    prjct: {},
    extra: [
      { id: 'custom-json', scope: { kind: 'project', id: 'p_example' }, scan: async () => [] },
      { id: 'unsafe\u001b[2J', scope: { kind: 'project', id: 'p_example' }, scan: async () => [] },
    ],
  } });
  const complete = commands.get('memory')?.getArgumentCompletions;
  assert.ok(complete);
  const values = async (prefix: string): Promise<string[] | null> => {
    const items = await complete(prefix);
    return items?.map(item => item.value) ?? null;
  };
  assert.deepEqual(await values(''), [
    'init', 'status', 'sources', 'sync', 'index', 'checkpoint', 'replay', 'rebuild', 'gc', 'prune', 'purge',
    'checkpoint-wal', 'migrate-curated',
  ]);
  assert.deepEqual(await values('st'), ['status']);
  assert.deepEqual(await values('s'), ['status', 'sources', 'sync']);
  assert.deepEqual(await values('sync '), ['sync custom-json', 'sync pi-session', 'sync prjct-observations']);
  assert.deepEqual(await values('sync p'), ['sync pi-session', 'sync prjct-observations']);
  assert.equal(await values('sync missing'), null);
  assert.equal(await values('index {'), null, 'JSON input must remain under direct editor control');
  const root = await complete('');
  assert.ok(root?.every(item => item.description?.startsWith('p · ')), 'every option carries the prjct mark');

  const provider = new CombinedAutocompleteProvider([{ name: 'memory', getArgumentCompletions: complete }], rootDir);
  const suggestions = await provider.getSuggestions(['/memory sync p'], 0, '/memory sync p'.length, {
    signal: AbortSignal.timeout(1_000),
  });
  assert.deepEqual(suggestions?.items.map(item => item.value), ['sync pi-session', 'sync prjct-observations']);
  const selected = suggestions?.items[0];
  assert.ok(selected);
  assert.deepEqual(provider.applyCompletion(['/memory sync p'], 0, '/memory sync p'.length, selected,
    suggestions?.prefix ?? ''), { lines: ['/memory sync pi-session'], cursorLine: 0, cursorCol: 23 });
  assert.deepEqual(await readdir(home).catch(() => []), []);
});

test('memory commands require explicit initialization and reject empty checkpoints', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-commands-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const home = join(root, 'home');
  const hooks = new Map<string, (event: any, ctx: any) => Promise<any>>();
  const commands = new Map<string, CommandDefinition>();
  const pi = {
    on(name: string, handler: (event: any, ctx: any) => Promise<any>) { hooks.set(name, handler); },
    registerTool() {},
    registerCommand(name: string, definition: CommandDefinition) { commands.set(name, definition); },
  } as unknown as ExtensionAPI;
  installMemory(pi, { home });
  const notices: string[] = [];
  const ctx = {
    cwd: root, mode: 'rpc', hasUI: false,
    sessionManager: { getSessionId: () => 'command-session' },
    ui: { notify(message: string) { notices.push(message); } },
  };
  await hooks.get('session_start')!({}, ctx);
  t.after(() => hooks.get('session_shutdown')!({}, ctx));
  const command = commands.get('memory')!.handler;

  await command('status', ctx);
  assert.match(notices.at(-1) ?? '', /initialized no/u);
  assert.match(notices.at(-1) ?? '', /session-references-v2/u);
  assert.doesNotMatch(notices.at(-1) ?? '', /"initialized"/u);
  assert.deepEqual(await readdir(home).catch(() => []), []);
  await assert.rejects(() => command('unknown', ctx), /Usage: \/memory init/u);
  assert.deepEqual(await readdir(home).catch(() => []), []);

  await command('init', ctx);
  assert.match(notices.at(-1) ?? '', /status initialized/u);
  const entries = await readdir(home);
  const projectId = entries.find(entry => /^p_[0-9a-f]{12}$/u.test(entry));
  assert.ok(projectId);
  assert.ok((await stat(join(home, projectId, 'memory', 'memory.sqlite'))).isFile());

  await command('status', ctx);
  assert.match(notices.at(-1) ?? '', /facts 0/u);

  await command('sync', ctx);
  assert.match(notices.at(-1) ?? '', /memory · sync/u);
  assert.doesNotMatch(notices.at(-1) ?? '', /"discovered"/u);

  const screens: string[] = [];
  const tui = {
    ...ctx, mode: 'tui', hasUI: true,
    ui: {
      notify(message: string) { notices.push(message); },
      async custom(factory: any) {
        const panel = factory({ terminal: { columns: 120, rows: 30 }, requestRender() {} }, { fg: (_tone: string, text: string) => text, bold: (text: string) => text }, undefined, () => undefined);
        screens.push(panel.render(120).join('\n'));
        throw Object.assign(new Error('Operation aborted'), { name: 'AbortError' });
      },
    },
  };
  const before = notices.length;
  await command('status', tui);
  assert.match(screens.at(-1) ?? '', /Memory .* sources?/u, 'the terminal gets the docked panel');
  assert.match(screens.at(-1) ?? '', /facts\s+0/u);
  assert.match(screens.at(-1) ?? '', /s Sync all sources · g Collect garbage · w Checkpoint WAL · R Rebuild index/u);
  assert.equal(notices.length, before, 'no modal text and no abort noise');

  await assert.rejects(() => command('checkpoint {}', ctx), /non-empty goal/u);
});

const commandHarness = async (t: { after(fn: () => unknown): void }) => {
  const rootDir = await mkdtemp(join(tmpdir(), 'pi-memory-command-'));
  t.after(() => rm(rootDir, { recursive: true, force: true }));
  const repo = join(rootDir, 'repo');
  await mkdir(repo, { recursive: true });
  await mkdir(join(repo, '.git'), { recursive: true });
  const commands = new Map<string, CommandDefinition>();
  const handlers = new Map<string, (event: any, context: any) => Promise<unknown>>();
  const notices: Array<{ level: string; text: string }> = [];
  const pi = {
    on(name: string, handler: (event: any, context: any) => Promise<unknown>) { handlers.set(name, handler); },
    registerTool() {},
    registerCommand(name: string, definition: CommandDefinition) { commands.set(name, definition); },
  } as unknown as ExtensionAPI;
  installMemory(pi, { home: join(rootDir, 'home') });
  const ctx = {
    cwd: repo, mode: 'rpc', hasUI: false,
    sessionManager: { getSessionId: () => 'cmd-session' },
    ui: { notify: (text: string, level = 'info') => notices.push({ level, text }) },
  };
  await handlers.get('session_start')!({}, ctx);
  t.after(async () => { await handlers.get('session_shutdown')?.({}, ctx); });
  const run = async (args: string): Promise<{ level: string; text: string } | undefined> => {
    await commands.get('memory')!.handler(args, ctx);
    return notices.at(-1);
  };
  return { run, notices };
};

test('/memory index without a payload answers with its usage, as it always has', async t => {
  const h = await commandHarness(t);
  // The handler renders the card and rethrows, so the host logs it too. That
  // double report is long-standing behaviour for every command error.
  await assert.rejects(h.run('index'), /not initialized|Usage: \/memory index/u);
  assert.equal(h.notices.at(-1)?.level, 'error');
});

test('/memory init needs no classifier configuration or secret prompt', async t => {
  const h = await commandHarness(t);
  const answer = await h.run('init');
  assert.notEqual(answer?.level, 'error');
  assert.match(answer?.text ?? '', /initialized/u);
  assert.doesNotMatch(h.notices.map(item => item.text).join('\n'), /rerank|credential|setup|jev/i);
  await assert.rejects(h.run('setup'), /Usage: \/memory init/u);
});

test('/memory index with a payload is accepted once memory exists', async t => {
  const h = await commandHarness(t);
  await h.run('init');
  const answer = await h.run('index {"namespace":"project.docs","externalId":"doc-1","text":"The release checklist lives in docs/release.md."}');
  assert.notEqual(answer?.level, 'error');
});
