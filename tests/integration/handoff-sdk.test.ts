import assert from 'node:assert/strict';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import {
  InMemoryCredentialStore,
  createAssistantMessageEventStream,
  type AssistantMessage,
  type Context,
  type Model,
  type SimpleStreamOptions,
} from '@earendil-works/pi-ai';
import {
  createAgentSession,
  DefaultResourceLoader,
  estimateTokens,
  defineTool,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { createHandoffController, installHandoffHooks } from '../../src/handoff/hooks.ts';
import { MemoryEngine } from '../../src/engine.ts';
import { installMemory } from '../../src/index.ts';

const usage = () => ({ input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } });

const textOf = (content: unknown): string => typeof content === 'string' ? content : Array.isArray(content)
  ? content.flatMap(block => block && typeof block === 'object' && typeof (block as { text?: unknown }).text === 'string'
    ? [(block as { text: string }).text] : []).join('\n')
  : '';

type ProviderCapture = Readonly<{
  model: string;
  context: Readonly<{ systemPrompt: string; messages: Context['messages'] }>;
  payload: unknown;
  sessionId?: string;
  cacheRetention?: string;
}>;

const offlineProvider = (captures: ProviderCapture[]) =>
  (model: Model<any>, context: Context, options?: SimpleStreamOptions) => {
    const stream = createAssistantMessageEventStream();
    void (async () => {
      const output: AssistantMessage = {
        role: 'assistant', content: [], api: model.api, provider: model.provider, model: model.id,
        usage: usage(), stopReason: 'pending', timestamp: Date.now(),
      };
      try {
        const proposed = { systemPrompt: context.systemPrompt, messages: context.messages, tools: context.tools };
        const replacement = await options?.onPayload?.(proposed, model);
        captures.push({
          model: model.id,
          context: { systemPrompt: context.systemPrompt ?? '', messages: structuredClone(context.messages) },
          payload: replacement ?? proposed,
          ...(options?.sessionId ? { sessionId: options.sessionId } : {}),
          ...(options?.cacheRetention ? { cacheRetention: options.cacheRetention } : {}),
        });
        stream.push({ type: 'start', partial: output });
        const latestUser = [...context.messages].reverse().find(message => message.role === 'user');
        const requestsTools = textOf(latestUser?.content).includes('MULTI_TOOL_LATEST');
        const results = context.messages.filter(message => message.role === 'toolResult');
        if (requestsTools && results.length < 2) {
          const calls = [
            { type: 'toolCall' as const, id: 'sdk_call_a', name: 'offline_one', arguments: {} },
            { type: 'toolCall' as const, id: 'sdk_call_b', name: 'offline_two', arguments: {} },
          ];
          for (const call of calls) {
            const contentIndex = output.content.length;
            output.content.push(call);
            stream.push({ type: 'toolcall_start', contentIndex, partial: output });
            stream.push({ type: 'toolcall_end', contentIndex, toolCall: call, partial: output });
          }
          output.stopReason = 'toolUse';
        } else {
          const contentIndex = 0;
          output.content.push({ type: 'text', text: 'offline complete' });
          stream.push({ type: 'text_start', contentIndex, partial: output });
          stream.push({ type: 'text_end', contentIndex, content: 'offline complete', partial: output });
          output.stopReason = 'stop';
        }
        stream.push({ type: 'done', reason: output.stopReason, message: output });
        stream.end();
      } catch (error) {
        output.stopReason = 'error';
        output.errorMessage = error instanceof Error ? error.message : String(error);
        stream.push({ type: 'error', reason: 'error', error: output });
        stream.end();
      }
    })();
    return stream;
  };

const createOfflineRuntime = async (root: string, captures: ProviderCapture[]) => {
  const modelRuntime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(), modelsPath: null, modelsStorePath: join(root, 'models-store.json'),
    refreshOnCreate: false,
  });
  modelRuntime.registerProvider('offline-handoff', {
    name: 'Offline handoff', baseUrl: 'http://127.0.0.1.invalid', apiKey: 'offline-only', api: 'offline-handoff-api',
    streamSimple: offlineProvider(captures),
    models: ['a', 'b'].map(id => ({
      id, name: `Offline ${id}`, api: 'offline-handoff-api', reasoning: false, input: ['text'] as const,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 100_000, maxTokens: 1_000,
    })),
  });
  await modelRuntime.setRuntimeApiKey('offline-handoff', 'offline-only');
  const modelA = modelRuntime.getModel('offline-handoff', 'a');
  const modelB = modelRuntime.getModel('offline-handoff', 'b');
  assert.ok(modelA && modelB);
  return { modelRuntime, modelA, modelB };
};

test('public Pi SDK sends bounded provider contexts across A→B→A and a multi-call tool loop', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-handoff-sdk-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const captures: ProviderCapture[] = [];
  const home = join(root, 'home');
  const initialized = await MemoryEngine.initializeProject(root, 'handoff-setup', { home });
  await initialized.engine.dispose();
  const { modelRuntime, modelA, modelB } = await createOfflineRuntime(root, captures);
  const tool = (name: 'offline_one' | 'offline_two', result: string) => defineTool({
    name, label: name, description: `Return ${result}`, parameters: Type.Object({}),
    execute: async () => ({ content: [{ type: 'text' as const, text: result }], details: {} }),
  });
  const resourceLoader = new DefaultResourceLoader({
    cwd: root, agentDir: join(root, 'agent'), systemPromptOverride: () => 'SDK_SYSTEM_POLICY',
    extensionFactories: [pi => installMemory(pi, {
      home,
      handoff: { maxTokens: 12_000, maxBytes: 65_536, maxMessages: 6, toolSchemaReserveTokens: 1_000 },
    })],
  });
  await resourceLoader.reload();
  const sessionManager = SessionManager.inMemory(root);
  sessionManager.appendMessage({ role: 'user', content: `OLD_PRIVATE_TRANSCRIPT ${'do-not-replay '.repeat(10_000)}`, timestamp: 1 });
  const { session } = await createAgentSession({
    cwd: root, agentDir: join(root, 'agent'), model: modelA, modelRuntime, resourceLoader,
    sessionManager, settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
    tools: ['offline_one', 'offline_two'], customTools: [tool('offline_one', 'RESULT_A'), tool('offline_two', 'RESULT_B')],
  });
  t.after(() => session.dispose());

  await session.prompt('READY_FOR_HANDOFF');
  await session.prompt('/memory checkpoint {"goal":"SDK_CHECKPOINT_GOAL","constraints":["KEEP_QUALIFICATION"],"done":[],"inProgress":[],"blocked":[],"decisions":[],"evidenceRefs":["ev_sdk"],"nextSteps":["run tools"]}');
  const afterOld = captures.length;
  await session.setModel(modelB);
  await session.prompt('MULTI_TOOL_LATEST preserve this latest instruction');
  const bRequests = captures.slice(afterOld);
  assert.equal(bRequests.length, 2);
  await session.setModel(modelA);
  await session.prompt('RETURN_TO_A_LATEST');
  const switched = [...bRequests, captures.at(-1)!];

  for (const request of switched) {
    const serialized = JSON.stringify(request.context.messages);
    const payload = JSON.stringify(request.payload);
    assert.equal(serialized.includes('OLD_PRIVATE_TRANSCRIPT'), false);
    assert.equal(payload.includes('OLD_PRIVATE_TRANSCRIPT'), false);
    assert.match(serialized, /SDK_CHECKPOINT_GOAL/);
    assert.match(payload, /SDK_CHECKPOINT_GOAL/);
    assert.ok(Buffer.byteLength(request.context.systemPrompt, 'utf8')
      + Buffer.byteLength(JSON.stringify(request.context.messages), 'utf8') <= 65_536);
    const totalTokens = estimateTokens({ role: 'user', content: request.context.systemPrompt } as never) + 1_000
      + request.context.messages.reduce((sum, message) => sum + estimateTokens(message as never), 0);
    assert.ok(totalTokens <= 12_000);
    // Pi 1.0.4 normalizes provider instructions/tools into system transcript items.
    // The handoff message budget bounds conversation messages, not SDK metadata.
    assert.ok(request.context.messages.filter(message => message.role !== 'system').length <= 6);
    assert.match(JSON.stringify(request.context), /SDK_SYSTEM_POLICY/);
  }
  const toolLoop = JSON.stringify(bRequests[1]!.context.messages);
  const toolLoopPayload = JSON.stringify(bRequests[1]!.payload);
  for (const expected of ['MULTI_TOOL_LATEST', 'sdk_call_a', 'sdk_call_b', 'RESULT_A', 'RESULT_B', 'KEEP_QUALIFICATION', 'ev_sdk']) {
    assert.match(toolLoop, new RegExp(expected));
    assert.match(toolLoopPayload, new RegExp(expected));
  }
  assert.deepEqual(bRequests.map(request => request.model), ['b', 'b']);
  assert.equal(captures.at(-1)?.model, 'a');
});

test('public Pi SDK preserves uninitialized history before and after switching without writes', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-handoff-sdk-unbound-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const captures: ProviderCapture[] = [];
  const { modelRuntime, modelA, modelB } = await createOfflineRuntime(root, captures);
  const resourceLoader = new DefaultResourceLoader({
    cwd: root, agentDir: join(root, 'agent'), systemPromptOverride: () => 'SDK_UNBOUND_SYSTEM',
    extensionFactories: [pi => installMemory(pi, { home: join(root, 'home') })],
  });
  await resourceLoader.reload();
  const sessionManager = SessionManager.inMemory(root);
  sessionManager.appendMessage({ role: 'user', content: 'UNBOUND_OLD_PRIVATE ' + 'old dump '.repeat(50_000), timestamp: 1 });
  const { session } = await createAgentSession({
    cwd: root, agentDir: join(root, 'agent'), model: modelA, modelRuntime, resourceLoader,
    sessionManager, settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
    tools: [],
  });
  t.after(() => session.dispose());
  await session.prompt('UNBOUND_BASELINE');
  const beforeSwitch = captures.length;
  await session.setModel(modelB);
  await session.prompt('UNBOUND_AFTER_SWITCH');
  const switched = captures.slice(beforeSwitch);
  assert.equal(switched.length, 1);
  assert.equal(switched[0]?.model, 'b');
  const context = JSON.stringify(switched[0]?.context.messages);
  const payload = JSON.stringify(switched[0]?.payload);
  for (const serialized of [context, payload]) {
    assert.match(serialized, /UNBOUND_BASELINE/);
    assert.match(serialized, /UNBOUND_AFTER_SWITCH/);
    assert.equal(serialized.includes('failed safely'), false);
    assert.equal(serialized.includes('UNBOUND_OLD_PRIVATE'), true);
  }
  assert.equal(JSON.stringify(captures[0]?.payload).includes('UNBOUND_OLD_PRIVATE'), true);
  assert.deepEqual(switched[0]?.context.messages.slice(0, captures[0]!.context.messages.length), captures[0]?.context.messages);
  assert.equal(captures[0]?.context.systemPrompt, switched[0]?.context.systemPrompt);
  assert.ok(captures[0]?.sessionId, 'host supplies a stable provider routing/cache session id');
  assert.equal(captures[0]?.sessionId, switched[0]?.sessionId);
  assert.equal(captures[0]?.cacheRetention, switched[0]?.cacheRetention);
  assert.deepEqual(await readdir(join(root, 'home')).catch(() => []), []);
});

test('public Pi SDK never transports original history after model activation cannot bind ownership', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-handoff-sdk-fault-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const captures: ProviderCapture[] = [];
  const { modelRuntime, modelA, modelB } = await createOfflineRuntime(root, captures);
  const resourceLoader = new DefaultResourceLoader({
    cwd: root, agentDir: join(root, 'agent'), systemPromptOverride: () => 'SDK_FAULT_SYSTEM',
    extensionFactories: [pi => {
      const unavailable = async () => {
        if (captures.length) throw new Error('project ownership unavailable');
        return undefined;
      };
      const controller = createHandoffController({
        engine: unavailable,
        budget: { maxTokens: 2_000, maxBytes: 16_000, maxMessages: 6, toolSchemaReserveTokens: 100 },
      });
      installHandoffHooks(pi, controller, unavailable);
    }],
  });
  await resourceLoader.reload();
  const { session } = await createAgentSession({
    cwd: root, agentDir: join(root, 'agent'), model: modelA, modelRuntime, resourceLoader,
    sessionManager: SessionManager.inMemory(root), settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
    tools: [],
  });
  t.after(() => session.dispose());
  await session.prompt('baseline before the switch');
  const beforeSwitch = captures.length;
  assert.equal(beforeSwitch, 1);
  await session.setModel(modelB);
  await session.prompt('PRIVATE_HISTORY_MUST_NOT_REPLAY');
  const afterFailure = captures.slice(beforeSwitch);
  assert.ok(afterFailure.length <= 1);
  for (const request of afterFailure) {
    const transported = JSON.stringify(request.context.messages);
    const payload = JSON.stringify(request.payload);
    assert.equal(transported.includes('PRIVATE_HISTORY_MUST_NOT_REPLAY'), false);
    assert.equal(payload.includes('PRIVATE_HISTORY_MUST_NOT_REPLAY'), false);
    assert.match(transported, /failed safely/);
    assert.match(payload, /failed safely/);
  }
});

test('public Pi SDK compacts automatically with pi-memory installed: compaction belongs to Pi', async t => {
  const root = await mkdtemp(join(tmpdir(), 'pi-memory-auto-compact-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const captures: ProviderCapture[] = [];
  const { modelRuntime, modelA } = await createOfflineRuntime(root, captures);
  const attempts: string[] = [];
  const resourceLoader = new DefaultResourceLoader({
    cwd: root, agentDir: join(root, 'agent'), systemPromptOverride: () => 'STABLE_SYSTEM',
    extensionFactories: [pi => {
      pi.on('session_before_compact', event => { attempts.push(event.reason); });
      installMemory(pi, { home: join(root, 'home') });
    }],
  });
  await resourceLoader.reload();
  const sessionManager = SessionManager.inMemory(root);
  sessionManager.appendMessage({ role: 'user', content: 'OLD_COMPACTION_INPUT ' + 'historical dump '.repeat(30_000), timestamp: 1 });
  const { session } = await createAgentSession({
    cwd: root, agentDir: join(root, 'agent'), model: modelA, modelRuntime, resourceLoader, sessionManager,
    settingsManager: SettingsManager.inMemory({ compaction: { enabled: true, reserveTokens: 99_999, keepRecentTokens: 1 } }),
    tools: [],
  });
  t.after(() => session.dispose());
  await session.prompt('LATEST_SMALL_REQUIREMENT');
  assert.ok(attempts.includes('threshold'), 'exercise actual host auto-compaction, not just a hook mock');
  assert.equal(sessionManager.getBranch().some(entry => entry.type === 'compaction'), true, 'Pi compacted; nothing cancelled it');
  assert.match(JSON.stringify(captures.at(-1)?.payload), /LATEST_SMALL_REQUIREMENT/);
  assert.doesNotMatch(JSON.stringify(captures.at(-1)?.payload), /OLD_COMPACTION_INPUT/);
});
