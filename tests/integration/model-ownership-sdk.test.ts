import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createAssistantMessageEventStream, InMemoryCredentialStore, type AssistantMessage } from '@earendil-works/pi-ai';
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from '@earendil-works/pi-coding-agent';
import { installMemory } from '../../src/index.ts';

test('public SDK can remember an earlier short user correction and recall it on the next turn', async t => {
  const root = await mkdtemp(join(tmpdir(), 'memory-short-rule-sdk-'));
  await mkdir(join(root, '.git'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
    modelsStorePath: join(root, 'models.json'), refreshOnCreate: false });
  const seen: string[] = [];
  runtime.registerProvider('offline-owner', { baseUrl: 'http://127.0.0.1.invalid', apiKey: 'offline', api: 'offline-owner',
    models: [{ id: 'fixture', name: 'Fixture', reasoning: true, input: ['text'], contextWindow: 100000, maxTokens: 1000,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
    streamSimple(model, context) {
      const stream = createAssistantMessageEventStream();
      seen.push(JSON.stringify(context.messages));
      const recorded = context.messages.some(message => message.role === 'toolResult' && message.toolName === 'memory_record');
      const message: AssistantMessage = { role: 'assistant', api: model.api, provider: model.provider, model: model.id, timestamp: Date.now(),
        stopReason: recorded ? 'stop' : 'toolUse',
        usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        content: recorded ? [{ type: 'text', text: 'Recorded.' }] : [{ type: 'toolCall', id: 'record-rule', name: 'memory_record',
          arguments: { action: 'remember', kind: 'constraint', statement: 'No auxiliary classifier.', userQuote: 'No JEV' } }],
      };
      queueMicrotask(() => { stream.push({ type: 'done', reason: message.stopReason as 'stop' | 'toolUse', message }); stream.end(); });
      return stream;
    },
  });
  const settings = SettingsManager.inMemory({ packages: [], extensions: ['-builtin:mcp'], cacheWarming: 'off' });
  const loader = new DefaultResourceLoader({ cwd: root, agentDir: join(root, 'agent'), settingsManager: settings,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    extensionFactories: [pi => installMemory(pi, { home: join(root, 'memory') })] });
  await loader.reload();
  const manager = SessionManager.inMemory(root);
  manager.appendMessage({ role: 'user', content: 'No JEV', timestamp: 1 });
  const { session } = await createAgentSession({ cwd: root, agentDir: join(root, 'agent'), settingsManager: settings, resourceLoader: loader,
    modelRuntime: runtime, model: runtime.getModel('offline-owner', 'fixture'), thinkingLevel: 'high', sessionManager: manager });
  t.after(async () => { await session.abort(); await session.extensionRunner.emit({ type: 'session_shutdown', reason: 'quit' }); session.dispose(); });
  await session.bindExtensions({ mode: 'rpc' });
  await session.prompt('Remember my earlier constraint.');
  const output = manager.getEntries().flatMap(entry => entry.type === 'message' && entry.message.role === 'toolResult' ? [entry.message] : []);
  assert.equal(output.length, 1);
  assert.equal(output[0]?.isError, false, JSON.stringify(output));
  assert.match(JSON.stringify(output[0]?.content), /supported/);
  await session.prompt('Continue the task.');
  assert.match(seen.at(-1) ?? '', /project_memory/);
  assert.match(seen.at(-1) ?? '', /No auxiliary classifier/);
});
