import { protectOutboundData } from '@prjct.app/pi-secrets/privacy';
import type { Api, AssistantMessage, Model } from '@earendil-works/pi-ai';
import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { CurationBlockError } from '../curation/types.ts';
import type { AnswerProvider, AnswerRequest, AnswerResult } from './comparison.ts';

const SYSTEM = 'Answer only from the provided evidence and prior facts. Quote supporting excerpts. Do not invent claims.';

const textOf = (message: AssistantMessage): string =>
  message.content.flatMap(block => 'type' in block && block.type === 'text' && 'text' in block ? [String(block.text)] : []).join('\n');

export const createSdkAnswerProvider = async (options: Readonly<{
  provider: string; model: string; maxOutputChars?: number;
}>): Promise<AnswerProvider> => {
  const runtime = await ModelRuntime.create({ refreshOnCreate: false, allowModelNetwork: false });
  const model = runtime.getModel(options.provider, options.model) as Model<Api> | undefined;
  if (!model) throw new CurationBlockError('missing_model', `Configured answer model ${options.provider}/${options.model} is not available.`);
  const auth = await runtime.checkAuth(options.provider);
  if (!auth) throw new CurationBlockError('missing_auth', `No credentials for answer provider ${options.provider}.`);
  const maxOutputChars = options.maxOutputChars ?? 4_000;
  return {
    provider: options.provider,
    model: options.model,
    answer: async (request: AnswerRequest, signal?: AbortSignal): Promise<AnswerResult> => {
      const maxTokens = Math.min(2048, Math.max(128, Math.ceil(maxOutputChars / 4)));
      const message = await runtime.completeSimple(model, {
        systemPrompt: SYSTEM,
        messages: [{ role: 'user', content: JSON.stringify(await protectOutboundData({
          condition: request.condition, query: request.query, evidence: request.evidence, priorFacts: request.priorFacts,
        })), timestamp: Date.now() }],
      }, { maxTokens, ...(signal === undefined ? {} : { signal }) });
      if (message.stopReason === 'error' || message.stopReason === 'aborted') {
        return { text: '', inputTokens: message.usage?.input ?? 0, outputTokens: message.usage?.output ?? 0, failed: true };
      }
      const text = textOf(message).slice(0, maxOutputChars);
      return {
        text,
        inputTokens: message.usage?.input ?? Math.ceil(request.evidence.length / 4),
        outputTokens: message.usage?.output ?? Math.ceil(text.length / 4),
      };
    },
  };
};
