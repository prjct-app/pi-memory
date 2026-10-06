import type { ExtensionCommandContext } from '@earendil-works/pi-coding-agent';
import { keyHasValidShape, openSecretPrompt, resolveKey, saveKey, type ResolvedKey, type SecretPromptSpec, type SecretStore } from '@prjct.app/pi-tui-kit';
import { openSecretStore } from '../security/credentials.ts';
import { DEFAULT_RERANK_MODEL, readRerankConfig, setRerankEnabled, TypeSafeRerankProvider } from '../retrieval/rerank.ts';

/** A line for the init/status card. Carries the fingerprint, never the key. */
export const evaluatorLine = (resolved: ResolvedKey, enabled: boolean): string => {
  if (resolved.state === 'usable') {
    return `rerank ${enabled ? 'on' : 'off'} · key ${resolved.fingerprint ?? 'set'} (${resolved.source})${resolved.verified ? ' · verified' : ''}`;
  }
  if (resolved.state === 'missing') return 'rerank off · no TypeSafe key · /memory setup';
  return `rerank off · ${resolved.detail}`;
};

/** Validate against the live service before storing, exactly as pi-qa does. */
const verify = async (key: string, model: string): Promise<string | undefined> => {
  try {
    await new TypeSafeRerankProvider({ apiKey: key, enabled: true, model })
      .judge(['connection check'], [{ key: 'c01', text: 'pi-memory evaluator setup', source: 'setup', kind: 'fact' }]);
    return undefined;
  } catch (error) {
    return (error instanceof Error ? error.message : String(error)).slice(0, 300);
  }
};

/**
 * Ask only when there is nothing to use. A key saved by any other prjct
 * extension is this one's key too, so re-prompting for something already
 * stored is the failure this shared credential exists to prevent. A broken or
 * unreachable key is the user's call to make through `/memory setup`, not a
 * hole to quietly refill.
 */
export type EvaluatorDeps = Readonly<{
  force?: boolean;
  /** Injected by tests; production reads the shared OS keyring. */
  store?: SecretStore;
  prompt?: (spec: SecretPromptSpec) => Promise<string | undefined>;
}>;

export const ensureEvaluator = async (
  ctx: Pick<ExtensionCommandContext, 'ui' | 'hasUI' | 'mode'>,
  root: string,
  options: EvaluatorDeps = {},
): Promise<{ resolved: ResolvedKey; enabled: boolean; prompted: boolean }> => {
  const store = options.store ?? await openSecretStore();
  const resolved = await resolveKey(store);
  const interactive = ctx.mode === 'tui' && ctx.hasUI;
  if (!options.force && resolved.state !== 'missing') {
    const enabled = resolved.state === 'usable' && (await readRerankConfig(root)).enabled === true;
    return { resolved, enabled, prompted: false };
  }
  if (!interactive) return { resolved, enabled: false, prompted: false };
  const model = process.env.PI_MEMORY_RERANK_MODEL ?? DEFAULT_RERANK_MODEL;
  const ask = options.prompt ?? ((spec: SecretPromptSpec) => openSecretPrompt(ctx, spec));
  const entered = await ask({
    title: 'TypeSafe evaluator key',
    message: 'Optional. Enables semantic reranking of memory lookups. Stored once in the OS keyring for every project and every prjct extension. Press esc to skip.',
    label: 'key',
    placeholder: 'paste TypeSafe key',
    validate: async value => {
      if (!keyHasValidShape(value)) return 'That value is not a valid TypeSafe API key.';
      const error = await verify(value, model);
      if (error) return error;
      await saveKey(store, value, true);
      return undefined;
    },
  });
  if (!entered) return { resolved, enabled: false, prompted: true };
  return { resolved: await resolveKey(store), enabled: await enable(root, true), prompted: true };
};

const enable = async (root: string, value: boolean): Promise<boolean> => {
  try {
    await setRerankEnabled(root, value);
    return value;
  } catch {
    // A project whose config cannot be written still has working memory; it
    // just does not get the optional stage.
    return false;
  }
};
