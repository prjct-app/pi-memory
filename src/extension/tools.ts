import { randomUUID } from 'node:crypto';
import { inspectSessionReferences, isSessionReference } from '../handoff/session-references.ts';
import { StringEnum } from '@earendil-works/pi-ai';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { Type, type TSchema } from 'typebox';
import { Compile } from 'typebox/compile';
import { problemsOf, schemaForModel } from '@prjct.app/pi-tui-kit';
import type { EvidenceRef } from '../contracts/evidence.ts';
import { assertEnglishStatement, isEnglish } from '../contracts/language.ts';
import { factIsValidAt, type MemoryKind, type MemoryStanding } from '../contracts/memory.ts';
import type { MemoryEngine } from '../engine.ts';
import type { MemorySearch } from './hooks.ts';
import { admitCapture } from '../retention/capture-gate.ts';
import { semanticConsolidationCandidates } from '../retention/consolidation.ts';
import { sha256 } from '../workspace/project-identity.ts';
import { renderMemoryCall, renderMemoryResult } from './renderers.ts';

export type ExtensionMemoryRuntime = Readonly<{
  /** The project scope, when this checkout already has memory. */
  engine(): Promise<MemoryEngine>;
  /** The project scope for a write; initializes a git repository's memory on first use. */
  writableEngine(): Promise<MemoryEngine>;
  /** Embeds new memories in the background so a record never waits on the encoder. */
  scheduleBackfill?(engine: MemoryEngine): void;
  /** Engines belonging to the active project; production currently returns one. */
  readable(): Promise<readonly MemoryEngine[]>;
  /** Project-local lookup across eligible ranking legs. */
  search: MemorySearch;
  /** The optional semantic reranker, resolved once. Absent when no key is configured. */
  reranker?(): Promise<Parameters<MemorySearch>[1]>;
  /** The English form of a statement, translating when needed; undefined when no model is reachable. */
  englishStatement?(text: string, signal?: AbortSignal): Promise<string | undefined>;
  stagedEvidence(): ReadonlyMap<string, EvidenceRef>;
  currentPrompt(): string;
}>;

const kinds = ['decision', 'fact', 'constraint', 'failure', 'correction', 'procedure', 'preference', 'learning'] as const;
const standings = ['candidate', 'supported', 'needs_review', 'contradicted', 'superseded'] as const;

const contextParameters = Type.Object({
  action: StringEnum(['lookup', 'inspect', 'feedback', 'consolidate'] as const),
  queries: Type.Optional(Type.Array(Type.String({ minLength: 1, maxLength: 1000 }), { maxItems: 4 })),
  ids: Type.Optional(Type.Array(Type.String({ minLength: 1, maxLength: 100 }), { maxItems: 32 })),
  asOf: Type.Optional(Type.String({ maxLength: 64 })),
  namespaces: Type.Optional(Type.Array(Type.String({ maxLength: 64 }), { maxItems: 16 })),
  kinds: Type.Optional(Type.Array(Type.String({ maxLength: 64 }), { maxItems: 16 })),
  signal: Type.Optional(StringEnum(['used', 'helpful', 'wrong', 'stale'] as const)),
  maxBytes: Type.Optional(Type.Integer({ minimum: 512, maximum: 32768 })),
}, { additionalProperties: false });

const recordParameters = Type.Object({
  action: StringEnum(['remember', 'resolve'] as const),
  kind: Type.Optional(StringEnum(kinds)),
  statement: Type.Optional(Type.String({ minLength: 1, maxLength: 8192 })),
  subject: Type.Optional(Type.String({ maxLength: 512 })),
  predicate: Type.Optional(Type.String({ maxLength: 256 })),
  object: Type.Optional(Type.String({ maxLength: 1024 })),
  entities: Type.Optional(Type.Array(Type.Object({ name: Type.String({ minLength: 1, maxLength: 256 }), type: Type.String({ minLength: 1, maxLength: 64 }), aliases: Type.Optional(Type.Array(Type.String({ maxLength: 256 }), { maxItems: 16 })) }, { additionalProperties: false }), { maxItems: 24 })),
  evidenceIds: Type.Optional(Type.Array(Type.String({ minLength: 1, maxLength: 100 }), { maxItems: 32 })),
  userQuote: Type.Optional(Type.String({ minLength: 1, maxLength: 4096 })),
  confidence: Type.Optional(Type.Number({ minimum: 0, maximum: 1 })),
  validAt: Type.Optional(Type.String({ maxLength: 64 })),
  invalidAt: Type.Optional(Type.String({ maxLength: 64 })),
  supersedes: Type.Optional(Type.Array(Type.String({ minLength: 1, maxLength: 100 }), { maxItems: 32 })),
  tags: Type.Optional(Type.Record(Type.String({ maxLength: 64 }), Type.String({ maxLength: 512 }))),
  factId: Type.Optional(Type.String({ minLength: 1, maxLength: 100 })),
  standing: Type.Optional(StringEnum(standings)),
  rationale: Type.Optional(Type.String({ minLength: 1, maxLength: 4096 })),
  replacementId: Type.Optional(Type.String({ minLength: 1, maxLength: 100 })),
}, { additionalProperties: false });

const lazyCompile = <T extends TSchema>(schema: T) => {
  const slot: { compiled?: { Check(value: unknown): boolean; Errors(value: unknown): Iterable<{ instancePath: string; message: string }> } } = {};
  return () => (slot.compiled ??= Compile(schema));
};
const contextValidator = lazyCompile(contextParameters);
const recordValidator = lazyCompile(recordParameters);
/** The full schema, limits included, since the model was shown only its shape. */
const checked = (validator: { Check(value: unknown): boolean; Errors(value: unknown): Iterable<{ instancePath: string; message: string }> }, params: unknown, tool: string): void => {
  if (validator.Check(params)) return;
  throw new Error(`Invalid ${tool} arguments: ${problemsOf(validator, params, tool).join('; ')}`);
};

const normalized = (value: string): string => value.normalize('NFC').replace(/\s+/gu, ' ').trim();
const compactItem = (value: unknown): unknown => {
  if (!value || typeof value !== 'object') return value;
  const item = value as Record<string, unknown>;
  const statement = typeof item.statement === 'string' ? item.statement : undefined;
  const title = typeof item.title === 'string' ? item.title : undefined;
  const includeTitle = title && statement && !normalized(statement).startsWith(normalized(title));
  return {
    ...(typeof item.id === 'string' ? { id: item.id } : {}),
    ...(typeof item.kind === 'string' ? { type: item.kind } : typeof item.namespace === 'string' ? { type: item.namespace } : {}),
    ...(typeof item.standing === 'string' ? { standing: item.standing } : {}),
    ...(includeTitle ? { title } : {}), ...(statement ? { statement } : {}),
    ...(typeof item.observedAt === 'string' ? { observedAt: item.observedAt } : {}),
    ...(typeof item.validAt === 'string' ? { validAt: item.validAt } : {}),
    ...(typeof item.invalidAt === 'string' ? { invalidAt: item.invalidAt } : {}),
    ...(typeof item.score === 'number' ? { score: Number(item.score.toFixed(4)) } : {}),
  };
};
const compactView = (details: unknown): unknown => {
  if (!details || typeof details !== 'object') return details;
  const value = details as Record<string, unknown>;
  if (!Array.isArray(value.items)) return details;
  return { status: value.status, items: value.items.map(compactItem),
    ...(typeof value.omitted === 'number' ? { omitted: value.omitted } : {}),
    ...(Array.isArray(value.gaps) && value.gaps.length ? { gaps: value.gaps } : {}) };
};
const result = (details: unknown) => ({ content: [{ type: 'text' as const, text: JSON.stringify(compactView(details)) }], details });
const required = (value: string | undefined, name: string): string => {
  if (!value?.trim()) throw new Error(`${name} is required for this action.`);
  return value;
};
const words = (text: string): Set<string> => new Set((text.toLocaleLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? [])
  .filter(word => !['the', 'and', 'for', 'that', 'this', 'with', 'from', 'para', 'con', 'una', 'por'].includes(word)));
/**
 * Stored statements are English while the conversation often is not, so a quote
 * and the memory it supports are frequently in different languages and share no
 * whole word. Comparing accent-stripped four-character stems keeps the guard
 * working across that gap: identifiers, numbers and cognates such as
 * publicamos/publish or releases/release still match, while an unrelated
 * sentence from elsewhere in the prompt still does not.
 */
const stems = (text: string): Set<string> => new Set([...words(text)]
  .map(word => word.normalize('NFD').replace(/\p{Diacritic}/gu, ''))
  .map(word => word.slice(0, 4)));
const related = (left: string, right: string): boolean => {
  const rightWords = words(right);
  const exact = [...words(left)].filter(word => rightWords.has(word)).length;
  if (exact >= 2) return true;
  const rightStems = stems(right);
  return [...stems(left)].filter(stem => rightStems.has(stem)).length >= 2;
};
const loose = (text: string): string => text.normalize('NFC').replace(/[\u2018\u2019\u201A\u2032`´]/gu, "'")
  .replace(/[\u201C\u201D\u201E\u2033«»]/gu, '"').replace(/[\u2013\u2014]/gu, '-').replace(/\s+/gu, ' ').trim().toLocaleLowerCase();
const verifiedQuote = (quote: string | undefined, prompt: string, statement: string): string | undefined => {
  const value = quote?.trim();
  if (!value) return undefined;
  if (value.length < 12 || words(value).size < 3) throw new Error('userQuote must contain at least 12 characters and three content words.');
  // Models re-type quotes: collapsed spaces, straight quotes for curly ones, a
  // different case. The words must still be the person's, in their order.
  if (!prompt.includes(value) && !loose(prompt).includes(loose(value))) throw new Error('userQuote must occur in the current user prompt.');
  if (!related(value, statement)) throw new Error('userQuote must be related to the memory statement.');
  return value;
};
/**
 * Translate rather than refuse, and refuse only when translation is impossible:
 * storing the memory in another language is the one outcome that is never
 * acceptable, because it is re-read into a prompt on every later turn.
 */
const englishOrThrow = async (
  runtime: ExtensionMemoryRuntime, field: string, text: string, signal?: AbortSignal,
): Promise<string> => {
  if (isEnglish(text)) return text;
  const translated = await runtime.englishStatement?.(text, signal);
  if (translated) return translated;
  assertEnglishStatement(field, text);
  return text;
};

const RESERVED_TAGS = new Set(['sourceDocumentKey', 'sourceRevision', 'sourceAdapter', 'topicId', 'semanticKey']);
const safeTags = (tags: Readonly<Record<string, string>> | undefined): Record<string, string> =>
  Object.fromEntries(Object.entries(tags ?? {}).filter(([key]) => !RESERVED_TAGS.has(key)));

export const installMemoryTools = (pi: ExtensionAPI, runtime: ExtensionMemoryRuntime): void => {
  pi.registerTool({
    name: 'memory_context', label: 'Memory context',
    description: 'Search or inspect bounded project memory; consolidate exact candidates or record positive retrieval feedback.',
    // The model reads the shape; the limits are checked here, on every call.
    parameters: schemaForModel(contextParameters),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      checked(contextValidator(), params, 'memory_context');
      if (params.action === 'inspect' && params.ids?.some(isSessionReference)) {
        if (!params.ids.every(isSessionReference)) throw new Error('Cannot mix session reference IDs and project memory IDs.');
        const details = inspectSessionReferences(ctx.sessionManager, params.ids, params.maxBytes ?? 4096);
        return { content: [{ type: 'text' as const, text: JSON.stringify(details) }], details };
      }
      // Nothing recorded yet is an empty memory, not a failure the agent must work around.
      const engine = await runtime.engine().catch(error => {
        if (error instanceof Error && /not initialized/iu.test(error.message)) return undefined;
        throw error;
      });
      if (!engine) {
        return result({ status: 'abstained', items: [], gaps: ['No memory has been recorded for this project yet.'] });
      }
      if (params.action === 'lookup') {
        // Only this path — a tool call the agent made on purpose — may reach a
        // reranker. The automatic per-turn hook stays lexical and offline.
        const rerank = await runtime.reranker?.() ?? {};
        return result(await runtime.search({ queries: params.queries ?? [],
          ...(params.asOf ? { asOf: params.asOf } : {}), namespaces: params.namespaces ?? ['memory', 'memory.topic'],
          ...(params.kinds ? { kinds: params.kinds } : {}), maxBytes: params.maxBytes ?? 1500,
          dense: true, signal }, rerank));
      }
      if (params.action === 'inspect') {
        const memory = await runtime.engine();
        const asOf = params.asOf ? Date.parse(params.asOf) : Date.now();
        if (!Number.isFinite(asOf)) throw new Error('asOf must be ISO-8601.');
        const maxBytes = params.maxBytes ?? 4096;
        const items: unknown[] = [];
        const gaps: string[] = [];
        for (const id of params.ids ?? []) {
          const fact = memory.projection.getFact(id);
          if (fact) {
            if (fact.scopeId !== memory.scopeId) { gaps.push(`${id} is not owned by this project.`); continue; }
            if (!factIsValidAt(fact, asOf)) { gaps.push(`${id} is not valid at asOf.`); continue; }
            items.push(fact);
            continue;
          }
          const topic = memory.projection.documentByKey({ namespace: 'memory.topic', externalId: id });
          if (!topic || topic.scopeId !== memory.scopeId) { gaps.push(`No requested memory ids exist for ${id}.`); continue; }
          const facts = (topic.metadata.facts ?? '').split(',').filter(Boolean).flatMap(factId => {
            const found = memory.projection.getFact(factId);
            return found && found.scopeId === memory.scopeId && factIsValidAt(found, asOf) ? [found] : [];
          });
          items.push({ id: topic.externalId, title: topic.title, statement: topic.text, facts, sources: topic.metadata.sources });
        }
        const packed = { status: items.length ? 'ok' : 'abstained', items, gaps };
        const encoded = JSON.stringify(packed);
        if (Buffer.byteLength(encoded) <= maxBytes) return result(packed);
        const clipped = { status: 'partial' as const, items: items.slice(0, 1), gaps: [...gaps, 'inspect exceeded maxBytes; returned the first topic or fact.'] };
        return result(clipped);
      }
      if (params.action === 'consolidate') {
        const candidates = await semanticConsolidationCandidates(engine.projection.activeFacts(engine.scopeId), {
          embed: texts => engine.embeddings.embed(texts, { inputType: 'passage' }),
        });
        return result({ status: candidates.length ? 'ok' : 'abstained', candidates,
          gaps: candidates.length ? [] : ['No mechanical consolidation candidates were found.'] });
      }
      if (!params.signal || !(params.ids?.length)) throw new Error('feedback requires ids and signal.');
      const memory = await runtime.engine();
      const query = (params.queries ?? []).join('\n');
      const applied = { count: 0 };
      const missing: string[] = [];
      for (const id of params.ids) {
        if (!memory.projection.getFact(id)) { missing.push(id); continue; }
        await memory.feedback(id, params.signal, query);
        applied.count += 1;
      }
      return result({ status: applied.count ? 'ok' : 'abstained', recorded: applied.count, signal: params.signal,
        gaps: missing.length ? [`No open scope holds ${missing.join(', ')}.`] : [] });
    },
    renderShell: "self",
    renderCall(args, theme, context) { return renderMemoryCall(theme, 'context', args, context?.isPartial !== false); },
    renderResult(output, options, theme, context) { return renderMemoryResult(theme, 'context', context?.args, output.details, options.expanded, Boolean(context?.isError)); },
  });

  pi.registerTool({
    name: 'memory_record', label: 'Record memory',
    description: 'Record selective durable knowledge or resolve memory using current-session evidence handles or an exact user quote. Write statement and rationale in English whatever language the conversation is in; put the original wording in userQuote, which is kept verbatim as evidence.',
    parameters: schemaForModel(recordParameters),
    async execute(_toolCallId, params, signal) {
      checked(recordValidator(), params, 'memory_record');
      const engine = await runtime.writableEngine();
      if (params.action === 'resolve') {
        const factId = required(params.factId, 'factId');
        const fact = engine.projection.getFact(factId);
        if (!fact) throw new Error(`Unknown memory ${factId}.`);
        const standing = params.standing as MemoryStanding | undefined;
        if (!standing || !['supported', 'needs_review', 'contradicted', 'superseded'].includes(standing)) throw new Error('resolve requires a supported, review, or terminal standing.');
        const rationale = required(params.rationale, 'rationale');
        const evidence = (params.evidenceIds ?? []).map(id => {
          const found = runtime.stagedEvidence().get(id);
          if (!found) throw new Error(`Evidence ${id} was not observed by this session.`);
          return found;
        });
        const englishRationale = await englishOrThrow(runtime, 'rationale', rationale, signal);
        const quote = verifiedQuote(params.userQuote, runtime.currentPrompt(), `${fact.statement} ${englishRationale}`);
        if (!quote && !evidence.some(item => related(item.excerpt, `${fact.statement} ${englishRationale}`))) {
          throw new Error('Resolving memory requires related current-session evidence or an exact user quote.');
        }
        await engine.resolveFact(factId, standing, englishRationale, params.replacementId);
        return result({ status: 'ok', factId, standing });
      }
      const written = required(params.statement, 'statement');
      // What is written here is injected into every later system prompt, so it
      // is stored in English regardless of the language of the conversation
      // that produced it. A statement in another language is translated rather
      // than refused; the user's own wording survives verbatim in userQuote.
      const statement = await englishOrThrow(runtime, 'statement', written, signal);
      const admission = admitCapture({
        statement, kind: params.kind ?? 'learning',
        existing: engine.projection.activeFacts(engine.scopeId, 200).map(item => ({ statement: item.statement, kind: item.kind })),
      });
      if (!admission.accept) return result({ status: 'abstained', gaps: [`Capture refused: ${admission.reason}.`] });
      const staged = runtime.stagedEvidence();
      const evidence = (params.evidenceIds ?? []).map(id => {
        const found = staged.get(id);
        if (!found) throw new Error(`Evidence ${id} was not observed by this session.`);
        return found;
      });
      const quote = verifiedQuote(params.userQuote, runtime.currentPrompt(), statement);
      const evidenceRelated = evidence.every(item => related(item.excerpt, statement));
      const allEvidence: EvidenceRef[] = [...evidence, ...(quote ? [{ id: `ev_${randomUUID()}`, origin: 'user_statement' as const,
        provenance: 'declared' as const, contentHash: sha256(quote), excerpt: quote, observedAt: new Date().toISOString() }] : [])];
      const entities = (params.entities ?? []).map(entity => ({ id: `ent_${sha256(`${engine.scopeId}\u0000${entity.type}\u0000${entity.name.toLocaleLowerCase()}`).slice(0, 24)}`,
        scopeId: engine.scopeId, name: entity.name, type: entity.type, aliases: entity.aliases ?? [] }));
      const recorded = await engine.recordFact({ kind: (params.kind ?? 'learning') as MemoryKind, statement,
        ...(params.subject ? { subject: params.subject } : {}), ...(params.predicate ? { predicate: params.predicate } : {}),
        ...(params.object ? { object: params.object } : {}), entities, evidence: allEvidence, episodeIds: [],
        confidence: params.confidence ?? (allEvidence.length ? 0.85 : 0.5), ...(params.validAt ? { validAt: params.validAt } : {}),
        ...(params.invalidAt ? { invalidAt: params.invalidAt } : {}), ...(params.supersedes ? { supersedes: params.supersedes } : {}),
        ...(!evidenceRelated && !quote ? { standing: 'needs_review' as const } : {}), tags: safeTags(params.tags) }, signal,
        runtime.scheduleBackfill ? { dense: false } : {});
      runtime.scheduleBackfill?.(engine);
      return result({ status: 'ok', id: recorded.fact.id, standing: recorded.fact.standing,
        dense: recorded.dense || Boolean(runtime.scheduleBackfill),
        gaps: recorded.dense || runtime.scheduleBackfill ? [] : ['Dense indexing deferred; memory and lexical index committed.'] });
    },
    renderShell: "self",
    renderCall(args, theme, context) { return renderMemoryCall(theme, 'record', args, context?.isPartial !== false); },
    renderResult(output, options, theme, context) { return renderMemoryResult(theme, 'record', context?.args, output.details, options.expanded, Boolean(context?.isError)); },
  });
};
