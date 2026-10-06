import type { MemoryKind, MemoryStanding } from '../contracts/memory.ts';
import { redactSecrets } from '../security/redact.ts';
import {
  ANALYSIS_ACTIONS, EPISTEMICS, type AnalysisProposal, type Epistemic, type EvidenceBundle, type ProposedFact,
} from './types.ts';

const KINDS: readonly MemoryKind[] = ['decision', 'fact', 'constraint', 'failure', 'correction', 'procedure', 'preference', 'learning'];
const STANDINGS: readonly MemoryStanding[] = ['candidate', 'supported', 'needs_review', 'contradicted', 'superseded'];

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && Array.isArray(value) === false;

const text = (value: unknown, max: number): string | undefined =>
  typeof value === 'string' && value.trim().length > 0 && Buffer.byteLength(value, 'utf8') <= max ? value.trim() : undefined;

const epistemicOf = (value: unknown): Epistemic | undefined =>
  typeof value === 'string' && (EPISTEMICS as readonly string[]).includes(value) ? value as Epistemic : undefined;

const standingFor = (fact: ProposedFact): MemoryStanding => {
  if (fact.epistemic === 'proposal' || fact.epistemic === 'hypothesis') return fact.standing === 'supported' ? 'needs_review' : fact.standing;
  if (fact.epistemic === 'conflict') return 'needs_review';
  return fact.standing;
};

const parseFact = (value: unknown, bundle: EvidenceBundle): ProposedFact => {
  if (!isObject(value)) throw new Error('Invalid analysis fact.');
  const action = typeof value.action === 'string' && (ANALYSIS_ACTIONS as readonly string[]).includes(value.action)
    ? value.action as ProposedFact['action'] : undefined;
  const kind = typeof value.kind === 'string' && (KINDS as readonly string[]).includes(value.kind) ? value.kind as MemoryKind : undefined;
  const epistemic = epistemicOf(value.epistemic);
  const statement = text(value.statement, 8192);
  const excerpt = text(value.excerpt, 500);
  const semanticKey = text(value.semanticKey, 128);
  const standing = typeof value.standing === 'string' && (STANDINGS as readonly string[]).includes(value.standing)
    ? value.standing as MemoryStanding : undefined;
  const confidence = typeof value.confidence === 'number' && Number.isFinite(value.confidence) ? value.confidence : undefined;
  const refs = Array.isArray(value.sourceRefs) && value.sourceRefs.length
    ? value.sourceRefs
    : [{ adapter: bundle.identity.adapter, namespace: bundle.identity.namespace, externalId: bundle.identity.externalId,
      revision: bundle.identity.revision, observedAt: bundle.identity.observedAt }];
  if (!action || !kind || !epistemic || !statement || !excerpt || !semanticKey || !standing || confidence === undefined) {
    throw new Error('Analysis fact is missing required fields.');
  }
  if (confidence < 0 || confidence > 1) throw new Error('Analysis confidence must be between 0 and 1.');
  if (!bundle.text.includes(excerpt)) throw new Error('Excerpt is not grounded in the source.');
  if (action === 'keep') {
    return { action, kind, epistemic, statement: redactSecrets(statement), confidence, standing, semanticKey,
      sourceRefs: [], excerpt: redactSecrets(excerpt), ...(typeof value.id === 'string' ? { id: value.id } : {}) };
  }
  const sourceRefs = refs.map(ref => {
    if (!isObject(ref)) throw new Error('Invalid source reference.');
    const adapter = text(ref.adapter, 256);
    const namespace = text(ref.namespace, 64);
    const externalId = text(ref.externalId, 512);
    const revision = text(ref.revision, 64);
    if (!adapter || !namespace || !externalId || !revision) throw new Error('Source reference is incomplete.');
    if (adapter !== bundle.identity.adapter || namespace !== bundle.identity.namespace || externalId !== bundle.identity.externalId) {
      throw new Error('Source reference does not match the analyzed identity.');
    }
    if (revision !== bundle.identity.revision) throw new Error('Source reference revision is stale.');
    return { adapter, namespace, externalId, revision, observedAt: bundle.identity.observedAt,
      ...(typeof ref.locator === 'string' && ref.locator.length <= 4096 ? { locator: ref.locator } : {}) };
  });
  const supersedes = Array.isArray(value.supersedes)
    ? value.supersedes.filter((id): id is string => typeof id === 'string' && id.startsWith('mem_')).slice(0, 32)
    : undefined;
  return {
    action, kind, epistemic, statement: redactSecrets(statement), confidence, standing, semanticKey,
    sourceRefs, excerpt: redactSecrets(excerpt),
    ...(typeof value.id === 'string' && /^mem_[a-z0-9_-]{8,64}$/.test(value.id) ? { id: value.id } : {}),
    ...(text(value.subject, 512) ? { subject: redactSecrets(text(value.subject, 512)!) } : {}),
    ...(text(value.predicate, 256) ? { predicate: redactSecrets(text(value.predicate, 256)!) } : {}),
    ...(text(value.object, 1024) ? { object: redactSecrets(text(value.object, 1024)!) } : {}),
    ...(typeof value.validAt === 'string' && Number.isFinite(Date.parse(value.validAt)) ? { validAt: value.validAt } : {}),
    ...(typeof value.invalidAt === 'string' && Number.isFinite(Date.parse(value.invalidAt)) ? { invalidAt: value.invalidAt } : {}),
    ...(supersedes?.length ? { supersedes } : {}),
  };
};

const assertNotRawCopy = (proposal: AnalysisProposal, sourceText: string): void => {
  const body = sourceText.trim();
  if (body.length < 80) return;
  const copied = [proposal.topic?.summary, ...proposal.facts.map(fact => fact.statement)]
    .some(textValue => textValue && textValue.trim() === body);
  if (copied) throw new Error('Analysis output copied the raw source body.');
};

export const parseProposal = (raw: unknown, bundle: EvidenceBundle): AnalysisProposal => {
  if (!isObject(raw)) throw new Error('Analysis output must be an object.');
  const noChange = raw.noChange === true;
  const conflicts = Array.isArray(raw.conflicts)
    ? raw.conflicts.flatMap(item => typeof item === 'string' && item.trim() ? [item.trim().slice(0, 1024)] : [])
    : [];
  const topic = isObject(raw.topic) && text(raw.topic.id, 64) && text(raw.topic.title, 256) && text(raw.topic.summary, 4000)
    ? { id: text(raw.topic.id, 64)!, title: redactSecrets(text(raw.topic.title, 256)!), summary: redactSecrets(text(raw.topic.summary, 4000)!) }
    : undefined;
  const facts = (Array.isArray(raw.facts) ? raw.facts : []).map(item => parseFact(item, bundle)).map(fact => ({ ...fact, standing: standingFor(fact) }));
  if (!noChange && !facts.some(fact => fact.action !== 'keep') && !topic && !conflicts.length) {
    throw new Error('Analysis produced no substance.');
  }
  if (topic && !noChange && !facts.length) throw new Error('A topic summary requires supporting facts.');
  const proposal = { noChange, facts, conflicts, ...(topic ? { topic } : {}) };
  assertNotRawCopy(proposal, bundle.text);
  return proposal;
};

export const assertPublishable = (proposal: AnalysisProposal, bundle: EvidenceBundle, allowedFactIds: ReadonlySet<string>): AnalysisProposal => {
  if (proposal.topic && !proposal.noChange && !proposal.facts.length) throw new Error('A topic summary requires supporting facts.');
  for (const fact of proposal.facts) {
    if (!bundle.text.includes(fact.excerpt)) throw new Error('Excerpt is not grounded in the source.');
    for (const id of fact.supersedes ?? []) {
      if (!allowedFactIds.has(id) && !bundle.currentFacts.some(item => item.id === id)) {
        throw new Error('Supersede target is not authorized for this source.');
      }
    }
    if (fact.action === 'discard' && fact.id && !allowedFactIds.has(fact.id) && !bundle.currentFacts.some(item => item.id === fact.id)) {
      throw new Error('Discard target is not authorized for this source.');
    }
    if (fact.action === 'revise' && fact.id && !allowedFactIds.has(fact.id) && !bundle.currentFacts.some(item => item.id === fact.id)) {
      throw new Error('Revise target is not authorized for this source.');
    }
  }
  assertNotRawCopy(proposal, bundle.text);
  return proposal;
};

export const extractJsonObject = (output: string): unknown => {
  const trimmed = output.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = (fenced ? fenced[1]! : trimmed).trim();
  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start < 0 || end <= start) throw new Error('Analysis output is not JSON.');
  return JSON.parse(candidate.slice(start, end + 1));
};
