import { isDeepStrictEqual } from 'node:util';
import type { SourceDocument } from '../contracts/documents.ts';
import { assertSourceDocument, documentKey } from '../contracts/documents.ts';
import { Projection, type VectorHit } from '../storage/projection.ts';
import type { ProjectionPort } from '../storage/ports.ts';
import { chunkDocument, type ChunkOptions } from './chunker.ts';
import type { EmbeddingProvider } from './providers.ts';

export type VectorQuery = Readonly<{
  text: string;
  limit?: number;
  signal?: AbortSignal;
}>;

export type VectorSearchHit = VectorHit & Readonly<{ text: string; namespace: string; source: string; externalId: string; uri?: string }>;

export class EmbeddingUnavailableError extends Error {
  override readonly cause: unknown;
  constructor(cause: unknown) {
    super(`Embedding provider unavailable: ${cause instanceof Error ? cause.message : String(cause)}`);
    this.name = 'EmbeddingUnavailableError';
    this.cause = cause;
  }
}

export interface VectorIndex {
  readonly provider: EmbeddingProvider;
  upsert(document: SourceDocument, signal?: AbortSignal): Promise<{ chunks: number; embedded: number }>;
  upsertAll(documents: readonly SourceDocument[], signal?: AbortSignal): Promise<{ chunks: number; embedded: number }>;
  remove(namespace: string, externalId: string): void;
  search(query: VectorQuery): Promise<VectorSearchHit[]>;
  backfill(signal?: AbortSignal): Promise<number>;
  dispose(): Promise<void>;
}

/** Query vectors kept per index: a prompt's queries repeat across budget retries and turns. */
export const QUERY_VECTOR_CACHE = 128;

type QueryWaiter = Readonly<{ text: string; resolve: (vector: number[] | undefined) => void; reject: (error: unknown) => void }>;

export class SqliteVectorIndex implements VectorIndex {
  readonly provider: EmbeddingProvider;
  private readonly projection: ProjectionPort;
  private readonly chunkOptions: ChunkOptions;
  private readonly queryVectors = new Map<string, Promise<number[] | undefined>>();
  private readonly queryBatch: { waiting: QueryWaiter[]; scheduled: boolean } = { waiting: [], scheduled: false };

  constructor(projection: ProjectionPort, provider: EmbeddingProvider, chunkOptions: ChunkOptions = {}) {
    this.projection = projection;
    this.provider = provider;
    this.chunkOptions = chunkOptions;
  }

  /**
   * A prompt searches up to four queries at once, and each used to cost its own
   * encoder pass, one after another (~58 ms each on the local model, several
   * per prompt with budget retries). Queries asked in the same tick now share
   * one pass, and a query already embedded is not embedded again.
   */
  private queryVector(text: string): Promise<number[] | undefined> {
    const cached = this.queryVectors.get(text);
    if (cached) {
      this.queryVectors.delete(text);
      this.queryVectors.set(text, cached);
      return cached;
    }
    const pending = new Promise<number[] | undefined>((resolve, reject) => {
      this.queryBatch.waiting.push({ text, resolve, reject });
      if (this.queryBatch.scheduled) return;
      this.queryBatch.scheduled = true;
      queueMicrotask(() => this.embedQueries());
    });
    // A failure is not remembered: the encoder may only be loading.
    pending.catch(() => { if (this.queryVectors.get(text) === pending) this.queryVectors.delete(text); });
    this.queryVectors.set(text, pending);
    const oldest = [...this.queryVectors.keys()].slice(0, Math.max(0, this.queryVectors.size - QUERY_VECTOR_CACHE));
    for (const key of oldest) this.queryVectors.delete(key);
    return pending;
  }

  private embedQueries(): void {
    const waiting = this.queryBatch.waiting.splice(0);
    this.queryBatch.scheduled = false;
    this.provider.embed(waiting.map(waiter => waiter.text), { inputType: 'query' }).then(
      vectors => waiting.forEach((waiter, index) => waiter.resolve(vectors[index])),
      error => waiting.forEach(waiter => waiter.reject(error)),
    );
  }

  async upsert(document: SourceDocument, signal?: AbortSignal): Promise<{ chunks: number; embedded: number }> {
    assertSourceDocument(document);
    signal?.throwIfAborted();
    const existing = this.projection.documentByKey(document);
    if (!existing || !isDeepStrictEqual(existing, document)) this.projection.upsertDocument(document);
    const chunks = chunkDocument(document, this.chunkOptions);
    // Preserve lexical rows and vectors when deterministic chunks and their
    // indexed metadata are unchanged. This avoids write amplification on
    // repeated source scans while still backfilling a previously failed model.
    if (!this.projection.chunksMatch(document, chunks, document.title)) {
      this.projection.replaceChunks(documentKey(document), chunks, document.title);
    }
    const missing = chunks.filter(chunk => !this.projection.hasVector(chunk.id, this.provider.model));
    if (!missing.length) return { chunks: chunks.length, embedded: 0 };
    const vectors = await this.provider.embed(missing.map(chunk => chunk.text), { signal, inputType: 'passage' })
      .catch(error => { throw new EmbeddingUnavailableError(error); });
    if (vectors.length !== missing.length) throw new EmbeddingUnavailableError('Provider returned an incomplete batch.');
    this.projection.storeVectors(missing.map((chunk, index) => ({ chunkId: chunk.id, vector: vectors[index]! })), this.provider.model);
    return { chunks: chunks.length, embedded: vectors.length };
  }

  // Same work as upsert() but with one projection transaction for the whole run
  // and provider batches of 64, so a bulk ingest pays neither a transaction nor
  // a model round-trip per document.
  async upsertAll(documents: readonly SourceDocument[], signal?: AbortSignal): Promise<{ chunks: number; embedded: number }> {
    if (!documents.length) return { chunks: 0, embedded: 0 };
    for (const document of documents) assertSourceDocument(document);
    signal?.throwIfAborted();
    this.projection.transaction(() => {
      for (const document of documents) {
        const existing = this.projection.documentByKey(document);
        if (!existing || !isDeepStrictEqual(existing, document)) this.projection.upsertDocument(document);
      }
    });
    const chunkLists = documents.map(document => chunkDocument(document, this.chunkOptions));
    const chunks = chunkLists.flat();
    const changed = chunkLists.filter((list, index) => {
      const document = documents[index]!;
      return !this.projection.chunksMatch(document, list, document.title);
    }).flat();
    const titles = new Map(documents.flatMap(document => document.title ? [[documentKey(document), document.title] as const] : []));
    if (changed.length) this.projection.replaceChunksBatch(changed, titles);
    const missing = chunks.filter(chunk => !this.projection.hasVector(chunk.id, this.provider.model));
    const batches = Array.from({ length: Math.ceil(missing.length / 64) }, (_, index) => missing.slice(index * 64, (index + 1) * 64));
    const progress = { embedded: 0 };
    for (const batch of batches) {
      signal?.throwIfAborted();
      const vectors = await this.provider.embed(batch.map(chunk => chunk.text), { signal, inputType: 'passage' })
        .catch(error => { throw new EmbeddingUnavailableError(error); });
      if (vectors.length !== batch.length) throw new EmbeddingUnavailableError('Provider returned an incomplete batch.');
      this.projection.storeVectors(batch.map((chunk, index) => ({ chunkId: chunk.id, vector: vectors[index]! })), this.provider.model);
      progress.embedded += batch.length;
    }
    return { chunks: chunks.length, embedded: progress.embedded };
  }

  remove(namespace: string, externalId: string): void {
    this.projection.deleteDocument(namespace, externalId, new Date().toISOString());
  }

  async search(query: VectorQuery): Promise<VectorSearchHit[]> {
    const limit = Math.max(1, Math.min(1000, query.limit ?? 20));
    query.signal?.throwIfAborted();
    const vector = await this.queryVector(query.text);
    query.signal?.throwIfAborted();
    if (!vector) return [];
    const hits = this.projection.vectorSearch(this.provider.model, vector.length, vector, limit);
    const chunks = new Map(this.projection.retrievalChunks(hits.map(hit => hit.chunkId)).map(chunk => [chunk.id, chunk]));
    return hits.flatMap(hit => {
      const chunk = chunks.get(hit.chunkId);
      return chunk ? [{ ...hit, text: chunk.text, namespace: chunk.namespace, source: chunk.document.source,
        externalId: chunk.document.externalId, ...(chunk.document.uri ? { uri: chunk.document.uri } : {}) }] : [];
    });
  }

  async backfill(signal?: AbortSignal): Promise<number> {
    const chunks = this.projection.unembeddedChunks(this.provider.model);
    const batches = Array.from({ length: Math.ceil(chunks.length / 64) }, (_, index) => chunks.slice(index * 64, (index + 1) * 64));
    for (const batch of batches) {
      signal?.throwIfAborted();
      const vectors = await this.provider.embed(batch.map(chunk => chunk.text), { signal, inputType: 'passage' })
        .catch(error => { throw new EmbeddingUnavailableError(error); });
      if (vectors.length !== batch.length) throw new EmbeddingUnavailableError('Provider returned an incomplete batch.');
      this.projection.storeVectors(batch.map((chunk, index) => ({ chunkId: chunk.id, vector: vectors[index]! })), this.provider.model);
    }
    return chunks.length;
  }

  async dispose(): Promise<void> {
    await this.provider.dispose?.();
  }
}

export const createVectorIndex = (projection: ProjectionPort, provider: EmbeddingProvider, options?: ChunkOptions): VectorIndex =>
  new SqliteVectorIndex(projection, provider, options);

class OwnedSqliteVectorIndex extends SqliteVectorIndex {
  private readonly ownedProjection: Projection;
  constructor(path: string, provider: EmbeddingProvider, options?: ChunkOptions) {
    const projection = new Projection(path);
    super(projection, provider, options);
    this.ownedProjection = projection;
  }
  override async dispose(): Promise<void> {
    try { await super.dispose(); } finally { this.ownedProjection.close(); }
  }
}

export const openVectorIndex = (options: Readonly<{ path: string; provider: EmbeddingProvider; chunking?: ChunkOptions }>): VectorIndex =>
  new OwnedSqliteVectorIndex(options.path, options.provider, options.chunking);
