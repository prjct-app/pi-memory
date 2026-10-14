import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { DatabaseSync, type SQLInputValue } from 'node:sqlite';
import type { EntryKind, HistoryEntry, SessionHeader } from './entry.ts';

/**
 * A derived index over Pi's session journals. The journals stay the only
 * source of truth: the index keeps search terms (a contentless FTS5 table)
 * and a byte pointer into the journal line, never a second copy of the text.
 * Deleting the database loses nothing; it is rebuilt from the journals.
 */

export const SCHEMA_VERSION = '1';

export type FileRow = Readonly<{ id: number; path: string; sessionId: string | null; cwd: string | null; startedAt: string | null; bytes: number }>;

export type IndexedRow = HistoryEntry & Readonly<{ offset: number; length: number; chars: number }>;

export type EntryHit = Readonly<{
  entryId: string; parentId: string | null; timestamp: string; kind: EntryKind; role: string; name: string | null;
  chars: number; offset: number; length: number; tokensBefore: number | null;
  path: string; sessionId: string | null; cwd: string | null; score: number;
}>;

export type Filters = Readonly<{
  sessionId?: string;
  /** Sessions whose cwd is this directory or below it. */
  under?: string;
  since?: string;
  until?: string;
  kinds?: readonly EntryKind[];
  role?: string;
}>;

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, session_id TEXT, cwd TEXT, started_at TEXT,
    bytes INTEGER NOT NULL DEFAULT 0);
  CREATE INDEX IF NOT EXISTS files_session ON files(session_id);
  CREATE INDEX IF NOT EXISTS files_cwd ON files(cwd);
  CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id), entry_id TEXT NOT NULL, parent_id TEXT,
    ts TEXT NOT NULL, kind TEXT NOT NULL, role TEXT NOT NULL, name TEXT, chars INTEGER NOT NULL,
    offset INTEGER NOT NULL, length INTEGER NOT NULL, tokens_before INTEGER, UNIQUE(file_id, entry_id));
  CREATE INDEX IF NOT EXISTS entries_entry ON entries(entry_id);
  CREATE INDEX IF NOT EXISTS entries_file_ts ON entries(file_id, ts);
  CREATE INDEX IF NOT EXISTS entries_kind_ts ON entries(kind, ts);
  CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    text, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2');
`;

const SELECT_HIT = `SELECT e.entry_id AS entryId, e.parent_id AS parentId, e.ts AS timestamp, e.kind, e.role, e.name, e.chars,
  e.offset, e.length, e.tokens_before AS tokensBefore, f.path, f.session_id AS sessionId, f.cwd`;

const likeEscape = (value: string): string => value.replace(/[\\%_]/gu, match => `\\${match}`);

const where = (filters: Filters): { sql: string; args: SQLInputValue[] } => {
  const parts: string[] = [];
  const args: SQLInputValue[] = [];
  if (filters.sessionId) { parts.push('f.session_id = ?'); args.push(filters.sessionId); }
  if (filters.under) { parts.push("(f.cwd = ? OR f.cwd LIKE ? ESCAPE '\\')"); args.push(filters.under, `${likeEscape(filters.under.replace(/\/+$/u, ''))}/%`); }
  if (filters.since) { parts.push('e.ts >= ?'); args.push(filters.since); }
  if (filters.until) { parts.push('e.ts <= ?'); args.push(filters.until); }
  if (filters.kinds?.length) { parts.push(`e.kind IN (${filters.kinds.map(() => '?').join(',')})`); args.push(...filters.kinds); }
  if (filters.role) { parts.push('e.role = ?'); args.push(filters.role); }
  return { sql: parts.length ? ` AND ${parts.join(' AND ')}` : '', args };
};

/** FTS5 query from free text: every term quoted, so punctuation and operators are literal. */
export const matchExpression = (query: string, any = false): string | undefined => {
  const terms = query.match(/[\p{L}\p{N}]+/gu) ?? [];
  return terms.length ? [...new Set(terms)].map(term => `"${term}"`).join(any ? ' OR ' : ' ') : undefined;
};

export const openHistoryStore = (path: string) => {
  mkdirSync(dirname(path), { recursive: true });
  const db = new DatabaseSync(path, { timeout: 10_000 });
  db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;');
  db.exec(SCHEMA);
  db.prepare('INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)').run('schema', SCHEMA_VERSION);

  const statements = {
    file: db.prepare('SELECT id, path, session_id AS sessionId, cwd, started_at AS startedAt, bytes FROM files WHERE path = ?'),
    addFile: db.prepare('INSERT OR IGNORE INTO files(path) VALUES (?)'),
    header: db.prepare('UPDATE files SET session_id = ?, cwd = ?, started_at = ? WHERE id = ?'),
    advance: db.prepare('UPDATE files SET bytes = max(bytes, ?) WHERE id = ?'),
    entry: db.prepare(`INSERT OR IGNORE INTO entries(file_id, entry_id, parent_id, ts, kind, role, name, chars, offset, length, tokens_before)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`),
    fts: db.prepare('INSERT INTO entries_fts(rowid, text) VALUES (?, ?)'),
    fileEntries: db.prepare('SELECT id FROM entries WHERE file_id = ?'),
    deleteFts: db.prepare('DELETE FROM entries_fts WHERE rowid = ?'),
    deleteEntries: db.prepare('DELETE FROM entries WHERE file_id = ?'),
    resetFile: db.prepare('UPDATE files SET bytes = 0 WHERE id = ?'),
    meta: db.prepare('SELECT value FROM meta WHERE key = ?'),
    setMeta: db.prepare('INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value'),
  };

  const transaction = <T>(work: () => T): T => {
    db.exec('BEGIN IMMEDIATE');
    try {
      const result = work();
      db.exec('COMMIT');
      return result;
    } catch (error) {
      db.exec('ROLLBACK');
      throw error;
    }
  };

  const file = (path: string): FileRow => {
    statements.addFile.run(path);
    return statements.file.get(path) as unknown as FileRow;
  };

  return {
    db,
    file,
    /** Appends the rows parsed from a journal span and moves the file cursor, atomically. */
    append(fileId: number, header: SessionHeader | undefined, rows: readonly IndexedRow[], bytes: number): number {
      return transaction(() => {
        if (header) statements.header.run(header.sessionId, header.cwd, header.startedAt, fileId);
        const added = rows.reduce((count, row) => {
          const inserted = statements.entry.run(fileId, row.entryId, row.parentId, row.timestamp, row.kind, row.role, row.name,
            row.chars, row.offset, row.length, row.tokensBefore);
          if (!inserted.changes) return count;
          statements.fts.run(inserted.lastInsertRowid, row.text);
          return count + 1;
        }, 0);
        statements.advance.run(bytes, fileId);
        return added;
      });
    },
    /** A journal that shrank was rewritten; its rows are dropped and indexed again from the start. */
    reset(fileId: number): void {
      transaction(() => {
        for (const row of statements.fileEntries.all(fileId) as { id: number }[]) statements.deleteFts.run(row.id);
        statements.deleteEntries.run(fileId);
        statements.resetFile.run(fileId);
      });
    },
    search(query: string, filters: Filters, limit: number): { hits: EntryHit[]; total: number; match: 'all' | 'any' } {
      const run = (expression: string) => {
        const scope = where(filters);
        const total = Number((db.prepare(`SELECT count(*) AS n FROM entries_fts JOIN entries e ON e.id = entries_fts.rowid
          JOIN files f ON f.id = e.file_id WHERE entries_fts MATCH ?${scope.sql}`).get(expression, ...scope.args) as { n: number }).n);
        const hits = db.prepare(`${SELECT_HIT}, bm25(entries_fts) AS score FROM entries_fts JOIN entries e ON e.id = entries_fts.rowid
          JOIN files f ON f.id = e.file_id WHERE entries_fts MATCH ?${scope.sql} ORDER BY score LIMIT ?`)
          .all(expression, ...scope.args, limit) as unknown as EntryHit[];
        return { hits, total };
      };
      const all = matchExpression(query);
      if (!all) return { hits: [], total: 0, match: 'all' };
      const exact = run(all);
      const anyTerm = matchExpression(query, true);
      if (exact.total || !anyTerm || anyTerm === all) return { ...exact, match: 'all' };
      return { ...run(anyTerm), match: 'any' };
    },
    /** Entries without a query: newest first, same filters. */
    list(filters: Filters, limit: number, offset: number): { hits: EntryHit[]; total: number } {
      const scope = where(filters);
      const total = Number((db.prepare(`SELECT count(*) AS n FROM entries e JOIN files f ON f.id = e.file_id WHERE 1 = 1${scope.sql}`)
        .get(...scope.args) as { n: number }).n);
      const hits = db.prepare(`${SELECT_HIT}, 0 AS score FROM entries e JOIN files f ON f.id = e.file_id WHERE 1 = 1${scope.sql}
        ORDER BY e.ts DESC LIMIT ? OFFSET ?`).all(...scope.args, limit, offset) as unknown as EntryHit[];
      return { hits, total };
    },
    find(entryId: string, sessionId?: string): EntryHit[] {
      return db.prepare(`${SELECT_HIT}, 0 AS score FROM entries e JOIN files f ON f.id = e.file_id
        WHERE e.entry_id = ?${sessionId ? ' AND f.session_id = ?' : ''} ORDER BY e.ts DESC LIMIT 20`)
        .all(...(sessionId ? [entryId, sessionId] : [entryId])) as unknown as EntryHit[];
    },
    /** Per-journal counts the garbage collector judges by. */
    journalStats(): { path: string; sessionId: string | null; cwd: string | null; startedAt: string | null;
      users: number; messages: number; replies: number; summaries: number }[] {
      return db.prepare(`SELECT f.path, f.session_id AS sessionId, f.cwd, f.started_at AS startedAt,
        coalesce(sum(e.kind = 'user'), 0) AS users, coalesce(sum(e.kind = 'custom'), 0) AS messages,
        coalesce(sum(e.kind = 'assistant'), 0) AS replies,
        coalesce(sum(e.kind IN ('compaction', 'branch_summary')), 0) AS summaries
        FROM files f LEFT JOIN entries e ON e.file_id = f.id GROUP BY f.id`).all() as never;
    },
    /** Drops a deleted journal from the index: its rows, its search terms, its cursor. */
    forget(path: string): void {
      const row = statements.file.get(path) as unknown as FileRow | undefined;
      if (!row) return;
      transaction(() => {
        for (const entry of statements.fileEntries.all(row.id) as { id: number }[]) statements.deleteFts.run(entry.id);
        statements.deleteEntries.run(row.id);
        db.prepare('DELETE FROM files WHERE id = ?').run(row.id);
      });
    },
    cwds(): string[] {
      return (db.prepare('SELECT DISTINCT cwd FROM files WHERE cwd IS NOT NULL').all() as { cwd: string }[]).map(row => row.cwd);
    },
    status() {
      const count = (sql: string) => Number((db.prepare(sql).get() as { n: number }).n);
      return {
        files: count('SELECT count(*) AS n FROM files'),
        sessions: count('SELECT count(DISTINCT session_id) AS n FROM files'),
        entries: count('SELECT count(*) AS n FROM entries'),
        summaries: count("SELECT count(*) AS n FROM entries WHERE kind IN ('compaction', 'branch_summary')"),
        indexedBytes: count('SELECT coalesce(sum(bytes), 0) AS n FROM files'),
      };
    },
    meta(key: string): string | undefined {
      return (statements.meta.get(key) as { value: string } | undefined)?.value;
    },
    setMeta(key: string, value: string): void {
      statements.setMeta.run(key, value);
    },
    /** One process at a time scans other sessions' journals; the lease expires if it dies. */
    lease(owner: string, ttlMs: number, now = Date.now()): boolean {
      return transaction(() => {
        const current = (statements.meta.get('backfill-lease') as { value: string } | undefined)?.value;
        const [holder, until] = current ? current.split('@') : [];
        if (current && holder !== owner && Number(until) > now) return false;
        statements.setMeta.run('backfill-lease', `${owner}@${now + ttlMs}`);
        return true;
      });
    },
    release(owner: string): void {
      transaction(() => {
        const current = (statements.meta.get('backfill-lease') as { value: string } | undefined)?.value;
        if (current?.startsWith(`${owner}@`)) db.prepare("DELETE FROM meta WHERE key = 'backfill-lease'").run();
      });
    },
    close(): void {
      try { db.close(); } catch { /* already closed */ }
    },
  };
};

export type HistoryStore = ReturnType<typeof openHistoryStore>;
