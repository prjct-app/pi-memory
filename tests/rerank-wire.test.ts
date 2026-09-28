import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { DEFAULT_RERANK_TIMEOUT_MS, TypeSafeRerankProvider } from '../src/retrieval/rerank.ts';

/**
 * The batching decision is only real on the wire. This drives the actual SDK
 * against a loopback endpoint and asserts what leaves the process: one request,
 * one copy of each query, one copy of each candidate.
 */
test('a whole shortlist leaves the process as one request with nothing sent twice', async t => {
  const seen: string[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', chunk => chunks.push(chunk as Buffer));
    req.on('end', () => {
      const body = Buffer.concat(chunks).toString('utf8');
      seen.push(body);
      const questions = Object.keys((JSON.parse(body) as { questions: Record<string, unknown> }).questions);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        model: 'jev-1.13.0',
        answers: Object.fromEntries(questions.map(name => [name,
          { type: 'noul', noul: name.endsWith('_instruction') ? 0.01 : 0.9 }])),
        usage: { input_tokens: 10, output_tokens: 0 },
      }));
    });
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { await new Promise<void>(resolve => { server.close(() => resolve()); }); });
  const { port } = server.address() as { port: number };
  const provider = new TypeSafeRerankProvider({ apiKey: 'k'.repeat(20), enabled: true, baseUrl: `http://127.0.0.1:${port}` });
  // Distinct, non-overlapping bodies: "item 1" would otherwise be found inside
  // "item 10" and the count below would be measuring the fixture, not the code.
  const candidates = Array.from({ length: 12 }, (_, index) => ({
    key: `key-${index}`, text: `candidate <${index}> body`, source: 'fixture', kind: 'fact',
  }));
  const judged = await provider.judge(['alpha query', 'beta query'], candidates);
  assert.equal(seen.length, 1, 'twelve candidates cost one request, not twelve');
  const body = seen[0]!;
  assert.equal(body.split('alpha query').length - 1, 1, 'the query is sent once, not once per candidate');
  assert.equal(body.split('The candidate addresses the subject').length - 1, 1, 'the rubric is sent once, not once per question');
  for (const candidate of candidates) assert.equal(body.split(candidate.text).length - 1, 1, `${candidate.key} is sent once`);
  assert.equal(judged.size, candidates.length);
  assert.equal(judged.get('key-7')?.relevant, 0.9);
  assert.equal(judged.get('key-7')?.instruction, 0.01);
});

test('a rejected credential surfaces as a failure the caller can degrade on', async t => {
  const server = createServer((_req, res) => {
    res.writeHead(401, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'invalid api key' } }));
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { await new Promise<void>(resolve => { server.close(() => resolve()); }); });
  const { port } = server.address() as { port: number };
  const provider = new TypeSafeRerankProvider({
    apiKey: 'k'.repeat(20), enabled: true, baseUrl: `http://127.0.0.1:${port}`, timeoutMs: 2_000,
  });
  await assert.rejects(provider.judge(['q'], [{ key: 'c1', text: 'body', source: 'fixture', kind: 'fact' }]));
});

/**
 * The rerank is an improvement over the fusion, never a reason for a memory
 * lookup to stall: the deadline is the default, not something a caller opts
 * into, and a slow answer is dropped in favour of the order already computed.
 */
test('the default deadline bounds a rerank that never answers', async t => {
  const asked = { count: 0 };
  const server = createServer(() => {
    asked.count += 1;
    /* holds the request open and never answers */
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => {
    server.closeAllConnections();
    await new Promise<void>(resolve => { server.close(() => resolve()); });
  });
  const { port } = server.address() as { port: number };
  const provider = new TypeSafeRerankProvider({ apiKey: 'k'.repeat(20), enabled: true, baseUrl: `http://127.0.0.1:${port}` });
  const started = Date.now();
  await assert.rejects(provider.judge(['q'], [{ key: 'c1', text: 'body', source: 'fixture', kind: 'fact' }]));
  const elapsed = Date.now() - started;
  assert.ok(elapsed < 4_000, `an unanswered rerank held the lookup for ${elapsed} ms`);
  assert.equal(asked.count, 1, 'a judgement is asked once: a retry is a second wait for an answer already given up on');
  assert.ok(DEFAULT_RERANK_TIMEOUT_MS <= 2_000, 'the default is a deadline, not a courtesy');
});
