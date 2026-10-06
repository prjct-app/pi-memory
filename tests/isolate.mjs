import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// SDK integration tests must never discover the developer's credentials or settings.
const root = mkdtempSync(join(tmpdir(), 'pi-isolated-tests-'));
process.env.PRJCT_HOME = join(root, 'prjct');
process.env.PI_CODING_AGENT_DIR = join(root, 'agent');
process.env.PI_MEMORY_HOME = join(root, 'memory');
for (const path of [process.env.PRJCT_HOME, process.env.PI_CODING_AGENT_DIR, process.env.PI_MEMORY_HOME]) mkdirSync(path, { recursive: true });
registerHooks({ resolve(specifier, context, nextResolve) {
  if (specifier.includes('@napi-rs/keyring')) throw new Error('Tests cannot access the OS keychain; inject an in-memory KeyStore.');
  return nextResolve(specifier, context);
} });
process.on('exit', () => rmSync(root, { recursive: true, force: true }));
