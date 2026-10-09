import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getAgentDir } from '@earendil-works/pi-coding-agent';

/**
 * Scopes are discovered from the marker files prjct publishes, not from
 * configuration each extension repeats. pi-team writes
 * `$PRJCT_HOME/teams/<id>/settings.json` for exactly this reason; reading it is
 * what binds a team's id to its name, and those two must travel together.
 */
export type TeamScope = Readonly<{ id: string; name: string }>;

const TEAM_ID = /^t_[A-Za-z0-9_-]+$/;
const TEAM_NAME = /^[a-z][a-z0-9-]{0,47}$/;

/**
 * Where pi-team keeps the mailbox and its journal. This is the Pi agent
 * directory, NOT `$PRJCT_HOME`: the durable team store lives under prjct, but
 * the mailbox is agent-local runtime state.
 */
export const teamMailboxRoot = (override?: string): string =>
  override ?? join(getAgentDir(), 'teams');

const readTeamMarker = async (home: string, id: string): Promise<TeamScope | undefined> => {
  const raw = await readFile(join(home, 'teams', id, 'settings.json'), 'utf8').catch(() => undefined);
  if (!raw) return undefined;
  const parsed = JSON.parse(raw) as { scope?: unknown; id?: unknown; name?: unknown };
  if (parsed.scope !== 'team' || parsed.id !== id) return undefined;
  // A team with no name has no mailbox to read; the artifact store alone is
  // still worth indexing, so it is reported rather than dropped.
  return typeof parsed.name === 'string' && TEAM_NAME.test(parsed.name) ? { id, name: parsed.name } : { id, name: '' };
};

export const discoverTeams = async (home: string): Promise<readonly TeamScope[]> => {
  const entries = (await readdir(join(home, 'teams'), { withFileTypes: true }).catch(() => []))
    .filter(entry => entry.isDirectory() && TEAM_ID.test(entry.name))
    .map(entry => entry.name)
    .sort();
  const markers = await Promise.all(entries.map(id => readTeamMarker(home, id)));
  return markers.flatMap(marker => marker ?? []);
};
