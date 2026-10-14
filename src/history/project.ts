import { homedir } from 'node:os';
import { dirname, relative, resolve, sep } from 'node:path';

/**
 * The project a session belongs to, for the default search scope: the
 * outermost directory above the cwd that has sessions of its own, so
 * ~/Apps/prjct covers prjct-be, prjct-fe and prjct-landing. A direct child of
 * home (~/Apps, ~/Desktop) only counts when it is the cwd itself: it holds
 * many unrelated projects. Outside home the cwd is the project.
 */
export const projectRoot = (cwd: string, sessionCwds: Iterable<string>, home = homedir()): string => {
  const target = resolve(cwd);
  const base = resolve(home);
  const inside = relative(base, target);
  if (!inside || inside.startsWith('..') || inside.startsWith(sep)) return target;
  const known = new Set([...sessionCwds].map(path => resolve(path)));
  const ancestors = (path: string): string[] => path === base || dirname(path) === path ? [] : [path, ...ancestors(dirname(path))];
  const candidates = ancestors(target).filter(path => path === target || (known.has(path) && dirname(path) !== base));
  return candidates.at(-1) ?? target;
};
