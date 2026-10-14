# Contributing

- Stable release branch: `main`. Integration branch: `develop`.
- Create feature branches from `develop` and target `develop` in normal pull requests.
- Deliver changes through a pull request using `.github/pull_request_template.md`.
- Use English for code, documentation, tests, issues, and pull requests.
- The session journals are the source of truth. The index holds search terms and pointers only; never store a second copy of the text, and never rewrite a journal except a `/memory gc confirm` deletion the user asked for.
- Use strict TypeScript and only APIs documented by Pi 1.0.4.
- Use immutable values: `npm run check` fails on any `let` under `src/`.
- Do not import host internals or start an MCP server. No hidden model calls, classifiers, daemons or context injection: the agent calls `memory_context` when it decides to.
- Keep runtime dependencies in `dependencies`; list Pi-provided packages in `peerDependencies`.
- Run `npm run check`, `npm test` and `npm pack --dry-run` before review.
- Publish this package independently and install from npm. Declare runtime package dependencies normally.
- Never push, open or merge a pull request, publish, or deploy without explicit authorization.
