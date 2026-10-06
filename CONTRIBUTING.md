# Contributing

- Stable release branch: `main`. Integration branch: `develop`.
- Create feature branches from `develop` and target `develop` in normal pull requests.
- Deliver changes through a pull request using `.github/pull_request_template.md`.
- Use English for code, documentation, tests, issues, and pull requests.
- Preserve stored statements, source language, constraints and verbatim evidence. The active model decides how to interpret them; do not insert automatic translation.
- Use strict TypeScript and only APIs documented by Pi 1.0.4.
- Use immutable values: `npm run check` fails on any `let` under `src/`.
- Do not import host internals or start an MCP server. Interactive memory uses no hidden model calls, classifiers or implicit daemon startup.
- An explicitly configured standalone memory daemon may perform autonomous extraction, synthesis, consolidation, and freshness review while Pi is closed. It must enforce durable job state, scoped access, revision checks, deadlines, and model/cost budgets.
- Pi's active agent owns interactive query expansion and final answers. Background analysis persists curated knowledge and provenance references, not raw source bodies or model transcripts.
- Daemon implementation is not authorization to install or activate a persistent service; activation requires explicit user authorization.
- Host observations may only receive native provenance from extension event handlers.
- Keep runtime dependencies in `dependencies`; list Pi-provided packages in `peerDependencies`.
- Run `npm run check`, `npm test`, `npm run test:integration`, and `npm pack --dry-run` before review.
- Publish this package independently and install from npm. Declare runtime package dependencies normally.
- Never push, open or merge a pull request, publish, or deploy without explicit authorization.
