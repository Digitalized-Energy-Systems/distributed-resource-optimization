# CLAUDE.md — distributed-resource-optimization (DRO)

Part of the energy scheduling benchmark workspace (`~/development/esb`, see its `CLAUDE.md` for
goal, rules and plan). Branch `restructuring`. Never `git push` unless Jonas asks.
Whenever new code is written here, first invoke the ponytail skill (`ponytail:ponytail`, level full)
— smallest change that fully solves it.

## Scope (IMPORTANT — check every change against it)

- Owns: distributed algorithms behind a generic problem interface (local cost/gradient, bounds,
  coupling constraint), carriers incl. the generic mango Role/carrier, loss handling inside rounds.
- Must not contain: energy-specific code (economic dispatch, generators, storage, PyPSA/pandapower,
  networks) or benchmark logic. Never import mango_energy_environments. The energy-specific code
  of `main` was moved out to energy-scheduling-benchmark (decision A2, done 2026-11-03).
- Use mango 2.4 features (topology, communication sim, scheduling, `@on_message`, tracked
  replies) before writing own code; what is native vs. own: workspace
  `docs/mango-2.4-features.md` → Status 2026-11-09.

Repo split (all three): see `README.md` § Scope of this package.
