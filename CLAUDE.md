# CLAUDE.md — distributed-resource-optimization (DRO)

Part of the energy scheduling benchmark workspace (`~/development/esb`, see its `CLAUDE.md` for
goal, rules and plan). Branch `restructuring`. Never `git push` unless Jonas asks.

## Scope (IMPORTANT — check every change against it)

- Owns: distributed algorithms behind a generic problem interface (local cost/gradient, bounds,
  coupling constraint), carriers incl. the generic mango Role/carrier, loss handling inside rounds.
- Must not contain: energy-specific code (economic dispatch, generators, storage, PyPSA/pandapower,
  networks) or benchmark logic. Never import mango_energy_environments. Energy-specific code on
  `main` is to be moved out to energy-scheduling-benchmark (decision A2).
- Use mango 2.4 features (topology, communication sim, scheduling, decorators) before writing own code.

Repo split (all three): see `README.md` § Scope of this package.
