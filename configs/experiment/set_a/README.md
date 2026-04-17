# Experiment Set A Configs

This directory holds the paper-facing synthetic benchmark suite definition for Experiment Set A.

Design choice:

- keep JSON config files so they map directly into the existing dataclass and CLI path
- resolve one run from three config bundles:
  - model config
  - train config
  - task config

Planned files:

- `suite_pilot.json`: minimal debugging matrix
- `suite_full.json`: full Set A matrix
- `tasks/*.json`: per-task deterministic generation defaults

The detailed spec for all fields lives in `docs/experiment_set_a.md`.
