# Experiment Script Conventions

Set A should use thin shell wrappers only for reproducible launch recipes.

Recommended wrappers:

- `run_experiment_set_a_pilot.sh`
- `run_experiment_set_a_full.sh`
- `eval_experiment_set_a.sh`
- `aggregate_experiment_set_a.sh`
- `plot_experiment_set_a.sh`
- `run_reproduction_required.sh`
- `run_reproduction_required_longctx.sh`
- `run_reproduction_required_small_8k.sh`
- `launch_reproduction_required_detached.sh`
- `status_reproduction_required_detached.sh`
- `snapshot_reproduction_progress.sh`
- `profile_detail_vs_transformer_decode.sh`
  - supports `DECODE_STEPS` to force longer incremental decode profiling even on short-answer synthetic tasks
- `run_reproduction_audit.sh`

Rules:

- keep real logic in Python modules under `src/srd`
- let shell scripts only resolve env vars, config paths, and output directories
- always pass explicit config paths and seeds
- write outputs under `outputs/set_a/`
