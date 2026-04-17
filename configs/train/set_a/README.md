# Experiment Set A Training Configs

Set A uses shared optimization defaults across all compared model families.

Rule:

- hold optimizer, schedule, token budget, and evaluation cadence constant for a fixed task/size/context combination
- vary only the model family or the targeted ablation setting

See `docs/experiment_set_a.md` for the full training-loop contract.
