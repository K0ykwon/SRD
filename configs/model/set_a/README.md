# Experiment Set A Model Configs

These configs define the aligned model families for Set A.

Rule:

- the shared local backbone is defined once per size tier
- family-specific configs should override only the intended long-range routing behavior

Current size targets:

- `small`: about 50M parameters
- `base`: about 150M parameters

Use the field definitions in `docs/experiment_set_a.md`.
