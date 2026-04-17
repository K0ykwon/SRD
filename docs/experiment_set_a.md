# Experiment Set A

Experiment Set A is the controlled synthetic structural-validation suite for SRD.

Its purpose is to test whether scheduled refresh-mediated long-range communication can replace continuous global attention on tasks where the distant dependency is compressible, while still exposing the cases where exact token-level recovery requires a stronger detail path or full attention.

## Scope

Experiment Set A compares five model families:

- `transformer_local`
- `transformer_full`
- `srd_refresh`
- `srd_refresh_sufficiency`
- `srd_refresh_sufficiency_detail`

All five must share the same token embedding, local block shape, MLP ratio, head count, and optimizer defaults wherever possible. Differences should come only from the intended long-range routing path.

## Repository Shape

Default repository additions for Set A:

```text
configs/
  experiment/
    set_a/
      suite_full.json
      suite_pilot.json
      tasks/
        easy_kv.json
        binding_kv.json
        delayed_kv.json
        needle_retrieval.json
        delayed_copy.json
        mixed_dependency.json
        multi_hop_segment_reasoning.json
  model/
    set_a/
      shared_backbone_small.json
      shared_backbone_base.json
      transformer_local.json
      transformer_full.json
      srd_refresh.json
      srd_refresh_sufficiency.json
      srd_refresh_sufficiency_detail.json
  train/
    set_a/
      pilot.json
      main.json
docs/
  experiment_set_a.md
outputs/
  README.md
  set_a/
    runs/
    aggregates/
    plots/
src/
  srd/
    data/
      synthetic_benchmarks.py
      set_a_tasks.py
      set_a_registry.py
    eval/
      benchmark_runner.py
      result_artifacts.py
      set_a_aggregate.py
      set_a_plot.py
    modeling/
      factory.py
      interfaces.py
      baseline_models.py
      block_refresh_model.py
      block_refresh_detail_model.py
    training/
      train.py
      losses.py
      loop.py
scripts/
  README.md
  run_experiment_set_a_pilot.sh
  run_experiment_set_a_full.sh
  eval_experiment_set_a.sh
  aggregate_experiment_set_a.sh
  plot_experiment_set_a.sh
tests/
  test_synthetic_benchmarks.py
  test_set_a_tasks.py
  test_set_a_aggregation.py
```

Default choice:

- keep runtime config loading JSON-backed because the current repo already uses JSON plus dataclasses
- extend `SRDConfig` and `SyntheticBenchmarkConfig` instead of introducing a second config system
- add Set A-specific JSON bundles that map cleanly into those dataclasses

## Config Surface

### Shared experiment config

One run should resolve to the following fields:

```json
{
  "experiment_set": "set_a",
  "run_name": "set_a_small_ctx2048_binding_kv_srd_refresh_sufficiency_seed11",
  "task": "binding_kv",
  "task_category": "compressible_exact_key_value_binding",
  "model_family": "srd_refresh_sufficiency",
  "model_size": "small",
  "context_length": 2048,
  "seed": 11,
  "train_config": "configs/train/set_a/main.json",
  "model_config": "configs/model/set_a/srd_refresh_sufficiency.json",
  "task_config": "configs/experiment/set_a/tasks/binding_kv.json"
}
```

### Model config fields

All model families must expose:

- `model_family`
- `model_type`
- `size_name`
- `vocab_size`
- `d_model`
- `num_layers`
- `num_heads`
- `mlp_ratio`
- `dropout_p`
- `max_seq_len`
- `local_window`
- `block_size`
- `refresh_interval_blocks`
- `refresh_slots`
- `bank_size`
- `sufficiency_loss_weight`
- `detail_enabled`
- `detail_topk`
- `detail_slots`
- `detail_scoring`
- `detail_anchor_first`
- `detail_anchor_last`
- `detail_saliency_slots`

Default Set A values:

- `detail_scoring = "dot"`
- `refresh_interval_blocks = 1`
- `refresh_slots = 2`
- `bank_size = 64` for `2k`, `128` for `4k`, `256` for `8k`
- `sufficiency_loss_weight = 0.1` for the default sufficiency run
- `detail_topk = 4`
- `detail_slots = 8`

### Training config fields

Each training preset should include:

- `optimizer = "adamw"`
- `lr`
- `weight_decay`
- `betas`
- `grad_clip_norm`
- `warmup_steps`
- `lr_schedule = "cosine"`
- `batch_size_tokens`
- `micro_batch_size`
- `gradient_accumulation_steps`
- `max_steps`
- `eval_every`
- `save_every`
- `log_every`
- `precision`
- `compile_model`
- `seed`
- `deterministic_algorithms`

Default Set A values:

- pilot: shorter schedule for implementation debugging
- main: long enough for clean task separation, identical across model families at fixed task/size/context

### Task config fields

Each task config should include:

- `task_name`
- `split_sizes`
- `vocab_size`
- `total_length`
- `segment_count`
- `segment_length`
- `distractor_density`
- `answer_span_length`
- `hop_count`
- `copy_span_length`
- `num_keys`
- `num_distractor_keys`
- `exact_recovery_required`
- `difficulty_levels`

## Unified APIs

### Model API

All five model families should satisfy one forward contract:

```python
class LongContextModel(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> dict:
        ...
```

Required forward outputs:

- `logits`: `[batch, seq, vocab]`
- `loss`: optional total loss if labels are provided
- `nll_loss`
- `sufficiency_loss`
- `auxiliary_losses`: dict
- `debug`: dict

Required `debug` keys:

- `segment_count`
- `bank_read_slots`
- `token_bank_access_count`
- `refresh_bank_access_count`
- `refresh_norm_mean`
- `refresh_norm_std`
- `detail_usage_rate`
- `detail_candidate_hit_rate`

Recommended helper methods:

```python
def compute_loss(self, batch: dict) -> dict: ...
def predict_answers(self, batch: dict) -> dict: ...
def reset_cache(self) -> None: ...
```

### Internal abstractions

Keep these as explicit subcomponents rather than hidden flags:

- local token backbone
- refresh state builder
- long-memory bank
- sufficiency head
- detail selector
- detail retriever

Default interface sketch:

```python
class RefreshStateBuilder(nn.Module):
    def forward(self, segment_hidden: torch.Tensor) -> torch.Tensor: ...


class LongMemoryInterface(nn.Module):
    def empty(self, batch_size: int, device: torch.device) -> torch.Tensor: ...
    def read(self, bank_states: torch.Tensor) -> torch.Tensor: ...
    def write(self, bank_states: torch.Tensor, entries: torch.Tensor) -> torch.Tensor: ...


class SufficiencyHead(nn.Module):
    def forward(self, refresh_context: torch.Tensor, next_segment_ids: torch.Tensor | None = None) -> dict: ...


class DetailRetriever(nn.Module):
    def select_slots(self, block_hidden: torch.Tensor) -> dict: ...
    def retrieve(self, query_hidden: torch.Tensor, past_slots: dict) -> dict: ...
```

## Task Suite

### Split policy

All Set A tasks should be deterministic generators:

- `train`: sample index range `[0, train_size)`
- `val`: sample index range `[train_size, train_size + val_size)`
- `test`: sample index range after `val`
- randomness comes only from `(base_seed, split_offset, sample_index)`

### Shared item format

Each generated item should return:

```python
{
  "input_ids": List[int],
  "labels": List[int],
  "answer_positions": List[int],
  "answer_tokens": List[int],
  "metadata": {
    "task_name": str,
    "split": str,
    "requires_exact_recovery": bool,
    "support_segment_indices": List[int],
    "query_segment_index": int
  }
}
```

### 1. `easy_kv`

Category:

- compressible easy binding-free retrieval

Required structure:

- early segments contain a single key/value binding
- later query asks for one key’s value
- answer is usually a single token or short value span

Difficulty knobs:

- `total_length`
- `segment_count`
- `gap_segments`
- `distractor_density`
- `value_length`
- `symbol_pool_size`

Target format:

- single-token or short-span answer

Metrics:

- `accuracy`
- `value_span_exact_match`

### 2. `binding_kv`

Category:

- compressible exact key-value binding

Required structure:

- early segments contain multiple candidate key/value bindings
- value shapes are intentionally similar
- later query asks for one key’s value
- answer requires exact value-span matching

Difficulty knobs:

- `total_length`
- `segment_count`
- `gap_segments`
- `num_keys`
- `num_distractor_keys`
- `distractor_density`
- `value_length`
- `symbol_pool_size`

Target format:

- short value span, with exact-match scoring as the primary metric

Metrics:

- `accuracy`
- `value_span_exact_match`

### 3. `needle_retrieval`

Category:

- compressible sparse-salience

Required structure:

- one salient token/span occurs in a distractor-heavy sequence
- later query asks for the planted needle

Difficulty knobs:

- `total_length`
- `segment_count`
- `distractor_density`
- `needle_span_length`
- `needle_segment_choices`

Target format:

- single token or short span

Metrics:

- `accuracy`
- `retrieval_hit_rate`

`retrieval_hit_rate` default:

- whether the model’s top predicted answer token/span overlaps the planted target
- for SRD+detail also log whether the gold source token lived inside retrieved detail candidates

### 4. `delayed_copy`

Category:

- non-compressible

Required structure:

- a source span appears early
- later prompt requires exact reproduction

Difficulty knobs:

- `total_length`
- `segment_count`
- `copy_span_length`
- `gap_segments`
- `alphabet_size`

Target format:

- multi-token exact span

Metrics:

- `exact_match`
- `token_accuracy`
- `normalized_edit_distance`

Default edit metric:

- Levenshtein distance divided by target span length

### 4. `mixed_dependency`

Category:

- mixed summary plus exact detail

Required structure:

- one answer component comes from compressible summary information
- one answer component requires exact recovery of a token/span/number

Default pattern:

- segments encode a rule and a category label that determine most of the answer
- one exact numeric or string field must be copied exactly from a distant support span

Difficulty knobs:

- `segment_count`
- `summary_rule_count`
- `detail_span_length`
- `distractor_density`
- `gap_segments`

Metrics:

- `summary_part_accuracy`
- `detail_part_accuracy`
- `joint_accuracy`

### 5. `multi_hop_segment_reasoning`

Category:

- compositional long-range reasoning

Required structure:

- segment A defines a rule or mapping
- segment B provides a value
- segment C asks a query that requires composing them
- optional additional hops insert extra intermediate mappings

Difficulty knobs:

- `segment_count`
- `hop_count`
- `rule_complexity`
- `num_candidate_rules`
- `distractor_density`

Metrics:

- `accuracy`
- `per_hop_failure_breakdown`

`per_hop_failure_breakdown` default:

- log which support segment type was missed or contradicted in the failed example

## Training Loop

Default Set A loop:

- optimizer: `AdamW`
- scheduler: linear warmup + cosine decay
- mixed precision: `bf16` on supported GPUs, else `fp16`, else `fp32`
- gradient accumulation: by token budget, not fixed examples
- checkpointing: save `latest`, `best_task_metric`, `best_val_loss`
- logging: JSONL plus stdout

Per-step logs:

- `step`
- `lr`
- `total_loss`
- `nll_loss`
- `answer_loss`
- `sufficiency_loss`
- `grad_norm`
- `tokens_per_second`
- `examples_per_second`
- `peak_memory_bytes`

Per-eval logs:

- task metrics
- average train/eval step time
- decode tokens/sec
- bank read slots
- refresh norm stats
- detail usage stats

Reproducibility controls:

- set Python, NumPy, and Torch seeds from the run seed
- fix split offsets
- store resolved config with each run
- store git commit and dirty flag with each artifact

Best checkpoint selection:

- primary: task-specific main metric on validation
- tie-breaker 1: lower answer loss
- tie-breaker 2: lower wall-clock per eval token

## Evaluation Protocol

For every `(task, model_family, model_size, context_length, seed)`:

1. train with the same task preset and optimization schedule
2. evaluate on held-out test split
3. record quality plus efficiency metrics
4. aggregate across three seeds

Main reported artifacts:

- `Table 1`: task quality by model family at `small`
- `Table 2`: quality plus efficiency at `base`
- `Figure 1`: context scaling curves by task
- `Figure 2`: efficiency-quality scatter or Pareto view

Required efficiency metrics:

- `peak_gpu_memory_bytes`
- `tokens_per_second`
- `decode_tokens_per_second`
- `wall_clock_step_seconds`

Required grouping keys:

- `experiment_set`
- `task_name`
- `task_category`
- `model_family`
- `model_size`
- `context_length`
- `seed`

## Minimal Set A Ablations

Keep ablations cheap and targeted:

- `sufficiency_loss_weight`: `0.0` vs `0.1`
- `detail_topk`: `0` vs `4`
- `refresh_interval_blocks`: `1` vs `2`

Scope rule:

- run ablations first on `small`
- run on `binding_kv`, `delayed_copy`, and `mixed_dependency`
- extend to `base` only after the pilot is stable

## Exact Run Matrix

### Main matrix

- model families: `5`
- sizes: `2` (`small`, `base`)
- contexts: `3` (`2048`, `4096`, `8192`)
- tasks: `5`
- seeds: `3` (`11`, `17`, `23`)

Total main runs:

- `5 * 2 * 3 * 5 * 3 = 450`

### Ablation matrix

Lightweight ablations on `small` only:

- tasks: `3` (`binding_kv`, `delayed_copy`, `mixed_dependency`)
- contexts: `2` (`2048`, `8192`)
- seeds: `3`

Runs by ablation family:

- sufficiency `0 vs default`: `2 SRD settings * 3 tasks * 2 contexts * 3 seeds = 36`
- detail `0 vs default`: `2 SRD-detail settings * 3 tasks * 2 contexts * 3 seeds = 36`
- refresh interval `1 vs 2`: `2 SRD-suff settings * 3 tasks * 2 contexts * 3 seeds = 36`

Total ablation runs:

- `108`

Grand total after full Set A:

- `558`

### Reduced pilot matrix

Run this first:

- sizes: `small`
- contexts: `2048`, `8192`
- tasks: `easy_kv`, `binding_kv`, `delayed_copy`, `mixed_dependency`
- model families: all five
- seeds: `11`

Pilot run count:

- `1 * 2 * 3 * 5 * 1 = 30`

Second pilot stage:

- same matrix plus seeds `17`, `23`

Stage-two count:

- `90`

## Artifact Schemas

### Per-run JSON

Required top-level keys:

```json
{
  "experiment_set": "set_a",
  "run_name": "string",
  "task_name": "binding_kv",
  "task_category": "compressible",
  "model_family": "srd_refresh_sufficiency",
  "model_size": "small",
  "context_length": 2048,
  "seed": 11,
  "status": "completed",
  "git_commit": "string",
  "git_dirty": false,
  "resolved_model_config": {},
  "resolved_train_config": {},
  "resolved_task_config": {},
  "metrics": {},
  "efficiency": {},
  "debug": {}
}
```

### Aggregate CSV

One row per seed-run.

Required columns:

- `experiment_set`
- `task_name`
- `task_category`
- `model_family`
- `model_size`
- `context_length`
- `seed`
- `main_metric_name`
- `main_metric_value`
- task-specific metric columns
- `peak_gpu_memory_bytes`
- `tokens_per_second`
- `decode_tokens_per_second`
- `wall_clock_step_seconds`
- `parameter_count`
- `trainable_parameter_count`

### Grouped summary CSV

One row per `(task_name, model_family, model_size, context_length)`.

Required summary fields:

- `seed_count`
- `metric_mean`
- `metric_std`
- task-specific metric means/stds
- efficiency means/stds

## Failure Analysis Hooks

Always log:

- `refresh_norm_mean`
- `refresh_norm_std`
- `refresh_cosine_to_previous`
- `detail_usage_frequency`
- `detail_candidate_count_mean`
- `detail_candidate_hit_rate`
- `length_bucket`
- `error_examples`

`error_examples` should store a small capped sample:

- input summary
- gold answer
- predicted answer
- support segment indices
- retrieved detail indices if applicable

## Prioritization

### Must build first

- deterministic five-task generator registry
- unified model forward/loss interface
- run config resolver for model/task/train bundles
- per-run JSON + aggregate CSV output
- small pilot runner

### Nice to have later

- richer plot styling
- per-hop reasoning confusion views
- beam-search or constrained decoding for copy tasks
- token-level retrieval saliency visualizations

### Likely failure points

- unfair parameter drift between model families
- copy-task label leakage from prompt formatting
- detail retrieval becoming an unrestricted token-memory path
- noisy efficiency numbers from inconsistent measurement setup

### Sanity checks before scale-up

- `transformer_local` should break first as distance exceeds local window
- `srd_refresh` should beat `transformer_local` on `easy_kv`
- if `easy_kv` works but `binding_kv` fails, the likely issue is missing binding fidelity rather than generic long-range routing failure
- sufficiency should help on `binding_kv` and `multi_hop_segment_reasoning`
- `srd_refresh_sufficiency_detail` should close part of the gap on `mixed_dependency`
- `transformer_full` should remain strongest on `delayed_copy`
