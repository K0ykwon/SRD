# PLANS.md

## Objective

Build a minimal, correct, measurable SRD prototype that can test whether scheduled refresh-only global interaction improves long-context efficiency-quality tradeoffs.

## Non-goals

The first phase does not include:

- massive-scale pretraining
- custom kernels
- production serving stack
- large benchmark sweep before correctness
- speculative novelty claims without evidence

## Background

SRD (Segment Refresh Decoder) is a decoder architecture with an explicit routing constraint:

- regular tokens follow a local-only path
- periodic refresh states alone can access long-range memory
- a shared bounded long-memory bank is updated at segment boundaries
- refresh integration may be restricted to upper layers in some variants
- a refresh sufficiency objective trains the bottleneck to carry useful future information
- evaluation should compare SRD against a simple local baseline and summary-memory-style baselines

The first prototype should favor clear tensor routing and measurable behavior over sophistication. Use standard PyTorch ops. Avoid custom CUDA kernels, ANN retrieval, and dynamic routing trees.

## Proposed Approach

Phase 1: repository setup and docs

Phase 2: local block + refresh block + long bank

Phase 3: full SRD model wiring

Phase 4: refresh sufficiency loss

Phase 5: tiny training and smoke tests

Phase 6: first benchmark harness and profiling

Phase 7: ablations and paper-facing outputs

## Work Breakdown

### Active extension — Experiment Set A scaffolding

Goal

- Define the first paper-facing controlled synthetic suite that cleanly separates compressible long-range dependence from non-compressible exact-detail dependence.
- Lock down a reusable repository structure, config surface, run matrix, and artifact schema before implementing the larger benchmark code path.

Files likely to change

- `PLANS.md`
- `docs/experiments.md`
- `docs/experiment_set_a.md`
- `configs/experiment/set_a/*`
- `configs/model/set_a/*`
- `configs/train/set_a/*`
- `outputs/README.md`

Risks

- The benchmark plan drifts away from the architectural invariants and quietly rewards direct token-memory access.
- The matrix becomes too large to debug cheaply before the first pilot run.
- Config surfaces fork from the existing `SRDConfig` / benchmark config path and create duplicate experiment logic.

Validation

- The suite definition keeps the local backbone aligned across all five target model families.
- Each task has deterministic generation, explicit difficulty knobs, and task-specific metrics.
- The full matrix and a reduced pilot matrix are both written down concretely.
- Artifact schemas are explicit enough for `Table 1`, `Table 2`, `Figure 1`, and `Figure 2` generation.

Exit criteria

- The repository contains an implementation-ready Experiment Set A spec and config scaffolding.
- Later implementation work can add code under the documented paths without revisiting experiment design.

### Active optimization — SRD overhead reduction

Goal

- Reduce prototype overhead in the SRD refresh path before continuing long synthetic sweeps.
- Keep the architecture unchanged while removing avoidable Python-loop and repeated-computation costs.

Decision

- Pause experiment execution after the first partial pilot/ablation results.
- Optimize the current implementation first, then resume comparisons on the same benchmark suite.

Concrete focus

- batch pre-refresh local-block execution across blocks instead of re-running the pre stack inside the block loop
- precompute next-block sufficiency targets instead of embedding the next block repeatedly in the loop
- replace repeated SRD list accumulation with fixed-shape output buffers and indexed writes
- use SDPA for refresh-bank and token-bank attention instead of explicit score/softmax/matmul steps
- reduce detail-path reallocation by keeping projected detail states in fixed buffers
- add a cheaper long-bank write fast path for the common one-refresh-at-a-time case
- split the old `delayed_kv` synthetic task into `easy_kv` and `binding_kv` so binding failures can be isolated from simple sparse retrieval
- preserve refresh-only routing and the explicit refresh bottleneck
- add a first decode-cache path so completed SRD blocks are cached and only the currently open block is recomputed during decode profiling
- add a true KV-cache path for `transformer_full` so dense-baseline decode comparisons are not penalized by prefix re-execution
- remove avoidable clone/copy overhead in decode state handling
- reduce benchmark-runner profiling overhead from per-step model snapshots and repeated CPU synchronization

Current optimization bundle

- keep only the baseline detail model as the active SRD variant
- restore and keep the real `transformer_full` KV-cache path active
- remove unnecessary decode-state clones in SRD and dense baselines
- lower benchmark bookkeeping overhead before rerunning the clean main suite
- add incremental `LocalBlock` caches so SRD decode stops recomputing the open-block pre stack from scratch
- cache the full SRD open-block `pre_blocks` path during decode and, for the base refresh model, cache the open-block `post_blocks` path as well

Validation

- targeted model tests still pass
- one short benchmark smoke run still completes and returns metrics
- output shapes and refresh routing behavior remain unchanged
- incremental decode logits match full-forward logits at divisible-prefix and block-completion checkpoints
- `transformer_full` incremental decode reuses layer KV caches and matches full-forward logits

### Active redesign — metric-credible Set A

Goal

- Replace the previous underpowered pilot recipe with a staged diagnostic suite that produces interpretable metrics.
- Separate routing success from exact binding failure before returning to copy-style tasks.

Decision

- Discard all previous `outputs/set_a/*` artifacts.
- Treat earlier tiny and ultracompact results as invalid for reporting because they mixed overwrite bugs, undertrained settings, and ceiling/floor tasks.

Concrete focus

- stage 0: calibration on tasks that should reliably learn (`easy_kv`, `needle_retrieval`)
- stage 1: binding diagnosis on paired tasks (`easy_kv`, `binding_kv`)
- stage 2: routing-plus-reasoning diagnosis (`needle_retrieval`, `multi_hop_segment_reasoning`)
- defer `delayed_copy` and `mixed_dependency` until detail-path fidelity improves
- add binding-specific metrics that distinguish exact success from wrong-catalog binding

Validation

- new suite configs exist for calibration, binding, and reasoning phases
- `binding_kv` emits prefix/suffix/catalog error metrics
- per-run JSON filenames are unique by run name

### Active redesign — stronger refresh without large efficiency regression

Goal

- Improve SRD quality on binding- and reasoning-heavy tasks without giving up the refresh bottleneck or destroying the current memory profile.

Decision

- prioritize architectural changes that preserve bounded bank size and small refresh counts
- avoid changes that simply widen the global path by brute force

Concrete focus

- role-structured refresh slots instead of one undifferentiated pooled summary
- binding-aware sufficiency auxiliary targets instead of summary-only future prediction
- importance-aware bank writes and merges instead of uniform average compression

Validation

- each change is implemented behind a config flag and can be ablated cleanly
- memory and throughput are re-measured after each change
- Stage 1 binding diagnostics improve before copy-style tasks are revisited

Decision update

- the first `importance-aware bank` variant underperformed the simpler `typed_slots + binding_aux` baseline on the 500-step compact diagnostic run
- remove `importance_bank` from active benchmark suites and exposed model-family configs for now
- keep the lower-level bank write/merge code path in source as dormant implementation material, but stop treating it as a current candidate
- narrow the next main comparison to the tasks SRD currently appears capable of solving: `easy_kv`, `needle_retrieval`, and `multi_hop_segment_reasoning`
- keep the active SRD comparison centered on `srd_refresh_sufficiency_detail` versus dense Transformer baselines
- use `1024` and `2048` contexts for the next long-context comparison
- add a follow-up comparison pass for `srd_refresh_sufficiency_detail` and a parameter-matched `~15M transformer_full` under the same `1024/2048` task set so the long-context table includes both the stronger SRD variant and a size-matched dense baseline

### Active diagnostic — Delayed KV only

Goal

- Run a narrow KV-only diagnostic that separates compressible retrieval from binding failure without mixing in unrelated tasks.

Decision

- restrict the next diagnostic suite to delayed-KV variants only
- use two task variants:
  - `binding_lite_kv` for the original compressible delayed-KV regime
  - `binding_heavy_kv` for the harder exact-binding regime
- compare `transformer_full` and `srd_refresh_sufficiency_detail`
- use short contexts `32`, `64`, `128`, `256` for fast iteration
- report both:
  - `retrieval_hit`
  - `binding_accuracy`

Metric convention

- `binding_accuracy` is exact gold value-span match
- `retrieval_hit` is an operationalized memory-hit metric: the predicted value span belongs to the in-context candidate-value catalog, even if bound to the wrong key

### Active redesign — scan-first detail path

Goal

- remove the current block-by-block structural bottleneck from the detail SRD path without relaxing the refresh-only routing rule
- preserve the paper claim that long-range interaction flows through bounded block-level bottlenecks rather than direct token-level global access

Decision

- treat Mamba-style scan parallelization as a structural reference, not as a requirement to adopt Mamba modules
- separate the redesign into three explicit deliverables before implementation:
  - a mechanism-preservation spec
  - a boundary document for what still counts as SRD versus what would collapse into a generic memory-token model
  - a code-level two-pass mapping from the current sequential block loop to a future scan-first implementation

Concrete focus

- represent inter-block carry as an explicit bounded block state instead of implicit sequential bank side effects
- split the current `detail retrieval -> post stack` loop into:
  - a block-parallel summary pass
  - a block-axis scan over compact states
  - a block-parallel conditioning pass
- move detail retrieval to a refinement stage that follows a compact prefix summary rather than driving the entire long-range path directly
- keep regular-token access local-only and keep any global path bounded, segment-triggered, and inspectable

Validation

- the redesign spec states which invariants are preserved and which changes would violate SRD's mechanism
- the future implementation plan names the files, intermediate tensors, and pass boundaries that would change
- the plan makes clear which parts are intended for research implementation next and which remain out of scope

Execution update

- added `docs/scan_first_redesign.md` as the mechanism-preserving redesign reference
- linked the redesign note from `README.md`, `docs/architecture.md`, and `docs/experiments.md`
- completed the first no-behavior-change refactor in `src/srd/modeling/block_refresh_detail_model.py`:
  - `encode` stage via `_encode_blocks_parallel(...)`
  - sequential-compatibility `scan` stage via `_scan_block_state(...)`
  - explicit conditioned `post` stage via `_apply_conditioned_post_blocks(...)`
- added an explicit per-block long-range state trace in the detail `forward()` path via `_scan_detail_block_sequence(...)`
- the current trace still follows sequential semantics, but the carry/fused-context tensors are now explicit bounded intermediates instead of only implicit loop state
- moved completed-block `prefill()` execution onto `_scan_completed_blocks_prefill(...)` so `forward()` and `prefill()` now expose the same staged inter-block state semantics
- added the first explicit carry recurrence hook behind `detail_scan_carry_mode`:
  - `legacy` preserves the old carry update
  - `affine` interpolates between the previous carry state and the new refresh write
- replaced online-only prefix-carry tracing with an explicit refresh-write trace plus `_scan_carry_sequence_from_refresh_writes(...)`
- added parity checks that the refresh-write scan reconstructs the same prefix carry trace as the online sequential execution
- unified `forward()` and completed-block `prefill()` around `_materialize_online_detail_block_pass(...)` so both now share the same first-pass online materialization structure
- split the shared online materialization path further into `_materialize_online_detail_block_step(...)` plus the pass-level orchestrator
- next structural step: make detail retrieval explicitly a refinement stage after coarse carry materialization, even before true parallel scan is implemented
- tried a bounded recent-candidate refinement option and discarded it because it risks degrading the original full-history detail path
- removed the recent-only refinement path from config, model code, tests, and docs so the repository again exposes only the original full-history detail retrieval behavior
- completed the cached-block long-context continuation:
  - `54 / 54` cached-block runs completed under `outputs/reproduction/cached_block_longctx_main`
  - final aggregate metric `0.9606`, forward throughput `233586.71 tok/s`, decode throughput `280.94 tok/s`, average peak memory `565.46 MiB`
  - `delayed_kv` and `needle_retrieval` reached `1.0000` across all `compact` and `small` cells
  - `delayed_copy` remains the main weak point at context `2048`, with both `compact` and `small` averaging `0.6667`
  - final committed summary is `docs/results_snapshot_2026-04-19_cached_block_final.md`
- removed the overengineered replay/selective-replay execution branches because they did not deliver stable throughput wins and added complexity to the active research path
- next implementation step is a true opt-in block-parallel forward path for the detail model:
  - keep the default sequential execution unchanged
  - restrict the experimental parallel path to `forward()` first
  - parallelize heavy block work as `all pre -> parallel refresh proposals -> compact carry scan -> parallel detail/post`
  - keep the refresh bottleneck explicit and keep regular-token access local-only
  - do not switch detached benchmark services onto the new path until it has separate smoke and profiler validation
- added a focused comparison suite for the opt-in parallel detail path:
  - `configs/experiment/set_a/suite_parallel_detail_vs_transformer_compact.json`
  - compares only `srd_refresh_sufficiency_detail_parallel` and `transformer_full`
  - compact scale, 1024/2048 contexts, required synthetic tasks, seed 11
  - first used with a short smoke train config to check plumbing
  - next run uses the same `reproduction_required_longctx_16gb` 2400-step setting as the active long-context experiment, with output kept separate from the smoke artifacts
- recorded the current partial result snapshot in `docs/results_snapshot_2026-04-18.md`; it summarizes completed per-run JSON files without committing raw output artifacts
- next decode-side parallelization step:
  - keep autoregressive token generation sequential
  - parallelize `prefill()` completed-block processing when `detail_forward_mode="parallel_scan"`
  - keep open-block `decode_step()` incremental and cache-based
  - fix the experimental parallel path so blocks with no long-range context receive no `carry_to_post` bias from a zero placeholder
- decode-side parallelization implementation update:
  - added `detail_decode_mode="cached_block"` as an opt-in decode mode
  - moved parallel completed-block `prefill()` onto the shared `parallel_scan` materialization helper
  - cached the open-block fused refresh/detail context so per-token decode can reuse post-stack local KV caches
  - rematerialized the completed open block once at the boundary before writing refresh/detail memory to reduce long-state drift
  - documented the approximation and cache layout in `docs/decode_parallelization.md`
- cached-block validation run:
  - run only `srd_refresh_sufficiency_detail_parallel` with `detail_decode_mode="cached_block"`
  - reuse the focused compact conditions from the previous parallel-vs-transformer suite: compact scale, contexts `1024/2048`, tasks `delayed_kv/needle_retrieval/delayed_copy`, seed `11`, 2400 training steps
  - keep artifacts separate under `outputs/reproduction/parallel_detail_cached_block_compact_full`
- cached-block longctx main run:
  - run only `srd_refresh_sufficiency_detail_parallel` with `detail_decode_mode="cached_block"`
  - match the main long-context reproduction matrix without sufficiency ablations: scales `compact/small`, contexts `1024/2048/4096`, tasks `delayed_kv/needle_retrieval/delayed_copy`, seeds `11/17/23`
  - keep artifacts separate under `outputs/reproduction/cached_block_longctx_main`
  - resume as two non-overlapping size shards against the same output directory:
    - `cached_block_longctx_compact.service` on GPU1
    - `cached_block_longctx_small.service` on GPU0
  - use `--skip-existing` so completed cached JSONs are reused and interrupted in-flight cells are redone once
- restarted detached reproduction jobs under the restored full-history-only code; `--skip-existing` still preserves completed run JSONs and only redoes interrupted in-flight cells
- next retrieval optimization path: keep full-history detail available, but add an optional coarse-to-fine grouped summary stage so refinement first selects a few detail groups and only then runs fine top-k inside those groups
- implemented the first grouped coarse-to-fine detail path behind `detail_coarse_group_size/detail_coarse_topk_groups`, with default `0/0` preserving the old full-history behavior
- initial compact smoke/profiler results did not show a consistent throughput win:
  - benchmark smoke regressed on both tasks tested
  - decode throughput generally regressed
  - prefill improved only in one cold-sensitive cell and regressed or mixed elsewhere
- keep grouped retrieval optional and non-default until a better coarse summary or routing policy is available
- hard constraint for subsequent work:
  - do not change the active default detail path
  - do not switch detached experiments onto experimental execution modes
  - any new structural optimization must remain opt-in behind config flags until it shows no quality regression and no throughput regression on the matched smoke/profiler cells
- removed the experimental `two_pass_replay` / selective replay / alternate refresh-write-source branches because they added complexity without proving a stable throughput win
- kept the current implementation sequential while making the future two-pass boundaries explicit in code

### Active redesign — general relation-preserving refresh

Goal

- improve binding preservation without introducing task-specific supervision

Decision

- avoid KV-specific candidate classification or task-name-conditioned loss branches for the next refresh improvement
- add a general `relation` refresh slot and a self-supervised relation auxiliary loss instead

Concrete focus

- typed refresh slots become `summary/entity/relation`
- relation slot is initialized from a generic pair-composition signal inside the current block
- relation auxiliary loss matches the refreshed relation slot against a detached relation target built from within-block token interactions

Decision update

- drop `srd_refresh_typed_slots_binding_aux` and `srd_refresh_typed_slots_relation_aux` from the active code path
- keep only `srd_refresh_sufficiency_detail` as the active SRD variant for the next round of experiments
- treat previous `aux` and `relation_aux` results as archived diagnostic evidence only

### Active redesign — detail-first structural fixes

Goal

- strengthen `srd_refresh_sufficiency_detail` on binding-heavy and long-context reasoning tasks without widening SRD into token-level global access

Decision

- stop adding new structural branches for this paper cycle
- remove exploratory `typed_carry_commit` and `carry_modulation` paths from the active code path
- focus the next work on implementation optimization and clean `detail` benchmarking

Concrete focus

- simplify the active SRD path back to `srd_refresh_sufficiency_detail`
- remove inactive architectural branches that complicate maintenance and comparisons
- optimize decode/train overhead in the surviving detail implementation
- rerun the clean main experiment suite only after the optimization pass

Validation

- `binding_lite_kv` stays strong
- `binding_heavy_kv` shows improved `retrieval_hit`, lower `off_catalog_rate`, and at least some recovery in `binding_accuracy`
- `needle_retrieval` does not regress materially
- memory stays bounded and decode throughput does not collapse relative to the current detail baseline

Stop conditions

- if typed carry commit hurts `binding_lite_kv` without improving `binding_heavy_kv`, revert it

### Active implementation — adaptive slot SRD

Goal

- add an `adaptive_slot_srd` variant that keeps fixed tensor shapes while learning the logical refresh capacity per segment
- preserve the existing SRD routing rule: only refresh-derived states may access long-range memory

Decision

- extend the existing block-refresh implementation rather than introducing a separate training stack
- keep physical slot count fixed with `refresh_slots_max`, and learn slot usage through differentiable gating
- default to soft sigmoid gating; keep hard gating optional and off by default
- store refresh memory as dense tensors with segment-granularity truncation support

Concrete focus

- refactor refresh-slot construction into a reusable learned-query pooling path over segment states
- add per-slot gate prediction plus budget and entropy regularization
- append gated refresh slots into the long-memory bank without Python-side per-slot structures
- add a compact memory-summary read path that runs once per segment and conditions later local processing
- surface adaptive-slot metrics in tiny training and benchmark summaries
- add a forward/backward smoke test for the new variant

Validation

- existing SRD and baseline model tests continue to pass unchanged
- the new adaptive-slot model preserves fixed-shape refresh tensors and refresh-only bank access
- a dummy segmented batch runs forward and backward with non-zero gradients through gates and sufficiency head
- training and evaluation summaries report gate-usage and memory-usage diagnostics

### Active optimization — adaptive slot memory and throughput

Goal

- reduce adaptive-slot SRD memory traffic and per-step overhead enough that its efficiency profile stays favorable against dense Transformer baselines
- keep the refresh-only routing invariant intact

Decision

- optimize the existing `adaptive_slot_srd` path in place instead of redesigning the model family
- prioritize fewer large temporary tensors, less repeated bank summarization work, and less Python-side segment bookkeeping

Concrete focus

- avoid redundant memory-summary reads when the carry state is unchanged within a block phase
- reduce dense bank append/copy overhead where recent-segment truncation is active
- keep slot pooling and gating batched, contiguous, and free of avoidable intermediate clones
- add a small benchmark-facing efficiency sanity check against `transformer_full`

Validation

- adaptive-slot forward/backward tests still pass
- tiny benchmark smoke run still completes
- measured decode or train throughput stays above the dense Transformer comparator on the same smoke setup

### Active packaging — reproducible synthetic benchmark bundle

Goal

- expose one explicit reproduction path for the required synthetic benchmarks:
  - Delayed KV
  - Needle
  - Delayed Copy
- expose one explicit sufficiency ablation path with a lambda sweep
- expose one explicit audit path that shows how task scoring maps into aggregate reporting

Decision

- reuse the existing large-suite runner and artifact writers instead of creating a parallel experiment stack
- keep shell wrappers thin and place the actual audit logic in `src/srd/eval`

Concrete focus

- add a dedicated reproduction experiment config covering the three required synthetic tasks
- add a dedicated lambda sweep over `sufficiency_loss_weight`
- add a score/aggregate audit module that sanity-checks task scoring and grouped aggregation with deterministic inputs
- add a short reproduction note describing the exact commands and expected output directories

Validation

- the reproduction bundle config is runnable through the existing suite driver
- the audit module exits cleanly and writes a human-readable summary
- tests cover the audit path and grouped aggregate expectations

### Active scaling — multi-million long-context reproduction

Goal

- move the required synthetic reproduction path out of the tiny regime and into the repository's paper-facing Set A scale bands
- make the default public reproduction recipe cover both low-million and tens-of-millions models at materially longer contexts

Decision

- reuse the Set A suite runner instead of adding another large-experiment code path
- target the existing `compact` and `small` backbones as the primary `~15M` and `~50M` classes
- keep the required task set unchanged: `delayed_kv`, `needle_retrieval`, and `delayed_copy`

Concrete focus

- add a dedicated long-context required-reproduction suite over `1024`, `2048`, and `4096` tokens
- keep the main comparison centered on `transformer_full`, `srd_refresh`, `srd_refresh_sufficiency`, and `srd_refresh_sufficiency_detail`
- keep the sufficiency lambda sweep explicit as a Set A ablation
- make the `compact` backbone valid at 4k context so the same suite can span low-million and tens-of-millions scales
- add a larger train preset and thin shell wrapper for the enlarged bundle
- document run counts and expected compute so this path is clearly distinct from the earlier smoke-scale reproduction bundle

Validation

- the new suite expands to the expected main and ablation run counts
- the 4k `compact` config builds successfully through the existing Set A model-config path
- the new script and docs point to one concrete runnable command and output directory

Execution update

- current workstation exposes a `16GB` RTX 5060 Ti and a `12GB` RTX 3060
- the default long-context public runner should therefore target the `16GB` card conservatively with `micro_batch_size=1`
- add a separate `small`-only `8k` suite instead of overloading the mixed `1k/2k/4k` suite further
- validate the new path with a real `MAX_RUNS` smoke execution before treating it as the default reproduction entry point
- no local `tmux` or `screen` is available in the current environment, so resilient execution should use detached shell launch plus on-disk run JSON checkpoints
- add suite-resume support keyed by existing run JSON files so interrupted long runs can restart without recomputing completed cells
- reorder the required suites so `srd_refresh_sufficiency_detail` runs before the weaker refresh and dense baselines

### Active diagnostic — separate decode profiling

Goal

- measure whether the SRD detail path actually wins where it is supposed to win: long-prefix incremental decode
- separate full-forward benchmark throughput from prefill/decode-specific throughput and memory

Decision

- add a dedicated profiling entry point instead of overloading the training benchmark runner further
- compare `srd_refresh_sufficiency_detail` and `transformer_full` on matched Set A cells
- report prefill tok/s, decode tok/s, and peak memory separately

Concrete focus

- build one small profiling module under `src/srd/eval`
- let it reuse Set A task config resolution and model-config building
- expose one shell wrapper for quick repeated runs
- write compact CSV/JSON artifacts under `outputs/profiling/`
- support decode-length overrides so profiling can measure many incremental decode steps even when the task answer span is short

Validation

- a short compact-cell profiling run completes for both families
- the output artifact includes separate prefill and decode timing fields
- the output artifact includes enough decode steps that one-step timing noise does not dominate
- the script is runnable independently of the main training benchmark suites

Execution update

- current implementation work focuses on low-risk overhead removal in `srd_refresh_sufficiency_detail` before rerunning the `detail vs transformer_full` comparison
- immediate code targets:
  - remove per-block `clone()` on detail-history slices in training/eval forward
  - parallelize completed-block `pre_blocks` work during `prefill()` instead of rerunning the pre stack inside the block loop
  - replace per-token full open-block `post_blocks` recomputation with bounded suffix recomputation and a one-time full recompute only when the block closes
  - shrink prefill/decode memory traffic by dropping dead detail-state buffers from decode state, computing next-token logits from the last token only, and writing detached detail KV caches under no-grad
  - keep completed-block semantics unchanged while preserving exact logits for existing forward/prefill/decode tests
- after the optimization patch:
  - rerun targeted model tests
  - rerun a focused `refresh_with_detail` vs `transformer_full` benchmark comparison

### Active scale-up — 150M detail-only run

Goal

- train only the `srd_refresh_sufficiency_detail` family at the repository's `~150M` Set A scale and measure whether scaling helps the active detail path before committing to larger comparison matrices

Decision

- use the existing Set A `base` shared backbone as the first 150M-scale backbone
- keep the active routing design unchanged: `refresh_with_detail` only, no new architectural branches
- run a detail-only training and eval sequence first, then decide whether a size-matched dense rerun is worth the cost

Concrete focus

- add a dedicated `~150M detail` model config that resolves to the Set A base backbone plus the active detail path
- add a dedicated train preset with a conservative token budget and accumulation defaults
- add a focused eval config that runs only `refresh_with_detail` on the current three-task synthetic comparison shell
- verify the resolved parameter count and record it in outputs

Validation

- resolved model parameter count lands near the intended `150M` scale and is reported in artifacts
- smoke validation runs without OOM at the configured micro-batch size
- targeted detail-model tests still pass after config additions
- the focused eval emits per-run JSON plus aggregate CSV/markdown artifacts
- if carry modulation improves reasoning but regresses sparse retrieval sharply, park it behind a config flag
- if structure-preserving compression costs too much decode throughput for negligible quality gain, keep the current bank path as default

### Phase 1 — Repo scaffolding

Goal

- Establish a minimal repository layout, architectural docs, agent instructions, scripts, tests, and CI that reflect SRD rather than a generic LM scaffold.

Files likely to change

- `README.md`
- `AGENTS.md`
- `PLANS.md`
- `docs/*`
- `scripts/*`
- `tests/*`
- `.github/workflows/ci.yml`

Risks

- Architecture gets described loosely enough that later code drifts into a generic memory-token design.
- Too much infra is added before the core model exists.

Validation

- Required files exist and describe SRD invariants consistently.
- Docs explicitly distinguish SRD from summary-token and memory-token approaches.

Exit criteria

- Repo structure is usable for model implementation.
- Architectural invariants are documented clearly enough to constrain later work.

### Phase 2 — Core SRD modules

Goal

- Implement the minimal core modules: local block, refresh block, and shared bounded long-memory bank.

Files likely to change

- `src/srd/modeling/local_block.py`
- `src/srd/modeling/refresh_block.py`
- `src/srd/modeling/long_bank.py`
- `tests/test_local_block.py`
- `tests/test_refresh_block.py`
- `tests/test_long_bank.py`
- `docs/architecture.md`

Risks

- Local block accidentally permits broader-than-intended context.
- Refresh block semantics become unclear about when bank reads and writes happen.
- Bank API becomes over-engineered before requirements are known.

Validation

- Shape tests for all core modules.
- Causal masking sanity checks for the local path.
- Bank growth and truncation tests.
- Refresh-only access invariance tests where feasible.

Exit criteria

- Core modules are readable, runnable, and have passing focused tests.
- Bank behavior is bounded and explicit in code.

### Phase 3 — Training path

Goal

- Wire the full SRD model so token states follow local computation, refresh states interact with the bank, and logits are produced for training and decode smoke tests.

Files likely to change

- `src/srd/config.py`
- `src/srd/modeling/srd_model.py`
- `src/srd/training/train.py`
- `src/srd/training/__init__.py`
- `tests/test_srd_model.py`
- `examples/minimal_infer.py`
- `examples/minimal_train.py`

Risks

- Model wiring obscures whether refresh updates occur before or after upper local layers.
- Bank lifecycle is reset or reused incorrectly across forward paths.
- Public config names become unstable too early.

Validation

- Forward shape tests on tiny batches.
- Tiny decode smoke check.
- Tests that refresh indices behave as expected for short sequences.

Exit criteria

- Full forward pass runs on CPU with clear outputs.
- The model exposes refresh states and bank states explicitly enough for debugging and evaluation.

### Phase 4 — Sufficiency objective

Goal

- Implement a first explicit refresh sufficiency loss that can be trained and ablated against plain LM loss.

Files likely to change

- `src/srd/training/losses.py`
- `src/srd/training/train.py`
- `docs/architecture.md`
- `docs/experiments.md`
- `README.md`

Risks

- Sufficiency loss is too weak and becomes a no-op.
- Sufficiency target leaks information in a way that breaks the intended SRD bottleneck.
- Loss design gets overly complex before a baseline variant is measured.

Validation

- Unit tests for loss computation shape and finite values.
- Tiny train step with sufficiency loss enabled.
- Ablation switch for with/without sufficiency objective.

Exit criteria

- Sufficiency path is explicit in code and can be toggled.
- Training code can report main loss and sufficiency loss separately.

### Phase 5 — Evaluation harness

Goal

- Build a minimal benchmark runner that measures correctness-adjacent efficiency signals and supports first baseline comparisons.

Files likely to change

- `src/srd/eval/metrics.py`
- `src/srd/eval/benchmark_runner.py`
- `docs/experiments.md`
- `scripts/eval_tiny.sh`
- `scripts/profile_decode.sh`

Risks

- Metrics are too noisy or too ad hoc to compare variants.
- Memory accounting differs across CPU and GPU paths without being documented.
- Benchmark harness grows into a full framework before small comparisons are stable.

Validation

- Tiny decode smoke check.
- Profiling output includes memory and tokens/sec.
- Throughput-per-memory is computed from explicit inputs.

Exit criteria

- Baseline and SRD variants can be run through one minimal harness.
- Efficiency metrics are logged in a way suitable for later plots.

### Phase 6 — Initial experiment set

Goal

- Run the first synthetic long-context experiment set that checks whether SRD helps on tasks where information must cross segment boundaries through the refresh bottleneck.

Files likely to change

- `docs/experiments.md`
- `configs/model/*`
- `configs/train/*`
- `configs/experiment/*`
- `src/srd/eval/benchmark_runner.py`
- `README.md`

Risks

- Small-scale results are dominated by variance.
- Baselines are unfairly weak or mismatched.
- Efficiency claims are made from unrepresentative runs.

Validation

- Repeated tiny or small runs where needed to estimate noise.
- Side-by-side reporting of quality and efficiency metrics.
- At least one comparison where local-only should degrade once dependency length exceeds the local window.
- Side-by-side comparison of SRD without sufficiency and SRD with sufficiency.

Exit criteria

- Synthetic benchmark families exist for delayed retrieval, sparse needle retrieval, and delayed copy.
- Per-run JSON, aggregate CSV, and markdown summary artifacts are generated by the benchmark path.
- First experiment table exists with clearly labeled caveats.
- Next architectural decisions are informed by data rather than intuition alone.

## Validation Plan

Validation for the first working prototype should include:

- unit tests for local block, refresh block, long bank, and full model wiring
- shape tests across representative tiny batch and sequence settings
- causal masking sanity checks for the local path
- bank growth and bank merge behavior tests
- refresh-only access invariance tests to guard against accidental direct long-bank access from regular tokens
- tiny train loss decrease check over a short smoke run
- tiny decode smoke check
- profiling for memory and tokens/sec

Normal-to-strict standard:

- Every touched module gets a targeted test or a clear reason it did not.
- Model or training edits should trigger at least one end-to-end smoke run.
- Report skipped checks explicitly.

## Experiment Plan

Initial experiments:

- local-only baseline vs SRD on boundary-crossing synthetic tasks
- SRD without sufficiency loss vs SRD with sufficiency loss
- varying segment length
- varying refresh count
- varying bank size
- upper-layer-only vs all-refresh-using layers

Primary metrics:

- validation loss / perplexity
- long-context task score
- peak memory
- decode tokens/sec
- throughput-per-memory
- context-length scaling slope

Execution notes:

- Start with the smallest runs that can reveal routing failures.
- Keep baseline implementations simple and interpretable.
- Report both absolute values and relative deltas where stable enough.

## Risks and Mitigations

- refresh path collapse
  Mitigation: monitor refresh-state norms, refresh loss magnitude, and ablate sufficiency weight.

- bank behaving like weak summary memory
  Mitigation: compare bank update rules and inspect whether bank reads materially affect downstream performance.

- local path dominating long path
  Mitigation: compare local-only baseline against SRD under matched settings and inspect gains specifically on long-gap tasks.

- efficiency gains disappearing at small scale
  Mitigation: treat small-scale efficiency as directional only and separate architectural correctness from scale claims.

- evaluation noise
  Mitigation: repeat critical tiny runs, keep setups fixed, and report hardware/context details with results.

## Open Questions

- What is the best default segment length for the first SRD runs?
- What teacher source should define the first sufficiency target?
- What bank merge rule should be the default: append-only, pooled merge, or gated overwrite?
- Should lower layers remain strictly local in all first-phase variants?
- What benchmark should have first priority?

## Progress

- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Phase 4 complete
- [x] Phase 5 complete
- [x] Phase 6 complete
- [x] Phase 7 started
- [x] Local-only baseline implemented
- [x] Summary-memory-style baseline implemented
- [x] Refresh sufficiency ablation implemented
- [x] Tiny benchmark harness stable
- [x] First result table recorded
- [x] Synthetic long-context benchmark suite implemented
- [x] Ablation sweep artifacts implemented
- [x] External baseline comparison suite implemented
- [x] Parameter-aware conventional baseline presets implemented
- [x] Paper-ready block-refresh SRD variant implemented
- [x] Block-refresh detail-memory variant implemented
- [x] Final comparison baselines and focused comparison suite implemented
- [x] Length-scaling comparison suite implemented and run
- [x] Parameter-scaling comparison suite implemented and later retired
- [x] Long-budget parameter-scaling suite implemented and later retired
- [x] Corrected compact 3-scale long-budget suite implemented and run

## Surprises & Discoveries

- Date: 2026-04-08
- Observation: Delaying refresh influence to later segments made the bank-access invariant much easier to preserve and test than scattering refreshed states back into ordinary token positions.
- Impact: The first prototype now enforces scheduled global interaction more cleanly in both code and tests.
- Follow-up: Compare this delayed-carry design against stronger upper-layer integration variants after the first baseline matrix is stable.

- Date: 2026-04-08
- Observation: The first synthetic suite and ablation artifacts run end to end, but the default local runs are still undertrained enough that task metrics remain near floor on several settings.
- Impact: The current benchmark path is valid for structural comparison and artifact generation, but not yet for strong performance claims.
- Follow-up: Increase training budget and tune benchmark difficulty before using these tasks to argue for or against SRD quality gains.

- Date: 2026-04-08
- Observation: Sparse long-context tasks became much more learnable once benchmark training combined explicit answer-span supervision with a smaller easy-mode symbol pool.
- Impact: The benchmark suite now produces non-trivial separations instead of only floor-level metrics.
- Follow-up: Keep the easy-mode path for structural validation, then add harder symbol-pool settings for scale-up experiments.

- Date: 2026-04-10
- Observation: Internal SRD ablations are no longer enough to support the main research claim because they do not distinguish scheduled refresh-only access from simpler conventional alternatives.
- Impact: The next comparison phase must include parameter-aware Transformer and summary-memory baselines in the same benchmark and reporting path.
- Follow-up: Add conventional local/full Transformer comparators, a direct summary-memory baseline, and expose parameter counts in the primary suite outputs.

- Date: 2026-04-11
- Observation: The first parameter-scaling sweeps used adjacent tiers that were too far apart, which made the scaling story harder to interpret and mixed architectural effects with budget jumps.
- Impact: The main scaling suite was reset to a tighter three-tier ladder: original `~200k` models as `small`, a new `~400k-480k` intermediate tier as `medium`, and the former `~1.2M` tier as `large`.
- Follow-up: Use only the corrected compact long-budget suite for main scaling claims and treat the retired wide-gap scaling results as superseded.

- Date: 2026-04-10
- Observation: The existing SRD prototype already has block-wise refresh behavior, but its public config surface is still framed around the older segment terminology.
- Impact: The first paper variant should expose a cleaner block-refresh model with explicit `block_size`, `refresh_slots`, and `refresh_enabled` settings while preserving the refresh-only memory constraint.
- Follow-up: Add a dedicated `srd_block_refresh` model path plus parameter-matched local-only / refresh-no-sufficiency / refresh-with-sufficiency configs and smoke validation.

- Date: 2026-04-10
- Observation: The first block-refresh paper variant still fails badly on delayed-copy-style precise recall tasks even when it is strong on some long-context retrieval settings.
- Impact: A second paper-facing variant should add only a tiny bounded detail path, not a return to dense cross-block attention.
- Follow-up: Add a small per-block detail-slot memory with sparse top-k retrieval and test whether it helps delayed_copy_long without destroying the strong needle_long behavior.

- Date: 2026-04-10
- Observation: The detail-memory extension improved both needle_long and delayed_copy_long, which makes it the strongest SRD-side candidate for a final baseline comparison pass.
- Impact: The next comparison phase should include additional conventional comparators that can match either token-level recurrent memory or latent-bottleneck behavior.
- Follow-up: Add minimal Transformer-XL-style and Perceiver-style baselines, then run a focused multi-model comparison grouped by parameter scale and throughput.

## Decision Log

- 2026-04-08:
  Decision: Synthetic benchmark training will upweight answer-token loss rather than relying on uniform LM loss over mostly filler tokens.
  Reason: The benchmark tasks are intentionally sparse, so without answer-focused weighting the optimization signal is dominated by easy filler reconstruction.
  Consequence: The benchmark path remains causal but becomes much more likely to learn the intended long-range dependency.

- 2026-04-08:
  Decision: The first end-to-end SRD path will generate refresh states by pooling segment-boundary token states into one or more refresh slots per segment, then use those slots as the only bank-reading states.
  Reason: This is the simplest implementation that preserves scheduled global interaction and avoids drifting into a generic token-level memory design.
  Consequence: The first model will operate on explicit segment structure and expose refresh summaries, bank reads, and sufficiency targets directly.

- 2026-04-08:
  Decision: The first sufficiency objective will reconstruct the next segment summary from the current segment refresh output.
  Reason: It is simple, testable, uses only in-repo signals, and directly encourages the refresh bottleneck to carry future-useful information.
  Consequence: Initial experiments can compare local-only, SRD without sufficiency loss, and SRD with sufficiency loss without requiring a teacher model.

- 2026-04-08:
  Decision: The bounded bank will compress overflow by merging the oldest pair of entries.
  Reason: This keeps the bank explicitly bounded without silently discarding the oldest history.
  Consequence: The first implementation exposes bank compression behavior directly and supports unit tests around merge semantics.

- 2026-04-08:
  Decision: The first validation benchmarks will be simple causal synthetic tasks scored from answer tokens at the end of the sequence.
  Reason: This keeps long-range dependence obvious, controllable, and easy to score while avoiding a heavier task framework.
  Consequence: The benchmark path can train with the existing LM loss and report exact-match or accuracy on explicit answer spans.

- 2026-04-08:
  Decision: Benchmark and sweep outputs will be emitted as per-run JSON plus aggregate CSV and markdown summary artifacts.
  Reason: This is the smallest artifact set that supports local analysis, quick comparison, and paper-style table building.
  Consequence: Single-benchmark runs, the synthetic suite, and ablations all share one structured reporting path.

- 2026-04-10:
  Decision: The first external comparison phase will add three conventional baselines: a local-window Transformer, a full-attention Transformer, and a summary-memory model where ordinary token states may read a shared summary bank directly.
  Reason: These baselines directly test whether SRD helps because of scheduled refresh-only interaction rather than because any extra memory path exists.
  Consequence: The main result table can compare SRD against both weaker and stronger conventional decoder-style alternatives, with parameter counts shown for every run.

- 2026-04-10:
  Decision: Baseline matching will be practical rather than exact, with matched presets kept close in width/depth and explicit parameter counts reported in every artifact.
  Reason: The repo needs fair conventional comparators now, but exact optimizer-free parameter matching across SRD, direct summary memory, and full-attention baselines would add unnecessary tuning machinery.
  Consequence: Comparison tables can stay honest about model size while keeping the implementation simple and reproducible.

- 2026-04-10:
  Decision: The first paper-specific SRD variant will be exposed as a dedicated `srd_block_refresh` model type that keeps the same refresh modules present across local-only and refresh-enabled ablations, toggling only `refresh_enabled` and `sufficiency_loss_weight`.
  Reason: This keeps the architecture cleanly identifiable while making the three primary ablations as closely parameter-matched as practical.
  Consequence: Local-only, refresh-no-sufficiency, and refresh-with-sufficiency comparisons can share one implementation and report equal or near-equal parameter counts directly.

- 2026-04-10:
  Decision: The detail-memory extension will remain strictly auxiliary: a tiny number of per-block detail slots, sparse global top-k retrieval, and simple gated fusion with the existing refresh carry.
  Reason: The goal is to recover some non-compressible detail without abandoning the refresh bottleneck or drifting into full cross-block attention.
  Consequence: The new variant can be compared cleanly against the existing paper ablations as `refresh_with_detail`, with detail disabled by default in the base block-refresh model.

- 2026-04-10:
  Decision: The final focused comparison will use the best current SRD candidate (`refresh_with_detail`, default detail settings) against existing baselines plus two new lightweight comparators: token-level recurrent memory and latent-bottleneck memory.
  Reason: This keeps the comparison relevant to the paper claim without reopening the full experiment matrix.
  Consequence: Final reporting can compare models by raw quality, throughput, and rough parameter/throughput-matched groupings on a small set of representative tasks.

- YYYY-MM-DD:
  Decision:
  Reason:
  Consequence:

## Outcomes & Retrospective

- What worked:
- What failed:
- What changed from the original plan:
- What should happen next:
