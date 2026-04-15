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

Execution update

- current implementation work focuses on low-risk overhead removal in `srd_refresh_sufficiency_detail` before rerunning the `detail vs transformer_full` comparison
- immediate code targets:
  - remove per-block `clone()` on detail-history slices in training/eval forward
  - replace repeated `torch.cat()` growth in detail decode caches with indexed writes into reusable buffers
  - replace repeated `torch.cat()` growth for open-block pre hidden states with fixed-capacity block buffers plus running sums
  - preserve exact logits for existing forward/prefill/decode tests
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
