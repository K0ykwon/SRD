# Roadmap

## Phase 0: Repository Initialization

- define SRD terminology and architectural invariants
- stand up a minimal PyTorch prototype
- add small tests and smoke scripts

## Phase 1: Prototype Correctness

- validate local-only routing behavior
- test scheduled refresh extraction and scatter
- compare bank update choices on synthetic tasks

## Phase 2: Training Objective Iteration

- implement stronger refresh sufficiency losses
- compare auxiliary-target variants
- measure optimization stability and failure modes

## Phase 3: Benchmarking

- establish baseline models
- run long-context quality and efficiency comparisons
- standardize memory and throughput reporting

## Phase 4: Scaling Decisions

- decide whether SRD warrants larger-scale implementation effort
- identify the most promising architectural simplifications
- refine the codebase around the winning path
