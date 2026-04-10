# AGENTS.md

## Purpose

This repository exists to prototype and evaluate SRD (Segment Refresh Decoder), a long-context generative modeling architecture centered on:

- scheduled global interaction
- refresh-only long-memory access
- sufficiency-based training
- the long-context quality-efficiency tradeoff

SRD is a research prototype. The priority is readable implementation and measurable behavior, not high ceremony or premature infrastructure.

## Read This First

Before making changes, read files in this order:

1. `README.md`
2. `docs/overview.md`
3. `docs/architecture.md`
4. `docs/experiments.md`
5. `PLANS.md`
6. relevant source files

## Repository Map

- `src/srd/modeling`: SRD architecture implementation, including local blocks, refresh blocks, the long-memory bank, and model assembly
- `src/srd/training`: losses, training logic, and tiny training entry points
- `src/srd/data`: dataset handling and synthetic data helpers
- `src/srd/eval`: benchmarks, profiling hooks, and evaluation metrics
- `configs`: model, train, and experiment configuration placeholders
- `tests`: unit tests and lightweight integration coverage
- `docs`: design notes, architecture constraints, setup, and experiment plans
- `scripts`: convenience commands for tiny runs and profiling

## Working Rules

Agents must:

- preserve architectural clarity over premature optimization
- avoid unrelated refactors
- prefer minimal diffs
- keep interfaces simple
- update docs when architecture or experiments change
- keep the refresh bottleneck explicit
- avoid silently turning SRD into a generic memory-token model

## Architectural Invariants

These are non-negotiable unless the user explicitly asks to change them:

- regular tokens must not directly access the long-range bank
- refresh access is periodic and segment-triggered
- the long-memory bank is shared and bounded
- the refresh sufficiency path must remain explicit in code
- efficiency claims must be backed by measurable metrics

## Commands

Best-known commands today:

- environment setup:
  `python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install torch pytest`
- unit tests:
  `PYTHONPATH=src pytest -q`
- formatting:
  `TODO: confirm formatter; suggested baseline is ruff format`
- linting:
  `TODO: confirm linter; suggested baseline is ruff check`
- tiny training run:
  `PYTHONPATH=src python3 -m srd.training.train`
- tiny evaluation run:
  `PYTHONPATH=src python3 -m srd.eval.benchmark_runner`
- profiling decode:
  `bash scripts/profile_decode.sh`

## Editing Policy

Safe to edit:

- `docs/`
- `src/srd/*`
- `tests/*`
- `configs/*`
- `scripts/*`

Use extra caution for:

- benchmark logic
- metrics definitions
- public config names
- README claims about novelty or efficiency

## Validation Checklist

Before finishing, agents should:

- run targeted unit tests for touched modules
- run at least one tiny end-to-end smoke test when model or training code changes
- report unrun checks clearly
- update docs if assumptions changed

## Planning Policy

- For any non-trivial feature, experiment campaign, architectural change, or refactor, read and update `PLANS.md` first.
- Treat `PLANS.md` as a living execution document.
- Do not ask for "next steps" after each small subtask; proceed phase by phase unless blocked.
- Log decisions and deviations in `PLANS.md`.

## Documentation Policy

Update `README.md` and relevant docs when:

- architecture changes
- loss functions change
- benchmark scope changes
- config conventions change

## Output Expectations

Agents must report:

- changed files
- commands run
- failed or skipped checks
- unresolved risks
- TODOs left behind

Style requirements:

- explicit
- short
- operational
- no vague general advice
