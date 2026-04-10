# Development Setup

This repository targets Python and PyTorch with a lightweight research workflow.

## Recommended Setup

1. Create and activate a virtual environment.
2. Install PyTorch for your platform.
3. Install testing tools such as `pytest`.
4. Add any profiling or experiment utilities as needed.

## Minimal Example

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytest
```

## Editable Install

TODO: confirm whether the project should ship a `pyproject.toml` immediately or keep setup fully ad hoc during the earliest prototype stage.

Until then, examples can use:

```bash
export PYTHONPATH=src
```

## Development Practices

- keep modules small and explicit
- prefer direct tensor-shape tests over elaborate fixtures
- use `PLANS.md` for non-trivial architecture and experiment changes

## Tooling Notes

TODO: confirm linting and formatting choices. Suggested minimal baseline: `ruff` plus `pytest`.

TODO: confirm whether experiment tracking should begin with plain files, TensorBoard, or a hosted system.
