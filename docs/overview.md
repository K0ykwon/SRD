# Overview

SRD (Segment Refresh Decoder) is a decoder architecture for long-context generation where long-range interaction happens only at scheduled refresh states.

## Core Thesis

If the model is forced to route distant information through explicit refresh states, it may preserve useful long-context behavior while reducing the cost and state footprint associated with globally connected decoding.

## Core Components

- local-only token computation
- periodic refresh positions
- a shared long-memory bank
- upper-layer use of refreshed states
- a refresh sufficiency training objective

## Intended Benefits

- clearer control over where non-local interaction occurs
- more interpretable long-range routing
- improved efficiency relative to broader global-access decoders
- a cleaner experimental handle on throughput-per-memory tradeoffs

## Prototype Scope

The initial codebase is focused on:

- architectural prototyping
- small-scale experiments
- correctness checks on routing behavior
- lightweight benchmarking hooks
