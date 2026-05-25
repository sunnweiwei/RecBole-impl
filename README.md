# RecBole JAX Migration Workspace

This workspace contains a stripped reference copy of RecBole under
`upstream/RecBole/`. Your task is to implement a complete JAX-native replacement
package at the root-level `recbole/` package.

## Goal

Build a drop-in RecBole replacement whose public behavior matches the reference
package, but whose implementation runs on JAX.

Existing RecBole-facing user code should keep working: imports, model lookup,
config keys, datasets, dataloaders, samplers, models, losses, layers, trainers,
evaluators, quick-start APIs, checkpointing, and end-to-end workflows must keep
the same semantics.

Use `upstream/RecBole/` as the source of truth while developing. Do not depend on
that upstream package at runtime.

## Required Architecture

Implement RecBole itself in JAX.

The final package should have the same public module layout as the reference,
with real JAX implementations under the corresponding RecBole modules:

```text
recbole/
  config/
  data/
    interaction.py
    dataset/
    dataloader/
  sampler/
  model/
    abstract_recommender.py
    loss.py
    layers.py
    general_recommender/
    sequential_recommender/
    context_aware_recommender/
    knowledge_aware_recommender/
    exlib_recommender/
  trainer/
  evaluator/
  quick_start/
  utils/
  jax_backend/                 # optional shared JAX helpers
```

Each reference model should have a corresponding model module in the matching
`recbole/model/` subpackage. That module should own the model's JAX trainable
state, initialization, forward pass, prediction path, full-sort path, loss
computation, and configuration-dependent branches.

Shared helper code is encouraged for common JAX layers, sparse operations,
random keys, PyTree state, initialization, masking, metrics, checkpointing, and
Optax integration. The model behavior itself should remain readable as
model-specific JAX code.

## What Not To Build

Do not make the migration primarily a fake PyTorch runtime.

The final solution should not depend on a broad compatibility facade such as a
mock `torch` package, a rewritten `nn.Module` hierarchy, an import hook, a
monkey-patched framework module, or a graph interpreter that runs mostly
unchanged PyTorch-style code.

Small adapters are allowed only at narrow RecBole boundaries, for example to
load legacy checkpoints or normalize user input containers. They must not be the
main model, data, trainer, evaluator, optimizer, autograd, or tensor execution
path.

## JAX Requirements

- Core model computation, losses, gradients, and optimizer updates should use
  JAX arrays and JAX-compatible functions.
- Trainable state should be represented as JAX PyTrees and exposed through a
  stable model-state interface such as `model.params`.
- Training should work with `jax.grad` / `jax.value_and_grad` and should be
  compatible with `jax.jit` where practical.
- Optimizers, learning-rate scheduling, gradient clipping, save/load, and resume
  behavior should preserve the original training semantics.
- Normal model execution must not require PyTorch to be installed.

## Parity Requirements

Equivalence means more than "the code runs." The JAX package should match the
reference behavior for:

- every model exposed by the reference package;
- every relevant model configuration option;
- `predict`, `full_sort_predict`, loss values, and gradients on deterministic
  inputs;
- shared layers, loss functions, metrics, samplers, and data utilities;
- dataloader fields, shapes, order, sampling behavior, last partial batches, and
  mode transitions;
- deterministic training trajectories when initialization, data, and randomness
  are controlled;
- checkpoint save, reload, evaluation, and continued training;
- sequential, context-aware, knowledge-aware, graph-based, traditional, and
  multi-stage training model families.

Preserve details such as padding semantics, masks, train/eval behavior,
negative sampling, dataloader state, random seeds, sparse graph normalization,
and checkpoint state explicitly.

## Runtime Success Criterion

The migrated package is successful only if normal RecBole usage works in a
Python environment where PyTorch is not installed.

A user should be able to import `recbole`, build datasets and dataloaders,
instantiate every supported model, run prediction, compute losses and gradients,
train, evaluate, save checkpoints, and reload checkpoints using JAX-backed code
paths.

## Environment

Work inside this workspace. If you need a Python environment, create a local
virtual environment here:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies into this workspace-local environment and keep development
focused on the files under this workspace.

## Validation Standard

An external contract test suite will compare this package against the reference
RecBole behavior. Passing those tests should mean that a user can replace the
original package with this JAX implementation without changing RecBole-facing
code.

Prefer fixing implementation details over relaxing tolerances. Numerical
tolerances are only for legitimate floating-point differences, not for missing
features, changed formulas, altered masks, skipped configuration paths, or
incomplete training/checkpoint behavior.
