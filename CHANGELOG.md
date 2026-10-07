# Changelog

## 0.3.2 (unreleased)

- Clarify installation, runtime ownership and parameter defaults; make README
  examples self-contained and report asynchronous failures.
- Move development instructions out of the package README.

## 0.3.1 — 2026-10-06

- Require Polygrad >=0.7.0; qualify native, Wasm and Python interchange against 0.7.0.

## 0.3.0 (local candidate)

- Migrate MLP, TabM and NAM to Polygrad 0.6 Model in a shared estimator implementation.
- Store versioned @2 Model bundles; reject legacy Instance artifacts explicitly.
- Own or borrow runtime contexts explicitly, restore best validation state, and reject unsupported training remainders.
- Align core dependency with 0.3 and add JS/Python/native/WASM migration checks.


## [0.2.0] - 2026-03-10

### Fixed

- Use the public `polygrad` runtime API instead of package-internal subpaths
- Keep `polygrad` runtime options out of saved model params and bundles
- Depend on the CommonJS `@wlearn/core` release
- Add package homepage and GitHub issue metadata

### Added

- Unified model classes: `MLPModel`, `TabMModel`, `NAMModel` via `createModelClass`
- Unified classes accept `task` parameter and auto-detect from labels
- Split classes (`MLPClassifier`, `MLPRegressor`, etc.) remain exported for explicit task-specific imports

## [0.1.0] - 2026-03-08

### Added

- MLPClassifier and MLPRegressor: configurable hidden sizes, activations (relu, gelu,
  silu), optimizers (SGD, Adam), mini-batch training, early stopping with
  validation_fraction and patience.
- TabMClassifier and TabMRegressor: parameter-efficient MLP ensembling via
  BatchEnsemble adapters (rank-1 weight perturbations). Configurable n_ensemble
  (default 32). Gorishniy et al. (2024), arXiv:2410.24210 (ICLR 2025).
- NAMClassifier and NAMRegressor: Neural Additive Models with per-feature MLPs.
  ExU (Exponential Unit) activation for sharp shape functions. Agarwal et al.
  (2021), arXiv:2004.13912 (NeurIPS 2021).
- Save/load via wlearn bundle format (IR + safetensors weights).
- Cross-language parity tests (JS predictions match Python polygrad Instance).
- 69 JS tests (MLP 32, TabM 18, NAM 19).
