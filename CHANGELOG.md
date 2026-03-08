# Changelog

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
