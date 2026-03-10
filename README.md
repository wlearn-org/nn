# @wlearn/nn

Neural tabular models for wlearn, powered by [polygrad](https://github.com/polygrad/polygrad).

## Models

- **MLPModel** -- Multi-layer perceptron with configurable hidden sizes, activations (relu, gelu, silu), optimizers (SGD, Adam), mini-batch training, and early stopping.
- **TabMModel** -- Parameter-efficient MLP ensembling via BatchEnsemble adapters. One model produces k implicit predictions with rank-1 weight perturbations. ICLR 2025.
- **NAMModel** -- Neural Additive Models. One small MLP per feature, summed for interpretable per-feature shape functions. Supports ExU activation. NeurIPS 2021.

All unified classes accept `task: 'classification'` or `task: 'regression'` and auto-detect from labels if omitted. Split classes (`MLPClassifier`, `MLPRegressor`, etc.) are also exported for backward compatibility.

## Installation

```
npm install @wlearn/nn
```

Requires `polygrad` as a peer dependency:

```
npm install polygrad
```

## Usage

```js
const { TabMModel } = require('@wlearn/nn')

const model = await TabMModel.create({
  task: 'classification',  // or 'regression'; auto-detected from labels if omitted
  hidden_sizes: [128],
  activation: 'relu',
  n_ensemble: 32,
  lr: 0.005,
  epochs: 100,
  optimizer: 'adam'
})

model.fit(X_train, y_train)
const predictions = model.predict(X_test)
const score = model.score(X_test, y_test)

// Save / load via wlearn bundle format
const bytes = model.save()
model.dispose()
```

## API

All models follow the wlearn estimator contract:

- `static async create(params)` -- async construction (WASM init)
- `fit(X, y)` -- train on data
- `predict(X)` -- predict labels
- `predictProba(X)` -- predict class probabilities (classifiers)
- `score(X, y)` -- evaluate (accuracy for classification, R2 for regression)
- `save()` -- serialize to wlearn bundle (Uint8Array)
- `dispose()` -- free resources

## Tests

```
npm test
```

69 tests: MLP (32), TabM (18), NAM (19).

## References

- Gorishniy et al. (2024). "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling." arXiv:2410.24210 (ICLR 2025).
- Agarwal et al. (2021). "Neural Additive Models." arXiv:2004.13912 (NeurIPS 2021).

## License

Apache-2.0
