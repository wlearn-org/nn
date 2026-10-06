# @wlearn/nn

Neural tabular models for [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)), powered by [polygrad](https://github.com/polygrad/polygrad).

## Models

- **MLPModel** -- Multi-layer perceptron with configurable hidden sizes, activations (relu, gelu, silu), optimizers (SGD, Adam), mini-batch training, and early stopping.
- **TabMModel** -- Parameter-efficient MLP ensembling via BatchEnsemble adapters. One model produces k implicit predictions with rank-1 weight perturbations. ICLR 2025.
- **NAMModel** -- Neural Additive Models. One small MLP per feature, summed for interpretable per-feature shape functions. Supports ExU activation. NeurIPS 2021.

All unified classes accept `task: 'classification'` or `task: 'regression'` and auto-detect from labels if omitted. Split classes (`MLPClassifier`, `MLPRegressor`, etc.) are available for explicit task-specific imports.

## Installation

```
npm install @wlearn/nn
```

Requires Polygrad **>=0.7.0**; tested with 0.7.0. npm installs this peer dependency automatically.
A caller-owned runtime can be reused across fits:

```js
const pg = await require('polygrad').create({ core: 'native', device: 'cpu' })
const { MLPRegressor } = require('@wlearn/nn')
const model = await MLPRegressor.create({ polygrad: pg, hidden_sizes: [32] })
```

## Usage

```js
const { readFileSync, writeFileSync } = require('fs')
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
writeFileSync('tabm.wlrn', model.save())
const restored = await TabMModel.load(readFileSync('tabm.wlrn'))
```

## API

All models follow the wlearn estimator contract:

- `static async create(params)` -- async construction (WASM init)
- `fit(X, y)` -- train on data
- `predict(X)` -- predict labels
- `predictProba(X)` -- predict class probabilities (classifiers)
- `score(X, y)` -- evaluate (accuracy for classification, R2 for regression)
- `save()` -- serialize to wlearn bundle bytes
- `static async load(bytes)` -- restore from a wlearn bundle
- `getParams()` / `setParams(p)` -- read or update hyperparameters
- `static defaultSearchSpace()` -- AutoML search-space IR
- `isFitted` -- fitted-state boolean
- `classes` / `nrClass` -- classifier label metadata
- `capabilities` -- supported estimator capabilities
- `dispose()` -- deterministic cleanup for long-running loops

## Tests

```
npm test
```

## References

- Gorishniy et al. (2024). "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling." arXiv:2410.24210 (ICLR 2025).
- Agarwal et al. (2021). "Neural Additive Models." arXiv:2004.13912 (NeurIPS 2021).

## License

Apache-2.0

## Polygrad 0.6 migration

All six estimators share lifecycle, training and artifact handling. A supplied
`polygrad` runtime is borrowed; otherwise the estimator creates and disposes its
own runtime. Native and synchronous WASM execution are supported. Asynchronous
WebGPU runtimes are rejected by the synchronous NN fitting API.

MLP and NAM support fixed-size batches. Short datasets use their row count as
the effective batch size. Training rows remaining after the validation split
must divide evenly into that batch size; otherwise `fit()` raises a validation
error before replacing a fitted model. TabM currently requires batch size 1 in
Polygrad's builder. The default AutoML search uses batch size 1 until variable
batch support is available upstream. Validation and prediction include their
last partial batch. Early stopping restores the best validation weights even
when the epoch budget ends before patience expires.

New WLRN type IDs end in `@2` and contain one Polygrad Model bundle. Legacy NN
`@1` Instance artifacts require retraining; their loaders report that explicitly.
`MLPRegressor.load(bytes, { polygrad: pg })` (and the other task-specific classes)
can reuse a caller's runtime. Runtime options are not serialized.

`npm run test:migration` checks native or WASM artifacts in both language
directions, pipeline composition, batching and ownership. Set `WLEARN_PYTHON`
to a Python executable with local `polygrad` and `wlearn` on `PYTHONPATH`,
`POLY_LIB` to the local shared library, and optionally `WLEARN_NN_TEST_CORE=wasm`.
The test writes run-owned temporary artifacts, not golden fixtures.
