# @wlearn/nn

Neural network models for tabular data in JavaScript: MLP, TabM and NAM,
trained with [Polygrad](https://github.com/polygrad/polygrad). Part of
[wlearn](https://github.com/wlearn-org/wlearn) ([wlearn.org](https://wlearn.org)).

## Models

- **MLPModel**: multi-layer perceptron with configurable hidden sizes,
  activations (relu, gelu, silu), SGD or Adam, mini-batches and early stopping.
- **TabMModel**: one network that acts as an ensemble of k networks through
  rank-1 weight adapters (BatchEnsemble). ICLR 2025.
- **NAMModel**: Neural Additive Model. One small network per feature, summed, so
  each feature's effect can be plotted. Supports ExU activations. NeurIPS 2021.

The unified classes take `task: 'classification'` or `task: 'regression'`, or
detect the task from the labels. Task-specific classes are also exported:
`MLPClassifier`, `MLPRegressor`, `TabMClassifier`, `TabMRegressor`,
`NAMClassifier` and `NAMRegressor`.

## Install

```sh
npm install @wlearn/nn
```

`polygrad` (0.7 or newer) is a peer dependency; npm 7 and newer install it
automatically.

## Quick start

```js
const { TabMModel } = require('@wlearn/nn')

async function main() {
  // Two classes separated by the sign of x0 + x1
  const X = Array.from({ length: 64 }, (_, i) => [Math.sin(i * 1.7), Math.cos(i * 2.3)])
  const y = X.map(([a, b]) => (a + b > 0 ? 1 : 0))

  const model = await TabMModel.create({
    task: 'classification', hidden_sizes: [32], n_ensemble: 8, lr: 0.01, epochs: 30
  })
  try {
    model.fit(X, y)
    console.log(model.score(X, y))  // accuracy

    const restored = await TabMModel.load(model.save())
    console.log(restored.predict(X.slice(0, 3)))
    restored.dispose()
  } finally {
    model.dispose()
  }
}

main().catch(error => { console.error(error); process.exitCode = 1 })
```

## Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `task` | detected | `'classification'` or `'regression'` |
| `hidden_sizes` | `[64]` | Hidden layer widths |
| `activation` | `'relu'` (NAM: `'exu'`) | `'relu'`, `'gelu'`, `'silu'`; NAM also `'exu'` |
| `optimizer` | `'adam'` | `'adam'` or `'sgd'` |
| `lr` | 0.01 | Learning rate |
| `epochs` | 100 | Maximum number of epochs |
| `batch_size` | 1 | MLP and NAM only; TabM uses 1 |
| `validation_fraction` | 0 | Above 0, enables early stopping |
| `patience` | 10 | Epochs without improvement before stopping |
| `n_ensemble` | 32 | TabM only |
| `seed` | 42 | Initialization and shuffling |
| `polygrad` | none | Runtime or runtime options, see below |

## API

All models follow the wlearn estimator interface:

- `static async create(params)` creates a model.
- `fit(X, y)` trains. `X` is an array of rows or a wlearn `DenseMatrix`.
- `predict(X)` returns predictions; classifiers add `predictProba(X)`,
  `classes` and `nrClass`.
- `score(X, y)` returns accuracy for classification and R2 for regression.
- `save()` returns the model as bytes; `static async load(bytes, { polygrad })`
  restores it.
- `getParams()` and `setParams(params)` read and change parameters.
- `isFitted`, `capabilities` and `static defaultSearchSpace()` (for wlearn
  AutoML).
- `dispose()` releases resources. Call it in long-running applications.

## Sharing a Polygrad runtime

By default each model creates and disposes its own Polygrad runtime. To reuse
one runtime, and its compiled kernels, across models:

```js
const polygrad = require('polygrad')
const { MLPRegressor } = require('@wlearn/nn')

async function main() {
  const X = Array.from({ length: 32 }, (_, i) => [i / 32, Math.sin(i / 4)])
  const y = X.map(([a, b]) => 3 * a - b)

  const pg = await polygrad.create({ core: 'native', device: 'cpu' })
  const model = await MLPRegressor.create({ polygrad: pg, hidden_sizes: [32], epochs: 20 })
  try {
    model.fit(X, y)
    console.log(model.predict(X.slice(0, 2)))
  } finally {
    model.dispose()
    await pg.dispose()
  }
}

main().catch(error => { console.error(error); process.exitCode = 1 })
```

A runtime you pass is never disposed by the model. Native and synchronous
WebAssembly runtimes are supported. Asynchronous WebGPU runtimes are rejected,
because `fit` is synchronous.

## Notes and limitations

- With `batch_size` above 1, the training rows left after the validation split
  must divide evenly into batches; otherwise `fit` raises a validation error and
  keeps the previous model. Prediction accepts any number of rows.
- Early stopping restores the weights from the best validation epoch.
- wlearn AutoML searches with `batch_size` 1.
- Models saved by `@wlearn/nn` before 0.3.0 use an older format and must be
  retrained; loading them raises an error that says so.
- The same models are available in Python as `wlearn.nn`
  (`pip install 'wlearn[nn]'`), and saved files load in both languages.

## References

- Gorishniy et al. (2024). "TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling." arXiv:2410.24210 (ICLR 2025).
- Agarwal et al. (2021). "Neural Additive Models." arXiv:2004.13912
  (NeurIPS 2021).

## License

Apache-2.0
