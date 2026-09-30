'use strict'

const {
  normalizeX, encodeBundle, decodeBundle, register,
  ValidationError, BundleError, BackendError, DisposedError, NotFittedError
} = require('@wlearn/core')
const { acquireRuntime } = require('./polygrad.js')

function matrix(X) {
  const m = normalizeX(X)
  const data = Float32Array.from(m.data)
  if (!data.every(Number.isFinite)) throw new ValidationError('X must contain finite float32 values')
  return { ...m, data }
}

function positive(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) throw new ValidationError(`${name} must be a positive integer`)
  return value
}

function probabilities(values, width) {
  const result = new Float64Array(values.length)
  for (let start = 0; start < values.length; start += width) {
    let max = -Infinity, total = 0
    for (let j = 0; j < width; j++) max = Math.max(max, values[start + j])
    for (let j = 0; j < width; j++) total += (result[start + j] = Math.exp(values[start + j] - max))
    for (let j = 0; j < width; j++) result[start + j] /= total
  }
  return result
}

// Padding is inference-only: these tabular families have independent rows.
function forward(model, data, rows, cols, batch, outputs) {
  const result = new Float64Array(rows * outputs)
  const input = new Float32Array(batch * cols)
  for (let start = 0; start < rows; start += batch) {
    const count = Math.min(batch, rows - start)
    input.fill(0)
    input.set(data.subarray(start * cols, (start + count) * cols))
    const values = model.forward({ x: input }).output
    if (values.length !== batch * outputs || !values.every(Number.isFinite)) {
      throw new BackendError('Polygrad returned invalid predictions')
    }
    result.set(values.subarray(0, count * outputs), start * outputs)
  }
  return result
}

function typeId(family, classifier, version = 2) {
  return `wlearn.nn.${family.toLowerCase()}.${classifier ? 'classifier' : 'regressor'}@${version}`
}

class NeuralEstimator {
  constructor(params, lease) {
    const { polygrad, ...stored } = params
    this._params = stored
    this._lease = lease
    this._model = null
    this._disposed = false
    this._classes = []
  }

  static async create(params = {}) {
    return new this(params, await acquireRuntime(params.polygrad))
  }

  _check(fitted = true) {
    if (this._disposed) throw new DisposedError()
    if (fitted && !this._model) throw new NotFittedError()
  }

  fit(X, y) {
    this._check(false)
    const { rows, cols, data } = matrix(X)
    const labels = Array.from(y)
    if (labels.length !== rows || !labels.every(Number.isFinite)) throw new ValidationError('y must have one finite value per row')
    const p = this._params
    const epochs = positive(p.epochs ?? 100, 'epochs')
    const patience = positive(p.patience ?? 10, 'patience')
    const requestedBatch = positive(p.batch_size ?? 1, 'batch_size')
    const fraction = p.validation_fraction ?? 0
    const lr = p.lr ?? 0.01
    if (!Number.isFinite(lr) || lr <= 0) throw new ValidationError('lr must be positive and finite')
    if (!Number.isFinite(fraction) || fraction < 0 || fraction >= 1) throw new ValidationError('validation_fraction must be in [0, 1)')
    const nVal = fraction ? Math.max(1, Math.floor(rows * fraction)) : 0
    const nTrain = rows - nVal
    if (!nTrain) throw new ValidationError('Validation split leaves no training rows')
    const batch = Math.min(requestedBatch, nTrain)
    if (nTrain % batch) throw new ValidationError('Polygrad tabular training requires complete fixed-size batches; choose batch_size dividing the training rows (or 1)')
    if (this.constructor.family === 'TabM' && batch !== 1) throw new ValidationError('Polygrad 0.6 TabM currently requires batch_size=1')
    const optimizer = p.optimizer ?? 'adam'
    if (!['adam', 'sgd'].includes(optimizer)) throw new ValidationError('optimizer must be adam or sgd')
    const classes = this.constructor.classifier ? [...new Set(labels)].sort((a, b) => a - b) : []
    if (this.constructor.classifier && classes.length < 2) throw new ValidationError('Classification requires at least two classes')
    const outputs = classes.length || 1
    const target = new Float32Array(rows * outputs)
    for (let i = 0; i < rows; i++) {
      if (classes.length) target[i * outputs + classes.indexOf(labels[i])] = 1
      else target[i] = labels[i]
    }
    if (!target.every(Number.isFinite)) throw new ValidationError('y exceeds float32 range')
    const hidden = p.hidden_sizes ?? p.hiddenSizes ?? [64]
    const spec = {
      activation: p.activation ?? (this.constructor.family === 'NAM' ? 'exu' : 'relu'),
      loss: classes.length ? 'cross_entropy' : 'mse', batch_size: batch, seed: p.seed ?? 42
    }
    if (this.constructor.family === 'NAM') Object.assign(spec, { n_features: cols, hidden_sizes: hidden, n_outputs: outputs })
    else Object.assign(spec, { layers: [cols, ...hidden, outputs], n_ensemble: p.n_ensemble ?? 32 })
    let model = null
    try {
      model = this._lease.runtime.models[this.constructor.family](spec)
      model.setOptimizer(optimizer, lr)
      const bx = new Float32Array(batch * cols), by = new Float32Array(batch * outputs)
      let seed = p.seed ?? 42, bestLoss = Infinity, bestWeights = null, stale = 0
      for (let epoch = 0; epoch < epochs; epoch++) {
        const order = Array.from({ length: nTrain }, (_, i) => i)
        for (let i = nTrain - 1; i > 0; i--) {
          seed = (seed * 1103515245 + 12345) & 0x7fffffff
          const j = Math.min(i, Math.floor(seed / 0x7fffffff * (i + 1)))
          const previous = order[i]
          order[i] = order[j]
          order[j] = previous
        }
        for (let start = 0; start < nTrain; start += batch) {
          for (let i = 0; i < batch; i++) {
            const row = order[start + i]
            bx.set(data.subarray(row * cols, (row + 1) * cols), i * cols)
            by.set(target.subarray(row * outputs, (row + 1) * outputs), i * outputs)
          }
          if (!Number.isFinite(model.trainStep({ x: bx, y: by }))) throw new BackendError('Non-finite training loss')
        }
        if (nVal) {
          let values = forward(model, data.subarray(nTrain * cols), nVal, cols, batch, outputs)
          if (classes.length) values = probabilities(values, outputs)
          let loss = 0
          for (let i = 0; i < nVal; i++) {
            loss += classes.length ? -Math.log(Math.max(1e-15, values[i * outputs + classes.indexOf(labels[nTrain + i])]))
              : (values[i] - labels[nTrain + i]) ** 2
          }
          loss /= nVal
          if (loss < bestLoss) {
            bestLoss = loss
            bestWeights = model.exportWeights({ includeOptimizer: false })
            stale = 0
          } else if (++stale >= patience) break
        }
      }
      // Restore the best validation state even when the epoch budget ends first.
      if (bestWeights) model.importWeights(bestWeights)
    } catch (error) {
      if (model) model.dispose()
      throw error
    }
    if (this._model) this._model.dispose()
    Object.assign(this, { _model: model, _classes: classes, _nFeatures: cols, _batchSize: batch })
    return this
  }

  _predict(X) {
    this._check()
    const m = matrix(X)
    if (m.cols !== this._nFeatures) throw new ValidationError(`Expected ${this._nFeatures} features, got ${m.cols}`)
    return forward(this._model, m.data, m.rows, m.cols, this._batchSize, this._classes.length || 1)
  }

  predict(X) {
    const values = this._predict(X)
    if (!this.constructor.classifier) return values
    const width = this._classes.length
    return Float64Array.from({ length: values.length / width }, (_, i) => {
      let best = 0
      for (let j = 1; j < width; j++) if (values[i * width + j] > values[i * width + best]) best = j
      return this._classes[best]
    })
  }

  score(X, y) {
    const pred = this.predict(X)
    if (y.length !== pred.length) throw new ValidationError('y length does not match predictions')
    if (this.constructor.classifier) return pred.reduce((sum, v, i) => sum + (v === y[i]), 0) / pred.length
    const mean = Array.from(y).reduce((a, b) => a + b, 0) / y.length
    const total = Array.from(y).reduce((sum, v) => sum + (v - mean) ** 2, 0)
    return total ? 1 - pred.reduce((sum, v, i) => sum + (v - y[i]) ** 2, 0) / total : 0
  }

  save() {
    this._check()
    return encodeBundle({
      typeId: typeId(this.constructor.family, this.constructor.classifier), params: this.getParams(),
      metadata: { nFeatures: this._nFeatures, batchSize: this._batchSize, nrClass: this._classes.length, classes: this._classes }
    }, [{ id: 'model', data: this._model.saveBundle({ includeOptimizer: false }) }])
  }

  static async load(bytes, options = {}) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return this._fromBundle(manifest, toc, blobs, options.polygrad)
  }

  static async _fromBundle(manifest, toc, blobs, polygrad) {
    if (manifest.typeId !== typeId(this.family, this.classifier)) throw new BundleError('NN requires a @2 Model bundle; legacy @1 Instance bundles must be retrained')
    const entry = toc.find(e => e.id === 'model'), meta = manifest.metadata || {}
    if (!entry || !Number.isSafeInteger(meta.nFeatures) || meta.nFeatures < 1 ||
        !Number.isSafeInteger(meta.batchSize) || meta.batchSize < 1 || !Array.isArray(meta.classes) ||
        !meta.classes.every(Number.isFinite) || new Set(meta.classes).size !== meta.classes.length ||
        (!this.classifier && meta.classes.length) || meta.nrClass !== meta.classes.length || (this.classifier && meta.nrClass < 2)) throw new BundleError('Invalid NN Model bundle metadata')
    const estimator = await this.create({ ...manifest.params, polygrad })
    try {
      estimator._model = estimator._lease.runtime.Model.fromBundle(blobs.subarray(entry.offset, entry.offset + entry.length))
      const bindings = estimator._model.bindings()
      for (const [name, shape] of [['x', [meta.batchSize, meta.nFeatures]], ['output', [meta.batchSize, meta.nrClass || 1]]]) {
        const b = bindings.find(b => b.name === name)
        if (!b || b.dtype !== 'float32' || JSON.stringify(b.shape) !== JSON.stringify(shape)) throw new BundleError('NN metadata does not match the Polygrad signature')
      }
      Object.assign(estimator, { _nFeatures: meta.nFeatures, _batchSize: meta.batchSize, _classes: [...meta.classes] })
      return estimator
    } catch (error) {
      estimator.dispose()
      throw error
    }
  }

  dispose() {
    if (this._disposed) return
    this._disposed = true
    if (this._model) this._model.dispose()
    this._model = null
    if (this._lease.owned) this._lease.runtime.dispose()
  }

  getParams() { return structuredClone(this._params) }
  setParams(p) {
    this._check(false)
    const { polygrad, ...rest } = p
    Object.assign(this._params, rest)
    return this
  }
  get isFitted() { return Boolean(this._model) && !this._disposed }
  get capabilities() {
    return { classifier: this.constructor.classifier, regressor: !this.constructor.classifier,
      predictProba: this.constructor.classifier, decisionFunction: false, sampleWeight: false, csr: false, earlyStopping: true }
  }
}

function createEstimator(family, classifier, searchSpace) {
  class Estimator extends NeuralEstimator {
    static family = family
    static classifier = classifier
    static defaultSearchSpace() { return structuredClone(searchSpace) }
  }
  Object.defineProperty(Estimator, 'name', { value: family + (classifier ? 'Classifier' : 'Regressor') })
  if (classifier) Object.defineProperties(Estimator.prototype, {
    predictProba: { value(X) { return probabilities(this._predict(X), this._classes.length) } },
    classes: { get() { return [...this._classes] } }, nrClass: { get() { return this._classes.length } }
  })
  register(typeId(family, classifier), (m, t, b) => Estimator._fromBundle(m, t, b))
  register(typeId(family, classifier, 1), () => { throw new BundleError('Legacy NN @1 Instance bundles require retraining for Polygrad 0.6') })
  return Estimator
}

module.exports = { createEstimator }
