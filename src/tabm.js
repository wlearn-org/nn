/**
 * TabMClassifier and TabMRegressor for wlearn using polygrad Instance backend.
 *
 * TabM: MLP with BatchEnsemble adapters (per-member rank-1 weight perturbations).
 * For each linear layer with shared weight W, ensemble member i computes:
 *   l_i(x) = s_i * (W @ (r_i * x)) + b_i
 * Final prediction: mean over k ensemble members.
 *
 * Reference: Gorishniy et al. (2024), arXiv:2410.24210 (ICLR 2025)
 */

const {
  encodeBundle, decodeBundle, register,
  DisposedError, NotFittedError
} = require('@wlearn/core')

let _Instance = null
let _OPTIM_SGD = null
let _OPTIM_ADAM = null

function getPolygrad() {
  if (!_Instance) {
    const pg = require('polygrad/src/instance')
    _Instance = pg.Instance
    _OPTIM_SGD = pg.OPTIM_SGD
    _OPTIM_ADAM = pg.OPTIM_ADAM
  }
  return { Instance: _Instance, OPTIM_SGD: _OPTIM_SGD, OPTIM_ADAM: _OPTIM_ADAM }
}

function softmax(logits) {
  const max = Math.max(...logits)
  const exp = logits.map(v => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(v => v / sum)
}

function lcg(seed) {
  let s = seed
  return function () {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function shuffleIndices(n, rng) {
  const idx = new Int32Array(n)
  for (let i = 0; i < n; i++) idx[i] = i
  for (let i = n - 1; i > 0; i--) {
    const j = (rng() * (i + 1)) | 0
    const tmp = idx[i]
    idx[i] = idx[j]
    idx[j] = tmp
  }
  return idx
}

const _UNFITTED = Symbol('unfitted')

// ─── TabMClassifier ───────────────────────────────────────────────────

class TabMClassifier {
  #instance = null
  #params = {}
  #nrClass = 0
  #classes = []
  #nFeatures = 0
  #batchSize = 1
  #fitted = false
  #disposed = false

  constructor(instanceOrSentinel, params, nrClass, classes, nFeatures, batchSize) {
    if (instanceOrSentinel === _UNFITTED) {
      this.#instance = null
      this.#params = { ...params }
      this.#fitted = false
    } else {
      this.#instance = instanceOrSentinel
      this.#params = { ...params }
      this.#nrClass = nrClass || 0
      this.#classes = classes ? [...classes] : []
      this.#nFeatures = nFeatures || 0
      this.#batchSize = batchSize || 1
      this.#fitted = true
    }
  }

  static async create(params = {}) {
    return new TabMClassifier(_UNFITTED, params)
  }

  fit(X, y) {
    if (this.#disposed) throw new DisposedError('TabMClassifier has been disposed.')

    const { Instance, OPTIM_SGD, OPTIM_ADAM } = getPolygrad()

    const { rows, cols, data } = this.#normalizeX(X)
    const nFeatures = cols
    this.#nFeatures = nFeatures

    const yArr = Array.isArray(y) ? y : [...y]
    const unique = [...new Set(yArr)].sort((a, b) => a - b)
    this.#classes = unique.map(Number)
    this.#nrClass = unique.length
    const classMap = new Map(unique.map((c, i) => [Number(c), i]))

    const yOnehot = new Float32Array(rows * this.#nrClass)
    for (let i = 0; i < rows; i++) {
      yOnehot[i * this.#nrClass + classMap.get(Number(yArr[i]))] = 1.0
    }

    // Early stopping: split train/val
    const valFrac = this.#params.validation_fraction || 0
    const patience = this.#params.patience || 10
    let nTrain = rows
    let nVal = 0
    let trainData = data, trainY = yOnehot
    let valData = null, valY = null
    if (valFrac > 0 && valFrac < 1) {
      nVal = Math.max(1, Math.floor(rows * valFrac))
      nTrain = rows - nVal
      trainData = data.subarray(0, nTrain * nFeatures)
      trainY = yOnehot.subarray(0, nTrain * this.#nrClass)
      valData = data.subarray(nTrain * nFeatures)
      valY = yOnehot.subarray(nTrain * this.#nrClass)
    }

    const hidden = this.#params.hidden_sizes || this.#params.hiddenSizes || [64]
    const activation = this.#params.activation || 'relu'
    const nEnsemble = this.#params.n_ensemble || 32
    const seed = this.#params.seed ?? 42
    const lr = this.#params.lr || 0.01
    const epochs = this.#params.epochs || 100
    const optimizer = this.#params.optimizer || 'adam'
    // NOTE: polygrad's backward pass currently only supports batch_size=1.
    const batchSize = this.#params.batch_size || 1

    const layers = [nFeatures, ...hidden, this.#nrClass]
    const spec = {
      layers, activation,
      loss: 'cross_entropy', batch_size: batchSize,
      seed, n_ensemble: nEnsemble
    }

    if (this.#instance) {
      this.#instance.free()
    }
    this.#instance = Instance.tabm(spec)

    const optimKind = optimizer === 'adam' ? OPTIM_ADAM : OPTIM_SGD
    this.#instance.setOptimizer(optimKind, lr)
    this.#batchSize = batchSize

    const nBatches = Math.floor(nTrain / batchSize)
    const rng = lcg(seed)
    const batchX = new Float32Array(batchSize * nFeatures)
    const batchY = new Float32Array(batchSize * this.#nrClass)
    let bestValLoss = Infinity
    let staleEpochs = 0
    let bestWeights = null

    for (let epoch = 0; epoch < epochs; epoch++) {
      const order = shuffleIndices(nTrain, rng)
      for (let b = 0; b < nBatches; b++) {
        for (let i = 0; i < batchSize; i++) {
          const si = order[b * batchSize + i]
          batchX.set(trainData.subarray(si * nFeatures, (si + 1) * nFeatures), i * nFeatures)
          batchY.set(trainY.subarray(si * this.#nrClass, (si + 1) * this.#nrClass), i * this.#nrClass)
        }
        this.#instance.trainStep({ x: batchX, y: batchY })
      }

      if (valData && nVal > 0) {
        let valLoss = 0
        const valBatchX = new Float32Array(batchSize * nFeatures)
        const nValBatches = Math.floor(nVal / batchSize)
        for (let b = 0; b < nValBatches; b++) {
          for (let i = 0; i < batchSize; i++) {
            const si = b * batchSize + i
            valBatchX.set(valData.subarray(si * nFeatures, (si + 1) * nFeatures), i * nFeatures)
          }
          const out = this.#instance.forward({ x: valBatchX })
          const logits = out.output
          for (let i = 0; i < batchSize; i++) {
            const row = []
            for (let c = 0; c < this.#nrClass; c++) row.push(logits[i * this.#nrClass + c])
            const probs = softmax(row)
            const si = (b * batchSize + i) * this.#nrClass
            let tc = 0
            for (let c = 0; c < this.#nrClass; c++) {
              if (valY[si + c] > 0) { tc = c; break }
            }
            valLoss -= Math.log(Math.max(probs[tc], 1e-7))
          }
        }
        valLoss /= Math.max(1, nValBatches * batchSize)
        if (valLoss < bestValLoss) {
          bestValLoss = valLoss
          staleEpochs = 0
          bestWeights = this.#instance.exportWeights()
        } else {
          staleEpochs++
          if (staleEpochs >= patience) {
            if (bestWeights) this.#instance.importWeights(bestWeights)
            break
          }
        }
      }
    }

    this.#fitted = true
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const { rows, cols, data } = this.#normalizeX(X)
    const result = new Float64Array(rows)
    const bs = this.#batchSize
    const nc = this.#nrClass
    const batchX = new Float32Array(bs * cols)

    for (let b = 0; b < rows; b += bs) {
      const actual = Math.min(bs, rows - b)
      batchX.fill(0)
      for (let i = 0; i < actual; i++) {
        batchX.set(data.subarray((b + i) * cols, (b + i + 1) * cols), i * cols)
      }
      const out = this.#instance.forward({ x: batchX })
      const logits = out.output
      for (let i = 0; i < actual; i++) {
        let maxIdx = 0
        for (let j = 1; j < nc; j++) {
          if (logits[i * nc + j] > logits[i * nc + maxIdx]) maxIdx = j
        }
        result[b + i] = maxIdx < this.#classes.length ? this.#classes[maxIdx] : maxIdx
      }
    }

    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    const { rows, cols, data } = this.#normalizeX(X)
    const nc = this.#nrClass
    const result = new Float64Array(rows * nc)
    const bs = this.#batchSize
    const batchX = new Float32Array(bs * cols)

    for (let b = 0; b < rows; b += bs) {
      const actual = Math.min(bs, rows - b)
      batchX.fill(0)
      for (let i = 0; i < actual; i++) {
        batchX.set(data.subarray((b + i) * cols, (b + i + 1) * cols), i * cols)
      }
      const out = this.#instance.forward({ x: batchX })
      const logits = out.output
      for (let i = 0; i < actual; i++) {
        const row = []
        for (let j = 0; j < nc; j++) row.push(logits[i * nc + j])
        const probs = softmax(row)
        for (let j = 0; j < nc; j++) {
          result[(b + i) * nc + j] = probs[j]
        }
      }
    }

    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = Array.isArray(y) ? y : [...y]
    let correct = 0
    for (let i = 0; i < preds.length; i++) {
      if (preds[i] === Number(yArr[i])) correct++
    }
    return correct / preds.length
  }

  save() {
    this.#ensureFitted()
    const irBytes = this.#instance.exportIR()
    const wBytes = this.#instance.exportWeights()

    return encodeBundle(
      {
        typeId: 'wlearn.nn.tabm.classifier@1',
        params: this.getParams(),
        metadata: {
          nrClass: this.#nrClass,
          classes: this.#classes,
          nFeatures: this.#nFeatures,
          batchSize: this.#batchSize
        }
      },
      [
        { id: 'ir', data: new Uint8Array(irBytes) },
        { id: 'weights', data: new Uint8Array(wBytes) }
      ]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return TabMClassifier._fromBundle(manifest, toc, blobs)
  }

  static _fromBundle(manifest, toc, blobs) {
    const { Instance } = getPolygrad()

    const irEntry = toc.find(e => e.id === 'ir')
    const wEntry = toc.find(e => e.id === 'weights')
    if (!irEntry || !wEntry) throw new Error('Bundle missing "ir" or "weights" artifact')

    const irBytes = blobs.subarray(irEntry.offset, irEntry.offset + irEntry.length)
    const wBytes = blobs.subarray(wEntry.offset, wEntry.offset + wEntry.length)

    const instance = Instance.fromIR(irBytes, wBytes)
    const params = manifest.params || {}
    const meta = manifest.metadata || {}

    return new TabMClassifier(
      instance, params,
      meta.nrClass || 0,
      meta.classes,
      meta.nFeatures || 0,
      meta.batchSize || 1
    )
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    if (this.#instance) {
      this.#instance.free()
      this.#instance = null
    }
    this.#fitted = false
  }

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  get isFitted() {
    return this.#fitted && !this.#disposed
  }

  get classes() {
    return [...this.#classes]
  }

  get nrClass() {
    return this.#nrClass
  }

  get capabilities() {
    return {
      classifier: true,
      regressor: false,
      predictProba: true,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: true
    }
  }

  static defaultSearchSpace() {
    return {
      hidden_sizes: { type: 'categorical', values: [[64], [128], [64, 64], [128, 64]] },
      activation: { type: 'categorical', values: ['relu', 'gelu', 'silu'] },
      n_ensemble: { type: 'categorical', values: [4, 8, 16, 32] },
      lr: { type: 'log_uniform', low: 1e-4, high: 1e-1 },
      epochs: { type: 'int_uniform', low: 10, high: 200 },
      optimizer: { type: 'categorical', values: ['adam', 'sgd'] }
    }
  }

  #ensureFitted() {
    if (this.#disposed) throw new DisposedError('TabMClassifier has been disposed.')
    if (!this.#fitted) throw new NotFittedError('TabMClassifier is not fitted.')
  }

  #normalizeX(X) {
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { rows, cols, data }
    }
    if (X && X.data) {
      const data = X.data instanceof Float32Array ? X.data : new Float32Array(X.data)
      return { rows: X.rows, cols: X.cols, data }
    }
    throw new Error('X must be number[][] or { data, rows, cols }')
  }
}

// ─── TabMRegressor ────────────────────────────────────────────────────

class TabMRegressor {
  #instance = null
  #params = {}
  #nFeatures = 0
  #batchSize = 1
  #fitted = false
  #disposed = false

  constructor(instanceOrSentinel, params, nFeatures, batchSize) {
    if (instanceOrSentinel === _UNFITTED) {
      this.#instance = null
      this.#params = { ...params }
      this.#fitted = false
    } else {
      this.#instance = instanceOrSentinel
      this.#params = { ...params }
      this.#nFeatures = nFeatures || 0
      this.#batchSize = batchSize || 1
      this.#fitted = true
    }
  }

  static async create(params = {}) {
    return new TabMRegressor(_UNFITTED, params)
  }

  fit(X, y) {
    if (this.#disposed) throw new DisposedError('TabMRegressor has been disposed.')

    const { Instance, OPTIM_SGD, OPTIM_ADAM } = getPolygrad()

    const { rows, cols, data } = this.#normalizeX(X)
    const nFeatures = cols
    this.#nFeatures = nFeatures

    const yArr = y instanceof Float32Array ? y : new Float32Array(Array.isArray(y) ? y : [...y])
    const nOutputs = yArr.length / rows

    // Early stopping: split train/val
    const valFrac = this.#params.validation_fraction || 0
    const patience = this.#params.patience || 10
    let nTrain = rows
    let nVal = 0
    let trainData = data, trainY = yArr
    let valData = null, valY = null
    if (valFrac > 0 && valFrac < 1) {
      nVal = Math.max(1, Math.floor(rows * valFrac))
      nTrain = rows - nVal
      trainData = data.subarray(0, nTrain * nFeatures)
      trainY = yArr.subarray(0, nTrain * nOutputs)
      valData = data.subarray(nTrain * nFeatures)
      valY = yArr.subarray(nTrain * nOutputs)
    }

    const hidden = this.#params.hidden_sizes || this.#params.hiddenSizes || [64]
    const activation = this.#params.activation || 'relu'
    const nEnsemble = this.#params.n_ensemble || 32
    const seed = this.#params.seed ?? 42
    const lr = this.#params.lr || 0.01
    const epochs = this.#params.epochs || 100
    const optimizer = this.#params.optimizer || 'adam'
    // NOTE: polygrad's backward pass currently only supports batch_size=1.
    const batchSize = this.#params.batch_size || 1

    const layers = [nFeatures, ...hidden, nOutputs]
    const spec = {
      layers, activation,
      loss: 'mse', batch_size: batchSize,
      seed, n_ensemble: nEnsemble
    }

    if (this.#instance) {
      this.#instance.free()
    }
    this.#instance = Instance.tabm(spec)

    const optimKind = optimizer === 'adam' ? OPTIM_ADAM : OPTIM_SGD
    this.#instance.setOptimizer(optimKind, lr)
    this.#batchSize = batchSize

    const nBatches = Math.floor(nTrain / batchSize)
    const rng = lcg(seed)
    const batchX = new Float32Array(batchSize * nFeatures)
    const batchY = new Float32Array(batchSize * nOutputs)
    let bestValLoss = Infinity
    let staleEpochs = 0
    let bestWeights = null

    for (let epoch = 0; epoch < epochs; epoch++) {
      const order = shuffleIndices(nTrain, rng)
      for (let b = 0; b < nBatches; b++) {
        for (let i = 0; i < batchSize; i++) {
          const si = order[b * batchSize + i]
          batchX.set(trainData.subarray(si * nFeatures, (si + 1) * nFeatures), i * nFeatures)
          batchY.set(trainY.subarray(si * nOutputs, (si + 1) * nOutputs), i * nOutputs)
        }
        this.#instance.trainStep({ x: batchX, y: batchY })
      }

      if (valData && nVal > 0) {
        let valLoss = 0
        const valBatchX = new Float32Array(batchSize * nFeatures)
        const nValBatches = Math.floor(nVal / batchSize)
        for (let b = 0; b < nValBatches; b++) {
          for (let i = 0; i < batchSize; i++) {
            const si = b * batchSize + i
            valBatchX.set(valData.subarray(si * nFeatures, (si + 1) * nFeatures), i * nFeatures)
          }
          const out = this.#instance.forward({ x: valBatchX })
          const preds = out.output
          for (let i = 0; i < batchSize; i++) {
            const si = b * batchSize + i
            for (let o = 0; o < nOutputs; o++) {
              const diff = preds[i * nOutputs + o] - valY[si * nOutputs + o]
              valLoss += diff * diff
            }
          }
        }
        valLoss /= Math.max(1, nValBatches * batchSize)
        if (valLoss < bestValLoss) {
          bestValLoss = valLoss
          staleEpochs = 0
          bestWeights = this.#instance.exportWeights()
        } else {
          staleEpochs++
          if (staleEpochs >= patience) {
            if (bestWeights) this.#instance.importWeights(bestWeights)
            break
          }
        }
      }
    }

    this.#fitted = true
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const { rows, cols, data } = this.#normalizeX(X)
    const bs = this.#batchSize
    const batchX = new Float32Array(bs * cols)

    batchX.fill(0)
    const n0 = Math.min(bs, rows)
    for (let i = 0; i < n0; i++) {
      batchX.set(data.subarray(i * cols, (i + 1) * cols), i * cols)
    }
    const out0 = this.#instance.forward({ x: batchX })
    const nOutputs = out0.output.length / bs

    const result = new Float64Array(rows * nOutputs)

    for (let i = 0; i < n0; i++) {
      for (let o = 0; o < nOutputs; o++) {
        result[i * nOutputs + o] = out0.output[i * nOutputs + o]
      }
    }

    for (let b = bs; b < rows; b += bs) {
      const actual = Math.min(bs, rows - b)
      batchX.fill(0)
      for (let i = 0; i < actual; i++) {
        batchX.set(data.subarray((b + i) * cols, (b + i + 1) * cols), i * cols)
      }
      const out = this.#instance.forward({ x: batchX })
      for (let i = 0; i < actual; i++) {
        for (let o = 0; o < nOutputs; o++) {
          result[(b + i) * nOutputs + o] = out.output[i * nOutputs + o]
        }
      }
    }

    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = y instanceof Float64Array ? y : new Float64Array(Array.isArray(y) ? y : [...y])
    const yMean = yArr.reduce((a, b) => a + b, 0) / yArr.length
    let ssRes = 0, ssTot = 0
    for (let i = 0; i < yArr.length; i++) {
      ssRes += (yArr[i] - preds[i]) ** 2
      ssTot += (yArr[i] - yMean) ** 2
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot
  }

  save() {
    this.#ensureFitted()
    const irBytes = this.#instance.exportIR()
    const wBytes = this.#instance.exportWeights()

    return encodeBundle(
      {
        typeId: 'wlearn.nn.tabm.regressor@1',
        params: this.getParams(),
        metadata: {
          nFeatures: this.#nFeatures,
          batchSize: this.#batchSize
        }
      },
      [
        { id: 'ir', data: new Uint8Array(irBytes) },
        { id: 'weights', data: new Uint8Array(wBytes) }
      ]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return TabMRegressor._fromBundle(manifest, toc, blobs)
  }

  static _fromBundle(manifest, toc, blobs) {
    const { Instance } = getPolygrad()

    const irEntry = toc.find(e => e.id === 'ir')
    const wEntry = toc.find(e => e.id === 'weights')
    if (!irEntry || !wEntry) throw new Error('Bundle missing "ir" or "weights" artifact')

    const irBytes = blobs.subarray(irEntry.offset, irEntry.offset + irEntry.length)
    const wBytes = blobs.subarray(wEntry.offset, wEntry.offset + wEntry.length)

    const instance = Instance.fromIR(irBytes, wBytes)
    const params = manifest.params || {}
    const meta = manifest.metadata || {}

    return new TabMRegressor(
      instance, params,
      meta.nFeatures || 0,
      meta.batchSize || 1
    )
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    if (this.#instance) {
      this.#instance.free()
      this.#instance = null
    }
    this.#fitted = false
  }

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  get isFitted() {
    return this.#fitted && !this.#disposed
  }

  get capabilities() {
    return {
      classifier: false,
      regressor: true,
      predictProba: false,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: true
    }
  }

  static defaultSearchSpace() {
    return {
      hidden_sizes: { type: 'categorical', values: [[64], [128], [64, 64], [128, 64]] },
      activation: { type: 'categorical', values: ['relu', 'gelu', 'silu'] },
      n_ensemble: { type: 'categorical', values: [4, 8, 16, 32] },
      lr: { type: 'log_uniform', low: 1e-4, high: 1e-1 },
      epochs: { type: 'int_uniform', low: 10, high: 200 },
      optimizer: { type: 'categorical', values: ['adam', 'sgd'] }
    }
  }

  #ensureFitted() {
    if (this.#disposed) throw new DisposedError('TabMRegressor has been disposed.')
    if (!this.#fitted) throw new NotFittedError('TabMRegressor is not fitted.')
  }

  #normalizeX(X) {
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { rows, cols, data }
    }
    if (X && X.data) {
      const data = X.data instanceof Float32Array ? X.data : new Float32Array(X.data)
      return { rows: X.rows, cols: X.cols, data }
    }
    throw new Error('X must be number[][] or { data, rows, cols }')
  }
}

// ─── Register loaders ─────────────────────────────────────────────────

register('wlearn.nn.tabm.classifier@1', (m, t, b) => TabMClassifier._fromBundle(m, t, b))
register('wlearn.nn.tabm.regressor@1', (m, t, b) => TabMRegressor._fromBundle(m, t, b))

module.exports = { TabMClassifier, TabMRegressor }
