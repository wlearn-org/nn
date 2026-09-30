'use strict'

const assert = require('node:assert/strict')
const { mkdtempSync, writeFileSync, readFileSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { join } = require('node:path')
const { execFileSync } = require('node:child_process')
const nn = require('../src')
const { decodeBundle, encodeBundle, load, Pipeline } = require('@wlearn/core')

function close(a, b) {
  assert.equal(a.length, b.length)
  for (let i = 0; i < a.length; i++) {
    assert(Number.isFinite(a[i]) && Number.isFinite(b[i]))
    assert(Math.abs(a[i] - b[i]) <= 1e-5, `${a[i]} != ${b[i]}`)
  }
}

async function main() {
  const runtime = await require('polygrad').create({ core: process.env.WLEARN_NN_TEST_CORE || 'native', device: 'cpu' })
  const out = mkdtempSync(join(process.env.WLEARN_TEST_OUTPUT || tmpdir(), 'nn-migration-'))
  const X = [[-1, 0], [0, 1], [1, 0], [0, -1], [.5, .5], [-.5, -.5]]
  const records = []
  try {
    for (const family of ['MLP', 'NAM', 'TabM']) for (const task of ['Classifier', 'Regressor']) {
      for (const batch of (family === 'TabM' ? [1] : [1, 2])) {
        const Class = nn[family + task]
        const y = task === 'Classifier' ? [10, 20, 20, 10, 20, 10] : [-1, 1, 1, -1, 1, -1]
        const model = await Class.create({ polygrad: runtime, hidden_sizes: [3], n_ensemble: 2, batch_size: batch, epochs: 2, optimizer: 'sgd' })
        try {
          model.fit(X, y)
          const input = X.slice(0, 5), pred = model.predict(input), bytes = model.save()
          const id = `${family}${task}-${batch}`
          const restored = await Class.load(bytes, { polygrad: runtime })
          try { close(pred, restored.predict(input)) } finally { restored.dispose() }
          model.setParams({ batch_size: 4 })
          assert.throws(() => model.fit(X, y), /fixed-size|TabM/)
          close(pred, model.predict(input))
          writeFileSync(join(out, id + '.wlrn'), bytes)
          records.push({ id, name: family + task, X: input, y: y.slice(0, 5), pred: [...pred],
            proba: task === 'Classifier' ? [...model.predictProba(input)] : null })
          const { manifest } = decodeBundle(bytes)
          assert(manifest.typeId.endsWith('@2'))
        } finally { model.dispose() }
      }
    }
    // The public default must handle arbitrary dataset sizes without padding or
    // dropping training rows. Keep explicit unsupported batch guards below.
    for (const family of ['MLP', 'NAM', 'TabM']) {
      for (const rows of [80, 96, 100, 150, 1000]) {
        const input = Array.from({ length: rows }, (_, i) => [i / rows, i % 2])
        const model = await nn[family + 'Classifier'].create({
          polygrad: runtime, hidden_sizes: [3], n_ensemble: 2, epochs: 1
        })
        try {
          model.fit(input, input.map(row => row[1]))
          assert.equal(model.predict(input).length, rows)
          assert(model.predictProba(input).every(Number.isFinite))
        } finally { model.dispose() }
      }
    }
    console.log('Default batching: 15 arbitrary-row-count classifier fits passed')
    const short = await nn.MLPRegressor.create({ polygrad: runtime, hidden_sizes: [2], epochs: 1, batch_size: 8 })
    try {
      short.fit(X.slice(0, 2), [1, 2])
      assert.equal(decodeBundle(short.save()).manifest.metadata.batchSize, 2)
      assert(short.predict(X).every(Number.isFinite))
    } finally { short.dispose() }
    // Borrowing an explicit runtime must never transfer its disposal ownership.
    const tensor = new runtime.Tensor([7])
    try { close(tensor.toArray(), [7]) } finally { tensor.dispose() }
    const legacy = encodeBundle({ typeId: 'wlearn.nn.mlp.regressor@1', params: {} }, [])
    await assert.rejects(load(legacy), /Legacy NN/)

    const estimator = await nn.MLPModel.create({ task: 'regression', polygrad: runtime, hidden_sizes: [2], epochs: 1 })
    const pipe = new Pipeline([['model', estimator]])
    try {
      await pipe.fit(X, [-1, 1, 1, -1, 1, -1])
      const loaded = await load(pipe.save())
      try { close(await pipe.predict(X), await loaded.predict(X)) } finally { loaded.dispose() }
    } finally { pipe.dispose() }

    writeFileSync(join(out, 'cases.json'), JSON.stringify(records))
    execFileSync(process.env.WLEARN_PYTHON || 'python3', [join(__dirname, 'interop.py'), out], { stdio: 'inherit' })
    for (const c of records) {
      const restored = await nn[c.name].load(readFileSync(join(out, c.id + '.py.wlrn')), { polygrad: runtime })
      try { close(c.pred, restored.predict(c.X)) } finally { restored.dispose() }
    }
    console.log(`10 JS → Python → JS Model bundle cases passed; artifacts: ${out}`)
  } finally { runtime.dispose() }

  let updates = 0, restoredWeight, runtimeDisposals = 0
  const borrowed = { Model: {}, models: { MLP: () => ({
    setOptimizer() {},
    trainStep() { updates++; return 1 },
    forward() { return { output: new Float32Array([updates, updates]) } },
    exportWeights() { return Uint8Array.of(updates) },
    importWeights(w) { restoredWeight = w[0] }, dispose() {}
  }) }, dispose() { runtimeDisposals++ } }
  const model = await nn.MLPRegressor.create({ polygrad: borrowed, batch_size: 2, epochs: 3, validation_fraction: 1 / 7 })
  model.fit(Array.from({ length: 7 }, () => [0]), Array(7).fill(0))
  model.dispose()
  assert.equal(updates, 9)
  assert.equal(restoredWeight, 3, 'restore best epoch even without patience exhaustion')
  assert.equal(runtimeDisposals, 0)
  console.log('Batch validation, short datasets, best-state restoration and borrowed ownership passed')
}
main().catch(error => { console.error(error); process.exitCode = 1 })
