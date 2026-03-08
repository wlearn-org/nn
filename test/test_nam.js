/**
 * Tests for @wlearn/nn (NAMClassifier, NAMRegressor)
 */

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

// ── Test Data Generators ───────────────────────────────────────────

function lcg(seed) {
  let s = seed
  return function () {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeBinaryData(seed = 42, n = 50, nFeatures = 2) {
  const rng = lcg(seed)
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push((rng() - 0.5) * 4)
    }
    X.push(row)
    y.push(row[0] + row[1] > 0 ? 1 : 0)
  }
  return { X, y }
}

function makeRegressionData(seed = 42, n = 50, nFeatures = 2) {
  const rng = lcg(seed)
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push((rng() - 0.5) * 4)
    }
    X.push(row)
    // NAM should learn additive functions well: y = f1(x1) + f2(x2)
    y.push(2.0 * row[0] + 3.0 * row[1] + 0.5)
  }
  return { X, y }
}

// ── Import ────────────────────────────────────────────────────────

const { NAMClassifier, NAMRegressor } = require('../src/index.js')
const { load } = require('@wlearn/core')

async function main() {

// ═══════════════════════════════════════════════════════════════════
// NAMClassifier
// ═══════════════════════════════════════════════════════════════════

console.log('\n=== NAMClassifier ===')

await test('create() returns unfitted model', async () => {
  const model = await NAMClassifier.create()
  assert(!model.isFitted, 'should not be fitted')
  model.dispose()
})

await test('throws before fit', async () => {
  const model = await NAMClassifier.create()
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')
  model.dispose()
})

await test('binary classification with relu', async () => {
  const { X, y } = makeBinaryData()
  const model = await NAMClassifier.create({
    hidden_sizes: [8], epochs: 80, lr: 0.01,
    optimizer: 'adam', seed: 42, activation: 'relu'
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')

  const preds = model.predict(X)
  assert(preds.length === X.length, `expected ${X.length} predictions, got ${preds.length}`)

  const acc = model.score(X, y)
  assert(acc > 0.9, `expected accuracy > 0.9, got ${acc}`)

  model.dispose()
})

await test('binary classification with exu', async () => {
  const { X, y } = makeBinaryData()
  const model = await NAMClassifier.create({
    hidden_sizes: [8], epochs: 80, lr: 0.01,
    optimizer: 'adam', seed: 42, activation: 'exu'
  })
  model.fit(X, y)
  assert(model.isFitted)

  const acc = model.score(X, y)
  assert(acc > 0.9, `expected accuracy > 0.9 with ExU, got ${acc}`)

  model.dispose()
})

await test('predictProba returns valid probabilities', async () => {
  const { X, y } = makeBinaryData()
  const model = await NAMClassifier.create({
    hidden_sizes: [8], epochs: 30, lr: 0.01,
    seed: 42, activation: 'relu'
  })
  model.fit(X, y)

  const proba = model.predictProba(X)
  assert(proba.length === X.length * 2, 'proba shape mismatch')

  for (let i = 0; i < X.length; i++) {
    const p0 = proba[i * 2]
    const p1 = proba[i * 2 + 1]
    assert(p0 >= 0 && p0 <= 1, `p0 out of range: ${p0}`)
    assert(p1 >= 0 && p1 <= 1, `p1 out of range: ${p1}`)
    const sum = p0 + p1
    assert(Math.abs(sum - 1.0) < 1e-5, `proba sum != 1: ${sum}`)
  }

  model.dispose()
})

await test('save and load roundtrip', async () => {
  const { X, y } = makeBinaryData()
  const model = await NAMClassifier.create({
    hidden_sizes: [8], epochs: 30, lr: 0.01,
    seed: 42, activation: 'relu'
  })
  model.fit(X, y)
  const predsOrig = model.predict(X)

  const bundle = model.save()
  assert(bundle instanceof Uint8Array, 'save should return Uint8Array')

  const loaded = await NAMClassifier.load(bundle)
  assert(loaded.isFitted, 'loaded model should be fitted')

  const predsLoaded = loaded.predict(X)
  for (let i = 0; i < predsOrig.length; i++) {
    assert(predsOrig[i] === predsLoaded[i], `prediction mismatch at ${i}`)
  }

  model.dispose()
  loaded.dispose()
})

await test('registry dispatch loads NAMClassifier', async () => {
  const { X, y } = makeBinaryData()
  const model = await NAMClassifier.create({
    hidden_sizes: [8], epochs: 20, lr: 0.01,
    seed: 42, activation: 'relu'
  })
  model.fit(X, y)
  const bundle = model.save()

  const loaded = await load(bundle)
  assert(loaded instanceof NAMClassifier, 'should be NAMClassifier')
  assert(loaded.isFitted)

  model.dispose()
  loaded.dispose()
})

await test('dispose prevents further use', async () => {
  const model = await NAMClassifier.create({
    hidden_sizes: [4], epochs: 5, seed: 42, activation: 'relu'
  })
  const { X, y } = makeBinaryData(42, 10)
  model.fit(X, y)
  model.dispose()

  assert(!model.isFitted)
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('capabilities', async () => {
  const model = await NAMClassifier.create()
  const caps = model.capabilities
  assert(caps.classifier === true)
  assert(caps.regressor === false)
  assert(caps.predictProba === true)
  assert(caps.earlyStopping === true)
  model.dispose()
})

await test('search space includes activation', async () => {
  const space = NAMClassifier.defaultSearchSpace()
  assert(space.activation, 'should have activation in search space')
  assert(space.hidden_sizes, 'should have hidden_sizes')
  assert(space.lr, 'should have lr')
})

// ═══════════════════════════════════════════════════════════════════
// NAMRegressor
// ═══════════════════════════════════════════════════════════════════

console.log('\n=== NAMRegressor ===')

await test('create() returns unfitted model', async () => {
  const model = await NAMRegressor.create()
  assert(!model.isFitted, 'should not be fitted')
  model.dispose()
})

await test('throws before fit', async () => {
  const model = await NAMRegressor.create()
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')
  model.dispose()
})

await test('regression on additive target with relu', async () => {
  const { X, y } = makeRegressionData()
  const model = await NAMRegressor.create({
    hidden_sizes: [16], epochs: 150, lr: 0.01,
    optimizer: 'adam', seed: 42, activation: 'relu'
  })
  model.fit(X, y)
  assert(model.isFitted)

  const r2 = model.score(X, y)
  assert(r2 > 0.9, `expected R^2 > 0.9, got ${r2}`)

  model.dispose()
})

await test('regression with exu activation', async () => {
  const { X, y } = makeRegressionData()
  const model = await NAMRegressor.create({
    hidden_sizes: [16], epochs: 150, lr: 0.01,
    optimizer: 'adam', seed: 42, activation: 'exu'
  })
  model.fit(X, y)
  assert(model.isFitted)

  const r2 = model.score(X, y)
  assert(r2 > 0.9, `expected R^2 > 0.9 with ExU, got ${r2}`)

  model.dispose()
})

await test('save and load roundtrip', async () => {
  const { X, y } = makeRegressionData()
  const model = await NAMRegressor.create({
    hidden_sizes: [8], epochs: 30, lr: 0.01,
    seed: 42, activation: 'relu'
  })
  model.fit(X, y)
  const predsOrig = model.predict(X)

  const bundle = model.save()
  assert(bundle instanceof Uint8Array)

  const loaded = await NAMRegressor.load(bundle)
  assert(loaded.isFitted)

  const predsLoaded = loaded.predict(X)
  for (let i = 0; i < predsOrig.length; i++) {
    const diff = Math.abs(predsOrig[i] - predsLoaded[i])
    assert(diff < 1e-5, `prediction mismatch at ${i}: ${predsOrig[i]} vs ${predsLoaded[i]}`)
  }

  model.dispose()
  loaded.dispose()
})

await test('registry dispatch loads NAMRegressor', async () => {
  const { X, y } = makeRegressionData()
  const model = await NAMRegressor.create({
    hidden_sizes: [8], epochs: 20, lr: 0.01,
    seed: 42, activation: 'relu'
  })
  model.fit(X, y)
  const bundle = model.save()

  const loaded = await load(bundle)
  assert(loaded instanceof NAMRegressor, 'should be NAMRegressor')
  assert(loaded.isFitted)

  model.dispose()
  loaded.dispose()
})

await test('dispose prevents further use', async () => {
  const model = await NAMRegressor.create({
    hidden_sizes: [4], epochs: 5, seed: 42, activation: 'relu'
  })
  const { X, y } = makeRegressionData(42, 10)
  model.fit(X, y)
  model.dispose()

  assert(!model.isFitted)
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('capabilities', async () => {
  const model = await NAMRegressor.create()
  const caps = model.capabilities
  assert(caps.classifier === false)
  assert(caps.regressor === true)
  assert(caps.predictProba === false)
  assert(caps.earlyStopping === true)
  model.dispose()
})

await test('search space includes activation', async () => {
  const space = NAMRegressor.defaultSearchSpace()
  assert(space.activation, 'should have activation in search space')
  assert(space.hidden_sizes, 'should have hidden_sizes')
  assert(space.lr, 'should have lr')
})

// ── Summary ───────────────────────────────────────────────────────

console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`)
if (failed > 0) process.exit(1)

} // end main

main()
