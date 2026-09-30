'use strict'

const { BackendError } = require('@wlearn/core')

async function acquireRuntime(options) {
  const owned = !(options && options.Model && options.models)
  const runtime = owned ? await require('polygrad').create(options) : options
  try {
    if (!runtime.Model || !runtime.models) throw new BackendError('NN requires the Polygrad 0.6 Model API')
    if (runtime.core === 'wasm' && runtime.device === 'webgpu') {
      throw new BackendError('NN fit is synchronous; use native or synchronous WASM Polygrad')
    }
    return { runtime, owned }
  } catch (error) {
    if (owned) runtime.dispose()
    throw error
  }
}

module.exports = { acquireRuntime }
