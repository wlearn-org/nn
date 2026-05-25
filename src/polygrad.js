'use strict'

const runtimePromises = new Map()

function splitPolygradParam(params = {}) {
  const source = params || {}
  const { polygrad, ...modelParams } = source
  return { polygradOptions: polygrad, modelParams }
}

function runtimeKey(options) {
  if (!options) return 'default'
  try {
    return JSON.stringify(options) || 'default'
  } catch (_) {
    return null
  }
}

async function loadPolygrad(options) {
  if (options && options.Instance) return options

  const polygrad = require('polygrad')
  const key = runtimeKey(options)
  if (key == null) return polygrad.create(options)

  if (!runtimePromises.has(key)) {
    runtimePromises.set(key, polygrad.create(options))
  }
  return runtimePromises.get(key)
}

function getPolygradParts(runtime) {
  if (!runtime || !runtime.Instance) {
    throw new Error('polygrad runtime is not initialized; create models with await Model.create()')
  }
  return {
    Instance: runtime.Instance,
    OPTIM_SGD: runtime.OPTIM_SGD,
    OPTIM_ADAM: runtime.OPTIM_ADAM
  }
}

module.exports = { loadPolygrad, getPolygradParts, splitPolygradParam }
