'use strict'

const { createEstimator } = require('./estimator.js')

const searchSpace = {
  hidden_sizes: { type: 'categorical', values: [[32], [64], [64, 32], [128]] },
  activation: { type: 'categorical', values: ['exu', 'relu', 'gelu'] },
  lr: { type: 'log_uniform', low: 1e-4, high: 1e-1 },
  epochs: { type: 'int_uniform', low: 10, high: 200 },
  optimizer: { type: 'categorical', values: ['adam', 'sgd'] }
}

const NAMClassifier = createEstimator('NAM', true, searchSpace)
const NAMRegressor = createEstimator('NAM', false, searchSpace)

module.exports = { NAMClassifier, NAMRegressor }
