'use strict'

const { createEstimator } = require('./estimator.js')

const searchSpace = {
  hidden_sizes: { type: 'categorical', values: [[64], [128], [64, 64], [128, 64]] },
  activation: { type: 'categorical', values: ['relu', 'gelu', 'silu'] },
  n_ensemble: { type: 'categorical', values: [4, 8, 16, 32] },
  lr: { type: 'log_uniform', low: 1e-4, high: 1e-1 },
  epochs: { type: 'int_uniform', low: 10, high: 200 },
  optimizer: { type: 'categorical', values: ['adam', 'sgd'] }
}

const TabMClassifier = createEstimator('TabM', true, searchSpace)
const TabMRegressor = createEstimator('TabM', false, searchSpace)

module.exports = { TabMClassifier, TabMRegressor }
