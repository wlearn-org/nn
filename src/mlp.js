'use strict'

const { createEstimator } = require('./estimator.js')

const searchSpace = {
  hidden_sizes: { type: 'categorical', values: [[64], [128], [64, 64], [128, 64]] },
  activation: { type: 'categorical', values: ['relu', 'gelu', 'silu'] },
  lr: { type: 'log_uniform', low: 1e-4, high: 1e-1 },
  epochs: { type: 'int_uniform', low: 10, high: 200 },
  optimizer: { type: 'categorical', values: ['adam', 'sgd'] },
  batch_size: { type: 'categorical', values: [1] }
}

const MLPClassifier = createEstimator('MLP', true, searchSpace)
const MLPRegressor = createEstimator('MLP', false, searchSpace)

module.exports = { MLPClassifier, MLPRegressor }
