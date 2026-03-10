const { MLPClassifier, MLPRegressor } = require('./mlp.js')
const { TabMClassifier, TabMRegressor } = require('./tabm.js')
const { NAMClassifier, NAMRegressor } = require('./nam.js')
const { createModelClass } = require('@wlearn/core')

const MLPModel = createModelClass(MLPClassifier, MLPRegressor, { name: 'MLPModel' })
const TabMModel = createModelClass(TabMClassifier, TabMRegressor, { name: 'TabMModel' })
const NAMModel = createModelClass(NAMClassifier, NAMRegressor, { name: 'NAMModel' })

module.exports = {
  // Unified classes (recommended)
  MLPModel, TabMModel, NAMModel,
  // Original split classes (backward compat)
  MLPClassifier, MLPRegressor,
  TabMClassifier, TabMRegressor,
  NAMClassifier, NAMRegressor
}
