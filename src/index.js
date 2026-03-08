const { MLPClassifier, MLPRegressor } = require('./mlp.js')
const { TabMClassifier, TabMRegressor } = require('./tabm.js')
const { NAMClassifier, NAMRegressor } = require('./nam.js')

module.exports = {
  MLPClassifier, MLPRegressor,
  TabMClassifier, TabMRegressor,
  NAMClassifier, NAMRegressor
}
