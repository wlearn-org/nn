"""Consume run-owned NN fixtures and return Python WLRN bundles to JavaScript."""
import json
from pathlib import Path
import sys

import numpy as np
import wlearn.nn as nn

root = Path(sys.argv[1])
for case in json.loads((root / 'cases.json').read_text()):
    model = getattr(nn, case['name']).load(root / (case['id'] + '.wlrn'))
    try:
        pred = model.predict(case['X'])
        assert np.isfinite(pred).all()
        np.testing.assert_allclose(pred, case['pred'], rtol=0, atol=1e-5)
        if case['proba'] is not None:
            proba = model.predict_proba(case['X'])
            assert np.isfinite(proba).all()
            np.testing.assert_allclose(proba, case['proba'], rtol=0, atol=1e-5)
        model.save(root / (case['id'] + '.py.wlrn'))
    finally:
        model.dispose()
print('10 Python NN bundle consumers passed')
