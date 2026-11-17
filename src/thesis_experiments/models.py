"""Classical small-feature baselines with sequence-grouped SVM calibration."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ClassicalModel:
    def __init__(self, kind, *, seed=11, **settings):
        self.kind, self.seed, self.settings = kind, seed, settings

    def fit(self, x, y, groups):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        if len(np.unique(y)) != 2:
            raise ValueError("classical fitting requires both classes")
        if self.kind == "forest":
            self.estimator = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion="gini",
                max_features="sqrt", random_state=self.seed, n_jobs=1, **self.settings).fit(x, y)
        elif self.kind == "svm":
            def estimator():
                return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False, **self.settings))
            oof = np.full(len(y), np.nan)
            self.fold_support = []
            folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self.seed)
            for fit, heldout in folds.split(x, y, groups):
                if set(groups[fit]) & set(groups[heldout]) or any(len(np.unique(y[i])) != 2 for i in (fit, heldout)):
                    raise ValueError("SVM folds require disjoint sequences and both classes")
                fold_model = estimator().fit(x[fit], y[fit])
                oof[heldout] = fold_model.decision_function(x[heldout])
                self.fold_support.append({"fit_sequences": sorted(set(groups[fit])),
                    "heldout_sequences": sorted(set(groups[heldout])),
                    "heldout_present": int(y[heldout].sum()), "heldout_absent": int((~y[heldout]).sum())})
            if not np.isfinite(oof).all():
                raise ValueError("incomplete SVM calibration predictions")
            self.calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=self.seed).fit(oof[:, None], y)
            self.estimator = estimator().fit(x, y)
        else:
            raise ValueError("unknown classical model")
        return self

    def predict(self, x):
        if self.kind == "svm":
            probability = self.calibrator.predict_proba(self.estimator.decision_function(x)[:, None])[:, 1]
        else:
            probability = self.estimator.predict_proba(x)[:, 1]
        return probability, None

    def save(self, path):
        import joblib
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(Path(path))
