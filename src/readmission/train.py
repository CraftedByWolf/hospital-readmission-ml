"""End-to-end training: data -> features -> model selection -> calibration -> report.

Run with:  python -m readmission.train
"""
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from . import __version__
from .config import load_config
from .data import GROUP, clean, download, load_raw, split, xy
from .evaluate import (best_threshold, fairness_report, net_savings, plot_all,
                       ranking_metrics, savings_curve)
from .features import build_preprocessor

log = logging.getLogger("readmission")


def candidate_models(cfg: dict, X_train, y_train, groups) -> dict:
    seed = cfg["split"]["random_state"]
    models = {
        "Baseline (prior rate)": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", DummyClassifier(strategy="prior"))]),
        "Logistic regression": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(C=0.1, max_iter=2000))]),
    }

    lgbm = Pipeline([
        ("prep", build_preprocessor(scale_numeric=False)),
        ("clf", LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1))])
    search = RandomizedSearchCV(
        lgbm,
        {
            "clf__n_estimators": randint(200, 900),
            "clf__learning_rate": loguniform(0.01, 0.1),
            "clf__num_leaves": randint(8, 64),
            "clf__min_child_samples": randint(20, 300),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__subsample_freq": [1],
            "clf__colsample_bytree": uniform(0.4, 0.6),
            "clf__reg_lambda": loguniform(1e-3, 30),
        },
        n_iter=cfg["tuning"]["n_iter"],
        cv=GroupKFold(n_splits=cfg["tuning"]["cv_folds"]),
        scoring="roc_auc", random_state=seed, n_jobs=1, verbose=0,
    )
    t = time.time()
    search.fit(X_train, y_train, groups=groups)
    log.info("LightGBM search: best CV AUC %.4f in %.0fs", search.best_score_, time.time() - t)
    models["LightGBM (tuned)"] = search.best_estimator_
    return models, search


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    models_dir, reports_dir = cfg["paths"]["models_dir"], cfg["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    # 1. Data
    raw_path = download(cfg["data"]["url"], cfg["data"]["raw_dir"], cfg["data"]["raw_file"])
    df = clean(load_raw(raw_path))
    train, valid, test = split(df, **cfg["split"])
    log.info("Split sizes  train=%d  valid=%d  test=%d", len(train), len(valid), len(test))
    X_tr, y_tr = xy(train)
    X_va, y_va = xy(valid)
    X_te, y_te = xy(test)

    # 2. Train candidates on train, compare on validation
    models, search = candidate_models(cfg, X_tr, y_tr, X_tr[GROUP])
    comparison = []
    for name, model in models.items():
        if name != "LightGBM (tuned)":
            model.fit(X_tr, y_tr)
        comparison.append({"model": name, **ranking_metrics(y_va, model.predict_proba(X_va)[:, 1])})
    comparison = pd.DataFrame(comparison)
    log.info("Validation comparison:\n%s", comparison.to_string(index=False))
    best_name = comparison.iloc[1:].sort_values("pr_auc", ascending=False)["model"].iloc[0]
    log.info("Selected model: %s", best_name)

    # 3. Calibrate probabilities on validation so a "20% risk" really means ~20%
    calibrated = CalibratedClassifierCV(FrozenEstimator(models[best_name]), method="sigmoid")
    calibrated.fit(X_va, y_va)
    p_va = calibrated.predict_proba(X_va)[:, 1]

    # 4. Pick the decision threshold that maximises expected net savings (validation)
    business = cfg["business"]
    threshold = best_threshold(y_va, p_va, business)
    log.info("Cost-optimal threshold on validation: %.2f", threshold)

    # 5. Final, untouched test-set evaluation
    preds = {name: m.predict_proba(X_te)[:, 1] for name, m in models.items()
             if not name.startswith("Baseline")}
    final_name = f"{best_name} + calibration"
    preds[final_name] = p_te = calibrated.predict_proba(X_te)[:, 1]
    test_metrics = {name: ranking_metrics(y_te, p) for name, p in preds.items()}
    business_result = net_savings(y_te, p_te, threshold, business)
    flag_all = net_savings(y_te, p_te, 0.0, business)
    curve = savings_curve(y_te, p_te, business)
    fairness = fairness_report(X_te, y_te, p_te, threshold)

    # Top-decile lift: how concentrated are readmissions among the highest-risk patients?
    top10 = p_te >= np.quantile(p_te, 0.9)
    lift = float(y_te[top10].mean() / y_te.mean())

    # 6. Explainability: which raw inputs move the model most?
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_te), size=min(8000, len(X_te)), replace=False)
    raw_cols = [c for c in X_te.columns if c not in ("encounter_id", GROUP, "weight")]
    perm = permutation_importance(calibrated, X_te.iloc[idx][raw_cols], y_te[idx],
                                  scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=1)
    importance = (pd.DataFrame({"feature": raw_cols, "importance": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    plot_all(y_te, preds, final_name, curve, threshold, importance, reports_dir / "figures")

    # 7. Persist model + metadata + reports
    joblib.dump(calibrated, models_dir / "model.joblib", compress=3)
    metadata = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best_name,
        "best_lgbm_params": {k.replace("clf__", ""): (float(v) if isinstance(v, (float, np.floating)) else v)
                             for k, v in search.best_params_.items()},
        "lgbm_cv_auc": float(search.best_score_),
        "threshold": threshold,
        "business_assumptions": business,
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "test_metrics": test_metrics[final_name],
        "top_decile_lift": lift,
        "business_result_test": business_result,
        "flag_everyone_test": flag_all,
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (reports_dir / "metrics.json").write_text(json.dumps(
        {"validation": comparison.to_dict(orient="records"), "test": test_metrics,
         "business": business_result, "flag_everyone": flag_all, "top_decile_lift": lift},
        indent=2))
    comparison.to_csv(reports_dir / "model_comparison_validation.csv", index=False)
    fairness.to_csv(reports_dir / "fairness_test.csv", index=False)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    curve.to_csv(reports_dir / "savings_curve_test.csv", index=False)

    log.info("TEST  %s", json.dumps(test_metrics[final_name]))
    log.info("TEST  top-decile lift %.2fx", lift)
    log.info("TEST  business at threshold %.2f: %s", threshold, json.dumps(business_result))
    log.info("Done. Model saved to %s", models_dir / "model.joblib")


if __name__ == "__main__":
    main()
