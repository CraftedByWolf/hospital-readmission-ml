"""Metrics, cost-based threshold selection, fairness slices and plots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def ranking_metrics(y, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(np.mean(y)),
    }


def net_savings(y, p, threshold: float, business: dict) -> dict:
    """Money saved by running the follow-up programme on everyone above `threshold`."""
    flagged = p >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    n_flag = int(flagged.sum())
    prevented = tp * business["intervention_success"]
    saved = prevented * business["readmission_cost"]
    spent = n_flag * business["intervention_cost"]
    return {
        "threshold": float(threshold),
        "flagged": n_flag,
        "flagged_pct": n_flag / len(y),
        "true_positives": tp,
        "readmissions_prevented": prevented,
        "net_savings": saved - spent,
        "net_savings_per_1000_patients": (saved - spent) / len(y) * 1000,
        "precision": float(precision_score(y, flagged, zero_division=0)),
        "recall": float(recall_score(y, flagged, zero_division=0)),
    }


def savings_curve(y, p, business: dict, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.61, 0.01), 2) if grid is None else grid
    return pd.DataFrame([net_savings(y, p, t, business) for t in grid])


def best_threshold(y, p, business: dict) -> float:
    curve = savings_curve(y, p, business)
    return float(curve.loc[curve["net_savings"].idxmax(), "threshold"])


def fairness_report(X_raw: pd.DataFrame, y, p, threshold: float) -> pd.DataFrame:
    """Performance per demographic group, to spot groups the model under-serves."""
    rows = []
    flagged = p >= threshold
    for col in ["race", "gender", "age"]:
        groups = X_raw[col].fillna("Missing").astype(str)
        for g in sorted(groups.unique()):
            m = (groups == g).to_numpy()
            if m.sum() < 200 or len(np.unique(y[m])) < 2:
                continue
            rows.append({
                "attribute": col, "group": g, "n": int(m.sum()),
                "readmit_rate": float(y[m].mean()),
                "roc_auc": float(roc_auc_score(y[m], p[m])),
                "flag_rate": float(flagged[m].mean()),
                "recall": float(recall_score(y[m], flagged[m], zero_division=0)),
            })
    return pd.DataFrame(rows)


def plot_all(y, preds: dict, final_name: str, curve: pd.DataFrame, threshold: float,
             importance: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(np.mean(y), color="k", ls="--", lw=1, label="Random")
    axes[1].set(title="Precision-recall (test set)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    frac_pos, mean_pred = calibration_curve(y, preds[final_name], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", label=final_name)
    ax.plot([0, 0.4], [0, 0.4], "k--", lw=1, label="Perfect")
    ax.set(title="Calibration (test set)", xlabel="Predicted risk", ylabel="Observed readmission rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["threshold"], curve["net_savings_per_1000_patients"])
    ax.axvline(threshold, color="r", ls="--", label=f"Chosen threshold {threshold:.2f}")
    ax.axhline(0, color="k", lw=1)
    ax.set(title="Net savings per 1,000 discharges vs. threshold (test set)",
           xlabel="Risk threshold", ylabel="USD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "savings_curve.png", dpi=120)
    plt.close(fig)

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set(title="Permutation importance (drop in ROC-AUC)", xlabel="AUC drop")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close(fig)
