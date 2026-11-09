"""Exploratory data analysis: what drives readmission in the raw data?

Run with:  python -m readmission.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_config  # noqa: E402
from .data import TARGET, clean, download, load_raw  # noqa: E402
from .features import EncounterFeatures  # noqa: E402


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["reports_dir"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw(download(cfg["data"]["url"], cfg["data"]["raw_dir"], cfg["data"]["raw_file"])))
    feats = EncounterFeatures().transform(df)
    feats[TARGET] = df[TARGET].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    base = feats[TARGET].mean()

    prior = feats.assign(prior=feats["number_inpatient"].clip(upper=5)).groupby("prior")[TARGET].mean()
    axes[0, 0].bar(prior.index.astype(str).str.replace("5", "5+"), prior.values)
    axes[0, 0].set(title="Readmission rate by prior-year inpatient stays", xlabel="Stays", ylabel="Rate")

    disc = feats.groupby("discharge_group")[TARGET].mean().sort_values()
    axes[0, 1].barh(disc.index, disc.values)
    axes[0, 1].set(title="Readmission rate by discharge destination", xlabel="Rate")

    age = feats.groupby("age_years")[TARGET].mean()
    axes[1, 0].plot(age.index, age.values, "o-")
    axes[1, 0].set(title="Readmission rate by age", xlabel="Age (bracket midpoint)", ylabel="Rate")

    diag = feats.groupby("diag_1_group")[TARGET].mean().sort_values()
    axes[1, 1].barh(diag.index, diag.values)
    axes[1, 1].set(title="Readmission rate by primary diagnosis group", xlabel="Rate")

    for ax in axes.flat:
        ax.axvline(base, color="r", ls="--", lw=1) if ax in (axes[0, 1], axes[1, 1]) \
            else ax.axhline(base, color="r", ls="--", lw=1)
    fig.suptitle(f"Diabetes 130-US hospitals: {len(df):,} encounters, "
                 f"{base:.1%} readmitted within 30 days (red line)")
    fig.tight_layout()
    fig.savefig(out / "eda_overview.png", dpi=120)

    summary = pd.DataFrame({
        "encounters": [len(df)], "patients": [df["patient_nbr"].nunique()],
        "readmit_30d_rate": [base],
        "patients_with_multiple_encounters": [(df["patient_nbr"].value_counts() > 1).sum()],
    })
    summary.to_csv(cfg["paths"]["reports_dir"] / "data_summary.csv", index=False)
    print(summary.T.to_string(header=False))


if __name__ == "__main__":
    main()
