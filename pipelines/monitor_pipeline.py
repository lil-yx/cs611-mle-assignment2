"""Model monitoring pipeline.

For a given snapshot month this pipeline:
  1. Loads predictions and joins actuals (when available) to compute AUC/Gini/bad-rate.
  2. Computes PSI on the score distribution versus all prior scored periods.
  3. Writes one monitoring row to the gold model_monitoring table.
  4. Saves two PNG plots under reports/monitoring/.

PSI thresholds (industry standard):
    < 0.10  — no significant shift
    0.10–0.25 — minor shift, investigate
    > 0.25  — major shift, retrain

Usage (from project root, inside Docker):
    python -m pipelines.monitor_pipeline --snapshotdate 2024-01-01
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")   # headless backend — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType, StringType

from utils.common.config import CONFIG
from utils.common.io import parquet_exists, read_all_partitions, read_parquet
from utils.common.log import get_logger
from utils.common.spark_session import get_spark
from utils.transformation.validation import validate_and_write

logger = get_logger("monitor_pipeline")

_N_BINS = 10
_REPORT_DIR = "reports/monitoring"


# ---------------------------------------------------------------------------
# PSI helpers
# ---------------------------------------------------------------------------

def _bin_scores(scores, n_bins=_N_BINS):
    """Return a count-per-bin array over [0, 1) with n_bins equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(scores, bins=bins)
    return counts


def _psi(reference_scores, current_scores, n_bins=_N_BINS):
    """Population Stability Index between two score arrays."""
    ref_counts = _bin_scores(reference_scores, n_bins).astype(float)
    cur_counts = _bin_scores(current_scores, n_bins).astype(float)

    # avoid zero divisions — replace zeros with a small epsilon
    eps = 0.0001
    ref_pct = np.maximum(ref_counts / ref_counts.sum(), eps)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), eps)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_score_dist(scores, bad_rate_actual, snapshot_date, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=20, range=(0, 1), edgecolor="white", color="#4C72B0")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="threshold=0.50")
    ax.set_xlabel("Default probability score")
    ax.set_ylabel("Count")
    title = f"Score distribution — {snapshot_date}"
    if not math.isnan(bad_rate_actual):
        title += f"\nActual bad rate: {bad_rate_actual:.1%}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"score_dist_{snapshot_date.replace('-', '_')}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.info("plot saved → %s", path)


def _plot_psi_bins(reference_scores, current_scores, psi_value, snapshot_date, out_dir):
    bins = np.linspace(0.0, 1.0, _N_BINS + 1)
    bin_labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(_N_BINS)]

    ref_pct = np.maximum(_bin_scores(reference_scores).astype(float) / max(len(reference_scores), 1), 0.0001)
    cur_pct = np.maximum(_bin_scores(current_scores).astype(float)  / max(len(current_scores), 1),  0.0001)

    x = np.arange(_N_BINS)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, ref_pct * 100, width, label="Reference", color="#4C72B0")
    ax.bar(x + width / 2, cur_pct * 100, width, label=f"Current ({snapshot_date})", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Score bin")
    ax.set_ylabel("% of population")
    ax.set_title(f"PSI score distribution — {snapshot_date}  (PSI={psi_value:.4f})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"psi_bins_{snapshot_date.replace('-', '_')}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.info("plot saved → %s", path)


def _plot_trend(spark, out_dir):
    """Performance and stability across every monitored month, from the gold table."""
    hist = read_all_partitions(spark, "gold", "model_monitoring").toPandas()
    hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"])
    hist = hist.sort_values("snapshot_date")
    dates = hist["snapshot_date"]
    cutoff = pd.Timestamp(CONFIG["ml"]["train_cutoff"])
    half_month = pd.Timedelta(days=15)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax in (ax1, ax2, ax3):
        if dates.min() < cutoff:
            ax.axvspan(dates.min() - half_month, cutoff - half_month, color="#E5E7EB",
                       label="in-sample (training window)")
        ax.grid(axis="y", alpha=0.3)

    ax1.plot(dates, hist["auc_roc"], marker="o", color="#2563EB", label="AUC-ROC")
    ax1.plot(dates, hist["gini"], marker="s", color="#D97706", label="Gini")
    oot = hist[dates >= cutoff]
    ax1.plot(oot["snapshot_date"], oot["gini"].rolling(3, min_periods=3).mean(), color="#92400E",
             linewidth=2.5, label="Gini, 3-month rolling (decision line)")
    ax1.axhline(0.57, color="#D97706", linestyle=":", linewidth=1, label="Gini amber = 0.57")
    ax1.set_ylabel("Performance")
    ax1.set_title("Model performance & stability across time")
    ax1.legend(loc="lower left", fontsize=8)

    psi = hist[hist["psi_score"].notna()]
    psi_colors = ["#16A34A" if p < 0.10 else "#D97706" if p < 0.25 else "#DC2626"
                  for p in psi["psi_score"]]
    ax2.bar(psi["snapshot_date"], psi["psi_score"], width=20, color=psi_colors)
    ax2.axhline(0.10, color="#D97706", linestyle="--", linewidth=1, label="0.10 investigate")
    ax2.axhline(0.25, color="#DC2626", linestyle="--", linewidth=1, label="0.25 retrain")
    if dates.max() >= cutoff:
        ax2.text(cutoff, 0.01, "OOT\nbaseline", ha="center", va="bottom", fontsize=8, color="#64748B")
    ax2.set_ylabel("PSI (score drift)")
    ax2.legend(loc="upper left", fontsize=8)

    ax3.plot(dates, hist["bad_rate_actual"] * 100, marker="o", color="#0F1B35", label="actual bad rate")
    ax3.plot(dates, hist["bad_rate_predicted"] * 100, marker="o", linestyle="--", color="#DC2626",
             label="predicted bad rate (score ≥ 0.5)")
    ax3.set_ylabel("Bad rate (%)")
    ax3.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "monitoring_trend.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("trend plot saved → %s (%d months)", path, len(hist))


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(snapshot_date):
    if not parquet_exists("gold", "predictions", snapshot_date):
        logger.warning("no predictions for %s — skipping", snapshot_date)
        return

    spark = get_spark("monitor_pipeline")
    os.makedirs(_REPORT_DIR, exist_ok=True)

    # ---- load predictions for this period ----
    pred_df  = read_parquet(spark, "gold", "predictions", snapshot_date)
    pred_pdf = pred_df.select("Customer_ID", "score", "prediction", "model_id").toPandas()

    n_scored   = len(pred_pdf)
    scores     = pred_pdf["score"].values.astype(float)
    model_id   = pred_pdf["model_id"].iloc[0]
    bad_rate_predicted = float((pred_pdf["prediction"] == 1).mean())

    # ---- join actuals if available ----
    # label_store is partitioned by the date the label becomes KNOWN (loan_start_date + ~6 months).
    # Predictions for snapshot_date X cover loans that started on X, so we must search
    # the full label_store for rows where loan_start_date == X.
    n_labelled       = 0
    bad_rate_actual  = float("nan")
    auc_roc          = float("nan")
    gini             = float("nan")

    all_labels_df = read_all_partitions(spark, "gold", "label_store")
    label_for_date = (
        all_labels_df
        .filter(col("loan_start_date").cast("string") == snapshot_date)
        .select("Customer_ID", "label")
        .toPandas()
    )
    if len(label_for_date) > 0:
        merged     = pred_pdf.merge(label_for_date, on="Customer_ID", how="inner")
        n_labelled = len(merged)
        if n_labelled > 0:
            bad_rate_actual = float(merged["label"].mean())
            y_true = merged["label"].values
            y_prob = merged["score"].values
            if y_true.sum() > 0 and (1 - y_true).sum() > 0:
                from sklearn.metrics import roc_auc_score
                auc_val = float(roc_auc_score(y_true, y_prob))
                auc_roc = round(auc_val, 4)
                gini    = round(2 * auc_val - 1, 4)
                logger.info(
                    "performance  AUC=%.4f  Gini=%.4f  bad_rate_actual=%.1f%%  bad_rate_predicted=%.1f%%",
                    auc_roc, gini, bad_rate_actual * 100, bad_rate_predicted * 100,
                )
    else:
        logger.info("labels not yet available for loans starting %s — performance metrics skipped", snapshot_date)

    # ---- PSI: compare current scores to all prior prediction periods ----
    psi_score = float("nan")
    # Reference = prior months scored AFTER train_cutoff. Training-window scores are
    # in-sample and over-confident, so including them inflates PSI with false alarms.
    date_str = snapshot_date.replace("-", "_")
    cutoff_str = CONFIG["ml"]["train_cutoff"].replace("-", "_")
    import glob as _glob
    from utils.common.config import table_dir
    all_pred_files = sorted(_glob.glob(table_dir("gold", "predictions") + "*.parquet"))
    prior_files = [
        f for f in all_pred_files
        if cutoff_str <= os.path.splitext(os.path.basename(f))[0][-10:] < date_str
    ]
    if prior_files:
        ref_df    = spark.read.parquet(*prior_files)
        ref_pdf   = ref_df.select("score").toPandas()
        ref_scores = ref_pdf["score"].values.astype(float)
        psi_score  = round(_psi(ref_scores, scores), 4)
        label = (
            "no shift" if psi_score < 0.10 else
            "minor shift" if psi_score < 0.25 else
            "MAJOR SHIFT — consider retraining"
        )
        logger.info("PSI=%.4f  (%s, reference n=%d)", psi_score, label, len(ref_scores))
        _plot_psi_bins(ref_scores, scores, psi_score, snapshot_date, _REPORT_DIR)
    else:
        logger.info("no out-of-time reference months before %s — PSI not computable", snapshot_date)

    # ---- plots ----
    _plot_score_dist(scores, bad_rate_actual, snapshot_date, _REPORT_DIR)

    # ---- write monitoring row ----
    row = pd.DataFrame([{
        "snapshot_date":       snapshot_date,
        "model_id":            model_id,
        "n_scored":            n_scored,
        "n_labelled":          n_labelled,
        "bad_rate_actual":     bad_rate_actual,
        "bad_rate_predicted":  bad_rate_predicted,
        "auc_roc":             auc_roc,
        "gini":                gini,
        "psi_score":           psi_score,
    }])

    result_df = (
        spark.createDataFrame(row)
        .withColumn("snapshot_date",      col("snapshot_date").cast("date"))
        .withColumn("model_id",           col("model_id").cast(StringType()))
        .withColumn("n_scored",           col("n_scored").cast(IntegerType()))
        .withColumn("n_labelled",         col("n_labelled").cast(IntegerType()))
        .withColumn("bad_rate_actual",    col("bad_rate_actual").cast(FloatType()))
        .withColumn("bad_rate_predicted", col("bad_rate_predicted").cast(FloatType()))
        .withColumn("auc_roc",            col("auc_roc").cast(FloatType()))
        .withColumn("gini",               col("gini").cast(FloatType()))
        .withColumn("psi_score",          col("psi_score").cast(FloatType()))
        .withColumn("monitored_at",       F.current_timestamp())
    )

    validate_and_write(result_df, "gold", "model_monitoring", snapshot_date, ["snapshot_date"])
    _plot_trend(spark, _REPORT_DIR)
    spark.stop()


def main():
    parser = argparse.ArgumentParser(description="monitor model performance for one snapshot month")
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(args.snapshotdate)


if __name__ == "__main__":
    main()
