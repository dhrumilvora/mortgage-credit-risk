from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def build_numeric_distribution_data(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> dict[str, dict]:
    """Build descriptive statistics for numerical features."""

    exclude = exclude or set()

    numeric = df.select_dtypes(include=np.number).drop(
        columns=exclude,
        errors="ignore",
    )

    results = {}

    for feature in numeric.columns:
        series = numeric[feature]

        results[feature] = {
            "count": int(series.count()),
            "missing_pct": float(series.isna().mean()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "skewness": float(series.skew()),
            "min": float(series.min()),
            "p01": float(series.quantile(0.01)),
            "p05": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
        }

    return results


def plot_numeric_distribution(
    series: pd.Series,
    feature: str,
    stats: dict,
    ax,
) -> None:
    """Plot the empirical distribution of a numerical feature."""

    values = series.dropna()

    ax.hist(
        values,
        bins=40,
        density=True,
        alpha=0.6,
        edgecolor="black",
    )

    ax.axvline(
        stats["mean"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {stats['mean']:.2f}",
    )

    ax.axvline(
        stats["median"],
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {stats['median']:.2f}",
    )

    ax.set_title(f"{feature} Distribution")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()


def build_numerical_distribution_report(
    df: pd.DataFrame,
    output_path: Path,
    exclude: set[str] | None = None,
) -> Path:
    """Build a PDF containing one numerical distribution per page."""

    exclude = exclude or set()

    numeric = df.select_dtypes(include=np.number).drop(
        columns=exclude,
        errors="ignore",
    )

    statistics = build_numeric_distribution_data(
        df,
        exclude=exclude,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for feature in numeric.columns:
            stats = statistics[feature]

            fig = plt.figure(figsize=(11, 8.5))

            ax = fig.add_axes([0.08, 0.48, 0.84, 0.42])

            plot_numeric_distribution(
                numeric[feature],
                feature,
                stats,
                ax,
            )

            summary_text = (
                f"Count: {stats['count']:,}\n"
                f"Missing: {stats['missing_pct']:.2%}\n"
                f"Mean: {stats['mean']:.2f}\n"
                f"Median: {stats['median']:.2f}\n"
                f"Std: {stats['std']:.2f}\n"
                f"Skewness: {stats['skewness']:.2f}\n\n"
                f"Min: {stats['min']:.2f}\n"
                f"P01: {stats['p01']:.2f}\n"
                f"P05: {stats['p05']:.2f}\n"
                f"P25: {stats['p25']:.2f}\n"
                f"P50: {stats['p50']:.2f}\n"
                f"P75: {stats['p75']:.2f}\n"
                f"P95: {stats['p95']:.2f}\n"
                f"P99: {stats['p99']:.2f}\n"
                f"Max: {stats['max']:.2f}"
            )

            fig.text(
                0.10,
                0.08,
                summary_text,
                fontsize=10,
                verticalalignment="bottom",
                family="monospace",
            )

            fig.suptitle(
                "Numerical Feature Distribution Report",
                fontsize=16,
                fontweight="bold",
            )

            pdf.savefig(fig)
            plt.close(fig)

    return output_path
