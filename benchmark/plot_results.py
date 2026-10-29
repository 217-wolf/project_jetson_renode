import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "benchmark/reports/results.csv"
PLOTS_DIR = Path("benchmark/reports/plots")

METRICS_FOR_HEATMAP = [
    "precision", "recall", "f1_score", "mAP50", "fps", "avg_confidence"
]


def load_results():
    with open(RESULTS) as f:
        return list(csv.DictReader(f))


def bar_plot(rows, metric, title, ylabel, filename):
    labels = [r["experiment"] for r in rows]
    values = [float(r[metric]) for r in rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="#4C72B0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


def summary_dashboard(rows, filename="summary_dashboard.png"):
    experiment_names = [r["experiment"] for r in rows]

    # surowe dane do table
    table_data = [
        [r[m] for m in METRICS_FOR_HEATMAP]
        for r in rows
    ]

    # dane do heatmapy znormalizowne do 1
    raw_matrix = np.array(
        [[float(r[m]) for m in METRICS_FOR_HEATMAP] for r in rows]
    )
    col_min = raw_matrix.min(axis=0)
    col_max = raw_matrix.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)
    normalized = (raw_matrix - col_min) / col_range

    fig, (ax_table, ax_heat) = plt.subplots(
        2, 1, figsize=(10, 2 + 0.6 * len(rows) * 2),
        gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # table
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_data,
        rowLabels=experiment_names,
        colLabels=METRICS_FOR_HEATMAP,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax_table.set_title("Benchmark results - raw metrics", fontsize=12, pad=20)

    # heatmap
    im = ax_heat.imshow(normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(METRICS_FOR_HEATMAP)))
    ax_heat.set_xticklabels(METRICS_FOR_HEATMAP, rotation=30, ha="right")
    ax_heat.set_yticks(range(len(experiment_names)))
    ax_heat.set_yticklabels(experiment_names)
    ax_heat.set_title(
        "Relative performance per metric (green = best, red = worst)",
        fontsize=11, pad=10
    )

    # surowe wartości w heatmape
    for i in range(len(experiment_names)):
        for j in range(len(METRICS_FOR_HEATMAP)):
            ax_heat.text(
                j, i, f"{raw_matrix[i, j]:.3f}",
                ha="center", va="center", fontsize=8, color="black"
            )

    fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label="normalized (0=worst, 1=best)")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


def main():
    rows = load_results()

    bar_plot(rows, "mAP50", "mAP50 by experiment", "mAP50", "map50.png")
    bar_plot(rows, "recall", "Recall by experiment", "Recall", "recall.png")
    bar_plot(rows, "precision", "Precision by experiment", "Precision", "precision.png")
    bar_plot(rows, "fps", "FPS by experiment", "FPS", "fps.png")
    bar_plot(rows, "latency_ms", "Latency by experiment", "Latency (ms)", "latency.png")

    summary_dashboard(rows)


if __name__ == "__main__":
    main()