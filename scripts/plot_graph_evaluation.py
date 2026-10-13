#!/usr/bin/env python3
"""Create report-ready PNG figures for the Long-Term Graph Detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

BLUE = "#2563eb"
GREEN = "#16a34a"
ORANGE = "#ea580c"
RED = "#dc2626"
SLATE = "#334155"
LABELS = {
    "confirmed_relationship": "Confirmed\nrelationship",
    "counter_enhanced_gae_alert": "Counter-enhanced\nGAE",
    "any_graph_signal": "Combined\nreview signal",
}


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def title(ax: plt.Axes, text: str, subtitle: str) -> None:
    ax.set_title(text, fontsize=18, fontweight="bold", pad=22)
    ax.text(
        0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, color="#64748b",
    )


def load_rows(hybrid_dir: str, labels_dir: str):
    table = ds.dataset(
        hybrid_dir, format="parquet", partitioning="hive", exclude_invalid_files=True
    ).to_table()
    rows = table.to_pylist()
    label_rows = ds.dataset(
        labels_dir, format="parquet", partitioning="hive"
    ).to_table(columns=["dataset_day", "user"]).to_pylist()
    positives = {
        (int(row["dataset_day"]), str(row["user"]))
        for row in label_rows if 17 <= int(row["dataset_day"]) <= 30
    }
    y_true = np.asarray(
        [(int(row["dataset_day"]), str(row["user_id"])) in positives for row in rows],
        dtype=bool,
    )
    scores = np.asarray([float(row["gae_score"]) for row in rows])
    return rows, y_true, scores


def metric_summary(metrics: dict, output: Path) -> None:
    keys = list(LABELS)
    names = ["precision", "recall", "f1"]
    colors = [BLUE, GREEN, ORANGE]
    x = np.arange(len(keys))
    width = 0.23
    fig, ax = plt.subplots(figsize=(12, 7))
    for index, (name, color) in enumerate(zip(names, colors)):
        values = [metrics[key][name] for key in keys]
        bars = ax.bar(x + (index - 1) * width, values, width, label=name.title(), color=color)
        ax.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=4, fontweight="bold")
    ax.set_xticks(x, [LABELS[key] for key in keys], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, loc="upper center")
    title(ax, "Long-Term Graph Detector Results", "Held-out Days 17–30")
    fig.text(
        0.5, 0.01,
        "84.06% is the supervised confirmed-relationship recurrence F1, not the GAE F1.",
        ha="center", fontsize=10, color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, output, "01_metric_summary.png")


def confusion_matrices(metrics: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, key in zip(axes, LABELS):
        item = metrics[key]
        matrix = np.asarray([[item["tn"], item["fp"]], [item["fn"], item["tp"]]])
        ax.imshow(matrix + 1, cmap="Blues", norm=LogNorm(vmin=1, vmax=matrix.max() + 1))
        row_totals = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        for row in range(2):
            for column in range(2):
                ax.text(
                    column, row,
                    f"{matrix[row, column]:,}\n({percentages[row, column]:.2%})",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if matrix[row, column] > matrix.max() / 20 else "#0f172a",
                )
        ax.set_xticks([0, 1], ["Predicted normal", "Predicted alert"])
        ax.set_yticks([0, 1], ["Actually normal", "Actually malicious"])
        ax.set_title(LABELS[key].replace("\n", " "), fontweight="bold")
    fig.suptitle("Confusion Matrices", fontsize=19, fontweight="bold")
    fig.text(
        0.5, 0.02, "Counts are shown with percentages normalized within each actual class.",
        ha="center", color="#64748b",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save(fig, output, "02_confusion_matrices.png")


def pr_curve_plot(y_true: np.ndarray, scores: np.ndarray, output: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    prevalence = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(recall, precision, color=ORANGE, linewidth=2.5, label=f"GAE (AP = {ap:.4f})")
    ax.axhline(prevalence, color="#64748b", linestyle="--", label=f"Random baseline = {prevalence:.5f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    title(
        ax, "Precision–Recall Curve: Counter-Enhanced GAE",
        "Held-out Days 17–30; appropriate primary ranking curve for severe class imbalance",
    )
    fig.tight_layout()
    save(fig, output, "03_precision_recall_curve.png")


def roc_curve_plot(y_true: np.ndarray, scores: np.ndarray, output: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(false_positive_rate, true_positive_rate, color=BLUE, linewidth=2.5, label=f"GAE (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", label="Random classifier")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    title(ax, "ROC Curve: Counter-Enhanced GAE", "Held-out Days 17–30")
    fig.tight_layout()
    save(fig, output, "04_roc_curve.png")


def training_loss_plot(training_log: str, output: Path) -> None:
    history = json.loads(Path(training_log).read_text())["history"]
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, losses, marker="o", linewidth=2.5, color=BLUE, label="Training loss")
    ax.scatter(epochs[-1], losses[-1], color=ORANGE, s=80, zorder=3, label="Selected checkpoint")
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy loss", xticks=epochs[::2])
    ax.grid(alpha=0.25)
    ax.legend()
    title(
        ax, "GAE Training Loss",
        "Validation loss was not recorded by this training run; it is intentionally not fabricated",
    )
    fig.tight_layout()
    save(fig, output, "05_training_loss.png")


def score_timeline(rows: list[dict], y_true: np.ndarray, threshold: float, output: Path) -> None:
    days = np.arange(17, 31)
    background_tail, malicious_max, malicious_mean = [], [], []
    row_days = np.asarray([int(row["dataset_day"]) for row in rows])
    row_scores = np.asarray([float(row["gae_score"]) for row in rows])
    for day in days:
        day_mask = row_days == day
        background_tail.append(np.quantile(row_scores[day_mask & ~y_true], 0.999))
        attacks = row_scores[day_mask & y_true]
        malicious_max.append(float(attacks.max()) if len(attacks) else np.nan)
        malicious_mean.append(float(attacks.mean()) if len(attacks) else np.nan)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(days, background_tail, marker="o", color=BLUE, label="Normal-user 99.9th percentile")
    ax.plot(days, malicious_max, marker="o", linewidth=2.5, color=RED, label="Malicious-user maximum")
    ax.plot(days, malicious_mean, marker="s", color=ORANGE, label="Malicious-user mean")
    ax.axhline(threshold, color=GREEN, linestyle="--", linewidth=2, label=f"Alert threshold {threshold:.6f}")
    ax.set(xlabel="Dataset day", ylabel="Counter-enhanced GAE score", xticks=days, ylim=(0, 1.005))
    ax.grid(alpha=0.25)
    ax.legend(ncols=2)
    title(ax, "Graph Anomaly Scores Over Time", "Missing malicious points indicate days with no labelled malicious user-day")
    fig.tight_layout()
    save(fig, output, "06_scores_over_time.png")


def threshold_tradeoff(y_true: np.ndarray, scores: np.ndarray, selected: float, output: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) > 0,
    )
    mask = thresholds >= 0.90
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds[mask], precision[mask], color=BLUE, label="Precision")
    ax.plot(thresholds[mask], recall[mask], color=GREEN, label="Recall")
    ax.plot(thresholds[mask], f1[mask], color=ORANGE, linewidth=2.5, label="F1")
    ax.axvline(selected, color=RED, linestyle="--", label=f"Frozen threshold {selected:.6f}")
    ax.set(xlabel="GAE alert threshold", ylabel="Metric", xlim=(0.90, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    title(
        ax, "Threshold Trade-off: Counter-Enhanced GAE",
        "Test labels are shown for evaluation only; the frozen threshold was selected on validation data",
    )
    fig.tight_layout()
    save(fig, output, "07_threshold_tradeoff.png")


def relationship_example(rows: list[dict], output: Path) -> None:
    row = next(item for item in rows if item["confirmed_relationship_alert"])
    match = row["relationship_matches"][0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    positions = [(2, 3), (6, 3), (10, 3)]
    labels = [
        (match["source_computer"], "Source computer"),
        (row["user_id"], "Acting user"),
        (match["destination_computer"], "Destination computer"),
    ]
    colors = ["#dbeafe", "#ffedd5", "#dcfce7"]
    for (x, y), (value, kind), color in zip(positions, labels, colors):
        circle = plt.Circle((x, y), 1.05, facecolor=color, edgecolor=SLATE, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.15, value, ha="center", va="center", fontweight="bold", fontsize=13)
        ax.text(x, y - 0.35, kind, ha="center", va="center", fontsize=10, color="#64748b")
    ax.annotate("", xy=(4.9, 3), xytext=(3.1, 3), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(8.9, 3), xytext=(7.1, 3), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(4, 3.55, "authentication context", ha="center", fontsize=11)
    ax.text(8, 3.55, "access relationship", ha="center", fontsize=11)
    ax.set_title("Example Confirmed Relationship Match", fontsize=19, fontweight="bold", pad=20)
    ax.text(
        6, 0.85,
        f"Dataset day {row['dataset_day']} • {row['relationship_match_count']} matching events • evidence {match['source_reference']}",
        ha="center", fontsize=11, color=SLATE,
    )
    ax.text(
        6, 0.35,
        "The high-confidence channel matches the full (user, source computer, destination computer) triple.",
        ha="center", fontsize=10, color="#64748b",
    )
    fig.tight_layout()
    save(fig, output, "08_relationship_example.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", default="outputs/graph_evaluation/hybrid_alerts")
    parser.add_argument("--labels", default="outputs/graph_evaluation/redteam_labels")
    parser.add_argument("--training-log", default="outputs/graph_evaluation/trained_model/training_log.json")
    parser.add_argument("--output", default="docs/images/graph_results")
    args = parser.parse_args()

    output = Path(args.output)
    report = json.loads((Path(args.hybrid) / "metrics.json").read_text())
    metrics = report["metrics"]
    rows, y_true, scores = load_rows(args.hybrid, args.labels)
    threshold = float(report["novelty_threshold"])
    metric_summary(metrics, output)
    confusion_matrices(metrics, output)
    pr_curve_plot(y_true, scores, output)
    roc_curve_plot(y_true, scores, output)
    training_loss_plot(args.training_log, output)
    score_timeline(rows, y_true, threshold, output)
    threshold_tradeoff(y_true, scores, threshold, output)
    relationship_example(rows, output)
    print(f"Created 8 figures in {output}")


if __name__ == "__main__":
    main()
