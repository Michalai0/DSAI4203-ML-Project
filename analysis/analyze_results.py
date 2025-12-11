"""
Analyze and visualize training results.
Reads saved JSON results from scripts/training/train.py and
generates comparison plots.
"""
import json
import os
from glob import glob
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def find_latest_results(results_dir: str) -> Optional[str]:
    """Return the most recently modified results file."""
    files = glob(os.path.join(results_dir, "results_*.json"))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_test_accuracy(results: Dict[str, Any], out_path: str):
    models = list(results.keys())
    accs = [results[m]["test_accuracy"] for m in models]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, accs, color="skyblue")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Test Accuracy Comparison")
    plt.ylim(0, max(accs) * 1.1 if accs else 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.2f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_training_curves(results: Dict[str, Any], out_path: str):
    plt.figure(figsize=(8, 5))

    for model_name, res in results.items():
        history = res.get("history", {})
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label=f"{model_name} Train")
        if val_loss:
            plt.plot(epochs, val_loss, label=f"{model_name} Val", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy_curves(results: Dict[str, Any], out_path: str):
    plt.figure(figsize=(8, 5))

    for model_name, res in results.items():
        history = res.get("history", {})
        train_acc = history.get("train_accuracy", [])
        val_acc = history.get("val_accuracy", [])
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, label=f"{model_name} Train")
        if val_acc:
            plt.plot(epochs, val_acc, label=f"{model_name} Val", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_class_metrics(
    report: Dict[str, Any],
    metric: str,
    out_path: str,
    top_k: Optional[int] = None,
):
    """
    Plot per-class metric bar chart from classification_report dict.
    metric: 'precision' | 'recall' | 'f1-score'
    """
    if not report:
        print("No classification report data; skip per-class plot.")
        return

    rows = []
    for cls, vals in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        if metric in vals:
            rows.append((cls, vals[metric]))

    if not rows:
        print(f"No metric '{metric}' found in report; skip.")
        return

    # Sort by metric descending
    rows.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]

    labels, values = zip(*rows)
    fig_height = max(6, 0.35 * len(labels) + 2)
    plt.figure(figsize=(10, fig_height))
    bars = plt.barh(range(len(values)), values, color="cornflowerblue")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} by class")
    plt.gca().invert_yaxis()  # Highest on top
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for idx, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{values[idx]:.3f}",
            va="center",
            ha="left",
        )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_precision_f1_combined(
    report: Dict[str, Any],
    out_path: str,
    top_k: Optional[int] = None,
):
    """Plot precision与f1-score同图（左右子图对比），便于对照每个类别。"""
    if not report:
        print("No classification report data; skip combined precision/f1 plot.")
        return

    rows = []
    for cls, vals in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        prec = vals.get("precision")
        f1 = vals.get("f1-score")
        if prec is not None and f1 is not None:
            rows.append((cls, prec, f1))

    if not rows:
        print("No precision/f1 data found; skip combined plot.")
        return

    # 按 f1-score 排序，便于观察表现最好的类别
    rows.sort(key=lambda x: x[2], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]

    labels = [r[0] for r in rows]
    precisions = [r[1] for r in rows]
    f1_scores = [r[2] for r in rows]

    fig_height = max(6, 0.35 * len(labels) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, fig_height), sharey=True)

    y_pos = np.arange(len(labels))

    axes[0].barh(y_pos, precisions, color="steelblue")
    axes[0].set_title("Precision by class")
    axes[0].set_xlabel("Precision")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    axes[1].barh(y_pos, f1_scores, color="salmon")
    axes[1].set_title("F1-score by class")
    axes[1].set_xlabel("F1-score")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    for idx in range(len(labels)):
        axes[0].text(
            precisions[idx] + 0.005,
            y_pos[idx],
            f"{precisions[idx]:.3f}",
            va="center",
            ha="left",
        )
        axes[1].text(
            f1_scores[idx] + 0.005,
            y_pos[idx],
            f"{f1_scores[idx]:.3f}",
            va="center",
            ha="left",
        )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main(results_file: Optional[str] = None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(project_root, "analysis")
    results_dir = os.path.join(analysis_dir, "results")
    figures_dir = os.path.join(analysis_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Locate results file
    if results_file is None:
        results_file = find_latest_results(results_dir)
        if results_file is None:
            print(f"No results file found in {results_dir}")
            return
        print(f"Using latest results: {results_file}")
    else:
        if not os.path.isfile(results_file):
            print(f"Specified results file not found: {results_file}")
            return

    data = load_results(results_file)
    results = data.get("results", {})
    metadata = data.get("metadata", {})

    print("Metadata:", metadata)
    print("Models:", list(results.keys()))

    # Plot and save figures
    base_name = os.path.splitext(os.path.basename(results_file))[0]

    acc_bar_path = os.path.join(figures_dir, f"{base_name}_test_accuracy.png")
    loss_curve_path = os.path.join(figures_dir, f"{base_name}_loss.png")
    acc_curve_path = os.path.join(figures_dir, f"{base_name}_accuracy.png")

    plot_test_accuracy(results, acc_bar_path)
    plot_training_curves(results, loss_curve_path)
    plot_accuracy_curves(results, acc_curve_path)

    # Per-class metrics for each model
    top_k = 30  # adjust here if needed
    for model_name, res in results.items():
        report = res.get("classification_report", {})
        cls_prec_path = os.path.join(figures_dir, f"{base_name}_{model_name}_precision_top{top_k}.png")
        cls_rec_path = os.path.join(figures_dir, f"{base_name}_{model_name}_recall_top{top_k}.png")
        cls_f1_path = os.path.join(figures_dir, f"{base_name}_{model_name}_f1_top{top_k}.png")
        cls_combined_path = os.path.join(figures_dir, f"{base_name}_{model_name}_precision_f1_top{top_k}.png")

        plot_class_metrics(report, "precision", cls_prec_path, top_k=top_k)
        plot_class_metrics(report, "recall", cls_rec_path, top_k=top_k)
        plot_class_metrics(report, "f1-score", cls_f1_path, top_k=top_k)
        plot_precision_f1_combined(report, cls_combined_path, top_k=top_k)

    print("Saved figures:")
    print(" -", acc_bar_path)
    print(" -", loss_curve_path)
    print(" -", acc_curve_path)
    for model_name in results.keys():
        print(f" - {base_name}_{model_name}_precision_top{top_k}.png")
        print(f" - {base_name}_{model_name}_recall_top{top_k}.png")
        print(f" - {base_name}_{model_name}_f1_top{top_k}.png")
        print(f" - {base_name}_{model_name}_precision_f1_top{top_k}.png")


if __name__ == "__main__":
    main()

