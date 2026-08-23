from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    results_dir = Path(__file__).resolve().parent
    csv_path = results_dir / "benchmark_comparison.csv"

    df = pd.read_csv(csv_path)

    metrics = df["Metric"].tolist()
    baseline = df["Baseline"].astype(float).tolist()
    our_model = df["Our Model"].astype(float).tolist()
    improvements = df["Improvement"].astype(str).str.replace("+", "", regex=False).astype(float).tolist()

    # Chart 1: Baseline vs Our model comparison
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9.5, 9))

    y = list(range(len(metrics)))
    bar_height = 0.36

    ax.barh([i - bar_height / 2 for i in y], baseline, bar_height, label="Baseline", color="#d65f5f")
    ax.barh([i + bar_height / 2 for i in y], our_model, bar_height, label="Our Model", color="#3f7fdb")

    ax.set_xlabel("Score", fontsize=13, color="black", fontweight="bold")
    ax.set_ylabel("Metrics", fontsize=13, color="black", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=12, color="black", fontweight="bold")
    ax.tick_params(axis="x", labelsize=12, colors="black")
    ax.tick_params(axis="y", labelsize=12, colors="black")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_color("black")
    ax.legend(frameon=False, labelcolor="black", prop={"weight": "bold"})
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_color("black")
    fig.tight_layout()

    comparison_path = results_dir / "benchmark_metrics_comparison.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Net improvement bar chart
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    x = list(range(len(metrics)))

    adjusted_improvements = []
    for metric, val in zip(metrics, improvements):
        if metric == "HR":
            adjusted_improvements.append(-val)
        else:
            adjusted_improvements.append(val)

    colors = ["#1ca34a" if val >= 0 else "#e52424" for val in adjusted_improvements]
    bars = ax2.bar(metrics, adjusted_improvements, color=colors)

    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.set_title("Per-Metric Improvement of Our Model", fontsize=17, weight="bold")
    ax2.set_ylabel("Net Improvement (HR inverted)", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=24, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, adjusted_improvements):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.006 if y >= 0 else -0.006
        ax2.text(bar.get_x() + bar.get_width() / 2, y + offset, f"{val:+.4f}", ha="center", va=va, fontsize=12)

    fig2.tight_layout()
    improvement_path = results_dir / "benchmark_improvement_graph.png"
    fig2.savefig(improvement_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved: {comparison_path}")
    print(f"Saved: {improvement_path}")


if __name__ == "__main__":
    main()
