#!/usr/bin/env python3
"""
Generate performance comparison plots from WoeBoost data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.switch_backend("Agg")
plt.style.use("default")


def load_performance_data():
    """Load performance data from JSON files."""
    data_dir = Path("outputs/data")

    try:
        with open(data_dir / "woeboost_performance_standard.json") as f:
            standard_data = json.load(f)
        with open(data_dir / "woeboost_performance_freethreaded.json") as f:
            freethreaded_data = json.load(f)
        return standard_data, freethreaded_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run ./run_performance_comparison.sh first to generate performance data")
        return None, None


def create_performance_chart(standard_data, freethreaded_data):
    """Create performance comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data
    standard_time = standard_data["total_time"]
    freethreaded_time = freethreaded_data["total_time"]
    standard_iter_time = standard_data.get(
        "avg_time_per_iteration", standard_data.get("average_time_per_iteration", 0.0)
    )
    freethreaded_iter_time = freethreaded_data.get(
        "avg_time_per_iteration",
        freethreaded_data.get("average_time_per_iteration", 0.0),
    )
    speedup = standard_time / freethreaded_time

    metrics = ["Training Time (s)", "Time per Iteration (s)", "Speedup Factor"]
    standard_values = [standard_time, standard_iter_time, 1.0]
    freethreaded_values = [freethreaded_time, freethreaded_iter_time, speedup]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        standard_values,
        width,
        label="Standard Python (GIL)",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        freethreaded_values,
        width,
        label="Free-threaded Python",
        color="#4ECDC4",
        alpha=0.8,
    )

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        if i < 2:  # Time metrics
            ax.text(
                bar1.get_x() + bar1.get_width() / 2.0,
                height1 + height1 * 0.01,
                f"{height1:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax.text(
                bar2.get_x() + bar2.get_width() / 2.0,
                height2 + height2 * 0.01,
                f"{height2:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        else:  # Speedup
            ax.text(
                bar1.get_x() + bar1.get_width() / 2.0,
                height1 + height1 * 0.01,
                f"{height1:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax.text(
                bar2.get_x() + bar2.get_width() / 2.0,
                height2 + height2 * 0.01,
                f"{height2:.1f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_xlabel("Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Values", fontsize=14, fontweight="bold")
    ax.set_title(
        "WoeBoost Performance: GIL vs Free-threaded Python",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_resource_chart(standard_data, freethreaded_data):
    """Create resource utilization chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate CPU usage based on task counts
    standard_cpu = (standard_data["n_tasks"] / 8) * 100  # Assume 8-core system
    freethreaded_cpu = min((freethreaded_data["n_tasks"] / 8) * 100, 100)
    speedup = standard_data["total_time"] / freethreaded_data["total_time"]

    categories = ["CPU Usage (%)", "Tasks Used", "Speedup Factor"]
    standard_values = [standard_cpu, standard_data["n_tasks"], 1.0]
    freethreaded_values = [freethreaded_cpu, freethreaded_data["n_tasks"], speedup]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        standard_values,
        width,
        label="Standard Python (GIL)",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        freethreaded_values,
        width,
        label="Free-threaded Python",
        color="#4ECDC4",
        alpha=0.8,
    )

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        if i == 2:  # Speedup factor
            ax.text(
                bar1.get_x() + bar1.get_width() / 2.0,
                height1 + height1 * 0.01,
                f"{height1:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax.text(
                bar2.get_x() + bar2.get_width() / 2.0,
                height2 + height2 * 0.01,
                f"{height2:.1f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        else:
            ax.text(
                bar1.get_x() + bar1.get_width() / 2.0,
                height1 + height1 * 0.01,
                f"{height1:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax.text(
                bar2.get_x() + bar2.get_width() / 2.0,
                height2 + height2 * 0.01,
                f"{height2:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_xlabel("Resource Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Values", fontsize=14, fontweight="bold")
    ax.set_title("Resource Utilization Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_metrics_table(standard_data, freethreaded_data):
    """Create metrics comparison table."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    # Calculate metrics
    time_speedup = standard_data["total_time"] / freethreaded_data["total_time"]
    iter_speedup = standard_data.get(
        "avg_time_per_iteration", standard_data.get("average_time_per_iteration", 1.0)
    ) / freethreaded_data.get(
        "avg_time_per_iteration",
        freethreaded_data.get("average_time_per_iteration", 1.0),
    )
    time_saved = standard_data["total_time"] - freethreaded_data["total_time"]
    percent_reduction = (time_saved / standard_data["total_time"]) * 100

    table_data = [
        ["Metric", "Standard Python (GIL)", "Free-threaded Python", "Improvement"],
        [
            "Training Time",
            f"{standard_data['total_time']:.3f}s",
            f"{freethreaded_data['total_time']:.3f}s",
            f"{time_speedup:.2f}× faster",
        ],
        [
            "Time per Iteration",
            f"{standard_data.get('avg_time_per_iteration', standard_data.get('average_time_per_iteration', 0.0)):.4f}s",
            f"{freethreaded_data.get('avg_time_per_iteration', freethreaded_data.get('average_time_per_iteration', 0.0)):.4f}s",
            f"{iter_speedup:.2f}× faster",
        ],
        [
            "Task Configuration",
            f"{standard_data['n_tasks']} tasks (GIL limited)",
            f"{freethreaded_data['n_tasks']} tasks (full parallelism)",
            f"{freethreaded_data['n_tasks'] / standard_data['n_tasks']:.0f}× more tasks",
        ],
        [
            "Python Version",
            standard_data["python_version"].split()[0],
            freethreaded_data["python_version"].split()[0],
            "Free-threading enabled",
        ],
        [
            "Time Saved",
            "—",
            f"{time_saved:.3f}s",
            f"{percent_reduction:.1f}% reduction",
        ],
    ]

    table = ax.table(cellText=table_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#E8E8E8")
        table[(0, i)].set_text_props(weight="bold")

    # Style improvement column
    if len(table_data[0]) >= 4:
        for i in range(1, len(table_data)):
            table[(i, 3)].set_facecolor("#E8F5E8")
            table[(i, 3)].set_text_props(weight="bold")

    plt.title("Performance Metrics Comparison", fontsize=16, fontweight="bold", pad=20)
    return fig


def main():
    """Generate all plots."""
    print("Loading performance data...")
    standard_data, freethreaded_data = load_performance_data()

    if not standard_data or not freethreaded_data:
        return

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(exist_ok=True)

    print("Generating performance comparison chart...")
    fig1 = create_performance_chart(standard_data, freethreaded_data)
    fig1.savefig(plots_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    print("Generating resource utilization chart...")
    fig2 = create_resource_chart(standard_data, freethreaded_data)
    fig2.savefig(plots_dir / "resource_utilization.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("Generating metrics table...")
    fig3 = create_metrics_table(standard_data, freethreaded_data)
    fig3.savefig(plots_dir / "metrics_table.png", dpi=450, bbox_inches="tight")
    plt.close(fig3)

    # Print summary
    speedup = standard_data["total_time"] / freethreaded_data["total_time"]
    print("\n✅ Plots generated successfully!")
    print("📊 Performance Summary:")
    print(f"   Standard Python: {standard_data['total_time']:.3f}s")
    print(f"   Free-threaded:   {freethreaded_data['total_time']:.3f}s")
    print(f"   Speedup:         {speedup:.2f}× faster")
    print("\n📁 Files saved to outputs/plots/")


if __name__ == "__main__":
    main()
