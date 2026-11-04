"""
Publication-Grade High-DPI Chart Generation for Retrieval Benchmarks and Conversational Evals.
Generates:
1. benchmarks/benchmark_summary_chart.png (20,000 Doc Qdrant + Reranker Benchmark on NVIDIA A100)
2. benchmarks/conversational_eval_chart.png (Golden Dataset Multi-Turn RAG Evaluation)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_benchmark_summary_chart(csv_path: str, output_png_path: str):
    """
    Generate High-DPI summary bar chart from h100_benchmark_results.csv.
    """
    df = pd.read_csv(csv_path)

    # Clean model labels for presentation
    model_display_names = {
        "infgrad/stella-base-en-v2": "Stella-Base-v2\n(Dense Baseline)",
        "ai-sage/Giga-Embeddings-instruct": "Giga-Embeddings\n(Instruct Dense)",
        "ai-sage/Giga-Embeddings-instruct + BGE-Base (Pre-trained)": "Giga + BGE-Base\n(Zero-Shot Reranker)",
        "ai-sage/Giga-Embeddings-instruct + BGE-Finetuned": "Giga + BGE-Finetuned\n(Domain Cross-Encoder)"
    }
    df["DisplayName"] = df["Model"].map(lambda x: model_display_names.get(x, x))

    metrics = ["Recall", "MRR", "NDCG"]
    metric_labels = ["Recall@3", "MRR@3", "NDCG@3"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Color palette: elegant navy, teal, and amber
    colors = ["#2b5c8f", "#17a2b8", "#f39c12"]

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        offsets = x + (i - 1) * width
        bars = ax.bar(offsets, df[metric], width, label=label, color=color, alpha=0.92, edgecolor="#ffffff", linewidth=1.2)

        # Add data value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#333333"
            )

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        "NVIDIA A100 Empirical Retrieval Benchmark (20,000 InsuranceQA Docs / 2,000 Test Queries)",
        fontsize=14,
        fontweight="bold",
        pad=18
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["DisplayName"], fontsize=10, fontweight="bold")
    ax.set_ylim(0.20, 0.46)
    ax.legend(frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotation highlighting the key reranker insight
    ax.annotate(
        "Zero-Shot Degradation:\n-9.96% MRR drop\n(lexical distribution shift)",
        xy=(2, 0.2659),
        xytext=(1.55, 0.22),
        arrowprops=dict(facecolor="#d9534f", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5,
        fontweight="semibold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf2e9", edgecolor="#e0b48f", alpha=0.95)
    )

    ax.annotate(
        "Domain Fine-Tuning Boost:\n+19.37% MRR vs. Zero-Shot\n(0.2659 -> 0.3174)",
        xy=(3, 0.3174),
        xytext=(2.65, 0.375),
        arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5,
        fontweight="semibold",
        color="#196f3d",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1", edgecolor="#a9dfbf", alpha=0.95)
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.close()
    print(f"Generated benchmark summary chart at {output_png_path}")


def generate_conversational_eval_chart(eval_csv_path: str, output_png_path: str):
    """
    Generate High-DPI comparison chart for Golden Dataset evaluation.
    """
    df = pd.read_csv(eval_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    # Subplot 1: Unfiltered vs. Filtered Retrieval Comparison
    ax1 = axes[0]
    metrics = ["Recall@3", "MRR@3", "Category Precision@3\n(Context Purity)"]
    unfiltered_vals = [
        df["unfiltered_recall_at_3"].mean(),
        df["unfiltered_mrr_at_3"].mean(),
        df["unfiltered_category_precision_at_3"].mean()
    ]
    filtered_vals = [
        df["filtered_recall_at_3"].mean(),
        df["filtered_mrr_at_3"].mean(),
        df["filtered_category_precision_at_3"].mean()
    ]

    x = np.arange(len(metrics))
    width = 0.28

    rects1 = ax1.bar(x - width/2, unfiltered_vals, width, label="Unfiltered Single-Turn RAG", color="#7f8c8d", alpha=0.88)
    rects2 = ax1.bar(x + width/2, filtered_vals, width, label="Multi-Turn Filtered + Reranked RAG", color="#2980b9", alpha=0.92)

    for bar in rects1:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for bar in rects2:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b4f72")

    ax1.set_title("Retrieval Quality & Purity: Unfiltered vs. Multi-Turn Filtered", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Metric Value (0.0 - 1.0)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 1.18)
    ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Subplot 2: Clarification Gate Accuracy & Grounding Rate (%)
    ax2 = axes[1]
    system_metrics = ["Clarification Gate\nAccuracy (Turn 1)", "Category Extraction\nAccuracy", "Citation Grounding\nFaithfulness Rate"]
    system_values = [
        df["t1_clarification_accuracy"].mean() * 100.0,
        df["category_match"].mean() * 100.0,
        df["citation_faithfulness_pct"].mean()
    ]
    bar_colors = ["#16a085", "#27ae60", "#2ecc71"]

    bars = ax2.bar(system_metrics, system_values, width=0.45, color=bar_colors, alpha=0.92, edgecolor="#ffffff", linewidth=1.2)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#145a32")

    ax2.set_title("Production Guardrail Reliability & Citation Grounding (%)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Reliability Percentage (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, 118)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("Golden Dataset Multi-Turn Evaluation Suite (15 Insurance Scenarios)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated conversational eval chart at {output_png_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    h100_csv = os.path.join(base_dir, "benchmarks", "h100_benchmark_results.csv")
    summary_chart = os.path.join(base_dir, "benchmarks", "benchmark_summary_chart.png")
    if os.path.isfile(h100_csv):
        generate_benchmark_summary_chart(h100_csv, summary_chart)

    eval_csv = os.path.join(base_dir, "benchmarks", "golden_eval_results.csv")
    eval_chart = os.path.join(base_dir, "benchmarks", "conversational_eval_chart.png")
    if os.path.isfile(eval_csv):
        generate_conversational_eval_chart(eval_csv, eval_chart)

