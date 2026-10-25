"""
scripts/analyze_data.py
Performs Exploratory Data Analysis (EDA) on the UCI Online Shoppers Purchasing Intention Dataset.
Generates statistical reports in JSON and visual figures in reports/figures/.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

def run_analysis():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_csv = os.path.join(base_dir, "data", "raw", "online_shoppers_intention.csv")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(raw_csv):
        print(f"[!] Error: Dataset not found at {raw_csv}. Run scripts/download_data.py first.")
        return False

    print(f"[*] Reading dataset from: {raw_csv}")
    df = pd.read_csv(raw_csv)

    total_sessions = len(df)
    duplicates_count = int(df.duplicated().sum())
    missing_count = int(df.isnull().sum().sum())

    # Standardize boolean target
    df["Revenue"] = df["Revenue"].astype(bool)

    # Class balance
    rev_counts = df["Revenue"].value_counts().to_dict()
    conv_count = int(rev_counts.get(True, 0))
    non_conv_count = int(rev_counts.get(False, 0))
    overall_conv_rate = float(conv_count / total_sessions) if total_sessions > 0 else 0.0

    print(f"    Total Sessions: {total_sessions:,}")
    print(f"    Duplicates: {duplicates_count}")
    print(f"    Missing Values: {missing_count}")
    print(f"    Converted: {conv_count:,} ({overall_conv_rate*100:.2f}%)")
    print(f"    Non-Converted: {non_conv_count:,} ({(1-overall_conv_rate)*100:.2f}%)")

    # Segment conversion rates
    def calc_rate_table(col):
        grouped = df.groupby(col)["Revenue"].agg(["count", "mean", "sum"]).reset_index()
        grouped.columns = [str(col), "total_sessions", "conversion_rate", "conversions"]
        grouped["conversion_rate"] = grouped["conversion_rate"].round(4)
        return grouped.to_dict(orient="records")

    month_rates = calc_rate_table("Month")
    visitor_rates = calc_rate_table("VisitorType")
    traffic_rates = calc_rate_table("TrafficType")
    weekend_rates = calc_rate_table("Weekend")
    os_rates = calc_rate_table("OperatingSystems")
    browser_rates = calc_rate_table("Browser")
    region_rates = calc_rate_table("Region")

    # Numerical feature stats
    num_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    num_stats = df[num_cols].describe().round(4).to_dict()

    # Correlation with Revenue
    correlations = {}
    for col in num_cols:
        corr_val = float(df[col].corr(df["Revenue"].astype(int)))
        correlations[col] = round(corr_val, 4)

    # Export EDA summary JSON
    eda_summary = {
        "dataset_name": "UCI Online Shoppers Purchasing Intention Dataset",
        "total_sessions": total_sessions,
        "duplicate_rows": duplicates_count,
        "missing_values": missing_count,
        "conversions": conv_count,
        "non_conversions": non_conv_count,
        "conversion_rate": round(overall_conv_rate, 4),
        "target_distribution": {
            "True": conv_count,
            "False": non_conv_count
        },
        "visitor_type_rates": visitor_rates,
        "month_rates": month_rates,
        "traffic_type_rates": traffic_rates,
        "weekend_rates": weekend_rates,
        "operating_systems_rates": os_rates,
        "browser_rates": browser_rates,
        "region_rates": region_rates,
        "numerical_stats": num_stats,
        "correlations_with_target": correlations,
        "data_leakage_note": (
            "PageValues has an extraordinarily high correlation with Revenue (~0.49). "
            "In e-commerce analytics, PageValues reflects the average value of a web page visited by the user "
            "prior to completing an e-commerce transaction. Including PageValues creates post-hoc target leakage "
            "for real-time pre-purchase prediction when a visitor is actively browsing. "
            "Hence, PageValues is excluded from the real-time operational classification model."
        )
    }

    eda_json_path = os.path.join(reports_dir, "eda_summary.json")
    with open(eda_json_path, "w") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"[+] Saved EDA summary JSON to {eda_json_path}")

    # Generate Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = ["#1e293b", "#0284c7"]

    # 1. Revenue Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["No Purchase (False)", "Purchase (True)"], [non_conv_count, conv_count], color=["#94a3b8", "#0284c7"], width=0.5)
    ax.set_title("Dataset Session Distribution by Revenue Outcome", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Sessions")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 150, f"{h:,} ({h/total_sessions*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, total_sessions * 0.95)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "revenue_distribution.png"), dpi=200)
    plt.close(fig)

    # 2. Conversion by Month
    m_df = pd.DataFrame(month_rates)
    # Order by natural calendar months represented
    cal_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m_df["Month"] = pd.Categorical(m_df["Month"], categories=cal_order, ordered=True)
    m_df = m_df.sort_values("Month")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(m_df["Month"], m_df["conversion_rate"] * 100, marker="o", color="#0284c7", linewidth=2.5, markersize=7)
    ax.set_title("Historical Conversion Rate (%) by Month", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", alpha=0.6)
    for x, y in zip(m_df["Month"], m_df["conversion_rate"] * 100):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "conversion_by_month.png"), dpi=200)
    plt.close(fig)

    # 3. Conversion by Visitor Type
    v_df = pd.DataFrame(visitor_rates)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(v_df["VisitorType"], v_df["conversion_rate"] * 100, color=["#0284c7", "#38bdf8", "#64748b"], width=0.45)
    ax.set_title("Conversion Rate (%) by Visitor Type", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Conversion Rate (%)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(v_df["conversion_rate"] * 100) * 1.25)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "conversion_by_visitor.png"), dpi=200)
    plt.close(fig)

    # 4. Correlation with Target Feature Bar Plot
    corr_series = pd.Series(correlations).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#ef4444" if v < 0 else ("#f59e0b" if k == "PageValues" else "#0284c7") for k, v in corr_series.items()]
    corr_series.plot(kind="barh", color=bar_colors, ax=ax)
    ax.set_title("Linear Correlation of Numerical Features with Revenue (Target)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Pearson Correlation Coefficient")
    for i, (k, v) in enumerate(corr_series.items()):
        ax.text(v + (0.01 if v >= 0 else -0.04), i, f"{v:+.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.25, 0.55)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "feature_correlations.png"), dpi=200)
    plt.close(fig)

    print(f"[+] Saved EDA figures to: {figures_dir}")
    print("[SUCCESS] Exploratory Data Analysis complete!")
    return True

if __name__ == "__main__":
    run_analysis()
