"""
app/services/analytics_service.py
Analytics engine for the dashboard: aggregates historical session metrics,
handles interactive filtering, builds chart datasets, and produces data-driven recommendations.
"""

import os
import pandas as pd
from flask import current_app
from app.models import PredictionHistory

_CACHED_DF = None


class AnalyticsService:
    @staticmethod
    def get_dataset() -> pd.DataFrame:
        """Loads and caches the raw dataset for dashboard slicing."""
        global _CACHED_DF
        if _CACHED_DF is not None:
            return _CACHED_DF

        csv_path = current_app.config.get("DATASET_PATH")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run scripts/download_data.py.")

        df = pd.read_csv(csv_path)
        df["Revenue"] = df["Revenue"].astype(bool)
        df["Weekend"] = df["Weekend"].astype(bool)
        df["OperatingSystems"] = df["OperatingSystems"].astype(str)
        df["Browser"] = df["Browser"].astype(str)
        df["Region"] = df["Region"].astype(str)
        df["TrafficType"] = df["TrafficType"].astype(str)
        _CACHED_DF = df
        return _CACHED_DF

    @classmethod
    def get_dashboard_data(
        cls,
        month: str = None,
        visitor_type: str = None,
        traffic_type: str = None,
        weekend: str = None
    ) -> dict:
        """Computes live KPIs and chart structures, filtered dynamically."""
        df = cls.get_dataset().copy()

        # Apply interactive filters if provided
        if month and month != "all":
            df = df[df["Month"].str.lower() == month.lower()]
        if visitor_type and visitor_type != "all":
            df = df[df["VisitorType"].str.lower() == visitor_type.lower()]
        if traffic_type and traffic_type != "all":
            df = df[df["TrafficType"] == str(traffic_type)]
        if weekend and weekend != "all":
            is_weekend = weekend.lower() == "true"
            df = df[df["Weekend"] == is_weekend]

        total_sessions = len(df)
        if total_sessions == 0:
            return cls._empty_dashboard_response()

        conversions = int(df["Revenue"].sum())
        non_conversions = total_sessions - conversions
        conversion_rate = float(conversions / total_sessions)

        # Visitor breakdown
        visitor_counts = df["VisitorType"].value_counts().to_dict()
        returning_visitors = int(visitor_counts.get("Returning_Visitor", 0))
        new_visitors = int(visitor_counts.get("New_Visitor", 0))
        other_visitors = int(visitor_counts.get("Other", 0))

        # Engagement averages
        avg_product_duration = float(df["ProductRelated_Duration"].mean())
        avg_product_pages = float(df["ProductRelated"].mean())
        avg_bounce_rate = float(df["BounceRates"].mean())
        avg_exit_rate = float(df["ExitRates"].mean())

        # 1. Monthly conversion chart
        month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_group = df.groupby("Month")["Revenue"].agg(["count", "mean"]).reindex(month_order).dropna()
        months_chart = {
            "labels": month_group.index.tolist(),
            "conversion_rates": [round(float(v) * 100, 2) for v in month_group["mean"]],
            "session_counts": [int(v) for v in month_group["count"]]
        }

        # 2. Visitor type chart
        vis_group = df.groupby("VisitorType")["Revenue"].agg(["count", "mean"])
        visitor_chart = {
            "labels": [str(l).replace("_", " ") for l in vis_group.index],
            "conversion_rates": [round(float(v) * 100, 2) for v in vis_group["mean"]],
            "session_counts": [int(v) for v in vis_group["count"]]
        }

        # 3. Traffic type chart (top 10 by session volume)
        traffic_vol = df["TrafficType"].value_counts().head(10).index
        traffic_group = df[df["TrafficType"].isin(traffic_vol)].groupby("TrafficType")["Revenue"].agg(["count", "mean"]).loc[traffic_vol]
        traffic_chart = {
            "labels": [f"Traffic {t}" for t in traffic_group.index],
            "conversion_rates": [round(float(v) * 100, 2) for v in traffic_group["mean"]],
            "session_counts": [int(v) for v in traffic_group["count"]]
        }

        # 4. Weekend conversion chart
        weekend_group = df.groupby("Weekend")["Revenue"].agg(["count", "mean"])
        weekend_chart = {
            "labels": ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"],
            "conversion_rates": [
                round(float(weekend_group.loc[False, "mean"]) * 100, 2) if False in weekend_group.index else 0.0,
                round(float(weekend_group.loc[True, "mean"]) * 100, 2) if True in weekend_group.index else 0.0
            ],
            "session_counts": [
                int(weekend_group.loc[False, "count"]) if False in weekend_group.index else 0,
                int(weekend_group.loc[True, "count"]) if True in weekend_group.index else 0
            ]
        }

        # 5. Product Duration Brackets vs Conversion
        bins = [-1, 60, 300, 900, 1800, float("inf")]
        labels = ["< 1 min", "1 - 5 mins", "5 - 15 mins", "15 - 30 mins", "30+ mins"]
        dur_series = pd.cut(df["ProductRelated_Duration"], bins=bins, labels=labels)
        dur_group = df.groupby(dur_series, observed=False)["Revenue"].agg(["count", "mean"])
        duration_chart = {
            "labels": labels,
            "conversion_rates": [round(float(v) * 100, 2) if pd.notnull(v) else 0.0 for v in dur_group["mean"]],
            "session_counts": [int(v) if pd.notnull(v) else 0 for v in dur_group["count"]]
        }

        # Database prediction logs stats
        db_total_preds = PredictionHistory.query.count()
        db_purchases = PredictionHistory.query.filter_by(predicted_class=1).count()
        db_conv_rate = (db_purchases / db_total_preds) if db_total_preds > 0 else 0.0

        # Business recommendations backed by data
        recommendations = cls._generate_business_recommendations(df, conversion_rate)

        return {
            "kpis": {
                "total_sessions": total_sessions,
                "conversions": conversions,
                "non_conversions": non_conversions,
                "conversion_rate": round(conversion_rate * 100, 2),
                "returning_visitors": returning_visitors,
                "new_visitors": new_visitors,
                "other_visitors": other_visitors,
                "avg_product_duration_secs": round(avg_product_duration, 1),
                "avg_product_duration_mins": round(avg_product_duration / 60.0, 1),
                "avg_product_pages": round(avg_product_pages, 1),
                "avg_bounce_rate": round(avg_bounce_rate * 100, 2),
                "avg_exit_rate": round(avg_exit_rate * 100, 2),
                "live_predictions_logged": db_total_preds,
                "live_predicted_conversions": db_purchases,
                "live_predicted_rate": round(db_conv_rate * 100, 2)
            },
            "charts": {
                "outcome_distribution": {
                    "labels": ["Non-Purchasing Sessions", "Purchased Sessions"],
                    "counts": [non_conversions, conversions]
                },
                "monthly_conversion": months_chart,
                "visitor_conversion": visitor_chart,
                "traffic_conversion": traffic_chart,
                "weekend_conversion": weekend_chart,
                "duration_conversion": duration_chart
            },
            "recommendations": recommendations,
            "active_filters": {
                "month": month or "all",
                "visitor_type": visitor_type or "all",
                "traffic_type": traffic_type or "all",
                "weekend": weekend or "all"
            }
        }

    @staticmethod
    def _generate_business_recommendations(df: pd.DataFrame, base_rate: float) -> list:
        """Derives actionable business recommendations based on statistical segment disparities."""
        recs = []

        # 1. Exit rate drop-off
        high_exit_df = df[df["ExitRates"] > 0.05]
        low_exit_df = df[df["ExitRates"] <= 0.02]
        if len(low_exit_df) > 50 and len(high_exit_df) > 50:
            low_exit_rate = low_exit_df["Revenue"].mean()
            high_exit_rate = high_exit_df["Revenue"].mean()
            recs.append({
                "title": "Mitigate Mid-Funnel Exit Hazard",
                "severity": "high",
                "badge": "Critical Funnel Fix",
                "metric": f"{high_exit_rate*100:.1f}% vs {low_exit_rate*100:.1f}%",
                "description": (
                    f"Sessions with exit rates > 5% convert at only {high_exit_rate*100:.1f}%, compared to "
                    f"{low_exit_rate*100:.1f}% for sessions with exit rates below 2%. "
                    "Action: Deploy timely exit-intent assistance, simplified checkout navigation, "
                    "and visible money-back guarantees on product pages."
                )
            })

        # 2. Seasonality surge
        nov_df = df[df["Month"] == "Nov"]
        feb_df = df[df["Month"] == "Feb"]
        if len(nov_df) > 50 and len(feb_df) > 50:
            nov_rate = nov_df["Revenue"].mean()
            feb_rate = feb_df["Revenue"].mean()
            recs.append({
                "title": "Capitalize on Q4 Peak Buying Velocity",
                "severity": "medium",
                "badge": "Marketing Optimization",
                "metric": f"Nov {nov_rate*100:.1f}% vs Feb {feb_rate*100:.1f}%",
                "description": (
                    f"November sessions show peak commercial intent with {nov_rate*100:.1f}% conversion, "
                    f"nearly 14x higher than February's {feb_rate*100:.1f}%. "
                    "Action: Allocate ad spend budget heavily toward October/November retargeting "
                    "and run early-bird holiday bundling to maximize purchase volume."
                )
            })

        # 3. New vs Returning visitor behavior
        new_df = df[df["VisitorType"] == "New_Visitor"]
        ret_df = df[df["VisitorType"] == "Returning_Visitor"]
        if len(new_df) > 50 and len(ret_df) > 50:
            new_rate = new_df["Revenue"].mean()
            ret_rate = ret_df["Revenue"].mean()
            recs.append({
                "title": "Welcome Incentive for First-Time Shoppers",
                "severity": "info",
                "badge": "Conversion Boost",
                "metric": f"New: {new_rate*100:.1f}% | Returning: {ret_rate*100:.1f}%",
                "description": (
                    f"New visitors convert at {new_rate*100:.1f}%, reflecting targeted inbound traffic. "
                    "Action: Implement personalized first-order discounts and immediate guest checkout "
                    "to reduce friction for new visitors before session decay occurs."
                )
            })

        # 4. Long browsing time conversion
        deep_df = df[df["ProductRelated_Duration"] > 900]
        if len(deep_df) > 50:
            deep_rate = deep_df["Revenue"].mean()
            recs.append({
                "title": "Engage High-Duration Consideration Sessions",
                "severity": "success",
                "badge": "Retention Strategy",
                "metric": f"{deep_rate*100:.1f}% conversion (15+ min)",
                "description": (
                    f"Visitors who spend 15+ minutes on product pages achieve a {deep_rate*100:.1f}% conversion rate. "
                    "Action: Introduce automated live chat prompts or stock scarcity indicators "
                    "('Only 3 items left!') after 8 minutes of browsing to prompt decision making."
                )
            })

        return recs

    @classmethod
    def _empty_dashboard_response(cls) -> dict:
        return {
            "kpis": {
                "total_sessions": 0, "conversions": 0, "non_conversions": 0,
                "conversion_rate": 0.0, "returning_visitors": 0, "new_visitors": 0,
                "other_visitors": 0, "avg_product_duration_secs": 0.0,
                "avg_product_duration_mins": 0.0, "avg_product_pages": 0.0,
                "avg_bounce_rate": 0.0, "avg_exit_rate": 0.0,
                "live_predictions_logged": 0, "live_predicted_conversions": 0, "live_predicted_rate": 0.0
            },
            "charts": {
                "outcome_distribution": {"labels": [], "counts": []},
                "monthly_conversion": {"labels": [], "conversion_rates": [], "session_counts": []},
                "visitor_conversion": {"labels": [], "conversion_rates": [], "session_counts": []},
                "traffic_conversion": {"labels": [], "conversion_rates": [], "session_counts": []},
                "weekend_conversion": {"labels": [], "conversion_rates": [], "session_counts": []},
                "duration_conversion": {"labels": [], "conversion_rates": [], "session_counts": []}
            },
            "recommendations": [],
            "active_filters": {}
        }
