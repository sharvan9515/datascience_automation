from __future__ import annotations

from typing import List, Dict, Any


class IntelligentModelSelector:
    """Heuristically rank ML algorithms based on dataset profile."""

    @staticmethod
    def _dataset_size(profile: Dict[str, Any]) -> int:
        """Return the number of rows inferred from the statistical summary."""
        stats = profile.get("statistical_summary", {})
        for metrics in stats.values():
            count = metrics.get("count")
            if isinstance(count, (int, float)):
                return int(count)
        return 0

    @classmethod
    def select_optimal_algorithms(
        cls, profile: Dict[str, Any] | None, task_type: str, timeseries_mode: bool = False
    ) -> List[str]:
        """Return a prioritized list of algorithm names."""
        if profile is None:
            profile = {}

        cm = profile.get("complexity_metrics", {})
        ratio = cm.get("feature_target_ratio", 0.0)
        imbalance = cm.get("class_imbalance") or {}
        noise = cm.get("noise_level", 0.0)
        n_rows = cls._dataset_size(profile)

        recommendations: List[str] = []

        if timeseries_mode:
            if task_type == "classification":
                return ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"]
            return ["XGBRegressor", "LGBMRegressor", "RandomForestRegressor"]

        if task_type == "classification":
            if n_rows < 1000:
                recommendations.extend(["LogisticRegression", "SVC"])
            else:
                recommendations.extend(["RandomForestClassifier", "XGBClassifier"])

            if imbalance and max(imbalance.values()) > 0.8:
                recommendations.insert(0, "RandomForestClassifier")
                recommendations.insert(0, "XGBClassifier")

            if ratio > 1:
                recommendations.append("LGBMClassifier")

            if noise > 0.1:
                recommendations.append("RandomForestClassifier")
        else:
            if n_rows < 1000:
                recommendations.extend(["LinearRegression", "SVR"])
            else:
                recommendations.extend(["RandomForestRegressor", "XGBRegressor"])

            if ratio > 1:
                recommendations.append("LGBMRegressor")

            if noise > 0.1:
                recommendations.append("RandomForestRegressor")

        # remove duplicates while preserving order
        seen = set()
        prioritized = []
        for alg in recommendations:
            if alg not in seen:
                prioritized.append(alg)
                seen.add(alg)
        return prioritized
