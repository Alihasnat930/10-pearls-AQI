"""
SHAP Explainability Service
Provides model interpretability using SHAP values
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")  # Use non-interactive backend
import base64
from io import BytesIO

import matplotlib.pyplot as plt

from backend.core.database import get_database
from backend.services.prediction_pipeline import AQIPredictor


class SHAPService:
    """Service for generating SHAP explanations"""

    def __init__(self):
        """Initialize SHAP service"""
        self.predictor = AQIPredictor()
        self.db = get_database()
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer for XGBoost model"""
        try:
            # Use TreeExplainer for XGBoost
            self.explainer = shap.TreeExplainer(self.predictor.xgboost_model)
            print("✅ SHAP TreeExplainer initialized")
        except Exception as e:
            print(f"⚠️ SHAP explainer initialization failed: {e}")
            self.explainer = None

    def explain(self, location: str, hours_ahead: int = 24, model_name: str = "xgboost") -> Dict:
        """
        Generate SHAP explanation for prediction

        Args:
            location: City name
            hours_ahead: Hours ahead to explain (default 24)
            model_name: Model to explain (xgboost or random_forest)

        Returns:
            Dictionary with SHAP values and explanation
        """
        try:
            if not self.explainer:
                self._initialize_explainer()

            # Get recent data
            df = self.db.get_recent_data(hours=168, table="live_data", location=location)

            if df.empty or len(df) < 50:
                raise ValueError(f"Insufficient data for {location}")

            # Prepare features
            features = self.predictor.prepare_features(df)

            if len(features) == 0:
                raise ValueError("Could not prepare features")

            # Get latest feature row
            X = features.iloc[-1:].values
            feature_names = features.columns.tolist()

            # Generate prediction
            if model_name == "xgboost":
                prediction = self.predictor.xgboost_model.predict(X)[0]
                explainer = self.explainer
            else:
                # For Random Forest, create new explainer
                prediction = self.predictor.random_forest_model.predict(X)[0]
                explainer = shap.TreeExplainer(self.predictor.random_forest_model)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X)

            # Get base value (expected value)
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]

            # Create SHAP value list with feature impacts
            shap_list = []
            feature_values = X[0]

            for i, (fname, sval, fval) in enumerate(
                zip(feature_names, shap_values[0], feature_values)
            ):
                shap_list.append(
                    {
                        "feature_name": fname,
                        "shap_value": float(sval),
                        "feature_value": float(fval),
                        "impact": "positive" if sval > 0 else "negative",
                    }
                )

            # Sort by absolute SHAP value
            shap_list_sorted = sorted(shap_list, key=lambda x: abs(x["shap_value"]), reverse=True)

            # Get top 10 features
            top_features = [item["feature_name"] for item in shap_list_sorted[:10]]

            # Generate waterfall plot
            plot_base64 = self._generate_waterfall_plot(
                shap_values[0], feature_values, feature_names, base_value, prediction
            )

            return {
                "prediction": float(prediction),
                "base_value": float(base_value),
                "shap_values": shap_list_sorted[:20],  # Return top 20
                "top_features": top_features,
                "explanation_plot": plot_base64,
                "model": model_name,
                "location": location,
            }

        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return None

    def _generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str],
        base_value: float,
        prediction: float,
    ) -> str:
        """
        Generate SHAP waterfall plot as base64 string

        Returns:
            Base64 encoded PNG image
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Sort by absolute SHAP value
            indices = np.argsort(np.abs(shap_values))[-15:]  # Top 15 features

            # Create waterfall data
            sorted_shap = shap_values[indices]
            sorted_features = [feature_names[i] for i in indices]
            sorted_values = feature_values[indices]

            # Plot
            colors = ["#ff6b6b" if val < 0 else "#51cf66" for val in sorted_shap]
            y_pos = np.arange(len(sorted_features))

            ax.barh(y_pos, sorted_shap, color=colors, alpha=0.7, edgecolor="black")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [f"{feat} = {val:.2f}" for feat, val in zip(sorted_features, sorted_values)],
                fontsize=10,
            )
            ax.set_xlabel("SHAP Value (Impact on AQI)", fontsize=12, fontweight="bold")
            ax.set_title(
                f"Top 15 Feature Contributions to AQI Prediction\nBase: {base_value:.1f} → Prediction: {prediction:.1f}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
            ax.grid(axis="x", alpha=0.3)

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#51cf66", label="Increases AQI"),
                Patch(facecolor="#ff6b6b", label="Decreases AQI"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")

            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)

            return image_base64

        except Exception as e:
            print(f"Plot generation error: {e}")
            return None

    def get_feature_importance(self, model_name: str = "xgboost") -> Dict:
        """
        Get overall feature importance

        Returns:
            Dictionary with feature importance rankings
        """
        try:
            # Get feature names
            with open("feature_columns.txt", "r") as f:
                feature_names = [line.strip() for line in f.readlines()]

            if model_name == "xgboost":
                # XGBoost feature importance
                importance = self.predictor.xgboost_model.get_score(importance_type="gain")

                # Convert to sorted list
                importance_list = [
                    {"feature": feat, "importance": float(imp)} for feat, imp in importance.items()
                ]
            else:
                # Random Forest feature importance
                importance = self.predictor.random_forest_model.feature_importances_

                importance_list = [
                    {"feature": fname, "importance": float(imp)}
                    for fname, imp in zip(feature_names, importance)
                ]

            # Sort by importance
            importance_list_sorted = sorted(
                importance_list, key=lambda x: x["importance"], reverse=True
            )

            return {
                "model": model_name,
                "feature_importance": importance_list_sorted,
                "top_10": importance_list_sorted[:10],
            }

        except Exception as e:
            print(f"Feature importance error: {e}")
            return None
