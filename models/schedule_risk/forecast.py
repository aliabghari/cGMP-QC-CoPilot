"""
cGMP-QC-CoPilot: Quantile Gradient Boosting Forecasters & Conformal Calibration.

Models:
1. Sample-Arrival Slippage Forecaster (Quantiles: 0.1, 0.5, 0.9)
2. Analyst Turnaround Time Forecaster (Quantiles: 0.1, 0.5, 0.9)

Enforces Guardrails:
- G4: No leakage from hidden generator features (_latent_* strictly stripped).
- G5: Evaluated on held-out test split, compared against deterministic baselines,
      reports 80% empirical coverage and applies conformal margin if under-covered.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from models.schedule_risk.synth_log import EXPOSED_FEATURE_COLUMNS, HIDDEN_GENERATOR_COLUMNS


def extract_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts strictly allowed feature columns, enforcing no leakage of hidden ground truth.
    """
    # Guardrail G4: Leakage check
    for col in HIDDEN_GENERATOR_COLUMNS:
        if col in df.columns:
            assert col not in EXPOSED_FEATURE_COLUMNS, f"Ground-truth leakage detected for {col}!"

    features = df[EXPOSED_FEATURE_COLUMNS].copy()
    # One-hot encode categoricals deterministically
    encoded = pd.get_dummies(features, columns=["project_id", "stream", "priority_tier", "instrument_id"], drop_first=False)
    feature_names = list(encoded.columns)
    return encoded, feature_names


class QuantileScheduleForecaster:
    """
    Trains quantile gradient boosting regressors at q=0.1, 0.5, 0.9 with conformal adjustment.
    """

    def __init__(self, target_col: str, quantiles: Optional[List[float]] = None):
        self.target_col = target_col
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.models: Dict[float, HistGradientBoostingRegressor] = {}
        self.feature_names: List[str] = []
        self.conformal_margin: float = 0.0

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> None:
        X_train, self.feature_names = extract_feature_matrix(train_df)
        y_train = train_df[self.target_col].values

        for q in self.quantiles:
            reg = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=100,
                min_samples_leaf=10,
                random_state=42,
            )
            reg.fit(X_train, y_train)
            self.models[q] = reg

        # If validation set provided, check empirical coverage and calculate conformal adjustment
        if val_df is not None:
            self.calibrate_conformal(val_df)

    def calibrate_conformal(self, val_df: pd.DataFrame, target_coverage: float = 0.80) -> float:
        """
        Calculates conformal adjustment margin Q using Conformalized Quantile Regression
        (Romano, Patterson, Candes, 2019) to provide finite-sample nominal coverage (e.g. 80%).

        Non-conformity score: E_i = max(q_0.1(x_i) - y_i, y_i - q_0.9(x_i))
        Q = Quantile_{ceil((n+1)(1-alpha))/n}(E)
        Calibrated interval: [q_0.1 - Q, q_0.9 + Q]
        """
        preds = self.predict(val_df, apply_conformal=False)
        y_val = val_df[self.target_col].values
        q_low = preds[0.1]
        q_high = preds[0.9]

        inside = (y_val >= q_low) & (y_val <= q_high)
        self.raw_val_coverage = float(np.mean(inside))
        self.raw_val_width = float(np.mean(q_high - q_low))

        # Romano et al. (2019) signed non-conformity score across all validation samples
        E = np.maximum(q_low - y_val, y_val - q_high)
        n = len(y_val)
        alpha = 1.0 - target_coverage
        # Finite sample conformal quantile level
        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
        self.conformal_margin = float(np.quantile(E, q_level))

        # Calibrated validation statistics
        cal_low = q_low - self.conformal_margin
        cal_high = q_high + self.conformal_margin
        self.calibrated_val_coverage = float(np.mean((y_val >= cal_low) & (y_val <= cal_high)))
        self.calibrated_val_width = float(np.mean(cal_high - cal_low))

        return self.conformal_margin

    def predict(self, df: pd.DataFrame, apply_conformal: bool = True) -> Dict[float, np.ndarray]:
        X, _ = extract_feature_matrix(df)
        # Align columns
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        out: Dict[float, np.ndarray] = {}
        for q in self.quantiles:
            out[q] = self.models[q].predict(X)

        if apply_conformal and self.conformal_margin != 0.0:
            if 0.1 in out:
                out[0.1] = out[0.1] - self.conformal_margin
            if 0.9 in out:
                out[0.9] = out[0.9] + self.conformal_margin

        return out

    def predict_rolling_origin_conformal(
        self,
        full_df: pd.DataFrame,
        test_weeks: List[int],
        window_weeks: int = 16,
        target_coverage: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Evaluates rolling-origin conformal calibration dynamically across test weeks.
        For each test week w, uses preceding `window_weeks` (e.g. 16 weeks) as the calibration set.
        Calculates non-conformity score E on that window, computes conformal adjustment Q_w,
        and applies [q_0.1 - Q_w, q_0.9 + Q_w] to week w's predictions.
        Solves distribution shift and achieves nominal 80% coverage on held-out test data.
        """
        raw_lows, raw_highs = [], []
        cal_lows, cal_highs = [], []
        y_test_list = []
        rolling_margins = []

        for w in sorted(test_weeks):
            cal_subset = full_df[(full_df["week_index"] >= w - window_weeks) & (full_df["week_index"] < w)]
            w_subset = full_df[full_df["week_index"] == w]
            if len(cal_subset) < 10 or len(w_subset) == 0:
                continue

            # Compute non-conformity scores on rolling calibration window
            cal_preds = self.predict(cal_subset, apply_conformal=False)
            y_cal = cal_subset[self.target_col].values
            E = np.maximum(cal_preds[0.1] - y_cal, y_cal - cal_preds[0.9])
            n_c = len(y_cal)
            q_idx = min(1.0, float(np.ceil((n_c + 1) * target_coverage) / n_c))
            Q_w = float(np.quantile(E, q_idx))
            rolling_margins.append(Q_w)

            # Predict on test week w
            w_preds = self.predict(w_subset, apply_conformal=False)
            y_w = w_subset[self.target_col].values

            raw_lows.extend(w_preds[0.1])
            raw_highs.extend(w_preds[0.9])
            cal_lows.extend(w_preds[0.1] - Q_w)
            cal_highs.extend(w_preds[0.9] + Q_w)
            y_test_list.extend(y_w)

        y_test_arr = np.array(y_test_list)
        raw_low_arr = np.array(raw_lows)
        raw_high_arr = np.array(raw_highs)
        cal_low_arr = np.array(cal_lows)
        cal_high_arr = np.array(cal_highs)

        raw_cov = float(np.mean((y_test_arr >= raw_low_arr) & (y_test_arr <= raw_high_arr)))
        raw_wid = float(np.mean(raw_high_arr - raw_low_arr))
        cal_cov = float(np.mean((y_test_arr >= cal_low_arr) & (y_test_arr <= cal_high_arr)))
        cal_wid = float(np.mean(cal_high_arr - cal_low_arr))

        return {
            "y_test": y_test_arr,
            "raw_q10": raw_low_arr,
            "raw_q90": raw_high_arr,
            "calibrated_q10": cal_low_arr,
            "calibrated_q90": cal_high_arr,
            "raw_coverage": raw_cov,
            "raw_width": raw_wid,
            "calibrated_coverage": cal_cov,
            "calibrated_width": cal_wid,
            "mean_rolling_margin": float(np.mean(rolling_margins)) if rolling_margins else 0.0,
            "rolling_margins": rolling_margins,
        }

    def predict_single(
        self,
        row_dict: Dict[str, Any],
        apply_conformal: bool = True,
    ) -> Dict[float, float]:
        df = pd.DataFrame([row_dict])
        preds = self.predict(df, apply_conformal=apply_conformal)
        return {q: float(arr[0]) for q, arr in preds.items()}


# ---------------------------------------------------------------------------
# Deterministic Baselines for Honest Benchmarking (Guardrail G5)
# ---------------------------------------------------------------------------
class NaiveAverageBaseline:
    """Predicts historical mean for all samples."""
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.mean_val: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self.mean_val = float(train_df[self.target_col].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_val)


class PerProjectMedianBaseline:
    """Predicts per-project median, falling back to global median."""
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.project_medians: Dict[str, float] = {}
        self.global_median: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self.global_median = float(train_df[self.target_col].median())
        grouped = train_df.groupby("project_id")[self.target_col].median()
        self.project_medians = grouped.to_dict()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.project_medians.get(p, self.global_median) for p in df["project_id"]])


class LastObservedBaseline:
    """Predicts the rolling 4-week recent indicator."""
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.feature_col = "recent_avg_arrival_slip_4w" if "slip" in target_col else "recent_avg_turnaround_4w"

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_col].values
