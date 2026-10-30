"""
4-Parameter Logistic (4-PL) Hill Curve Parallelism & 96-Well Spatial Edge-Effect Diagnostic Engine
Agent 2: Principal Analytical Scientist & Hybrid Physics+ML Modeling Specialist
Regulatory Basis: USP <1032>, <1033>, <1034> & ICH Q2(R2)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def hill_4pl(conc_ng_ml: np.ndarray, a_min: float, d_max: float, c_ec50: float, b_slope: float) -> np.ndarray:
    """
    Standard 4-Parameter Logistic (4-PL) Dose-Response Function:
    y(x) = D + (A - D) / (1 + (x / C)^B)
    where A = lower asymptote, D = upper asymptote, C = EC50, B = Hill slope.
    """
    return d_max + (a_min - d_max) / (1.0 + (conc_ng_ml / c_ec50) ** b_slope)


def evaluate_bioassay_4pl_and_plate_edge(
    include_outer_edge_wells: bool = False
) -> Dict[str, Any]:
    """
    Computes 4-PL dose-response curves for Reference Standard (RS-04),
    In-Spec Release Batch (PROJ-001 PPQ #2), and Thermally Degraded / Edge-Biased sample,
    plus a full 8x12 (96-well) spatial residual heatmap showing evaporative edge effects.
    """
    # 11-point serial dilution (0.1 ng/mL to 10,000 ng/mL)
    conc_points = np.logspace(-1, 4, 11)
    conc_smooth = np.logspace(-1, 4, 120)

    # Reference Standard (RS-04) Parameters
    ref_params = {"A": 12.0, "D": 195.0, "EC50": 42.0, "B": 1.18}

    # Test Batch (PROJ-001 PPQ Lot #2):
    # If outer perimeter wells are included, evaporative concentration in Row A/H inflates upper asymptote & distorts Hill slope
    if include_outer_edge_wells:
        test_params = {"A": 15.8, "D": 218.5, "EC50": 37.8, "B": 1.49}
    else:
        # Clean Inner-60 Wells (PBS Moat in Outer-36 Wells)
        test_params = {"A": 12.4, "D": 193.8, "EC50": 43.1, "B": 1.16}

    # Positive Control Stressed / Non-Parallel Degraded Sample (for comparison)
    stressed_params = {"A": 18.0, "D": 162.0, "EC50": 78.5, "B": 0.72}

    # Smooth curves
    df_curves = pd.DataFrame({
        "Concentration_ng_mL": conc_smooth,
        "Log10_Conc": np.round(np.log10(conc_smooth), 3),
        "Reference_Standard_RLU": np.round(
            hill_4pl(conc_smooth, ref_params["A"], ref_params["D"], ref_params["EC50"], ref_params["B"]), 2
        ),
        "Test_Sample_PPQ_RLU": np.round(
            hill_4pl(conc_smooth, test_params["A"], test_params["D"], test_params["EC50"], test_params["B"]), 2
        ),
        "Forced_Degraded_NonParallel_RLU": np.round(
            hill_4pl(conc_smooth, stressed_params["A"], stressed_params["D"], stressed_params["EC50"], stressed_params["B"]), 2
        ),
    })

    # Discrete 11-point observations with deterministic pseudo-replicate noise
    rng = np.random.default_rng(42)
    df_points = pd.DataFrame({
        "Concentration_ng_mL": np.round(conc_points, 3),
        "Log10_Conc": np.round(np.log10(conc_points), 3),
        "Ref_Observed_RLU": np.round(
            hill_4pl(conc_points, ref_params["A"], ref_params["D"], ref_params["EC50"], ref_params["B"])
            + rng.normal(0, 2.1, size=len(conc_points)),
            2,
        ),
        "Test_Observed_RLU": np.round(
            hill_4pl(conc_points, test_params["A"], test_params["D"], test_params["EC50"], test_params["B"])
            + rng.normal(0, 3.8 if include_outer_edge_wells else 1.9, size=len(conc_points)),
            2,
        ),
    })

    # USP <1034> Parallelism & Relative Potency Metrics
    slope_ratio = round(test_params["B"] / ref_params["B"], 3)
    upper_asymptote_ratio = round(test_params["D"] / ref_params["D"], 3)
    relative_potency_pct = round((ref_params["EC50"] / test_params["EC50"]) * 100.0, 1)

    parallelism_pass = (0.80 <= slope_ratio <= 1.25) and (0.90 <= upper_asymptote_ratio <= 1.10)

    # 96-Well Spatial Residual Heatmap (8 Rows A-H x 12 Columns 1-12)
    rows_labels = ["Row A", "Row B", "Row C", "Row D", "Row E", "Row F", "Row G", "Row H"]
    cols_labels = [f"Col {c}" for c in range(1, 13)]

    residual_matrix = np.zeros((8, 12))
    for r in range(8):
        for c in range(12):
            is_outer_row = r in (0, 7)
            is_outer_col = c in (0, 11)
            is_corner = is_outer_row and is_outer_col
            # Deterministic spatial thermal/evaporative bias in % residual RLU
            base_noise = float(np.sin((r + 1) * 1.7 + (c + 1) * 0.9) * 1.4)
            if is_corner:
                edge_bias = 13.8
            elif is_outer_row or is_outer_col:
                edge_bias = 9.4
            else:
                edge_bias = 0.0
            residual_matrix[r, c] = round(base_noise + edge_bias, 2)

    df_plate_heatmap = pd.DataFrame(residual_matrix, index=rows_labels, columns=cols_labels)

    outer_wells_vals = [
        residual_matrix[r, c]
        for r in range(8)
        for c in range(12)
        if r in (0, 7) or c in (0, 11)
    ]
    inner_wells_vals = [
        residual_matrix[r, c]
        for r in range(1, 7)
        for c in range(1, 11)
    ]

    return {
        "df_curves": df_curves,
        "df_points": df_points,
        "df_plate_heatmap": df_plate_heatmap,
        "metrics": {
            "mode": "All 96 Wells (Outer Perimeter Included)" if include_outer_edge_wells else "Inner-60 Wells Only (PBS Moat in Outer-36)",
            "ref_ec50_ng_ml": ref_params["EC50"],
            "test_ec50_ng_ml": test_params["EC50"],
            "relative_potency_pct": relative_potency_pct,
            "hill_slope_ref": ref_params["B"],
            "hill_slope_test": test_params["B"],
            "hill_slope_ratio": slope_ratio,
            "upper_asymptote_ratio": upper_asymptote_ratio,
            "parallelism_gate_status": (
                "PASS (USP <1034> Parallelism Verified)"
                if parallelism_pass
                else "FAIL — System Suitability Parallelism Gate Breached (Outer-Well Evaporation Bias)"
            ),
            "outer_36_mean_residual_pct": round(float(np.mean(outer_wells_vals)), 2),
            "inner_60_mean_residual_pct": round(float(np.mean(inner_wells_vals)), 2),
        },
    }


if __name__ == "__main__":
    clean = evaluate_bioassay_4pl_and_plate_edge(include_outer_edge_wells=False)
    print("Inner-60 Metrics:", clean["metrics"])
    edge = evaluate_bioassay_4pl_and_plate_edge(include_outer_edge_wells=True)
    print("Outer-Included Metrics:", edge["metrics"])
