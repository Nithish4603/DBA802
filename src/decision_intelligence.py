import numpy as np
import pandas as pd


def risk_level(p):
    if p < 0.25:
        return "Low"
    if p < 0.50:
        return "Moderate"
    if p < 0.75:
        return "High"
    return "Critical"


def recommendation(level):
    return {
        "Low": "Continue monitoring.",
        "Moderate": "Increase logging and inspect suspicious endpoints.",
        "High": "Escalate to security operations and isolate suspicious traffic.",
        "Critical": "Trigger immediate incident-response review and containment procedures.",
    }[level]


def build_decision_output(meta_test, predictions, probabilities):
    out = meta_test.reset_index(drop=True).copy()
    out["predicted_label"] = np.asarray(predictions, dtype=int)
    out["attack_probability"] = np.asarray(probabilities, dtype=float)
    out["risk_level"] = out["attack_probability"].map(risk_level)
    out["recommended_action"] = out["risk_level"].map(recommendation)
    return out
