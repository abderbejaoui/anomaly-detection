"""Generate a short French explanation + recommendation from a DiagnosticResult.

Strategy:
  - Compute the percentage gap between observed and expected NDVI on the trailing window.
  - Look at the weather summary against known seasonal norms (rain ~ 25 mm/30d in Tunisia
    interior in summer is roughly 'normal'; LST > 35 C anomaly is 'thermal stress').
  - Pick the dominant explanatory factor and emit one of a handful of templates.
  - Recommendation severity scales with status.
"""

from __future__ import annotations

from typing import Any

from .diagnose import DiagnosticResult


# Heuristic thresholds (Tunisia coastal/interior context).
_RAIN_DRY_30D_MM = 5.0          # below this, "très sec sur 30j"
_RAIN_DRY_90D_MM = 25.0         # below this, "saison sèche prolongée"
_LST_HOT_ANOMALY_C = 4.0        # above this, "stress thermique marqué"
_LST_HOT_ABS_C = 38.0           # absolute hot threshold


def _pct_gap(observed: list[float], expected: list[float]) -> float:
    if not observed or not expected:
        return 0.0
    obs_avg = sum(observed) / len(observed)
    exp_avg = sum(expected) / len(expected)
    if exp_avg <= 1e-6:
        return 0.0
    return 100.0 * (obs_avg - exp_avg) / exp_avg


def _explanation_from_weather(
    weather: dict[str, float], gap_pct: float
) -> tuple[str, list[str]]:
    """Return (cause_short, list of factor strings)."""
    factors: list[str] = []
    cause = "stress hydrique"
    rain_30 = weather.get("rain_cum_30d")
    rain_90 = weather.get("rain_cum_90d")
    lst_anom = weather.get("lst_anomaly_30d")
    lst_abs = weather.get("lst_c")

    if rain_30 is not None and rain_30 < _RAIN_DRY_30D_MM:
        factors.append(f"pluie 30j très faible ({rain_30:.0f} mm)")
        cause = "stress hydrique aggravé par la sécheresse récente"
    elif rain_90 is not None and rain_90 < _RAIN_DRY_90D_MM:
        factors.append(f"pluie 90j déficitaire ({rain_90:.0f} mm)")
        cause = "déficit pluviométrique cumulé"

    if lst_anom is not None and lst_anom > _LST_HOT_ANOMALY_C:
        factors.append(f"température sol +{lst_anom:.1f}°C / normale")
        cause = "stress thermique"
    elif lst_abs is not None and lst_abs > _LST_HOT_ABS_C:
        factors.append(f"température sol élevée ({lst_abs:.0f}°C)")
        if "thermique" not in cause:
            cause = "stress thermique"

    if not factors and gap_pct < -8:
        cause = "anomalie inexpliquée par la météo"

    return cause, factors


def explication_text(result: DiagnosticResult) -> str:
    gap = _pct_gap(result.ndvi_observe, result.ndvi_attendu)
    abs_gap = abs(gap)
    cause, factors = _explanation_from_weather(result.weather_summary, gap)

    direction = "sous" if gap < 0 else "au-dessus de"
    header = (
        f"NDVI {abs_gap:.0f}% {direction} l'attendu sur les 3 dernières semaines"
    )

    if result.statut == "vert":
        return (
            f"{header}. Comportement dans la plage normale pour cette parcelle "
            f"compte tenu de la météo (vert)."
        )

    factor_clause = ""
    if factors:
        factor_clause = " (" + ", ".join(factors) + ")"

    if result.statut == "orange":
        return (
            f"{header}{factor_clause}. Cause probable : {cause}. "
            f"À surveiller (orange)."
        )

    # rouge
    return (
        f"{header}{factor_clause}. Cause probable : {cause}. "
        f"Anomalie marquée (rouge)."
    )


def recommandation_text(result: DiagnosticResult) -> str:
    if result.statut == "vert":
        return "RAS - poursuivre la conduite habituelle."
    if result.statut == "orange":
        return "Inspection visuelle dans 48h et vérification de l'irrigation."
    # rouge
    rain_30 = result.weather_summary.get("rain_cum_30d")
    if rain_30 is not None and rain_30 < _RAIN_DRY_30D_MM:
        return (
            "Inspection urgente sous 24h, irrigation d'appoint immédiate "
            "et vérification du système hydraulique."
        )
    return (
        "Inspection urgente sous 24h, vérifier irrigation, "
        "ravageurs et état général des arbres."
    )


def to_response_payload(result: DiagnosticResult) -> dict[str, Any]:
    """Render a DiagnosticResult into the exact JSON shape the brief asks for."""
    return {
        "statut": result.statut,
        "anomaly_score": result.anomaly_score,
        "ndvi_observe": result.ndvi_observe,
        "ndvi_attendu": result.ndvi_attendu,
        "explication": explication_text(result),
        "recommandation": recommandation_text(result),
        "_dates": result.dates,
        "_tier": result.tier,
        "_parcel_id": result.parcel_id,
        "_systeme": result.system,
        "_weather_summary": result.weather_summary,
    }
