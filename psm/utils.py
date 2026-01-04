import math
import pandas as pd


def round_up_to_increment(value: float, increment: float = 0.25) -> float:
    """
    Rounds UP to the nearest increment (default 0.25).

    NOTE: This model intentionally rounds UP for conservative staffing.
    Example: 2.01 -> 2.25
    """
    if value is None:
        return None
    return math.ceil(value / increment) * increment


def interpolate_staffing(df: pd.DataFrame, visits: float) -> dict:
    """
    Linearly interpolates staffing ratios from the staffing_ratios.csv table.
    Returns raw interpolated values (not rounded yet).
    """
    df = df.sort_values("ave_patients_day").reset_index(drop=True)

    # Clamp below min or above max
    if visits <= df["ave_patients_day"].min():
        row = df.iloc[0].to_dict()
        return row
    if visits >= df["ave_patients_day"].max():
        row = df.iloc[-1].to_dict()
        return row

    # Find bounding rows
    lower = df[df["ave_patients_day"] <= visits].iloc[-1]
    upper = df[df["ave_patients_day"] >= visits].iloc[0]

    if lower["ave_patients_day"] == upper["ave_patients_day"]:
        return lower.to_dict()

    x0 = lower["ave_patients_day"]
    x1 = upper["ave_patients_day"]
    weight = (visits - x0) / (x1 - x0)

    interpolated = {}
    for col in df.columns:
        if col == "ave_patients_day":
            interpolated[col] = visits
        else:
            interpolated[col] = lower[col] + weight * (upper[col] - lower[col])

    return interpolated

