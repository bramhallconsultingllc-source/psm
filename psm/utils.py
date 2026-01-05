import math


def round_up_to_increment(value: float, increment: float = 0.25) -> float:
    """
    Always rounds UP to the nearest increment.

    Example:
        2.01 -> 2.25
        2.25 -> 2.25
        2.26 -> 2.50
    """
    return math.ceil(value / increment) * increment
