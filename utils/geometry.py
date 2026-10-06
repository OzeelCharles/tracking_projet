import numpy as np


def side(p: tuple[float, float], x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Retourne la position relative d'un point par rapport à une ligne (x1,y1)-(x2,y2).
    > 0 : d'un côté, < 0 : de l'autre, = 0 : sur la ligne.
    """
    x, y = p
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def calculate_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Calcule la distance euclidienne entre deux points."""
    return np.linalg.norm(pt1 - pt2)
