from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class PitchData:
    """Data class for storing pitch information."""
    type: str
    speed: float
    confidence: float
    timestamp: float
    vertical_break: float
    horizontal_break: float
    spin_rate: int
    release_pos: Tuple[float, float]
    heatmap_points: List[Tuple[float, float]] 