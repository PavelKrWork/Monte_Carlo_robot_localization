from dataclasses import dataclass

# 2D particles cloud
@dataclass
class State:
    x: float
    y: float
    theta: float

# Land mark coords initialization
@dataclass
class Coord:
    x: float
    y: float

@dataclass
class LandMark:
    left_coord: Coord
    right_coord: Coord