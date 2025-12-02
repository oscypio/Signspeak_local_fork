from pydantic import BaseModel, Field
from typing import List, Optional

# This model represents a single {x, y, z, visibility} point


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None

# This model represents the handedness information


class HandednessInfo(BaseModel):
    score: float
    index: int
    categoryName: str
    displayName: str

# This is the main model for a single frame of data sent from the other backend


class FrameData(BaseModel):
    timestamp: float
    sequenceNumber: int
    receivedAt: float
    landmarks: List[List[LandmarkPoint]]
    handedness: List[List[HandednessInfo]]