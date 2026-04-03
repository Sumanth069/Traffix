from pydantic import BaseModel
from typing import Literal

Action = Literal[
    "SWITCH_SIGNAL",
    "KEEP_SIGNAL",
    "PRIORITIZE_LANE_N",
    "PRIORITIZE_LANE_S",
    "PRIORITIZE_LANE_E",
    "PRIORITIZE_LANE_W"
]

class Observation(BaseModel):
    north_queue: int
    south_queue: int
    east_queue: int
    west_queue: int
    signal_state: str  # "NS_GREEN" or "EW_GREEN"
    avg_wait: float
    violations: int
    time_step: int

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
