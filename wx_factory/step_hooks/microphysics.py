import numpy
import torch

from ..common import Configuration
from ..microphysics import KesslerMicrophysics
from . import step_hook


class MicrophysicsHook(step_hook.StepHook):
    def __init__(self, geometry, operators, metric, config):
        self.geom = geometry
        self.ops = operators
        self.metric = metric
        self.config: Configuration = config
        self.gscp = KesslerMicrophysics(geometry, operators, metric, config)

    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        dt = self.config.dt
        self.gscp.step(Q, dt) # type: ignore
        return Q
