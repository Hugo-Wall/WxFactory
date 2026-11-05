import math
from copy import copy

import numpy
import torch
from mpi4py import MPI
from torch import Tensor

from ..common.definitions import p0

class RHSSedimentation:
    """
    Class to compute the right-hand side of the sedimentation term in the Kessler microphysics scheme.
    """

    def __init__(self, geometry, operators, metric, process_topology, config, velqr: Tensor):
        self.geometry = geometry
        self.operators = operators
        self.metric = metric
        self.process_topology = process_topology
        self.config = config
        self.velqr = velqr

    def __call__(self, state: Tensor, *, explicit=True) -> Tensor: # type: ignore
        # Compute contravariant velocity field
        pass

def kessler(theta: Tensor, qv: Tensor, qc: Tensor, qr: Tensor, rho: Tensor, pk: Tensor, dt: float) -> None:
    f2x = 17.27
    f5 = 237.3 * f2x * 250_000 / 1003.0
    xk = 0.2875 # kappa (r/cp)
    psl = p0 / 100.0 # pressure at sea level (mb)
    rhoqr = 1000.0 # density of liquid water (kg/m^3)

    r = 0.001 * rho
    rhalf = torch.sqrt(1.123 / r) # TODO: use actual air density at ground level?
    pc = 3.8 / (pk ** (1.0 / xk) * psl)

    # Liquid water terminal velocity (m/s) following KW eq. 2.15
    velqr = 36.34 * (qr * r) ** 0.1364 * rhalf

    assert dt > 0.0, "kessler called with non-positive dt"

    dt_max = dt

    # TODO: compute dt_max based on CFL condition for sedimentation

    rainsplit = math.ceil(dt / dt_max)
    dt0 = dt / rainsplit

    for i in range(rainsplit):
        # TODO: accumulate precipitation?

        # TODO: compute sedimentation term using DFR
        sed = torch.zeros_like(qr)

        # Adjustment terms

        # Autoconversion and accretion rates following KW eq. 2.13a,b
        qrprod = qc - (qc - dt0 * torch.clamp(0.001 * (qc - 0.001), min=0.0)) / (1.0 + dt0 * 2.2 * qr ** 0.875)
        qc = torch.clamp(qc - qrprod, min=0.0)
        qr = torch.clamp(qr + qrprod + sed, min=0.0)

        # Saturation vapor mixing ratio (gm/gm) following KW eq. 2.11
        qvs = pc * torch.exp(f2x * (pk * theta - 273.0) / (pk * theta - 36.0))
        prod = (qv - qvs) / (1.0 + qvs * f5 / (pk * theta - 36.0) ** 2)


        # Evaporation rate following KW eq. 2.14a,b
        ern = torch.minimum(dt0*(((1.6+124.9*(r*qr)**.2046)
            *(r*qr)**.525)/(2_550_000.0*pc
            /(3.8 * qvs) + 540_000.0))*(torch.clamp(qvs-qv, min=0.0)
            /(r*qvs)),
            torch.minimum(torch.clamp(-prod-qc, min=0.0),qr))

        theta = theta + 2_500_000.0 / (1003.0 * pk) * (torch.maximum(prod, -qc) - ern)
        qv = torch.clamp(qv - torch.maximum(prod, -qc) + ern, min=0.0)
        qc = qc + torch.maximum(prod, -qc)
        qr = qr - ern

        # Recalculate liquid water terminal velocity
        velqr = 36.34 * (qr * r) ** 0.1364 * rhalf
