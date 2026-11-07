import math
from copy import copy

import numpy
import torch
from mpi4py import MPI
from torch import Tensor

from ..common.definitions import p0
from ..common.matmul import DFRDerivative, apply_op
from ..common import Configuration
from ..geometry import (
    DFROperators,
    CubedSphere3D,
    Metric5D,
)
from ..init.initialize import Topo
from ..pde.euler_interfaces import interface_slices

def kessler(
        theta: Tensor,  # potential temperature (K)
        qv: Tensor,     # water vapor mixing ratio (gm/gm)
        qc: Tensor,     # cloud water mixing ratio (gm/gm)
        qr: Tensor,     # rain  water mixing ratio (gm/gm)
        rho: Tensor,    # dry air density (not mean state as in KW) (kg/m^3)
        pk: Tensor,     # Exner function  (not mean state as in KW) (p/p0)^(Rd/cpd)
        dt: float,      # time step (s)
        z: Tensor,      # heights of solution points (m)
    ) -> None:
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

    rainsplit = math.ceil(dt / dt_max)
    dt0 = dt / rainsplit

    for i in range(rainsplit):

        # Sedimentation term using DFR
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

class RHSSedimentation:
    """
    Class to compute the right-hand side of the sedimentation term in the Kessler microphysics scheme.
    """

    idx_rho = 0
    idx_qr = 1

    def __init__(
        self,
        geometry: CubedSphere3D,
        operators: DFROperators,
        metric: Metric5D,
        config: Configuration,
    ):
        self.geom = geometry
        self.ops = operators
        self.metric = metric
        self.config = config

    def _trace(self, state: Tensor) -> Tensor:
        """
        Returns (N_VARS, E3+2, E2, E1, 2N^2) tensor of interface values of the state variables
        slice(None, width) is bottom interface, slice(width, None) is top interface
        """
        field = self.metric.faces[2]
        trace = apply_op(state, self.ops.extrap_z) #TODO: only extrapolate downward? logarithmic extrapolation?
        interior = (..., slice(1, -1), slice(None), slice(None), slice(None)),
        face = state.new_zeros((state.shape[0],) + field.height.shape)
        face[interior] = trace

        width = self.geom.itf_size
        #TODO: fix boundary conditions for sedimentation
        face[..., 0, :, :, width:] = face[..., 1, :, :, :width]
        face[..., -1, :, :, :width] = face[..., -2, :, :, width:]
        return face


    # state.shape = (N_VARS, E3, E2, E1, N^3)
    def __call__(self, state: Tensor) -> Tensor:  # type: ignore
        volume_field = self.metric.volume
        face_field = self.metric.faces[2]

        face = self._trace(state)

        def terminal_velocity(rho: Tensor, rho_qr: Tensor) -> Tensor:
            qr = rho_qr / rho
            r = 0.001 * rho
            rhalf = torch.sqrt(1.123 / r) # TODO: use actual air density at ground level?
            velqr = 36.34 * (qr * r) ** 0.1364 * rhalf
            return velqr

        # Compute terminal velocities at solution points
        volume_rho = state[self.idx_rho]
        volume_rho_qr = state[self.idx_qr]
        volume_velqr = terminal_velocity(volume_rho, volume_rho_qr)

        # Compute terminal velocities at interface points
        face_rho = face[self.idx_rho]
        face_qr = face[self.idx_qr]
        face_velqr = terminal_velocity(face_rho, face_qr)

        # Compute contravariant sedimentation velocity vector at solution points and interface points
        radial_volume = volume_field.radial
        physical_volume = radial_volume * volume_velqr
        volume_contravariant_velocity = volume_field.raise_velocity(
            volume_field.covariant_from_physical(physical_volume)
        )

        radial_faces = face_field.radial
        physical_faces = radial_faces * face_velqr
        face_contravariant_velocity = face_field.raise_velocity(
            face_field.covariant_from_physical(physical_faces)
        )

        # Compute flux at solution points and Riemann flux at interfaces
        volume_flux = volume_field.volume_jacobian * volume_contravariant_velocity[2] * volume_rho_qr
        lower, upper = interface_slices(2, self.geom.num_solpts)
        common_face_flux = face_field.volume_jacobian[upper] * face_contravariant_velocity[2][upper] * face_qr[upper]
        face_flux = torch.zeros_like(face[:-1])
        face_flux[lower], face_flux[upper] = common_face_flux, common_face_flux
        interior = (..., slice(1, -1), slice(None), slice(None), slice(None)),

        #TODO: make it so that only rho_qr is updated
        result = torch.zeros_like(state)
        result[self.idx_qr] -= (apply_op(volume_flux, self.ops.derivative_z) + apply_op(face_flux[interior], self.ops.correction_DU)) / volume_field.volume_jacobian
        return result
