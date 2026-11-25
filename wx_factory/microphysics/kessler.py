import math
from copy import copy

import numpy
import torch
from mpi4py import MPI
from torch import Tensor

from ..common.definitions import (
    p0,
    Rd,
    cpd,
    idx_rho,
    idx_rho_theta,
)
from ..common import Configuration
from ..common.matmul import DFRDerivative, apply_op
from ..geometry import (
    DFROperators,
    CubedSphere3D,
    Metric5D,
    Metric5DFields,
)
from ..pde.euler_interfaces import interface_slices, pressure


class RhsSedimentation:
    # global_idx_rho: int = 0
    # global_idx_rho_qr: int = 7
    idx_rho: int = 0
    idx_rho_qr: int = 1
    n_vars: int = 2

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
        self.shape = (self.n_vars,) + self.metric.volume.height.shape
        self.mid_faces = (
            (..., slice(1, -1), slice(None)),
            (..., slice(1, -1), slice(None), slice(None)),
            (..., slice(1, -1), slice(None), slice(None), slice(None)),
        )

    def _trace(self, state: Tensor) -> Tensor:
        field = self.metric.faces[2]
        # TODO: only extrapolate downward? logarithmic extrapolation?
        trace = apply_op(state, self.ops.extrap_z)
        face = state.new_zeros((state.shape[0],) + field.height.shape)
        face[self.mid_faces[2]] = trace

        width = self.geom.itf_size
        face[..., 0, :, :, width:] = face[..., 1, :, :, :width]
        face[self.idx_rho, -1, :, :, :width] = 1.0  # non-zero to avoid division-by-zero
        return face

    def _terminal_velocity(self, rho: Tensor, rho_qr: Tensor) -> Tensor:
        qrr = torch.clamp(0.001 * rho_qr, min=0.0)
        rhalf = torch.sqrt(1.123 / rho)  # TODO: use actual air density at ground level?
        velqr = 36.34 * qrr ** 0.1364 * rhalf
        return velqr

    def _contravariant_velocity(
        self, rho: Tensor, rho_qr: Tensor, field: Metric5DFields
    ) -> Tensor:
        velqr = self._terminal_velocity(rho, rho_qr)
        radial = field.radial
        physical_velocity = -velqr * radial
        contravariant_velocity = field.raise_velocity(
            field.covariant_from_physical(physical_velocity)
        )
        return contravariant_velocity

    # TODO: refactor to use that rho is constant over time step
    def __call__(self, state: Tensor) -> Tensor:
        # if state.numel() != math.prod(self.shape):
        # full = state.reshape((-1,) + self.metric.volume.height.shape)
        # result = torch.zeros_like(state)
        # augmented = torch.stack(
        #     (full[self.global_idx_rho], full[self.global_idx_rho_qr])
        # )
        # result[self.global_idx_rho_qr] = self(augmented)[self.idx_rho_qr]
        # return result.reshape_as(state)
        given_shape = state.shape
        # TODO: rename to state if above part is removed
        augmented = state.reshape(self.shape)

        volume_field = self.metric.volume
        face_field = self.metric.faces[2]

        face = self._trace(augmented)

        volume_rho = augmented[self.idx_rho]
        volume_rho_qr = augmented[self.idx_rho_qr]
        volume_contravariant_velocity = self._contravariant_velocity(
            volume_rho, volume_rho_qr, volume_field
        )
        volume_flux = (
            volume_field.volume_jacobian
            * volume_rho_qr
            * volume_contravariant_velocity[2]
        )

        face_rho = face[self.idx_rho]
        face_rho_qr = face[self.idx_rho_qr]
        face_contravariant_velocity = self._contravariant_velocity(
            face_rho, face_rho_qr, face_field
        )
        lower, upper = interface_slices(2, self.geom.num_solpts)
        common_face_flux = (
            face_field.volume_jacobian[lower]
            * face_rho_qr[lower]
            * face_contravariant_velocity[2][lower]
        )
        face_flux = torch.zeros_like(face[1:])
        face_flux[lower], face_flux[upper] = common_face_flux, common_face_flux

        result = torch.zeros_like(augmented)
        result[self.idx_rho_qr:] -= (
            apply_op(volume_flux, self.ops.derivative_z)
            + apply_op(face_flux[self.mid_faces[2]], self.ops.correction_DU)
        ) / volume_field.volume_jacobian

        return result.reshape(given_shape)


class KesslerMicrophysics:
    idx_rho_qv: int = 5
    idx_rho_qc: int = 6
    idx_rho_qr: int = 7

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
        self.rhs_sedimentation = RhsSedimentation(geometry, operators, metric, config)
        self.integrator = None  # TODO: add this

    def _kessler(
        self,
        theta: Tensor,  # potential temperature (K)
        qv: Tensor,  # water vapor mixing ratio (gm/gm)
        qc: Tensor,  # cloud water mixing ratio (gm/gm)
        qr: Tensor,  # rain  water mixing ratio (gm/gm)
        rho: Tensor,  # dry air density (not mean state as in KW) (kg/m^3)
        pk: Tensor,  # Exner function  (not mean state as in KW) (p/p0)^(Rd/cpd)
        dt: float,  # time step (s)
        # z: Tensor,  # heights of solution points (m)
    ) -> None:
        f2x = 17.27
        f5 = 237.3 * f2x * 250_000 / 1003.0
        xk = 0.2875  # kappa (r/cp)
        psl = p0 / 100.0  # pressure at sea level (mb)
        rhoqr = 1000.0  # density of liquid water (kg/m^3)

        r = 0.001 * rho
        pc = 3.8 / (pk ** (1.0 / xk) * psl)

        # TODO: compute dt_max based on CFL condition and terminal velocity
        dt_max = dt

        rainsplit = math.ceil(dt / dt_max)
        dt0 = dt / rainsplit

        for i in range(rainsplit):
            # Sedimentation term using DFR
            sed = dt0 * self.rhs_sedimentation(torch.stack((rho, rho * qr))) / rho

            # Autoconversion and accretion rates following KW eq. 2.13a,b
            qrprod = qc - (qc - dt0 * torch.clamp(0.001 * (qc - 0.001), min=0.0)) / (
                1.0 + dt0 * 2.2 * qr**0.875
            )
            qc[...] = torch.clamp(qc - qrprod, min=0.0)
            qr[...] = torch.clamp(
                qr + qrprod + sed[self.rhs_sedimentation.idx_rho_qr], min=0.0
            )

            # Saturation vapor mixing ratio (gm/gm)
            qvs = pc * torch.exp(f2x * (pk * theta - 273.0) / (pk * theta - 36.0))
            prod = (qv - qvs) / (1.0 + qvs * f5 / (pk * theta - 36.0) ** 2)

            # Evaporation rate following KW eq. 2.14a,b
            ern = torch.minimum(
                dt0
                * (
                    ((1.6 + 124.9 * (r * qr) ** 0.2046) * (r * qr) ** 0.525)
                    / (2_550_000.0 * pc / (3.8 * qvs) + 540_000.0)
                )
                * (torch.clamp(qvs - qv, min=0.0) / (r * qvs)),
                torch.minimum(torch.clamp(-prod - qc, min=0.0), qr),
            )

            # Saturation adjustment following KW eq. 3.10
            theta[...] = theta + 2_500_000.0 / (1003.0 * pk) * (
                torch.maximum(prod, -qc) - ern
            )
            qv[...] = torch.clamp(qv - torch.maximum(prod, -qc) + ern, min=0.0)
            qc[...] = qc + torch.maximum(prod, -qc)
            qr[...] = qr - ern

    def step(self, state: Tensor, dt: float) -> None:
        state = state.reshape((-1,) + self.metric.volume.height.shape)
        rho = state[idx_rho]
        rho_theta = state[idx_rho_theta]

        rho_inv = 1.0 / rho
        theta_v = rho_theta * rho_inv
        qv = torch.clamp(state[self.idx_rho_qv] * rho_inv, min=0.0)
        qc = torch.clamp(state[self.idx_rho_qc] * rho_inv, min=0.0)
        qr = torch.clamp(state[self.idx_rho_qr] * rho_inv, min=0.0)
        pk = (pressure(rho_theta) / p0) ** (Rd / cpd)

        Rv = 461.52 #TODO: move to definitions.py
        theta = theta_v / (1.0 + Rd / Rv * qv)
        self._kessler(theta, qv, qc, qr, rho, pk, dt)
        assert torch.all(theta > 0.0) and not torch.isnan(theta).any(), "Potential temperature must be positive and not NaN"
        assert torch.all(qv >= 0.0) and not torch.isnan(qv).any(), "Water vapor mixing ratio must be non-negative and not NaN"
        assert torch.all(qc >= 0.0) and not torch.isnan(qc).any(), "Cloud water mixing ratio must be non-negative and not NaN"
        assert torch.all(qr >= 0.0) and not torch.isnan(qr).any(), "Rain water mixing ratio must be non-negative and not NaN"
        theta_v = theta * (1.0 + Rd / Rv * qv)

        state[idx_rho_theta] = rho * theta_v
        state[self.idx_rho_qv] = rho * qv
        state[self.idx_rho_qc] = rho * qc
        state[self.idx_rho_qr] = rho * qr
