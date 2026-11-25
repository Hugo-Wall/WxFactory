from functools import cache

import numpy
from scipy.integrate import solve_ivp
import torch
import xarray as xr
from numpy.typing import NDArray
from torch import Tensor

from ..common import Configuration
from ..common.definitions import (
    Rd,
    cpd,
    cvd,
    gravity,
    kappa,
    idx_rho,
    idx_rho_theta,
    p0,
)

Rv = 461.52

z_tr = 12_000.0
height = 1200.0
theta_0 = 300.0
theta_tr = 343.0
T_tr = 213.0

def _compute_theta(z):
    return numpy.where(
        z < z_tr,
        theta_0 + (theta_tr - theta_0) * (z / z_tr) ** 1.25,
        theta_tr * numpy.exp(gravity / (cpd * T_tr) * (z - z_tr))
    )

def _compute_saturation_pressure(T_b):
    T = T_b - 273.15
    return 0.61121e3 * numpy.exp((18.678 - T / 234.5) * (T / (257.14 + T)))

def _compute_relative_humidity(z, p_b, T_b):
    p_s = _compute_saturation_pressure(T_b)
    q_s = Rd / Rv * p_s / (p_b - p_s)
    return numpy.where(
        z < height,
        0.014 / q_s,
        numpy.where(
            z < z_tr,
            1.0 - 0.75 * (z / z_tr) ** 1.25,
            0.25
        )
    )

def _compute_vapor_pressure(p_s, rh):
    return p_s * rh

def _compute_vapor_mixing_ratio(z, p_b, T_b, RH):
    p_s = _compute_saturation_pressure(T_b)
    p_v = _compute_vapor_pressure(p_s, RH)
    q_v = Rd / Rv * p_v / (p_b - p_v)
    return numpy.where(z < height, 0.014, q_v)

def _compute_temperature(p_b, theta_b):
    return theta_b * (p_b / p0) ** kappa

def _compute_rho(z, pressure):
    theta = _compute_theta(z)
    T_b = _compute_temperature(pressure, theta)
    rh = _compute_relative_humidity(z, pressure, T_b)
    q_v = _compute_vapor_mixing_ratio(z, pressure, T_b, rh)
    rho = pressure / (Rd * T_b * (1.0 + Rv / Rd * q_v))
    rho = rho * (1.0 + q_v)
    return rho

@cache
def _dense_pressure():
    result = solve_ivp(
        fun=lambda z, p: -gravity * _compute_rho(z, p),
        t_span=(0.0, 24_000.0),
        y0=[p0],
        method='Radau',
        atol=1e-10,
        rtol=1e-10,
        dense_output=True
    )
    return result.sol

def squall_line_initial_background_sounding(z: NDArray[numpy.float64]):
    given_shape = z.shape
    z = z.reshape(-1)
    p = _dense_pressure()(z)
    theta = _compute_theta(z)
    T_b = _compute_temperature(p, theta)
    rh = _compute_relative_humidity(z, p, T_b)
    q_v = _compute_vapor_mixing_ratio(z, p, T_b, rh)
    theta_v = theta * (1.0 + Rd / Rv * q_v)

    theta_v = theta_v.reshape(given_shape)
    q_v = q_v.reshape(given_shape)
    p = p.reshape(given_shape)
    return (
        torch.from_numpy(theta_v),
        torch.from_numpy(q_v),
        torch.from_numpy(p)
    )

def squall_line_initial_state(geom, metric, param: Configuration):
    x1, x3 = geom.X1, geom.X3  # physical x and z, (ne3, ne2, ne1, ns**3)

    uu = torch.clamp(-12 + 0.0048 * x3, max=0.0)
    ww = torch.zeros_like(x1)
    θ, qv, p = squall_line_initial_background_sounding(x3)
    exner = (p / p0) ** (Rd / cpd)

    qc = torch.zeros_like(x1)
    qr = torch.zeros_like(x1)

    xc, zc = 0.0, 2000.0
    rc, rx, rz = 1.0, 10_000.0, 1500.0
    θc = 3.0

    r = torch.sqrt((x1 - xc) ** 2 / rx**2 + (x3 - zc) ** 2 / rz**2)
    Δθ = torch.where(r <= rc, θc * torch.cos(0.5 * torch.pi * r)**2, 0.0)
    θ += Δθ

    ρ = p0 / (Rd * θ) * exner ** (cvd / Rd)

    Q = torch.zeros((8,) + geom.grid_shape_3d_new, dtype=x1.dtype)
    Q[idx_rho] = ρ
    physical_momentum = torch.stack((ρ * uu, torch.zeros_like(ρ), ρ * ww))
    Q[1:4] = metric.volume.covariant_from_physical(physical_momentum / ρ)
    Q[idx_rho_theta] = ρ * θ
    Q[5] = ρ * qv
    Q[6] = ρ * qc
    Q[7] = ρ * qr
    return Q
