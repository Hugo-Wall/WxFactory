"""Dense linear algebra on the Krylov Hessenberg matrix (matrix exponential, triangular solve).

The Krylov Hessenberg matrix is tiny (at most ``mmax + 1`` square), so these run on the host with
scipy even when the Krylov vectors live on a GPU: the cost is negligible and scipy covers ``expm``
and ``solve_triangular`` directly. Results come back on the device of the input matrix.
"""

from typing import Any

import numpy
import scipy.linalg
import torch


def _device(value: torch.Tensor | Any) -> torch.device:
    return value.device if isinstance(value, torch.Tensor) else torch.device("cpu")


def _to_host(value: torch.Tensor | Any) -> numpy.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else numpy.asarray(value)


def expm(a) -> torch.Tensor:
    """Matrix exponential of a small (Hessenberg) matrix, evaluated on the host."""
    return torch.as_tensor(scipy.linalg.expm(_to_host(a)), device=_device(a))


def solve_triangular(a, b, **kwargs) -> torch.Tensor:
    """Triangular solve of a small system, evaluated on the host."""
    return torch.as_tensor(scipy.linalg.solve_triangular(_to_host(a), _to_host(b), **kwargs), device=_device(a))
