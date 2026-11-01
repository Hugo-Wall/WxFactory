"""Different methods to solve a system of equations."""

from .exode import OperatorPartition, exode, exode_needs_partition
from .exponential_solver import (
    EXPONENTIAL_SOLVER_REGISTRY,
    ExponentialSolverRequest,
    ExponentialSolverResult,
    register_exponential_solver,
    resolve_exponential_solver,
)
from .fgmres import fgmres
from .global_operations import global_dotprod, global_inf_norm, global_norm
from .kiops import kiops
from .nonlin import KrylovJacobian, newton_krylov
from .pmex import pmex

__all__ = [
    "EXPONENTIAL_SOLVER_REGISTRY",
    "ExponentialSolverRequest",
    "ExponentialSolverResult",
    "KrylovJacobian",
    "OperatorPartition",
    "exode",
    "exode_needs_partition",
    "fgmres",
    "global_dotprod",
    "global_inf_norm",
    "global_norm",
    "kiops",
    "newton_krylov",
    "pmex",
    "register_exponential_solver",
    "resolve_exponential_solver",
]
