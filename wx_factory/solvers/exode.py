"""Evaluate exponential and phi-function actions through an auxiliary ODE."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy
from mpi4py import MPI
from torch import Tensor

from ..context import Context


@dataclass
class ExodeStartup:
    """First accepted auxiliary step from the preceding successful solve."""

    first_accepted: float | None = None


@dataclass(frozen=True)
class OperatorPartition:
    """Splitting A = explicit + implicit for additive Runge-Kutta methods.

    solve(r, a) returns z such that z - a * implicit(z) = r.
    """

    explicit: Callable[[Tensor], Tensor]
    implicit: Callable[[Tensor], Tensor]
    solve: Callable[[Tensor, float], Tensor]


def _method(name: str):
    # The integrator package also imports the exponential solvers.
    from ..integrators.butcher import METHODS

    method_name = name.upper()
    if method_name not in METHODS:
        raise ValueError(f"Unknown EXODE method {name!r}; choose from {', '.join(METHODS)}")
    return METHODS[method_name]


def exode_needs_partition(method: str) -> bool:
    from ..integrators.explicit_runge_kutta import AdditiveRungeKutta

    method_class = _method(method)
    return isinstance(method_class, type) and issubclass(method_class, AdditiveRungeKutta)


def exode(
    tau_out,
    A,
    u,
    method="BS3(2)",
    controller="deadbeat",
    rtol=1e-3,
    atol=1e-6,
    context: Context | None = None,
    startup: ExodeStartup | None = None,
    initial_step: Callable[[], float] | None = None,
    partition: OperatorPartition | None = None,
):
    """Solve y' = A(y) + sum_j t**j/j! u[j+1], starting from u[0].

    Additive methods treat partition.explicit and the polynomial forcing explicitly,
    and partition.implicit implicitly.
    """
    tau_out = float(tau_out)
    if not math.isfinite(tau_out) or tau_out <= 0:
        raise ValueError("EXODE requires a positive, finite output time")
    method_class = _method(method)
    additive = exode_needs_partition(method)
    if additive and partition is None:
        raise ValueError(f"EXODE method {method!r} needs an explicit/implicit partition of the operator")

    def add_forcing(time, tendency):
        for order, vector in enumerate(u[1:]):
            tendency.add_(vector, alpha=time**order / math.factorial(order))
        return tendency

    explicit = partition.explicit if additive else A

    def rhs(time, state):
        return add_forcing(time, explicit(state))

    comm = context.comm if context is not None else MPI.COMM_WORLD
    first_step = startup.first_accepted if startup is not None else None
    if first_step is None:
        first_step = initial_step() if initial_step is not None else tau_out
    if not math.isfinite(first_step) or first_step <= 0:
        raise ValueError("EXODE initial step must be positive and finite")
    # All ranks must share the startup step and the zero-state shortcut.
    controls = numpy.array([min(first_step, tau_out), 0.0 if u[0].any() else 1.0])
    comm.Allreduce(MPI.IN_PLACE, controls, op=MPI.MIN)
    first_step, zero_start = float(controls[0]), bool(controls[1])

    initial_derivatives = None
    if zero_start:
        # Linearity gives A(0) = 0; only polynomial forcing contributes at startup.
        forcing = add_forcing(0.0, u[0].clone())
        initial_derivatives = (forcing, u[0].clone()) if additive else (forcing,)

    functions = (rhs, lambda time, state: partition.implicit(state), partition.solve) if additive else (rhs,)
    solver = method_class(
        *functions,
        0.0,
        u[0].clone(),
        tau_out,
        controller=controller or None,
        first_step=first_step,
        rtol=rtol,
        atol=atol,
        context=context,
        initial_derivatives=initial_derivatives,
    )
    first_accepted = None
    while solver.status == "running":
        previous_time = solver.t
        solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"EXODE integration failed at t={solver.t}")
        if first_accepted is None:
            first_accepted = solver.t - previous_time

    if startup is not None:
        startup.first_accepted = first_accepted
    stats = (
        solver.nfev,
        solver.failed_steps,
        getattr(solver, "nsolve", 0),
        solver.estimate_error(),
        solver.error_norm_old,
        solver.h_previous,
        solver.h,
    )
    return solver.y, stats
