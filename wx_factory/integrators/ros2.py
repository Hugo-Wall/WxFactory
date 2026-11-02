from collections.abc import Callable
from time import time

import numpy
import torch

from ..common.configuration import Configuration
from ..jacobian import FiniteDifferenceJacobian, FiniteDifferenceRosenbrock
from ..solvers import fgmres
from .integrator import Integrator


class Ros2(Integrator):
    Q_flat: numpy.ndarray
    A: FiniteDifferenceRosenbrock
    b: numpy.ndarray

    def __init__(self, param: Configuration, rhs_handle: Callable, *, context=None, preconditioner=None) -> None:
        super().__init__(param, context=context, preconditioner=preconditioner)
        self.rhs_handle = rhs_handle
        self.tol = param.tolerance
        self.gmres_restart = param.gmres_restart

    def __prestep__(self, Q: numpy.ndarray, dt: float) -> None:
        if hasattr(self.rhs_handle, "double_precision_rhs"):
            return
        rhs = self.rhs_handle(Q)
        self.Q_flat = torch.ravel(Q)
        self.A = FiniteDifferenceRosenbrock(dt, Q, rhs, self.rhs_handle)
        self.b = self.A(self.Q_flat) + torch.ravel(rhs) * dt

    def __step__(self, Q: torch.Tensor, dt: float):
        self._prepare_preconditioner(Q, 1.0, dt / 2)
        if hasattr(self.rhs_handle, "double_precision_rhs"):
            return self._euler_step(Q, dt)
        maxiter = 20000 // self.gmres_restart
        if self.preconditioner is not None:
            maxiter = 400 // self.gmres_restart

        operator = _CountedOperator(self.A)
        t0 = time()
        Qnew, norm_r, norm_b, num_iter, flag, _ = self._solve_linear(
            operator,
            self.b,
            x0=self.Q_flat,
            tol=self.tol,
            restart=self.gmres_restart,
            maxiter=maxiter,
        )
        self._report(flag, num_iter, operator.count, time() - t0, norm_r / norm_b)

        self.failure_flag = flag

        return torch.reshape(Qnew, Q.shape)

    def _euler_step(self, Q, dt):
        working, rhs = self._implicit_state_and_rhs(Q, self.rhs_handle)
        value = rhs(working)
        if getattr(self.param, "jacobian", None) == "analytic" or getattr(rhs, "partition", None) == "implicit":
            jacobian = rhs.linearize(working)
        else:
            jacobian = FiniteDifferenceJacobian(1.0, working, rhs)
        right = dt * value.flatten()
        self.failure_flag = 0
        if hasattr(jacobian, "solve"):
            delta = jacobian.solve(right, dt).flatten()
        else:

            def rosenbrock(vector):
                vector = vector.to(working.dtype)
                return vector - dt / 2 * jacobian(vector)

            operator = _CountedOperator(rosenbrock)
            t0 = time()
            delta, norm_r, norm_b, num_iter, flag, _ = fgmres(
                operator,
                right,
                tol=self.tol,
                restart=self.gmres_restart,
                maxiter=10,
                preconditioner=self.preconditioner,
                verbose=self.verbose_solver,
                context=self.context,
                krylov_dtype=Q.dtype,
            )
            self._report(flag, num_iter, operator.count, time() - t0, norm_r / norm_b)
            self.failure_flag = flag
            if flag:
                raise RuntimeError("Euler ROS2 linear solve did not converge")
        return (working + delta.reshape_as(working)).to(Q.dtype)

    def _report(self, flag, num_iter, matvecs, elapsed, relative_residual):
        if self.context.comm.rank == 0:
            result_type = "convergence" if flag == 0 else "stagnation/interruption"
            print(
                f"FGMRES {result_type} at iteration {num_iter} ({matvecs} matvecs) in {elapsed:4.3f} s"
                f" to a solution with relative residual {relative_residual: .2e}"
            )


class _CountedOperator:
    """Linear operator that counts its applications, including FGMRES's true-residual checks."""

    def __init__(self, operator):
        self.operator = operator
        self.count = 0

    def __call__(self, vector):
        self.count += 1
        return self.operator(vector)


REGISTRY = {
    "ros2": lambda cfg, rhs, prec, ctx: Ros2(cfg, rhs.full, preconditioner=prec, context=ctx),
}
