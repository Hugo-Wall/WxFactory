import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from mpi4py import MPI

from wx_factory.integrators.butcher import METHODS
from wx_factory.integrators.explicit_runge_kutta import global_rms_norm
from wx_factory.solvers.exode import ExodeStartup, exode
from wx_factory.solvers.exponential_solver import ExponentialSolverRequest, resolve_exponential_solver


class ExodeTestCase(unittest.TestCase):
    def test_phi_actions_match_augmented_matrix_exponential(self):
        matrix = torch.diag(torch.tensor([-2.0, 0.5], dtype=torch.float64))
        vectors = torch.tensor([[1.0, -2.0], [0.3, 0.1], [-0.5, 0.2], [0.1, -0.3]], dtype=torch.float64)
        context = SimpleNamespace(comm=MPI.COMM_SELF)
        for count, time, method in ((1, 1.0, "BS3(2)"), (4, 0.25, "DP5(4)"), (4, 1.5, "DP5(4)"), (4, 0.5, "M4(3)")):
            with self.subTest(count=count, time=time):
                u = vectors[:count].clone()
                size = matrix.shape[0]
                augmented = matrix.new_zeros((size + count - 1,) * 2)
                augmented[:size, :size] = matrix
                initial = matrix.new_zeros(augmented.shape[0])
                initial[:size] = u[0]
                if count > 1:
                    augmented[:size, size:] = u[1:].flip(0).T
                    for i in range(size, augmented.shape[0] - 1):
                        augmented[i, i + 1] = 1
                    initial[-1] = 1
                expected = (torch.matrix_exp(time * augmented) @ initial)[:size]
                actual, _ = exode(time, lambda x: matrix @ x, u, method=method, rtol=1e-9, atol=1e-11, context=context)
                torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
                torch.testing.assert_close(u, vectors[:count], rtol=0, atol=0)

    def test_registry_uses_requested_time_and_context_communicator(self):
        context = SimpleNamespace(comm=MPI.COMM_SELF)
        vectors = torch.tensor([[1.0, 2.0], [0.3, -0.5], [-0.2, 0.1]], dtype=torch.float64)
        time = 0.25
        request = ExponentialSolverRequest(
            [time], lambda x: torch.zeros_like(x), vectors, 1e-8, 32, context, exode_controller="", announce=False
        )
        with patch("wx_factory.integrators.explicit_runge_kutta.global_rms_norm", wraps=global_rms_norm) as norm:
            result = resolve_exponential_solver("exode")(request)
        expected = vectors[0] + time * vectors[1] + time**2 / 2 * vectors[2]
        torch.testing.assert_close(result.value, expected, rtol=2e-14, atol=2e-14)
        self.assertTrue(norm.called)
        self.assertTrue(all(call.args[2] is context.comm for call in norm.call_args_list))

    def test_completed_solve_releases_operator_without_cyclic_collection(self):
        class Operator:
            def __call__(self, value):
                return -value

        enabled = gc.isenabled()
        gc.disable()
        try:
            operator = Operator()
            reference = weakref.ref(operator)
            exode(
                1.0,
                operator,
                torch.ones((1, 2), dtype=torch.float64),
                context=SimpleNamespace(comm=MPI.COMM_SELF),
            )
            del operator
            self.assertIsNone(reference())
        finally:
            if enabled:
                gc.enable()

    def test_failed_integration_does_not_return_a_partial_solution(self):
        solver = SimpleNamespace(status="running", t=0.0)
        solver.step = Mock(side_effect=lambda: setattr(solver, "status", "failed"))
        startup = ExodeStartup(0.1)
        with patch.dict(METHODS, {"BS3(2)": Mock(return_value=solver)}):
            with self.assertRaisesRegex(RuntimeError, "EXODE integration failed"):
                exode(1.0, lambda x: x, torch.ones((2, 2), dtype=torch.float64), startup=startup)
            self.assertEqual(startup.first_accepted, 0.1)

    def test_startup_reuses_first_accepted_step_and_is_scoped(self):
        cold_step = Mock(return_value=0.5)
        trials = []

        def make_solver(*args, **kwargs):
            trials.append(kwargs["first_step"])
            solver = SimpleNamespace(
                t=0.0,
                status="running",
                nfev=2,
                failed_steps=1,
                error_norm_old=0.1,
                h_previous=0.0,
                h=0.1,
                y=args[2],
                estimate_error=lambda: 0.0,
            )

            def step():
                solver.t += 0.2 if solver.t == 0 else 0.8
                if solver.t == 1.0:
                    solver.status = "finished"

            solver.step = step
            return solver

        solve = resolve_exponential_solver("exode")
        independent_solve = resolve_exponential_solver("exode")
        request = ExponentialSolverRequest(
            1.0,
            lambda x: x,
            torch.ones((1, 2)),
            1e-6,
            32,
            SimpleNamespace(comm=MPI.COMM_SELF),
            initial_step=cold_step,
            announce=False,
        )
        with patch.dict(METHODS, {"BS3(2)": make_solver}):
            for solver in (solve, solve, independent_solve):
                solver(request)
        self.assertEqual(trials, [0.5, 0.2, 0.5])
        self.assertEqual(cold_step.call_count, 2)

    def test_h211b_uses_previous_accepted_step_after_rejection(self):
        from wx_factory.integrators.explicit_runge_kutta import limiter

        solver = METHODS["BS3(2)"](
            lambda t, y: -y,
            0.0,
            torch.ones(1, dtype=torch.float64),
            10.0,
            first_step=0.1,
            controller=(0.25, 0.25, 0.25, 0.8),
        )
        with patch(
            "wx_factory.integrators.explicit_runge_kutta.global_rms_norm",
            side_effect=[0.2, 0.3, 2.0, 0.4],
        ):
            solver.step()
            first_step = solver.h_previous
            second_step = solver.h
            solver.step()
            expected_factor = 0.8**0.5 * (0.3 * 0.2) ** (-1 / 8) * (second_step / first_step) ** (-0.25)
            self.assertAlmostEqual(solver.h, second_step * limiter(expected_factor, 2))
            trial_step = solver.h
            solver.step()
        accepted_step = trial_step * limiter(0.8 * 2.0 ** (-1 / 2), 2)
        self.assertAlmostEqual(solver.h_previous, accepted_step)
        self.assertLessEqual(solver.h, accepted_step)
        self.assertEqual(solver.failed_steps, 1)

    def test_registry_tolerance_controls_relative_accuracy(self):
        import math

        context = SimpleNamespace(comm=MPI.COMM_SELF)
        errors = []
        for tolerance in (1e-4, 1e-7):
            request = ExponentialSolverRequest(
                [1.0],
                lambda x: -2 * x,
                torch.tensor([[1e6]], dtype=torch.float64),
                tolerance,
                32,
                context,
                announce=False,
            )
            result = resolve_exponential_solver("exode")(request)
            errors.append(abs(result.value.item() / (1e6 * math.exp(-2)) - 1))
        self.assertLess(errors[1], errors[0] / 100)
        self.assertLess(errors[1], 2e-6)

    def test_nonfinite_trial_is_retried_and_invalid_controls_are_rejected(self):
        import math

        def rhs(t, y):
            return torch.where(y.abs() > 2, math.nan, -10 * y)

        solver = METHODS["BS3(2)"](
            rhs,
            0.0,
            torch.ones(1, dtype=torch.float64),
            1.0,
            first_step=1.0,
            rtol=1e-6,
            atol=1e-8,
        )
        while solver.status == "running":
            solver.step()
        self.assertEqual(solver.status, "finished")
        self.assertGreater(solver.failed_steps, 0)
        torch.testing.assert_close(solver.y, torch.tensor([math.exp(-10)], dtype=torch.float64), rtol=1e-3, atol=1e-8)
        for controls in ({"rtol": -1}, {"atol": 0}, {"atol": math.nan}, {"max_step": 0}, {"first_step": math.nan}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                METHODS["BS3(2)"](rhs, 0.0, torch.ones(1, dtype=torch.float64), 1.0, **({"first_step": 0.1} | controls))

    def test_small_time_interval_has_no_absolute_step_floor(self):
        import math

        solver = METHODS["BS3(2)"](
            lambda t, y: -2e14 * y,
            0.0,
            torch.ones(1, dtype=torch.float64),
            1e-14,
            first_step=1e-14,
            rtol=1e-7,
            atol=1e-9,
        )
        while solver.status == "running":
            solver.step()
        self.assertEqual(solver.status, "finished")
        self.assertGreater(solver.failed_steps, 0)
        torch.testing.assert_close(solver.y, torch.tensor([math.exp(-2)], dtype=torch.float64), rtol=2e-6, atol=1e-9)

    def test_additive_pairs_match_augmented_matrix_exponential(self):
        from wx_factory.solvers.exode import OperatorPartition

        explicit = torch.tensor([[-0.3, 1.0, 0.0], [-1.0, -0.2, 0.5], [0.0, -0.5, 0.1]], dtype=torch.float64)
        implicit = torch.tensor([[-50.0, 0.0, 0.0], [5.0, -2.0, 0.0], [0.0, 0.0, -400.0]], dtype=torch.float64)
        partition = OperatorPartition(
            explicit=lambda x: explicit @ x,
            implicit=lambda x: implicit @ x,
            solve=lambda r, a: torch.linalg.solve(torch.eye(3, dtype=r.dtype) - a * implicit, r),
        )
        vectors = torch.tensor([[0.0, 0.0, 0.0], [0.3, -1.0, 2.0], [-0.5, 0.2, 0.1]], dtype=torch.float64)
        augmented = torch.zeros((5, 5), dtype=torch.float64)
        augmented[:3, :3] = explicit + implicit
        augmented[:3, 3:] = vectors[1:].flip(0).T
        augmented[3, 4] = 1
        context = SimpleNamespace(comm=MPI.COMM_SELF)
        for method in ("ARK3(2)4L[2]SA", "ARK4(3)6L[2]SA", "ARK4(3)7L[2]SA", "ARK5(4)8L[2]SA", "ARK5(4)8L[2]SAb"):
            for start in (vectors[0], torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)):
                with self.subTest(method=method, start=start.any().item()):
                    u = vectors.clone()
                    u[0] = start
                    initial = torch.cat((start, torch.tensor([0.0, 1.0], dtype=torch.float64)))
                    expected = (torch.matrix_exp(augmented) @ initial)[:3]
                    operator = Mock(side_effect=lambda x: (explicit + implicit) @ x)
                    actual, stats = exode(
                        1.0, operator, u, method=method, rtol=1e-9, atol=1e-11, context=context, partition=partition
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
                    operator.assert_not_called()
                    self.assertGreater(stats[2], 0)

    def test_additive_step_is_not_limited_by_implicit_stiffness(self):
        from wx_factory.solvers.exode import OperatorPartition

        stiff = -1e6
        partition = OperatorPartition(
            explicit=lambda x: -x, implicit=lambda x: stiff * x, solve=lambda r, a: r / (1 - a * stiff)
        )
        u = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        context = SimpleNamespace(comm=MPI.COMM_SELF)
        actual, stats = exode(
            1.0, None, u, method="ARK3(2)4L[2]SA", rtol=1e-6, atol=1e-8, context=context, partition=partition
        )
        self.assertLess(stats[0], 2000)
        self.assertLess(actual.abs().item(), 1e-6)
        with self.assertRaisesRegex(ValueError, "partition"):
            exode(1.0, lambda x: x, u, method="ARK3(2)4L[2]SA", context=context)

    def test_zero_start_skips_the_operator_at_the_initial_state(self):
        calls = []

        def operator(x):
            calls.append(x.clone())
            return -x

        vectors = torch.tensor([[0.0, 0.0], [1.0, -2.0]], dtype=torch.float64)
        exode(
            1.0, operator, vectors, method="BS3(2)", rtol=1e-8, atol=1e-10, context=SimpleNamespace(comm=MPI.COMM_SELF)
        )
        self.assertTrue(all(call.any() for call in calls))

    def test_additive_tables_satisfy_coupled_order_conditions(self):
        from wx_factory.integrators.explicit_runge_kutta import AdditiveRungeKutta

        for name, method in METHODS.items():
            if not issubclass(method, AdditiveRungeKutta):
                continue
            with self.subTest(method=name):
                c = method.C
                for weights, order in (
                    (method.B, method.order),
                    (method.B + method.E[:-1], method.error_estimator_order),
                ):
                    conditions = [weights.sum() - 1, weights @ c - 1 / 2]
                    if order >= 3:
                        conditions.append(weights @ c**2 - 1 / 3)
                        conditions.extend(weights @ table @ c - 1 / 6 for table in (method.A, method.A_implicit))
                    self.assertLess(max(abs(x) for x in conditions), 1e-12)
                for table in (method.A, method.A_implicit):
                    numpy_rows = table.sum(axis=1)
                    self.assertLess(abs(numpy_rows - c).max(), 1e-12)

    def test_explicit_tables_satisfy_order_conditions_up_to_fourth_order(self):
        import numpy

        def residuals(A, b, c, order):
            conditions = [(1, b.sum() - 1), (2, b @ c - 1 / 2), (3, b @ c**2 - 1 / 3), (3, b @ A @ c - 1 / 6)]
            conditions += [(4, b @ c**3 - 1 / 4), (4, b @ (c * (A @ c)) - 1 / 8), (4, b @ A @ c**2 - 1 / 12)]
            conditions += [(4, b @ A @ A @ c - 1 / 24)]
            return [abs(value) for level, value in conditions if level <= order]

        for name, method in METHODS.items():
            with self.subTest(method=name):
                s = method.n_stages
                A = numpy.zeros((s + 1, s + 1))
                A[:s, : method.A.shape[1]] = method.A
                A[s, :s] = method.B
                c = numpy.append(method.C, 1.0)
                main = numpy.append(method.B, 0.0)
                embedded = method.E.copy()
                embedded[:s] += method.B
                self.assertLess(max(residuals(A, main, c, min(method.order, 4))), 1e-12)
                self.assertLess(max(residuals(A, embedded, c, min(method.error_estimator_order, 4))), 1e-12)

    def test_every_explicit_pair_solves_phi_actions(self):
        from wx_factory.integrators.explicit_runge_kutta import AdditiveRungeKutta

        matrix = torch.tensor([[-2.0, 1.0], [-1.0, -0.5]], dtype=torch.float64)
        u = torch.tensor([[1.0, -2.0], [0.3, 0.1]], dtype=torch.float64)
        augmented = torch.zeros((3, 3), dtype=torch.float64)
        augmented[:2, :2] = matrix
        augmented[:2, 2] = u[1]
        expected = (torch.matrix_exp(augmented) @ torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64))[:2]
        for name, method in METHODS.items():
            if issubclass(method, AdditiveRungeKutta):
                continue
            with self.subTest(method=name):
                actual, _ = exode(
                    1.0,
                    lambda x: matrix @ x,
                    u,
                    method=name,
                    rtol=1e-7,
                    atol=1e-9,
                    context=SimpleNamespace(comm=MPI.COMM_SELF),
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
