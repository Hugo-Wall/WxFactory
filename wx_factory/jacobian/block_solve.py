"""Application and inversion of the vertical column operator."""

import functools

import torch

# Index letters shared by the einsum calls below:
#   c = column, e = vertical element, i = block row, j = block column.


def blocks_matvec(lower, diag, upper, x_col, *, periodic=False):
    """Apply the column blocks, including periodic end couplings when requested."""
    out = torch.einsum("ceij,cej->cei", diag, x_col)
    out[:, 1:] += torch.einsum("ceij,cej->cei", lower[:, 1:], x_col[:, :-1])
    out[:, :-1] += torch.einsum("ceij,cej->cei", upper[:, :-1], x_col[:, 1:])
    if periodic:
        out[:, 0] += torch.einsum("cij,cj->ci", lower[:, 0], x_col[:, -1])
        out[:, -1] += torch.einsum("cij,cj->ci", upper[:, -1], x_col[:, 0])
    return out


#: CUDA compute capabilities with strongly rate-limited FP64 arithmetic.
_RATE_LIMITED_FP64 = frozenset({(5, 0), (5, 2), (6, 1), (7, 5), (8, 6), (8, 9), (12, 0)})


@functools.cache
def _fp64_is_rate_limited(device_index: int) -> bool:
    """Return whether the CUDA device has strongly rate-limited FP64 arithmetic.

    Unknown compute capabilities use the conservative float64 path.
    """
    return torch.cuda.get_device_capability(device_index) in _RATE_LIMITED_FP64


def factor_in_float32(b) -> bool:
    """Select float32 factorization for rate-limited CUDA devices."""
    if b.dtype != torch.float32 or b.device.type != "cuda":
        return False
    return _fp64_is_rate_limited(b.device.index if b.device.index is not None else torch.cuda.current_device())


class BlockThomasFactor:
    """Factors of ``I - dt/2 J1`` for repeated right-hand sides.

    Cache D_e^-1, L_e D_(e-1)^-1 and D_e^-1 U_e, where D_e is the Schur complement at element e.
    """

    def __init__(self, lower, diag, upper, dt, *, refine=False, blocks64=None, check_errors=True):
        """Use blocks64 for refinement when supplied.

        With check_errors=False, the caller must check failures before solving.
        """
        self.lower, self.diag, self.upper = lower, diag, upper
        self.blocks64 = blocks64
        self.half_dt = 0.5 * dt
        self.refine = refine
        self.work_dtype = torch.float32 if refine else torch.float64
        _, num_elements, block_size, _ = diag.shape
        identity = torch.eye(block_size, dtype=self.work_dtype, device=diag.device)[None]
        sub = -self.half_dt * lower.to(self.work_dtype)
        sup = -self.half_dt * upper.to(self.work_dtype)
        pivot = identity - self.half_dt * diag[:, 0].to(self.work_dtype)
        inverse, info = torch.linalg.inv_ex(pivot)
        failures = info != 0
        inverses, multipliers = [inverse], []
        for e in range(1, num_elements):
            multipliers.append(sub[:, e] @ inverses[-1])
            pivot = identity - self.half_dt * diag[:, e].to(self.work_dtype) - multipliers[-1] @ sup[:, e - 1]
            inverse, info = torch.linalg.inv_ex(pivot)
            failures |= info != 0
            inverses.append(inverse)
        self.inverses = torch.stack(inverses, dim=1)
        self.multipliers = multipliers
        self.reduced_upper = [inverses[e] @ sup[:, e] for e in range(num_elements - 1)]
        self.failures = failures.any()
        # Defer the host synchronization until the block sweep is complete.
        if check_errors and self.failures.item():
            raise torch.linalg.LinAlgError("Singular block in vertical column factorization")

    def _solve(self, rhs):
        forward = [rhs[:, 0]]
        for e, multiplier in enumerate(self.multipliers, start=1):
            forward.append(torch.baddbmm(rhs[:, e], multiplier, forward[-1], alpha=-1))
        reduced = self.inverses @ torch.stack(forward, dim=1)
        result = [None] * reduced.shape[1]
        result[-1] = reduced[:, -1]
        for e in range(len(self.reduced_upper) - 1, -1, -1):
            result[e] = torch.baddbmm(reduced[:, e], self.reduced_upper[e], result[e + 1], alpha=-1)
        return result

    def solve(self, b):
        vector_rhs = b.ndim == 3
        right = b.to(self.work_dtype)
        if vector_rhs:
            right = right[..., None]
        result = self._solve(right)
        if self.refine:
            lower, diag, upper = self.blocks64 or tuple(block.double() for block in (self.lower, self.diag, self.upper))
            x = torch.stack(result, dim=1).double()
            coupled = torch.einsum("ceij,cejk->ceik", diag, x)
            coupled[:, 1:] += torch.einsum("ceij,cejk->ceik", lower[:, 1:], x[:, :-1])
            coupled[:, :-1] += torch.einsum("ceij,cejk->ceik", upper[:, :-1], x[:, 1:])
            residual = (right.double() - x + self.half_dt * coupled).to(self.work_dtype)
            correction = self._solve(residual)
            result = [value + delta for value, delta in zip(result, correction)]
        result = torch.stack(result, dim=1).to(b.dtype)
        return result[..., 0] if vector_rhs else result


class CyclicBlockFactor:
    """Block Thomas factors with a Woodbury correction for periodic columns."""

    def __init__(self, lower, diag, upper, dt):
        lower, diag, upper = (value.double() for value in (lower, diag, upper))
        self.factor = BlockThomasFactor(lower, diag, upper, dt)
        num_columns, num_elements, block_size, _ = diag.shape
        identity = torch.eye(block_size, dtype=diag.dtype, device=diag.device)
        right = diag.new_zeros((num_columns, num_elements, block_size, 2 * block_size))
        right[:, 0, :, :block_size] = identity
        right[:, -1, :, block_size:] = identity
        self.response = self.factor.solve(right)
        self.bottom, self.top = -dt / 2 * lower[:, 0], -dt / 2 * upper[:, -1]
        closure = torch.cat((self.bottom @ self.response[:, -1], self.top @ self.response[:, 0]), dim=1)
        closure += torch.eye(2 * block_size, dtype=diag.dtype, device=diag.device)
        lu, pivots, info = torch.linalg.lu_factor_ex(closure, check_errors=False)
        if (info != 0).any().item():
            raise torch.linalg.LinAlgError("Singular periodic vertical column correction")
        self.closure = lu, pivots

    def solve(self, b):
        particular = self.factor.solve(b.double())[..., None]
        boundary_rhs = torch.cat((self.bottom @ particular[:, -1], self.top @ particular[:, 0]), dim=1)
        correction = torch.linalg.lu_solve(*self.closure, boundary_rhs)
        return (particular - self.response @ correction[:, None])[..., 0].to(b.dtype)


def factor_column_blocks(lower, diag, upper, dt, *, periodic=False, **options):
    """Factor the column operator in the precision used by the direct solver."""
    if periodic:
        return CyclicBlockFactor(lower, diag, upper, dt)
    return BlockThomasFactor(lower, diag, upper, dt, refine=factor_in_float32(diag), **options)


def block_thomas_solve(lower, diag, upper, b, dt):
    """Solve ``(I - dt/2 J1) x = b`` with one refinement step for float32 CUDA factors."""
    return BlockThomasFactor(lower, diag, upper, dt, refine=factor_in_float32(b)).solve(b)


def cyclic_block_solve(lower, diag, upper, b, dt):
    """Solve periodic columns using a Woodbury correction to block Thomas."""
    return CyclicBlockFactor(lower, diag, upper, dt).solve(b)
