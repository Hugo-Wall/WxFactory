"""Element block-Jacobi preconditioner for the implicit shallow-water systems."""

import torch
from torch import Tensor

from .preconditioner import Preconditioner
from .registry import PreconditionerContext, register_preconditioner

# Target 32 MiB per float64 probe field.
_PROBE_BUDGET = 2**22


class ElementBlockJacobi(Preconditioner):
    """Inverse element blocks of ``alpha I - beta J``.

    Each block couples the 3 N^2 element unknowns, including their own interface-flux contributions.
    """

    def __init__(self, rhs, dtype, shape: tuple, param) -> None:
        super().__init__(dtype, shape, param)
        self.rhs = rhs
        self.inverse = None

    def prepare(self, state: Tensor, alpha: float, beta: float) -> None:
        rhs = self.rhs.double_precision_rhs()
        base = state.to(torch.float64).reshape(rhs.shape)
        linearization = rhs.linearize(base)
        grid = base.shape[1:]
        nodes = grid[-1]
        size = 3 * nodes
        # Element independence lets one probe excite the same local degree of freedom on all elements.
        batch = max(1, min(size, _PROBE_BUDGET // base.numel()))
        columns = []
        for start in range(0, size, batch):
            probes = range(start, min(start + batch, size))
            directions = base.new_zeros((3, len(probes)) + grid)
            for column, probe in enumerate(probes):
                directions[probe // nodes, column, ..., probe % nodes] = 1.0
            columns.append(linearization.element_action(directions))
        jacobian = torch.cat(columns, dim=1).permute(2, 3, 0, 4, 1).reshape(-1, size, size)
        identity = torch.eye(size, dtype=torch.float64, device=base.device)
        self.inverse = torch.linalg.inv(alpha * identity - beta * jacobian).to(state.dtype)
        self.grid = grid

    def __apply__(self, vec: Tensor, x0: Tensor | None = None, verbose: int | None = None) -> Tensor:
        num_elem_1, num_elem_2, nodes = self.grid
        values = vec.reshape((3,) + self.grid).to(self.inverse.dtype)
        columns = values.permute(1, 2, 0, 3).reshape(num_elem_1 * num_elem_2, 3 * nodes, 1)
        result = (self.inverse @ columns).reshape(num_elem_1, num_elem_2, 3, nodes).permute(2, 0, 1, 3)
        return result.reshape(vec.shape).to(vec.dtype)


@register_preconditioner("element_block_jacobi")
def _build(ctx: PreconditionerContext) -> ElementBlockJacobi:
    if ctx.config.equations != "shallow_water":
        raise ValueError("The element_block_jacobi preconditioner supports the shallow-water equations only")
    return ElementBlockJacobi(ctx.rhs.full, ctx.context.real_dtype, ctx.fields_shape, ctx.config)
