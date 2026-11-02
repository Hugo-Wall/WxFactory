"""Direct action of the shallow-water DFR Jacobian."""

import torch

from ..common.definitions import gravity


class ShallowWaterLinearization:
    def __init__(self, rhs, state):
        self.rhs = rhs
        self.state = state
        self.shape = state.shape
        self.velocity = rhs.metric.volume.raise_velocity(state[1:3])
        exchange = rhs._start_exchange(state)
        partial = rhs._volume_derivatives(rhs._volume_terms(state, self.velocity))
        self.interfaces = rhs._interfaces(rhs._finish_exchange(exchange))
        dx, dy = rhs._add_face_corrections(partial, tuple(face.common_values() for face in self.interfaces))
        self.absolute_curl = dx[2] - dy[2] + rhs.metric.volume.frame_vorticity

    def _volume(self, perturbation):
        rhs = self.rhs
        delta_velocity = rhs.metric.volume.raise_velocity(perturbation[1:3])
        delta_energy = (
            self.velocity[0] * perturbation[1] + self.velocity[1] * perturbation[2] + gravity * perturbation[0]
        )
        volume_terms = tuple(
            torch.stack(
                (
                    rhs.metric.sqrtG
                    * (perturbation[0] * self.velocity[direction] + self.state[0] * delta_velocity[direction]),
                    delta_energy,
                    perturbation[2 - direction],
                )
            )
            for direction in range(2)
        )
        return delta_velocity, rhs._volume_derivatives(volume_terms)

    def _tendency(self, delta_velocity, dx, dy):
        delta_curl = dx[2] - dy[2]
        return torch.stack(
            (
                -(dx[0] + dy[0]) / self.rhs.metric.sqrtG,
                -dx[1] + delta_velocity[1] * self.absolute_curl + self.velocity[1] * delta_curl,
                -dy[1] - delta_velocity[0] * self.absolute_curl - self.velocity[0] * delta_curl,
            )
        )

    def __call__(self, vector):
        rhs = self.rhs
        perturbation = vector.reshape(self.shape)
        exchange = rhs._start_exchange(perturbation, perturbation=True)
        delta_velocity, partial = self._volume(perturbation)
        faces = rhs._finish_exchange(exchange)
        common = tuple(
            interface.linearize(face[left], face[right])
            for interface, face, left, right in zip(self.interfaces, faces, rhs.left, rhs.right)
        )
        dx, dy = rhs._add_face_corrections(partial, common)
        return self._tendency(delta_velocity, dx, dy).flatten()

    def element_action(self, vector):
        """Apply element-diagonal Jacobian blocks, including each element's own interface traces.

        Inputs shaped (3, batch..., grid) carry independent perturbations along the batch axes.
        """
        rhs = self.rhs
        perturbation = vector.reshape(self.shape) if vector.ndim == 1 else vector
        delta_velocity, partial = self._volume(perturbation)
        from_left, from_right = [], []
        for interface, face, left, right in zip(self.interfaces, rhs._face_traces(perturbation), rhs.left, rhs.right):
            zero = torch.zeros_like(face[left])
            from_left.append(interface.linearize(face[left], zero))
            from_right.append(interface.linearize(zero, face[right]))
        dx, dy = rhs._add_face_corrections(partial, from_left, from_right)
        return self._tendency(delta_velocity, dx, dy).reshape(vector.shape)
