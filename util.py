from typing import Sequence

import cirq
import numpy as np
from numpy.typing import ArrayLike

_pauli_bit_map = {
    cirq.I: (0, 0),
    cirq.X: (1, 0),
    cirq.Z: (0, 1),
    cirq.Y: (1, 1)
}

_bit_pauli_map = {value: key for key, value in _pauli_bit_map.items()}


def bit_to_pauli(qubits: Sequence[cirq.LineQubit], bitstring: ArrayLike) -> cirq.PauliString:
    if len(qubits) * 2 != len(bitstring):
        raise ValueError('incompatible bitstring length and no. of qubits')
    seq = cirq.PauliString()
    for i, q in enumerate(qubits):
        seq = seq * cirq.PauliString(_bit_pauli_map[tuple(bitstring[2 * i:2 * i + 2])](q))
    return cirq.PauliString(seq)


def beta_cirq(v, u):
    u = np.array(u)
    v = np.array(v)
    qubits = cirq.LineQubit.range(len(u) // 2)
    scalar = bit_to_pauli(qubits, v) * bit_to_pauli(qubits, (u + v) % 2) * bit_to_pauli(qubits, u)
    assert scalar == cirq.PauliString() * scalar.coefficient, f"Pauli string {scalar} is not proportional to identity"
    return int(-np.angle(scalar.coefficient) / np.pi * 2) % 4
