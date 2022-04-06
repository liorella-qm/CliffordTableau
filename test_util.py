import cirq

from util import bit_to_pauli, beta_cirq


def test_bit_to_pauli_string():
    qs = cirq.LineQubit.range(2)
    result = bit_to_pauli(qs, [0, 1, 1, 0])
    print(result)
    assert result == cirq.PauliString(cirq.Z(qs[0]) * cirq.X(qs[1]))
    assert bit_to_pauli(qs, [1, 1, 0, 0]) * bit_to_pauli(qs, [0, 0, 1, 1]) == bit_to_pauli(qs, [1, 1, 1, 1])
    assert (bit_to_pauli(qs, [1, 1, 0, 0]) * bit_to_pauli(qs, [0, 0, 1, 1])).coefficient == 1+0*1j
    assert bit_to_pauli(qs, [1, 0, 1, 0]) * bit_to_pauli(qs, [0, 1, 0, 1]) == -bit_to_pauli(qs, [1, 1, 1, 1])
    print(bit_to_pauli(qs, [1, 0, 1, 0]) * bit_to_pauli(qs, [1, 1, 1, 1]) * bit_to_pauli(qs, [0, 1, 0, 1]))
    print(bit_to_pauli(qs, [1, 0, 1, 0]) * bit_to_pauli(qs, [1, 1, 1, 1]) * bit_to_pauli(qs, [0, 1, 0, 1]) == -cirq.PauliString())


def test_beta_cirq():
    assert beta_cirq([1, 0, 1, 0], [0, 1, 0, 1]) == 2
    assert beta_cirq([1, 1, 0, 0], [0, 0, 1, 1]) == 0

