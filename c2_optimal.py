from typing import Optional, List, Union, Tuple

import numpy as np
import cirq

from clifford_tableau import CliffordTableau

_generate_table = False
_test_2_design = False  # careful, this takes about 20 minutes
size_c2 = 11520

_c1_ops = [
    ('I',),  # XZ ++
    ('X',),  # XZ +-
    ('Z',),  # XZ -+
    ('Y',),  # XZ --

    ('-SX',),  # XY ++
    ('SX',),  # XY +-
    ('Y', 'SX'),  # XY -+
    ('Y', '-SX'),  # XY --

    ('X', '-SY'),  # ZX ++
    ('-SY',),  # ZX +-
    ('SY',),  # ZX -+
    ('X', 'SY'),  # ZX --

    ('-SX', '-SY'),  # ZY ++
    ('SX', '-SY'),  # ZY +-
    ('-SX', 'SY'),  # ZY -+
    ('SX', 'SY'),  # ZY --

    ('SY', 'SX'),  # YX ++
    ('-SY', '-SX'),  # YX +-
    ('SY', '-SX'),  # YX -+
    ('-SY', 'SX'),  # YX --

    ('-SX', 'SY', 'SX'),  # YZ ++
    ('SX', 'SY', 'SX'),  # YZ +-
    ('-SX', '-SY', 'SX'),  # YZ -+
    ('-SX', 'SY', '-SX'),  # YZ --
]

_s1_ops = [
    ('I',),
    ('SY', 'SX'),
    ('-SX', '-SY'),
]

_s1x2_ops = [
    ('SX',),
    ('SX', 'SY', 'SX'),
    ('-SY',)
]

_s1y2_ops = [
    ('SY',),
    ('Y', 'SX'),
    ('-SX', '-SY', 'SX')
]

_gate_from_op = {
    'I': cirq.I,
    'X': cirq.X,
    'Y': cirq.Y,
    'Z': cirq.Z,
    'SX': cirq.X ** 0.5,
    'SY': cirq.Y ** 0.5,
    '-SX': cirq.X ** -0.5,
    '-SY': cirq.Y ** -0.5,
}


class C2Op:
    """
    A representation of a two-qubit Clifford operator as a series of native gate operations
    """
    def __init__(self, index: Union[int, Tuple[int, int]], native_clifford: str):
        """

        Args:
            index: either an integer between 0 and 11519 or a pair of integers, the first between 0 and 15 and the
            second between 0 and 719.
            In the first case, the Clifford will be built natively. In the second case, it will be built as a
            concatenation of the Pauli string between 0 and 15 and of the symplectic matrix.
            native_clifford: either 'CZ', 'ISWAP' or 'CNOT'.
        """
        self.index = index
        self.native_clifford = native_clifford
        if not isinstance(index, int):
            raise NotImplementedError()

        if index < 576:
            # single qubit class
            q1c1, q0c1 = np.unravel_index(index, (24, 24))
            self.q0 = (_c1_ops[q0c1],)
            self.q1 = (_c1_ops[q1c1],)
        elif index < 576 + 5184:
            # CNOT class
            index -= 576
            q1s1, q0s1, q1c1, q0c1 = np.unravel_index(index, (3, 3, 24, 24))
            print(q1s1, q0s1, q1c1, q0c1)
            self.q0 = (_c1_ops[q0c1], 'cz', _s1_ops[q0s1])
            self.q1 = (_c1_ops[q1c1], 'cz', _s1y2_ops[q1s1])
        elif index < 576 + 2 * 5184:
            # iSWAP class
            index -= 576 + 5184
            q1s1, q0s1, q1c1, q0c1 = np.unravel_index(index, (3, 3, 24, 24))
            self.q0 = (_c1_ops[q0c1], 'cz', ('SY',), 'cz', _s1y2_ops[q0s1])
            self.q1 = (_c1_ops[q1c1], 'cz', ('-SX',), 'cz', _s1x2_ops[q1s1])
        elif index < 2 * 576 + 2 * 5184:
            # SWAP class
            index -= 576 + 2 * 5184
            q1c1, q0c1 = np.unravel_index(index, (24, 24))
            self.q0 = (_c1_ops[q0c1], 'cz', ('-SY',), 'cz', ('SY',), 'cz', ('I',))
            self.q1 = (_c1_ops[q1c1], 'cz', ('SY',), 'cz', ('-SY',), 'cz', ('SY',))
        else:
            raise ValueError(f'index={index} is out of range')

    def __str__(self):
        s = 'q0\t' + str(self.q0) + '\n'
        s += 'q1\t' + str(self.q1)
        return s

    def to_cirq(self) -> cirq.Circuit:
        qs = cirq.LineQubit.range(2)
        qc = cirq.Circuit()
        for q0g, q1g in zip(self.q0, self.q1):

            if q0g == 'cz':
                qc.append(cirq.CZ(*qs))
            else:
                for i in range(max(len(q0g), len(q1g))):
                    try:
                        qc.append(_gate_from_op[q0g[i]](qs[0]))
                    except IndexError:
                        qc.append(_gate_from_op['I'](qs[0]))
                    try:
                        qc.append(_gate_from_op[q1g[i]](qs[1]))
                    except IndexError:
                        qc.append(_gate_from_op['I'](qs[1]))
        return qc

    def to_clifford_tableau(self) -> CliffordTableau:
        return CliffordTableau(self.to_cirq())

    def to_simple_tableau(self):
        raise NotImplementedError()

    def to_binary_op(self):
        raise NotImplementedError()


if __name__ == '__main__':
    for i in range(576, 576 + 24):
        c = C2Op(i)
        print(c)
        print(c.to_clifford_tableau())
        print('\n')

#
#
# if _generate_table:
#     print('starting...')
#     c2_unitaries = [clifford_to_unitary(index_to_clifford(i)) for i in range(size_c2)]
#     np.savez_compressed('c2_unitaries', c2_unitaries)
#     print('done')
# else:
#     ld = np.load('../raw/c2_unitaries.npz')
#     c2_unitaries = ld['arr_0']
#
#
# def is_phase(unitary):
#     if np.abs(np.abs(unitary[0, 0]) - 1) < 1e-10:
#         if np.max(np.abs(unitary / unitary[0, 0] - np.eye(4))) < 1e-10:
#             return True
#     else:
#         return False
#
#
# def unitary_to_index(unitary):
#     matches = []
#     prod_unitaries = unitary.conj().T @ c2_unitaries
#     eye4 = np.eye(4)
#     for i in range(size_c2):
#         if np.abs(np.abs(prod_unitaries[i, 0, 0]) - 1) < 1e-10:
#             if np.max(np.abs(prod_unitaries[i] / prod_unitaries[i, 0, 0] - eye4)) < 1e-10:
#                 matches.append(i)
#     assert len(matches) == 1, f"algrithm failed, found {len(matches)} matches > 1"
#
#     return matches[0]
#
#
# def generate_clifford_truncations(seq_len: int,
#                                   truncations_positions: Optional[List] = None,
#                                   seed: Optional[int] = None):
#     if seed is not None:
#         np.random.seed(seed)
#
#     if truncations_positions is None:
#         truncations_positions = range(seq_len)
#     else:
#         set_truncations = set(truncations_positions)
#         set_truncations.add(seq_len)
#         truncations_positions.append(sorted(list(set_truncations)))
#
#     # generate total sequence:
#     main_seq = [index_to_clifford(index) for index in np.random.randint(size_c2, size=seq_len)]
#     main_seq_unitaries = [clifford_to_unitary(seq) for seq in main_seq]
#     truncations_plus_inverse = []
#
#     # generate truncations:
#     for pos in truncations_positions:
#         trunc = main_seq[:pos + 1]
#         trunc_unitary = main_seq_unitaries[:pos + 1]
#         trunc_unitary_prod = np.eye(4)
#         for unitary in trunc_unitary:
#             trunc_unitary_prod = unitary @ trunc_unitary_prod
#         inverse_unitary = trunc_unitary_prod.conj().T
#         inverse_clifford = index_to_clifford(unitary_to_index(inverse_unitary))
#         trunc.append(inverse_clifford)
#         truncations_plus_inverse.append(trunc)
#     return truncations_plus_inverse
#
#
# def _clifford_seq_to_qiskit_circ(clifford_seq):
#     qc = QuantumCircuit(2)
#     for clifford in clifford_seq:
#         qc += _clifford_to_qiskit_circ(clifford)
#         qc.barrier()
#     return qc
#
#
# if __name__ == '__main__':
#     if _test_2_design:
#         sum_2d = 0
#         for i in range(size_c2):
#             if i % 10 == 0:
#                 print(i)
#             for j in range(size_c2):
#                 sum_2d += np.abs(np.trace(c2_unitaries[i].conj().T @ c2_unitaries[j])) ** 4
#         print("2 design ? ")
#         print(sum_2d / size_c2 ** 2)
