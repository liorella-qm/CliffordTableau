import numpy as np
import stim

from simple_tableau import SimpleTableau


def stim_to_simple(tableau: stim._stim_march_sse2.Tableau) -> SimpleTableau:
    int_bit_map = {
        0: [0, 0],
        1: [1, 0],
        2: [1, 1],
        3: [0, 1]
    }
    n = len(tableau)
    g = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    alpha = np.zeros(2 * n, dtype=np.uint8)
    for j in range(0, 2 * n, 2):
        pauli_string_x = tableau.x_output(j // 2)
        pauli_string_z = tableau.z_output(j // 2)
        alpha[j] = pauli_string_x.sign.real < 0
        alpha[j + 1] = pauli_string_z.sign.real < 0
        for i in range(0, 2 * n, 2):
            g[i:i + 2, j] = int_bit_map[pauli_string_x[i // 2]]
            g[i:i + 2, j+1] = int_bit_map[pauli_string_z[i // 2]]
    return SimpleTableau(g, alpha)
