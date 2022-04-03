from functools import reduce
from typing import Dict, Tuple, List, Union

import numpy as np
from numpy.typing import ArrayLike


def g(x0z0, x1z1):
    if np.all(x0z0 == [0, 0]):
        return 0
    if np.all(x0z0 == [0, 1]):
        return x1z1[0] * (1 - 2 * x1z1[1])
    if np.all(x0z0 == [1, 0]):
        return x1z1[1] * (2 * x1z1[0] - 1)
    if np.all(x0z0 == [1, 1]):
        return x1z1[1] - x1z1[0]


def _beta(v, u):
    u = np.array(u)
    v = np.array(v)
    uz = u[1::2]
    ux = u[::2]
    vz = v[1::2]
    vx = v[::2]

    return (np.dot(vz + uz, vx + ux) % 2 - np.dot(vz, vx) % 2 - np.dot(uz, ux) % 2 + 2 * (np.dot(vx, uz) % 2)) % 4


def _compose_alpha(g1, alpha1, g2, alpha2):
    n = len(alpha1) // 2
    alpha21 = []
    two_alpha21 = 2 * alpha1 % 4
    for i in range(2 * n):
        a = g1[:, i]
        for j in range(0, 2 * n, 2):
            ax = a[j]
            az = a[j + 1]
            two_alpha21[i] += 2 * ax * alpha2[j] + 2 * az * alpha2[j + 1] % 4
            if ax * az:
                s = (_beta(g2[:, j], g2[:, j + 1]) + 1) % 4
                assert s % 2 == 0, 'expression does not divide by 2'
                two_alpha21[i] += s % 4
        assert two_alpha21[i] % 2 == 0
        alpha21.append(two_alpha21[i] // 2 % 2)
    return np.array(alpha21).astype(np.uint8)


class SimpleTableau:
    def __init__(self, g, alpha):
        g = np.array(g, dtype=np.uint8)
        alpha = np.array(alpha, dtype=np.uint8)
        if not len(alpha) % 2 == 0:
            raise ValueError(f'alpha needs to have an even length but length is {len(alpha)}')
        g_shape = g.shape
        if not (len(g_shape) == 2 and g_shape[0] == g_shape[1] and g_shape[0] % 2 == 0):
            raise ValueError(f'g has shape {g_shape}, which is not an even square matrix')
        if not len(alpha) % 2 == 0:
            raise ValueError(f'alpha has len {len(alpha)}, which is not even')
        self._n = len(alpha) // 2
        if not _is_symplectic(g, self._n):
            raise ValueError('g is not a symplectic matrix')
        self._np_repr = np.vstack((
            g, alpha
        )).astype(np.uint8)

    @property
    def g(self):
        return self._np_repr[:-1, :]

    @property
    def alpha(self):
        return self._np_repr[-1]

    @property
    def n(self):
        return self._n

    def __str__(self):
        # todo: work with more than single digit qubit number
        st = " " * 2 + "|" + " ".join(f"x{i} z{i}" for i in range(self._n)) + "\n"
        st += "-" * 2 + "+" + "-" * 6 * self._n + "\n"
        for i in range(2 * self._n):
            st += f"z{i // 2}|" if i % 2 else f"x{i // 2}|"
            st += "  ".join(str(entry) for entry in self._np_repr[i]) + "\n"
        st += "s |" + "  ".join(str(entry) for entry in self._np_repr[i + 1]) + "\n"
        return st

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return np.array_equal(self._np_repr, other._np_repr)

    def then(self, other: 'SimpleTableau') -> 'SimpleTableau':
        if self.n != other.n:
            raise ValueError(f'number of qubits of self={self.n} and of other={other.n} is incompatible')
        g12 = other.g @ self.g % 2
        alpha12 = _compose_alpha(self.g, self.alpha, other.g, other.alpha)
        return SimpleTableau(g12, alpha12)


_single_qubit_gate_conversions = {
    'id': (np.identity(2), np.zeros(2)),
    'h': (np.array([[0, 1], [1, 0]]), np.zeros(2)),
    'x': (np.identity(2), np.array([0, 1])),
    'z': (np.identity(2), np.array([1, 0])),
    'y': (np.identity(2), np.array([1, 1])),
    's': (np.array([[1, 0], [1, 1]]), np.zeros(2)),
    'sx': (np.array([[1, 1], [0, 1]]), np.array([0, 1])),
    'sy': (np.array([[0, 1], [1, 0]]), np.array([1, 0])),
    'msy': (np.array([[0, 1], [1, 0]]), np.array([0, 1]))
}

_two_qubit_gate_conversions = {
    'cnot': (np.array([[1, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 1]]).T,
             np.zeros(4)),
    'iswap': (np.array([[0, 1, 1, 1],
                        [1, 1, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]]).T,
              np.zeros(4)),  # not 100% sure this is correct
    'swap': (np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0]]).T,
             np.zeros(4)),
    'cz': (np.array([[1, 0, 0, 1],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0],
                     [0, 0, 0, 1]]).T,
           np.zeros(4))
}

known_names = set(_single_qubit_gate_conversions.keys()).union(set(_two_qubit_gate_conversions.keys()))


def generate_from_name(name: str, target: Union[int, Tuple[int, int]], n=None) -> SimpleTableau:
    """
    Generate a `SimpleTableau` object from a name.
    Args:
        name: gate name, must be in `known_names`
        target: For a single qubit gate, the qubit on which the gate operates, from 0 to n-1. For two qubit gates,
                a tuple of the form (control, target) on which the gate operates
        n: number of total qubits on which gate operates

    Returns: the `SimpleTableau` object

    """
    if n is None:
        n = np.max(target) + 1
    g = np.identity(2 * n)
    alpha = np.zeros(2 * n)
    if name in _single_qubit_gate_conversions:
        if not isinstance(target, int) or target >= n:
            raise ValueError(f'invalid target {target} for single qubit gate {name} on {n} qubits')
        _embed_single_qubit_gate(alpha, g, name, target)
        return SimpleTableau(g, alpha)

    elif name in _two_qubit_gate_conversions:
        if not isinstance(target, tuple) or max(target) >= n:
            raise ValueError(f'invalid target {target} for two qubit gate {name} on {n} qubits')
        _embed_two_qubit_gate(alpha, g, name, target)
        return SimpleTableau(g, alpha)
    else:
        raise ValueError(f'unknown gate {name}')


def _embed_two_qubit_gate(alpha, g, name, target):
    g2q, alpha2q = _two_qubit_gate_conversions[name]
    g[2 * target[0]: 2 * target[0] + 2, 2 * target[0]: 2 * target[0] + 2] = g2q[:2, :2]
    g[2 * target[0]: 2 * target[0] + 2, 2 * target[1]: 2 * target[1] + 2] = g2q[:2, 2:4]
    g[2 * target[1]: 2 * target[1] + 2, 2 * target[0]: 2 * target[0] + 2] = g2q[2:4, :2]
    g[2 * target[1]: 2 * target[1] + 2, 2 * target[1]: 2 * target[1] + 2] = g2q[2:4, 2:4]
    alpha[2 * target[0]: 2 * target[0] + 2] = alpha2q[:2]
    alpha[2 * target[1]: 2 * target[1] + 2] = alpha2q[2:4]


def _embed_single_qubit_gate(alpha, g, name, target):
    g[2 * target:2 * target + 2, 2 * target:2 * target + 2] = _single_qubit_gate_conversions[name][0]
    alpha[2 * target:2 * target + 2] = _single_qubit_gate_conversions[name][1]


def _lambda(n):
    return np.diag([1] + [0, 1] * (n - 1), 1) + np.diag([1] + [0, 1] * (n - 1), -1)


def _is_symplectic(mat, n):
    lhs = mat @ _lambda(n) @ mat.T % 2
    return np.all(lhs == _lambda(n))
