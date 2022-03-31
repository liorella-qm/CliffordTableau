from typing import Dict, Tuple, List

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


def beta(v, u):
    u = np.array(u)
    v = np.array(v)
    uz = u[1::2]
    ux = u[::2]
    vz = v[1::2]
    vx = v[::2]

    return (np.dot(vz + uz, vx + ux) % 2 - np.dot(vz, vx) % 2 - np.dot(uz, ux) % 2 + 2 * (np.dot(vx, uz) % 2)) % 4


def compose_alpha(g1, alpha1, g2, alpha2):
    n = len(alpha1) // 2
    alpha21 = []
    two_alpha21 = 2 * alpha1 % 4
    for i in range(2 * n):
        a = g1[:, i]
        #         print('a=', a)
        for j in range(0, 2 * n, 2):
            ax = a[j]
            az = a[j+1]
            two_alpha21[i] += 2 * ax * alpha2[j] + 2 * az * alpha2[j+1] % 4
            if ax * az:
                #                 print('g2x=', g2[:, j])
                #                 print('g2z=', g2[:, j+1])
                #                 print('beta=', beta(g2[:, j], g2[:, j+1]))
                s = (beta(g2[:, j], g2[:, j + 1]) + 1) % 4
                assert s % 2 == 0, 'expression does not divide by 2'
                two_alpha21[i] += s % 4
        assert two_alpha21[i] % 2 == 0
        alpha21.append(two_alpha21[i] // 2 % 2)  # not sure
    return np.array(alpha21).astype(np.uint8)


class SimpleTableau:
    def __init__(self, g, alpha):
        if not len(alpha) % 2 == 0:
            raise ValueError(f'alpha needs to have an even length but length is {len(alpha)}')
        self._n = len(alpha) // 2
        self._np_repr = np.vstack((
            g, alpha
        )).astype(np.uint8)

    #         self._np_repr = np.vstack((np.identity(2 * n, dtype=np.uint8),
    #                                   np.zeros((1, 2 * n), dtype=np.uint8)
    #                                   )).astype(np.uint8)

    def set_x_image(self, qubit, image):
        """
        set the image of x as a vector with 2*n+1 binary entries, (x0, z0, x1, z1, ..., sign)
        """
        raise NotImplementedError()

    def set_z_image(self, qubit, image):
        """
        set the image of z as a vector with 2*n+1 binary entries, (x0, z0, x1, z1, ..., sign)
        """
        raise NotImplementedError()

    def set_g(self, mat):
        self._np_repr[:-1, :] = np.array(mat).astype(np.uint8)

    def set_alpha(self, vec):
        self._np_repr[-1] = np.array(vec).astype(np.uint8)

    @property
    def g(self):
        return self._np_repr[:-1, :]

    @property
    def alpha(self):
        return self._np_repr[-1]

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


def compose(t1: SimpleTableau, t2: SimpleTableau) -> SimpleTableau:
    g12 = t2.g @ t1.g % 2
    alpha12 = compose_alpha(t1.g, t1.alpha, t2.g, t2.alpha)
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


def generate_from_names(names: List[str]):
    if len(names) > 1:
        raise NotImplementedError('currently only single qubits supported')
    return SimpleTableau(*_single_qubit_gate_conversions[names[0]])
