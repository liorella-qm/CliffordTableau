from pprint import pprint

import cirq
import numpy as np

from clifford_tableau import CliffordTableau
from simple_tableau import SimpleTableau, generate_from_names, compose, beta


def test_beta():
    expectation = [
        [0, 0, 0, 0],
        [0, 0, 3, 1],
        [0, 1, 0, 3],
        [0, 3, 1, 0]
    ]
    dd = [[0, 0], [1, 0], [0, 1], [1, 1]]  # i, x, z, y
    res = []
    for i in range(4):
        res.append([])
        for j in range(4):
            print((dd[i], dd[j], beta(dd[i], dd[j])))
            res[i].append(beta(dd[i], dd[j]))
            assert res[i][j] == expectation[i][j]

    # assert beta([1, 0], [0, 1]) == 3


def test_creation():
    t1 = SimpleTableau([[1, 0], [0, 1]], [0, 0])
    print(t1)
    # todo: assert


def test_from_name():
    s = generate_from_names(['s'])
    print(s)
    # todo: assert


def test_eq():
    id1 = generate_from_names(['id'])
    id2 = generate_from_names(['id'])
    s = generate_from_names(['s'])
    assert id1 == id2
    assert id1 != s


def test_compose():
    i2 = generate_from_names(['id'])
    s = generate_from_names(['s'])
    z = generate_from_names(['z'])
    h = generate_from_names(['h'])
    sx = generate_from_names(['sx'])
    x = generate_from_names(['x'])
    sy = generate_from_names(['sy'])
    msy = generate_from_names(['msy'])
    y = generate_from_names(['y'])
    assert compose(h, h) == i2
    assert compose(s, s) == z
    assert compose(sx, sx) == x
    assert compose(sy, sy) == y
    assert compose(s, h) == SimpleTableau(np.array([[1, 1], [1, 0]]).T, [1, 0])
    assert compose(x, y) == z
    assert compose(y, x) == z
    assert compose(sy, msy) == i2
    assert compose(msy, sy) == i2
    assert compose(sy, x) == h
    assert compose(x, msy) == h