import numpy as np

from simple_tableau import SimpleTableau, _beta, generate_from_name


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
            res[i].append(_beta(dd[i], dd[j]))
            assert res[i][j] == expectation[i][j]


def test_creation():
    t1 = SimpleTableau([[1, 0], [0, 1]], [0, 0])
    print(t1)
    # todo: assert


def test_eq():
    id1 = generate_from_name('id', 0)
    id2 = generate_from_name('id', 0)
    s = generate_from_name('s', 0)
    assert id1 == id2
    assert id1 != s


def test_from_name_single_qubit():
    s = generate_from_name('s', 0)
    print(s)
    assert s.g.shape == (2, 2)
    assert len(s.alpha) == 2
    assert np.array_equal(s.g, np.array([[1, 0], [1, 1]], dtype=np.uint8))
    assert np.array_equal(s.alpha, np.zeros_like(s.alpha))


def test_from_name_single_qubit_on_2_qubit_operator():
    s = generate_from_name('s', 0, n=2)
    assert s.g.shape == (4, 4)
    assert len(s.alpha) == 4
    assert np.array_equal(s.g, np.array([[1, 0, 0, 0],
                                         [1, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], dtype=np.uint8))
    assert np.array_equal(s.alpha, np.zeros_like(s.alpha))

    s = generate_from_name('s', 1, n=2)
    assert s.g.shape == (4, 4)
    assert len(s.alpha) == 4
    assert np.array_equal(s.g, np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 1, 1]], dtype=np.uint8))
    assert np.array_equal(s.alpha, np.zeros_like(s.alpha))
    y1 = generate_from_name('y', 1, n=2)


def test_from_name_two_qubit():
    gate = generate_from_name('cnot', (0, 1))
    assert gate == SimpleTableau(np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 1, 0, 1]]).T,
                                 np.zeros(4))
    # test that flipping control and target works
    gate = generate_from_name('cnot', (1, 0))
    assert gate == SimpleTableau(np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 1],
                                           [1, 0, 1, 0],
                                           [0, 0, 0, 1]]).T,
                                 np.zeros(4))
    generate_from_name('swap', (0, 1))
    generate_from_name('swap', (1, 0))
    # generate_from_name('iswap', (1, 0))
    # generate_from_name('iswap', (0, 1))
    generate_from_name('cz', (0, 1))
    generate_from_name('cz', (1, 0))


def test_compose_single_qubit():
    i2 = generate_from_name('id', 0)
    s = generate_from_name('s', 0)
    z = generate_from_name('z', 0)
    h = generate_from_name('h', 0)
    sx = generate_from_name('sx', 0)
    x = generate_from_name('x', 0)
    sy = generate_from_name('sy', 0)
    msy = generate_from_name('msy', 0)
    y = generate_from_name('y', 0)
    y0 = generate_from_name('y', 0, n=2)
    y1 = generate_from_name('y', 1, n=2)
    assert h.then(h) == i2
    assert s.then(s) == z
    assert sx.then(sx) == x
    assert sy.then(sy) == y
    assert s.then(h) == SimpleTableau(np.array([[1, 1], [1, 0]]).T, [1, 0])
    assert x.then(y) == z
    assert y.then(x) == z
    assert sy.then(msy) == i2
    assert msy.then(sy) == i2
    assert sy.then(x) == h
    assert x.then(msy) == h

    assert y1.then(y0) == y0.then(y1)
    assert y1.then(y0) == SimpleTableau(np.identity(4), [1, 1, 1, 1])


def test_compose_two_qubit():
    # flipping control and target with hadamards
    h0 = generate_from_name('h', 0, 2)
    h1 = generate_from_name('h', 1, 2)
    cnot_ab = generate_from_name('cnot', (0, 1))
    cnot_ba = generate_from_name('cnot', (1, 0))
    assert h0.then(h1).then(cnot_ab).then(h0).then(h1) == cnot_ba

    # cnot commutation identities
    x0 = generate_from_name('x', 0, 2)
    x1 = generate_from_name('x', 1, 2)
    z0 = generate_from_name('z', 0, 2)
    z1 = generate_from_name('z', 1, 2)
    assert x0.then(cnot_ab) == cnot_ab.then(x0).then(x1)
    assert z0.then(cnot_ab) == cnot_ab.then(z0)
    assert z1.then(cnot_ab) == cnot_ab.then(z0).then(z1)
    assert x1.then(cnot_ab) == cnot_ab.then(x1)

    # swap identity
    swap = generate_from_name('swap', (0, 1), 2)
    swap_reverse = generate_from_name('swap', (1, 0), 2)
    assert swap_reverse == swap
    assert cnot_ab.then(cnot_ba).then(cnot_ab) == swap

    # cz identities
    cz = generate_from_name('cz', (0, 1), 2)
    cz_reverse = generate_from_name('cz', (1, 0), 2)
    assert cz == cz_reverse
    assert h1.then(cnot_ab).then(h1) == cz
    assert h0.then(cnot_ba).then(h0) == cz
