import numpy as np
import numpy.random
import stim

from simple_tableau import SimpleTableau, _beta, generate_from_name, stim_to_simple


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

    _beta([1, 1, 0, 0], [0, 0, 1, 1])


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
    print('\n', s)
    assert s.g.shape == (2, 2)
    assert len(s.alpha) == 2
    assert np.array_equal(s.g, np.array([[1, 0], [1, 1]], dtype=np.uint8))
    assert np.array_equal(s.alpha, np.zeros_like(s.alpha))


def test_from_name_single_qubit_on_2_qubit_operator():
    s = generate_from_name('s', 0, n=2)
    print('\n', s)
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
    # single qubit in parallel
    sy0 = generate_from_name('sy', 0, 2)
    sy1 = generate_from_name('sy', 1, 2)
    msy0 = generate_from_name('msy', 0, 2)
    msy1 = generate_from_name('msy', 1, 2)
    id0 = generate_from_name('id', 0, 2)
    id1 = generate_from_name('id', 1, 2)
    y0 = generate_from_name('y', 0, 2)
    y1 = generate_from_name('y', 1, 2)
    assert sy0.then(sy1).then(sy0).then(sy1) == y0.then(y1)
    assert sy0.then(sy1).then(msy0).then(msy1) == id0.then(id1)

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


def test_inverse():
    s = generate_from_name('s', 0)
    assert s.inverse() == SimpleTableau([[1, 0],
                                         [1, 1]],
                                        [1, 0])
    s0 = generate_from_name('s', 0, 2)
    s1 = generate_from_name('s', 1, 2)
    sy0 = generate_from_name('sy', 0, 2)
    sy1 = generate_from_name('sy', 1, 2)
    msy0 = generate_from_name('msy', 0, 2)
    msy1 = generate_from_name('msy', 1, 2)
    h0 = generate_from_name('h', 0, 2)
    h1 = generate_from_name('h', 1, 2)
    id0 = generate_from_name('id', 0, 2)
    id1 = generate_from_name('id', 1, 2)
    assert sy0.then(sy1).inverse() == msy0.then(msy1)
    assert s0.then(sy1).inverse() == msy1.then(s0.inverse())

    cz = generate_from_name('cz', (0, 1), 2)
    assert cz.inverse() == cz

    # random circuit
    circ = h0.then(cz).then(h1).then(s0).then(msy0).then(cz).then(msy1)
    assert circ.then(circ.inverse()) == id0.then(id1)


def test_convert_from_stim_basic():
    h_stim = stim.Tableau.from_named_gate('H')
    h_st = stim_to_simple(h_stim)
    assert h_st == generate_from_name('h', 0)


def test_convert_from_stim_random_basic():
    # todo: seed
    c_stim = stim.Tableau.random(2)
    c_st = stim_to_simple(c_stim)


def test_random_circuits_then_1q():
    for _ in range(100):
        c_stim1 = stim.Tableau.random(1)
        c_stim2 = stim.Tableau.random(1)
        assert stim_to_simple(c_stim1).then(stim_to_simple(c_stim2)) == stim_to_simple(c_stim1.then(c_stim2))


def test_random_circuits_inverse_1q():
    for _ in range(100):
        c_stim = stim.Tableau.random(1)
        assert stim_to_simple(c_stim).inverse() == stim_to_simple(c_stim.inverse())


def test_random_circuits_then_2q():
    for i in range(1000):
        print(i)
        c_stim1 = stim.Tableau.random(2)
        c_stim2 = stim.Tableau.random(2)
        c_st1 = stim_to_simple(c_stim1)
        c_st2 = stim_to_simple(c_stim2)
        print(c_stim1)
        print(c_st1)
        print(c_stim2)
        print(c_st2)
        assert c_stim1.then(c_stim2).inverse() == c_stim2.inverse().then(c_stim1.inverse())
        assert c_st1.then(c_st2).inverse() == c_st2.inverse().then(c_st1.inverse())
        if stim_to_simple(c_stim1).then(stim_to_simple(c_stim2)) != stim_to_simple(c_stim1.then(c_stim2)):
            print('expected')
            print(stim_to_simple(c_stim1.then(c_stim2)))
            print('actual')
            print(stim_to_simple(c_stim1).then(stim_to_simple(c_stim2)))
            stim_to_simple(c_stim1).then(stim_to_simple(c_stim2))
            assert False


def test_random_circuits_inverse_2q():
    for _ in range(1000):
        c_stim = stim.Tableau.random(2)
        c_st = stim_to_simple(c_stim)
        print(c_stim)
        print(c_st)
        assert stim_to_simple(c_stim).inverse() == stim_to_simple(c_stim.inverse())
        assert c_st.inverse().then(c_st) == SimpleTableau(np.identity(4), np.zeros(4))


def test_compose_pauli_clifford_2q():
    """
    test that pauli and then clifford simply adds the pauli to the clifford

    """
    for _ in range(1000):
        c_stim = stim.Tableau.random(2)
        c_st = stim_to_simple(c_stim)
        c_st = SimpleTableau(c_st.g, np.zeros(4))  # convert to a tableau with positive signs
        rand_sign = np.random.randint(0, 2, 4, dtype=np.uint8)
        rand_pauli = SimpleTableau(np.identity(4), rand_sign)
        added = SimpleTableau(c_st.g, rand_sign)
        assert rand_pauli.then(c_st) == added


def test_left_coset_equal_to_right_coset():
    """
    test that the right coset is simply the symplectic matrix with all the possible signs
    (i.e the left coset is equal to the right coset)
    Returns:

    """
    for _ in range(1000):
        c_stim = stim.Tableau.random(2)
        c_st = stim_to_simple(c_stim)
        c_st = SimpleTableau(c_st.g, np.zeros(4))  # convert to a tableau with positive signs
        left_coset_signs = set()
        right_coset_signs = set()

        for i in range(16):
            binary = np.array([int(s) for s in np.binary_repr(i, 4)], dtype=np.uint8)
            pauli = SimpleTableau(np.identity(4), binary)
            left_coset_signs.add(tuple(binary))
            prod = c_st.then(pauli)
            right_coset_signs.add(tuple(prod.alpha))
        assert left_coset_signs == right_coset_signs

