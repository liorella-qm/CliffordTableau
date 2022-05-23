# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:58:22 2022

@author: Charlie Guinn
"""

import numpy as np
from scipy.linalg import expm
import cirq
import pickle

q1, q2 = cirq.LineQubit.range(2)

I = np.matrix([[1, 0], [0, 1]])  # identity
X = np.matrix([[0, 1], [1, 0]])  # pi x
X2 = expm(-1j * X * np.pi / 4)  # pi/2 x
mX2 = expm(1j * X * np.pi / 4)  # -pi/2 x
Y = np.matrix([[0, -1j], [1j, 0]])  # pi y
Y2 = expm(-1j * Y * np.pi / 4)  # pi/2 y
mY2 = expm(1j * Y * np.pi / 4)  # -pi/2 y
Z = np.matrix([[1, 0], [0, -1]])  # pi z
CNOT = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
SWAP = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
iSWAP = np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

# cirq definitions
X_1 = cirq.X(q1)
X2_1 = cirq.X(q1) ** 0.5
mX2_1 = cirq.X(q1) ** -0.5
Y_1 = cirq.Y(q1)
Y2_1 = cirq.Y(q1) ** 0.5
mY2_1 = cirq.Y(q1) ** -0.5
X_2 = cirq.X(q2)
X2_2 = cirq.X(q2) ** 0.5
mX2_2 = cirq.X(q2) ** -0.5
Y_2 = cirq.Y(q2)
Y2_2 = cirq.Y(q2) ** 0.5
mY2_2 = cirq.Y(q2) ** -0.5
CNOT_12 = [cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=-1)(q1),
           cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1)(q2),
           cirq.ISWAP(q1, q2) ** 0.5,
           cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=1, z_exponent=0)(q1),
           cirq.ISWAP(q1, q2) ** 0.5,
           cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0.5)(q1),
           cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0.5, z_exponent=1)(q2)
           ]  # compilation of CNOT in terms of phased XZ and sqiSWAP

iSWAP_12 = [cirq.ISWAP(q1, q2) ** 0.5,
            cirq.ISWAP(q1, q2) ** 0.5]  # compilation of iSWAP in terms of phased XZ and sqiSWAP

SWAP_12 = [cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5)(q1),
           cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0)(q2),
           cirq.ISWAP(q1, q2) ** 0.5,
           cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0.5, z_exponent=1)(q1),
           cirq.PhasedXZGate(axis_phase_exponent=-1, x_exponent=0.5, z_exponent=1)(q2),
           cirq.ISWAP(q1, q2) ** 0.5,
           cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0)(q1),
           cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0)(q2),
           cirq.ISWAP(q1, q2) ** 0.5,
           cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5)(q1),
           cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1)(q2)
           ]  # compilation of SWAP in terms of phased XZ and sqiSWAP

# reduced set of single qubit Clifford unitaries
C1_reduced = [I,
              mX2,
              mY2 @ X,
              mY2 @ mX2,
              X2 @ Y2,
              X2 @ Y2 @ mX2]

# reduced set of single qubit Clifford Cirq definitions on qubit 1
C1_reduced_q1 = [[],
                 cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=-0.5, z_exponent=0)(q1),
                 cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=1)(q1),
                 cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=-0.5)(q1),
                 cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5)(q1),
                 cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0.5)(q1)]

# reduced set of single qubit Clifford Cirq definitions on qubit 2
C1_reduced_q2 = [[],
                 cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=-0.5, z_exponent=0)(q2),
                 cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=1)(q2),
                 cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=-0.5)(q2),
                 cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5)(q2),
                 cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0.5)(q2)]

# full single qubit Clifford group unitaries
C1_full = [I,
           X,
           Y,
           X @ Y,
           Y2 @ X2,
           mY2 @ X2,
           Y2 @ mX2,
           mY2 @ mX2,
           X2 @ Y2,
           mX2 @ Y2,
           X2 @ mY2,
           mX2 @ mY2,
           X2,
           mX2,
           Y2,
           mY2,
           X2 @ Y2 @ mX2,
           X2 @ mY2 @ mX2,
           Y2 @ X,
           mY2 @ X,
           X2 @ Y,
           mX2 @ Y,
           X2 @ Y2 @ X2,
           mX2 @ Y2 @ mX2]

# S1 unitaries, see https://www.nature.com/articles/nature13171
S1 = [I,
      X2 @ Y2,
      mY2 @ mX2]

# S1 Cirq definitions on qubit 1
S1_q1 = [[],
         cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5)(q1),
         cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=-0.5)(q1)]

# S1 Cirq definitions on qubit 2
S1_q2 = [[],
         cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5)(q2),
         cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=-0.5)(q2)]

paulis = [I, X, Y, Z]
twoQBPaulis = [np.kron(i, j) for i in paulis for j in paulis]

# ordered list of two qubit Pauli operator labels
pauliTable = ['II',
              'IX',
              'IY',
              'IZ',
              'XI',
              'XX',
              'XY',
              'XZ',
              'YI',
              'YX',
              'YY',
              'YZ',
              'ZI',
              'ZX',
              'ZY',
              'ZZ']

# ordered list of Clifford tableau columns for each Pauli product
symplecticTable = [[0, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 1],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0],
                   [1, 1, 0, 0],
                   [1, 1, 0, 1],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1],
                   [1, 0, 1, 1],
                   [0, 0, 1, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1],
                   [0, 0, 1, 1]]


def get_pauli_prod(m):
    # input: tensor product of two Pauli matrices
    # output: tableau column of m

    for i, p in enumerate(twoQBPaulis):
        prod = m @ p
        if np.trace(prod) > 3.9:
            return symplecticTable[i], 0
        if np.trace(prod) < -3.9:
            return symplecticTable[i], 1


# The following functions output unitaries for two-qubit Cliffords using the reduced
# single qubit set. These will be used to make symplectic matrices.
def make_C1_r(i, j):
    return np.matrix(np.kron(C1_reduced[i], C1_reduced[j]))


def make_CNOT_r(i, j, k, l):
    return np.matrix(np.kron(S1[k], S1[l]) @ CNOT @ np.kron(C1_reduced[i], C1_reduced[j]))


def make_iSWAP_r(i, j, k, l):
    return np.matrix(np.kron(S1[k], S1[l]) @ iSWAP @ np.kron(C1_reduced[i], C1_reduced[j]))


def make_SWAP_r(i, j):
    return np.matrix(SWAP @ np.kron(C1_reduced[i], C1_reduced[j]))


# The following functions output unitaries for two-qubit Cliffords using the full
# single qubit set. These are used to verify the full set is a unitary 2-design
def make_C1_f(i, j):
    return np.matrix(np.kron(C1_full[i], C1_full[j]))


def make_CNOT_f(i, j, k, l):
    return np.matrix(np.kron(S1[k], S1[l]) @ CNOT @ np.kron(C1_full[i], C1_full[j]))


def make_iSWAP_f(i, j, k, l):
    return np.matrix(np.kron(S1[k], S1[l]) @ iSWAP @ np.kron(C1_full[i], C1_full[j]))


def make_SWAP_f(i, j):
    return np.matrix(SWAP @ np.kron(C1_full[i], C1_full[j]))


# The following functions output Cirq objects for two-qubit Cliffords using the reduced
# single qubit set. 
def make_C1_XZ(i, j):
    return cirq.Circuit([C1_reduced_q1[i], C1_reduced_q2[j]])


def make_CNOT_XZ(i, j, k, l):
    return cirq.Circuit([C1_reduced_q1[i], C1_reduced_q2[j],
                         CNOT_12,
                         S1_q1[k], S1_q2[l]])


def make_iSWAP_XZ(i, j, k, l):
    return cirq.Circuit([C1_reduced_q1[i], C1_reduced_q2[j],
                         iSWAP_12,
                         S1_q1[k], S1_q2[l]])


def make_SWAP_XZ(i, j):
    return cirq.Circuit([C1_reduced_q1[i], C1_reduced_q2[j],
                         SWAP_12])


def get_symplectic_and_phase(m):
    # Turns a two-qubit unitary into the full tableau representation
    # input: two-qubit unitary m
    # outputs: s: symplectic matrix corresponding to m
    #         p: phase vector corresponding to m 
    s = np.zeros([4, 4])
    p = np.zeros(4)

    s[:, 0], p[0] = get_pauli_prod(m.H @ np.kron(X, I) @ m)
    s[:, 1], p[1] = get_pauli_prod(m.H @ np.kron(I, X) @ m)
    s[:, 2], p[2] = get_pauli_prod(m.H @ np.kron(Z, I) @ m)
    s[:, 3], p[3] = get_pauli_prod(m.H @ np.kron(I, Z) @ m)

    return s, p


def symplectic_to_int(m):
    # Turns a symplectic matrix into a unique integer. Used to verify all 720 symplectics
    # are made
    # input: symplectic matrix
    # output: 16 bit integer
    flat_m = m.ravel()
    num = 0

    for i in range(16):
        num += 2 ** i * flat_m[i]

    return num


def phase_to_int(p):
    # Turns a phase vector into a unique integer
    # input: phase vector
    # output: 4 bit integer
    num = 0

    for i in range(4):
        num += 2 ** i * p[i]

    return num


def full_matrix_to_int(m, p):
    # Turns a tableau into a unique integer
    # inputs: m: symplectic matrix
    #        p: phase vector
    # output: 20 bit integer
    flat_m = m.ravel()
    num = 0

    for i in range(16):
        num += 2 ** i * flat_m[i]

    for i in range(4):
        num += 2 ** (i + 16) * p[i]

    return num


def is_symplectic(m):
    # Checks if a matrix is symlectic
    # input: matrix
    # output: Boolean, True if m is symplectic
    Omega = np.matrix([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]])
    test = m.T @ Omega @ m

    if test.all() == Omega.all():
        return True
    else:
        return False


def make_symplectic_cliffords():
    # makes full set of synplectic matrices using reduced single qubit Clifford set
    # output: cliffords: list of unitaries for reduced two-qubit Clifford group
    #         commands: list of instructions to make gates, used for debug
    #         circuits: list of Cirq objects for reduced two-qubit Clifford group 
    cliffords = []
    commands = []
    circuits = []

    # single qubit class
    for i in range(6):
        for j in range(6):
            cliffords.append(make_C1_r(i, j))
            commands.append(['C1\'s', i, j])
            circuits.append(make_C1_XZ(i, j))

    # CNOT class
    for i in range(6):
        for j in range(6):
            for k in range(3):
                for l in range(3):
                    cliffords.append(make_CNOT_r(i, j, k, l))
                    commands.append(['CNOT\'s', i, j, k, l])
                    circuits.append(make_CNOT_XZ(i, j, k, l))

    # iSWAP class
    for i in range(6):
        for j in range(6):
            for k in range(3):
                for l in range(3):
                    cliffords.append(make_iSWAP_r(i, j, k, l))
                    commands.append(['iSWAP\'s', i, j, k, l])
                    circuits.append(make_iSWAP_XZ(i, j, k, l))

    # SWAP class
    for i in range(6):
        for j in range(6):
            cliffords.append(make_SWAP_r(i, j))
            commands.append(['SWAP\'s', i, j])
            circuits.append(make_SWAP_XZ(i, j))

    return cliffords, commands, circuits


def make_all_cliffords():
    # makes full two-qubit clifford group
    # output: cliffords: list of unitaries for two-qubit Clifford group
    #         commands: list of instructions to make gates, used for debug
    cliffords = []
    commands = []

    # single qubit class
    for i in range(24):
        for j in range(24):
            cliffords.append(make_C1_f(i, j))
            commands.append(['C1\'s', i, j])

    # CNOT class
    for i in range(24):
        for j in range(24):
            for k in range(3):
                for l in range(3):
                    cliffords.append(make_CNOT_f(i, j, k, l))
                    commands.append(['CNOT\'s', i, j, k, l])

    # iSWAP class
    for i in range(24):
        for j in range(24):
            for k in range(3):
                for l in range(3):
                    cliffords.append(make_iSWAP_f(i, j, k, l))
                    commands.append(['iSWAP\'s', i, j, k, l])

    # SWAP class
    for i in range(24):
        for j in range(24):
            cliffords.append(make_SWAP_f(i, j))
            commands.append(['SWAP\'s', i, j])

    return cliffords, commands


# make the reduced Cliffords
cliffords, commands, circuits = make_symplectic_cliffords()

# make the full Cliffords
# cliffords_full,commands_full = make_all_cliffords()

# Pull unitaries directly from Cirq object list, also used for debug
cirq_unitaries = [np.matrix(c.unitary([q1, q2])) for c in circuits]


def check2Design(unitaries):
    # checks if full two-qubit Clifford group generated is unitary 2-design
    # input: list of unitaries
    # output: value that had better be 2 if you did this right
    check = 0

    numUs = len(unitaries)

    for i in range(numUs):
        for j in range(numUs):
            check += (np.absolute(np.trace(np.conjugate(unitaries[i].T) @ unitaries[j]))) ** 4 / (numUs ** 2)
        if (i % 100 == 0):
            print(i, "/ 11520")

    print(check)


# make the symplectics
symplectics = []  # for reduced set
phases = []  # for reduced set

tableaus = []  # for full set, this is actually just storing the symplectic part
full_phases = []  # for full set

# populate symplectic matrix list
for m in cirq_unitaries:
    s, p = get_symplectic_and_phase(m)
    symplectics.append(s)
    phases.append(p)

'''    
#populate full tableau list
for m in cliffords_full:
    s,p = get_symplectic_and_phase(m)
    tableaus.append(s)
    full_phases.append(p)
'''

# test to see if they are symplectic
for s in symplectics:
    if is_symplectic(s) == False:
        print('oops, something isn\'t symplectic!')

'''       
# test to see if they are symplectic
for s in tableaus:
    if is_symplectic(s) == False:
        print('oops, something isn\'t symplectic!')
'''

# test for duplicates using the conversion of symplectics to unique integers
array_nums = []
dupe_count = 0
for i, s in enumerate(symplectics):
    new_num = symplectic_to_int(s)
    if new_num in array_nums:
        print('oops, there is a duplicate symplectic!')
        print(i, array_nums.index(new_num))
    array_nums.append(new_num)

'''
# test for duplicates
array_nums = []
dupe_count = 0
for i in range(11520):
    new_num = full_matrix_to_int(tableaus[i],full_phases[i])
    
    #symplectics_to_check = symplectics[i:]
    if new_num in array_nums:
        dupe_count+=1
        print('oops, there is a duplicate symplectic!') 
        print(i,array_nums.index(new_num))
       #f=1
    array_nums.append(new_num)
'''
print(dupe_count, 'duplicate symplectics')

# The following line takes 20-30 mins
# check2Design(cliffords_full) #2 design verified 5/4/22 1.9999999987445924

data_to_export = {'symplectics': symplectics,  # list of symplectics
                  'phases': phases,  # list of phases
                  'circuits': circuits,  # list of Cirq objects
                  'unitaries': cirq_unitaries}  # list of unitaries

with open('symplectic_compilation_XZ.pkl', 'wb') as f:
    pickle.dump(data_to_export, f)

with open('symplectic_compilation_XZ.pkl', 'rb') as f:
    test_load = pickle.load(f)
