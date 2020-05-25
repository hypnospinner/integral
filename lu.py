import math
import operator
import numpy as np

# SWAP OPERATIONS

# swap rows


def swapr(matrix, fr, to):
    if fr != to:
        copy = np.copy(matrix)
        matrix[fr], matrix[to] = copy[to],  copy[fr]
    return matrix

# swap columns


def swapc(matrix, fr, to):
    if(fr != to):
        copy = np.copy(matrix)
        matrix[::, fr], matrix[::, to] = copy[::, to], copy[::, fr]
    return matrix

# swap col and row at the same time


def swapcr(matrix, frr, tor, frc, toc):
    matrix = swapr(matrix, frr, tor)
    matrix = swapc(matrix, frc, toc)
    return matrix


def pivot(L, U, P, Q, d):
    absU = abs(U)

    if absU[d:, d:].sum() < 1e-10:
        return False, L, U, P, Q

    i, j = np.where(absU[d:, d:] == absU[d:, d:].max())
    i[0] += d
    j[0] += d

    L = swapcr(L, i[0], d, j[0], d)
    U = swapcr(U, i[0], d, j[0], d)
    P = swapr(P, i[0], d)
    Q = swapc(Q, j[0], d)

    return True, L, U, P, Q


def factorize2(matrix):
    l = len(matrix)
    U = np.copy(matrix)
    L = np.zeros((l, l))
    P, Q = np.eye(l), np.eye(l)

    for i in range(l-1):

        success, L, U, P, Q = pivot(L, U, P, Q, i)

        if success == False:
            break

        T = np.eye(l)
        for k in range(i+1, l):
            L[k, i] = U[k, i] / U[i, i]
            T[k, i] = (-1) * L[k, i]

        U = np.dot(T, U)

    L = L + np.eye(l)

    return L, U, P, Q


def solveSLAE(matrix, b):
    L, U, P, Q = factorize2(matrix)
    x = np.zeros(len(b))
    y = np.zeros(len(b))

    pb = np.dot(P, b)

    l = len(pb)

    # Ly = Pb
    for k in range(l):
        y[k] = pb[k]

        for j in range(k):
            y[k] -= y[j] * L[k, j]

    # Ux = y
    for k in range(l-1, -1, -1):
        x[k] = y[k]

        for j in range(k+1, l):
            x[k] -= x[j] * U[k, j]

        x[k] /= U[k, k]

    return np.dot(Q, x)