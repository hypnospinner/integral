import func7
from numpy import array, sum, append, zeros, vstack, multiply
from lu import solveSLAE
from math import factorial, log, ceil, pow, acos, acosh, asin, asinh, sqrt, cos, sin, pi, sinh, cosh


def cardano2(coefs):
    coefs = coefs[::-1]
    a, b, c = coefs[0], coefs[1], coefs[2]

    Q = ((a ** 2) - 3 * b) / 9
    R = (2 * (a ** 3) - 9 * a * b + 27 * c) / 54

    S = (Q ** 3) - (R ** 2)

    if S > 0:

        fi = acos(R / sqrt(Q ** 3)) / 3

        x_1 = -2 * sqrt(Q) * cos(fi) - a / 3
        x_2 = -2 * sqrt(Q) * cos(fi + (2 * pi / 3)) - a / 3
        x_3 = -2 * sqrt(Q) * cos(fi - (2 * pi / 3)) - a / 3

        return array([x_1, x_2, x_3], float)

    elif S < 0:

        if Q > 0:
            fi = acosh(abs(R) / sqrt(Q ** 3)) / 3

            x_1 = -2 * (1 if R > 0 else -1) * sqrt(Q) * cosh(fi) - a / 3

            return array([x_1], float)

        elif Q < 0:
            fi = asinh(abs(R) / sqrt(abs(Q) ** 3))

            x_1 = -2 * (1 if R > 0 else -1) * sqrt(abs(Q)) * sinh(fi) - a / 3

            return array([x_1], float)

    else:

        x_1 = -2 * (1 if R > 0 else -1) * sqrt(Q) - a / 3
        x_2 = (1 if R > 0 else -1) * sqrt(Q) - a / 3

        return array([x_1, x_2], float)


def newton_cotes_iqf(l, u, f, moments):
    units = array([l, (l + u) / 2, u])

    f_matrix = array([units ** i for i in range(len(units))])

    coefs = solveSLAE(f_matrix, moments(l, u)[:3])

    return array([coefs[i] * f(units[i]) for i in range(len(units))]).sum()


def composite_qf(l, u, f, moments, L, method, exact, opth=1):
    intervals = opth
    S, H = [], []
    error = 1
    flag = False

    while error > 1e-6:
        h = (u - l) / intervals

        integral = 0.0

        for i in range(intervals):
            print(f"{i}: {l + i * h} -> {l + (i + 1) * h}")
            integral += method(l + i * h, l + (i + 1) * h, f, moments)

        S = append(S, array([integral]))
        H = append(H, array([h]))

        speed = 3
        t = len(H) - 1

        if len(H) > 2:
            speed = -log(abs((S[t] - S[t - 1]) /
                             (S[t - 1] - S[t - 2]))) / log(L)

            error = abs(S[t-2] - S[t-1]) / ((L ** 4) - 1)

            print(f"Speed: {speed}")

        intervals *= L

    return integral, S


def gauss_qf(l, u, f, moments):
    result = 0.0
    units = array([l, (l + u) / 2, u])
    mom6 = moments(l, u)
    n = 3

    A = array([
        mom6[i:i + n] for i in range(n)
    ], float)

    y = array(-mom6[3:6])
    temp = solveSLAE(A, y)

    roots = cardano2(temp)

    roots.sort()

    A = array([roots ** i for i in range(n)])

    coefs = solveSLAE(A, mom6[:3])

    return sum([coefs[i] * f(roots[i]) for i in range(n)])


##############################################################################

L = 2

iqf = newton_cotes_iqf(func7.a, func7.b, func7.f, func7.moments)

print(f"Default Newton Cotes IQF: {iqf}")
print(f"Exact value: {func7.exact}")
print(f"Absolute Error: {abs(func7.exact - iqf)}")
print(f"Methodological Error: 1.10395")

print(f"\n-----------------------------------------------\n")

print(f"Speed for CIQF")

cqf_int, S = composite_qf(func7.a, func7.b, func7.f,
                          func7.moments, 2, newton_cotes_iqf, func7.exact)

lenS = len(S)
print(f"Composite Newton Cotes QF: {cqf_int}")
print(f"Error: {(S[lenS-2] - S[lenS-3]) / (2 ** 4 - 1)}")
print(f"Absolute Error: {abs(cqf_int - func7.exact)}")

print(f"\n-----------------------------------------------\n")

step = (func7.b - func7.a) / 2
speed = -log(abs((S[lenS - 1] - S[lenS - 2]) /
                 (S[lenS - 2] - S[lenS - 3]))) / log(2)

hopt = 0.95 * ((func7.b - func7.a) / L ** 2) * \
    ((1e-6 / abs(S[2] - S[1] / (L ** speed - 1))) ** (1 / speed))

hopt = ceil((func7.b - func7.a) / hopt)

optciqf_int, optS = composite_qf(func7.a, func7.b, func7.f,
                                 func7.moments, 2, newton_cotes_iqf, func7.exact, hopt)

print(f"Composite IQF with optimal step: {optciqf_int}")
print(f"Error: {abs((optciqf_int - cqf_int) / (2 ** 4 - 1))}")
print(f"Absolute Error: {abs(optciqf_int - func7.exact)}")

print(f"\n-----------------------------------------------\n")

gcqf_int, gS = composite_qf(func7.a, func7.b, func7.f,
                            func7.moments, 2, gauss_qf, func7.exact)
lengS = len(gS)

print(f"Composite Gauss QF: {gcqf_int}")
print(f"Error: {(gS[lengS-2] - gS[lengS-3]) / (2 ** 4 - 1)}")
print(f"Absolute Error: {abs(gcqf_int - func7.exact)}")

print(f"\n-----------------------------------------------\n")
