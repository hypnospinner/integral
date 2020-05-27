import func7
from math import log, sqrt, cos, sin, acos, ceil, acosh, cosh, sinh, asinh, pi
from numpy import array, multiply, vstack, zeros
from lu import solveSLAE
import func1


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


def gauss_step(l, u, moments, f):
    result = 0.0
    units = array([l, (l + u) / 2, u])
    mom6 = moments(l, u)

    A = array([
        mom6[:3],
        mom6[1:4],
        mom6[2:5],
    ], float)

    y = array(-mom6[3:6])
    solution = solveSLAE(A, y)

    p = cardano2(solution)

    p.sort()

    A = array([[1, 1, 1]], float)
    A = vstack((A, p))
    A = vstack((A, multiply(p, p)))

    solution = solveSLAE(A, mom6[:3])

    for i, v in enumerate(solution):
        result += v * f(p[i])

    return result


def gauss_qf(l, u, moments, f, intervals):
    step = (u - l) / intervals
    result = 0.0

    for i in range(intervals):
        u = l + step
        # print(f"({l}, {u})")
        result += gauss_step(l, u, moments, f)

        l += step

    return result


def gauss_comp_prec(l, u, moments, f, precision=1e-6):
    error = 1
    L = 2
    degree = 4
    intervals = 1
    while error > precision:
        results = zeros(3)

        for i in range(3):
            results[i] = gauss_qf(l, u, moments, f, intervals)
            intervals *= L

        o = (results[2] - results[1])
        z = (results[1] - results[0])

        speed = - log(abs((results[2] - results[1]) /
                          (results[1] - results[0]))) / log(L)
        error = abs((results[1] - results[0]) / ((L ** degree) - 1))
        integral = results[0] + (results[1] - results[0]) / \
            (1 - (1 / (L ** degree)))

    return integral, error, (u - l) / intervals, results[1]


def gauss_opt_step(l, u, moments, f, err, step):
    L = 2
    degree = 4
    intervals = ceil((u - l) / (step * L * ((0.00001 / err) ** (1 / degree))))

    return gauss_qf(l, u, moments, f, intervals)


intervals = 128


print("Function 7")

print(f"\n************************ Gauss default ***********************\n")

r1 = gauss_step(func7.a, func7.b, func7.moments, func7.f)
r128 = gauss_qf(func7.a, func7.b, func7.moments, func7.f, intervals)
print(f"Intervals - 1: {r1}")
print(f"Absolute Error: {abs(func7.exact - r1)}")
print(f"Intervals - {intervals}: {r128}")
print(f"Absolute Error: {abs(func7.exact - r128)}")

print(f"\n******** Gauss Composite QF with specified precision *********\n")

integral, err, step, meter = gauss_comp_prec(func7.a, func7.b, func7.moments, func7.f)

print(f"Result: {integral}\nError: {err}\nStep: {step}")

print(f"\n***************** Gauss QF with optimal step *****************\n")

integral = gauss_opt_step(func7.a, func7.b, func7.moments, func7.f, err, step)

print(f"Result: {integral}")
print(f"Absolute Error: {abs(func7.exact - integral)}")
print(f"Methodological Error: {abs(integral - meter) / (2 ** 4 - 1)}")
