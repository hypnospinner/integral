from numpy import array, vstack, multiply, zeros
from lu import solveSLAE
from math import log, ceil
import func1
import func7


def newton_cotes_step(l, u, moments, f):
    result = 0.0
    units = array([l, (l + u) / 2, u])
    mom3 = moments(l, u)[:3]

    A = array([1, 1, 1], float)
    A = vstack((A, units))
    A = vstack((A, multiply(units, units)))

    s = solveSLAE(A, mom3)

    for i, v in enumerate(s):
        result += v * f(units[i])

    return result


def newton_cotes_qf(l, u, moments, f, intervals):
    step = (u - l) / intervals
    result = 0.0

    for i in range(intervals):
        u = l + step

        result += newton_cotes_step(l, u, moments, f)

        l += step

    return result


def newton_cotes_comp_prec(l, u, moments, f, precision=1e-6):
    error = 1
    L = 2
    degree = 4
    intervals = 1
    while error > precision:
        results = zeros(3)

        for i in range(3):
            # print(f"Calculating for intervals: {intervals}")
            results[i] = newton_cotes_qf(l, u, moments, f, intervals)
            intervals *= L

        speed = - log((results[2] - results[1]) /
                      (results[1] - results[0])) / log(L)

        print(speed)

        error = abs((results[1] - results[0]) / ((L ** degree) - 1))
        integral = results[0] + (results[1] - results[0]) / \
            (1 - (1 / (L ** degree)))

    return integral, error, (u - l) / intervals, results[1]


def newton_cotes_opt_step(l, u, moments, f, err, step):
    L = 2
    degree = 4
    intervals = ceil((u - l) / (step * L * ((0.00001 / err) ** (1 / degree))))

    return newton_cotes_qf(l, u, moments, f, intervals)


intervals = 128

print("Function 7")

print(f"\n************************ Newton-Cotes default ***********************\n")

intergal = newton_cotes_step(func7.a, func7.b, func7.moments, func7.f)
print(f"Absolute Error: {abs(func7.exact - intergal)}")
print(f"Methodological Error: 1.10395")

print(f"\n******** Newton-Cotes Composite QF with specified precision *********\n")

print("Speed:")

integral, err, step, meter = newton_cotes_comp_prec(
    func7.a, func7.b, func7.moments, func7.f)

print()

print(f"Result: {integral}\nAbsolute Error: {abs(func7.exact - integral)}\nMethodological Error: {err}")

print(f"\n***************** Newton-Cotes QF with optimal step *****************\n")

integral = newton_cotes_opt_step(
    func7.a, func7.b, func7.moments, func7.f, err, step)

print(f"Result: {integral}")
print(f"Absolute Error: {abs(func7.exact - integral)}")
print(f"Methodological Error: {abs(meter - integral) / (2 ** 4 - 1)}")
