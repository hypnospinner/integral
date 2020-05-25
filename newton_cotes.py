from numpy import array, vstack, multiply
from lu import solveSLAE
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

intervals = 128

print(f"Intervals - 1: {newton_cotes_step(func1.a, func1.b, func1.moments, func1.f)}")
print(f"Intervals - {intervals}: {newton_cotes_qf(func1.a, func1.b, func1.moments, func1.f, intervals)}")
print(f"Exact: {func1.exact}")

print(f"Intervals - 1: {newton_cotes_step(func7.a, func7.b, func7.moments, func7.f)}")
print(f"Intervals - {intervals}: {newton_cotes_qf(func7.a, func7.b, func7.moments, func7.f, intervals)}")
print(f"Exact: {func7.exact}")
