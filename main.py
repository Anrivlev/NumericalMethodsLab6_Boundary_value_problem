import numpy as np
import matplotlib.pyplot as plt

# y'' + p(x)y' + q(x)y = f(x), 0 < x < l
# alpha1 * y(0) + beta1 * y'(0) = gamma1
# alpha1 * y(l) + beta1 * y'(l) = gamma1


def least_squares(x, y):
    n = len(x)

    sumx = x.sum()
    sumy = y.sum()
    xy = x * y
    sumxy = xy.sum()
    xx = x * x
    sumxx = xx.sum()

    b = (n * sumxy - sumx*sumy) / (n * sumxx - sumx**2)
    a = (sumy - b * sumx) / n
    return a, b


def solve1(p, q, f, alpha, beta, gamma, x0, xend, h):
    xrange = np.linspace(x0, xend, int(((xend - x0) // h) + 1))

    a = np.zeros(len(xrange))
    b = np.zeros(len(xrange))
    c = np.zeros(len(xrange))
    p_array = np.vectorize(p)(xrange)
    q_array = np.vectorize(q)(xrange)
    f_array = np.vectorize(f)(xrange)

    b[0] = alpha[0] * (h ** 2) - beta[0] * h
    c[0] = beta[0] * h
    f_array[0] = gamma[0] * (h ** 2)

    a[1:-1] = 1 - ((p_array[1:-1] * h) / 2)
    b[1:-1] = -2 + q_array[1:-1] * (h ** 2)
    c[1:-1] = 1 + ((p_array[1:-1] * h) / 2)
    f_array[1:-1] = f_array[1:-1] * (h ** 2)

    a[-1] = - beta[1] * h
    b[-1] = alpha[1] * (h**2) + beta[1] * h
    f_array[-1] = gamma[1] * (h**2)

    return solve3diagonal(a, b, c, f_array)


def solve2(p, q, f, alpha, beta, gamma, x0, xend, h):
    xrange = np.linspace(x0, xend,  int(((xend - x0) // h) + 1))

    a = np.zeros(len(xrange))
    b = np.zeros(len(xrange))
    c = np.zeros(len(xrange))
    f_array = np.zeros(len(xrange))

    b[0] = q(x0) * h ** 2 - 2 + (2 * h * alpha[0]) / beta[0] - (alpha[0] * p(x0) * h ** 2) / beta[0]
    c[0] = 2
    f_array[0] = f(x0) * h ** 2 + (2 * h * gamma[0]) / beta[0] - (p(x0) * gamma[0] * h ** 2) / beta[0]

    a[1:-1] = 1 - ((p(xrange[1:-1]) * h) / 2)
    b[1:-1] = -2 + q(xrange[1:-1]) * (h ** 2)
    c[1:-1] = 1 + ((p(xrange[1:-1]) * h) / 2)
    f_array[1:-1] = f(xrange[1:-1]) * (h ** 2)

    a[-1] = 2
    b[-1] = q(xend) * h**2 - 2 - (2 * h * alpha[1]) / beta[1] - (alpha[1] * p(xend) * h**2) / beta[1]
    f_array[-1] = f(xend) * h**2 - (2 * h * gamma[1]) / beta[1] - (p(xend) * gamma[1] * h**2) / beta[1]

    return solve3diagonal(a, b, c, f_array)


def solve3diagonal(a, b, c, f):
    # assuming that len(a) == len(b) == len(c) == len(f) == n + 1
    A = np.zeros(len(f))
    B = np.zeros(len(f))
    A[0] = -c[0] / b[0]
    B[0] = f[0] / b[0]
    for i in range(1, len(f) - 1):
        A[i] = -c[i] / (b[i] + a[i] * A[i - 1])
        B[i] = (f[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1])
    A[-1] = 0
    B[-1] = (f[-1] - a[-1] * B[-2]) / (b[-1] + a[-1] * A[-2])

    y = np.zeros(len(f))
    y[-1] = B[-1]
    for i in range(len(f) - 2, -1, -1):
        y[i] = B[i] + A[i] * y[i + 1]
    return y


def solve3diagonal_thompson(a, b, c, f):
    # assuming that len(a) == len(b) == len(c) == len(f) == n + 1
    C = np.zeros(len(f))
    F = np.zeros(len(f))
    C[0] = c[0] / b[0]
    for i in range(1, len(f) - 1):
        C[i] = c[i] / (b[i] - a[i] * C[i - 1])
    F[0] = f[0] / b[0]
    for i in range(1, len(f)):
        F[i] = (f[i] - a[i] * F[i - 1]) / (b[i] - a[i] * C[i - 1])
    y = np.zeros(len(f))
    y[-1] = F[-1]
    for i in range(len(f) - 2, -1, -1):
        y[i] = F[i] - C[i] * y[i + 1]
    return y


def main1():
    x0 = 0
    xend = 1

    alpha = np.array([0, 6])
    beta = np.array([1, 1])
    gamma = np.array([0, 8.3761043995962756169530107919075058250981216852045948197363873469])

    # u'(0) = 0
    # 6u(1) + u'(1) = 8.3761

    p = np.cosh
    q = np.sinh
    f = lambda x: np.cosh(x) + x * np.sinh(x)
    solution = lambda x: np.exp(-np.sinh(x)) + x

    h = 0.01
    xrange = np.arange(x0, xend, h)

    # y = solve1(p, q, f, alpha, beta, gamma, x0, xend, h)
    y = solve2(p, q, f, alpha, beta, gamma, x0, xend, h)

    number_of_methods = 1

    plt.subplot(2, number_of_methods, 1)
    plt.title("")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, y, color='k', label='Численное значение')
    plt.plot(xrange, solution(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.show()


def main2():
    x0 = 0
    xend = 1

    alpha = np.array([0, 6])
    beta = np.array([1, 1])
    gamma = np.array([0, 8.3761043995962756169530107919075058250981216852045948197363873469])

    p = np.cosh
    q = np.sinh
    f = lambda x: np.cosh(x) + x * np.sinh(x)
    solution = lambda x: np.exp(-np.sinh(x)) + x

    hmin = 0.0001
    hmax = 0.005
    number_of_steps = 100
    hrange = np.linspace(hmin, hmax, number_of_steps)

    error = dict()
    error[solve1] = np.zeros(len(hrange))
    error[solve2] = np.zeros(len(hrange))

    for i, h in zip(range(len(hrange)), hrange):
        sol = np.vectorize(solution)(np.linspace(x0, xend, int(((xend - x0) // h) + 1)))
        for key in error:
            error[key][i] = np.max(np.abs(key(p, q, f, alpha, beta, gamma, x0, xend, h) - sol))

    hrange = np.log10(hrange)
    for key in error:
        error[key] = np.log10(error[key])

    plt.suptitle('Зависимость логарифма абсолютной погрешности от логарифма шага интегрирования')
    for key, i in zip(error, range(1, len(error) + 1)):
        plt.subplot(1, len(error), i)
        plt.title(key.__name__)
        plt.xlabel("log(h)")
        plt.ylabel("log(max(|Δu|))")
        plt.grid()
        plt.plot(hrange, error[key], color='k')

    for key in error:
        coeffs = least_squares(hrange, error[key])
        print(key.__name__, ": ", coeffs[0], " + ", coeffs[1], "x", sep="")

    plt.show()


if __name__ == '__main__':
    main2()
