import math


def f_x(x):
    return math.log(x, 2) + 5


def reverse(f):
    return lambda x: -f(x)


def f_x_first(x):
    return 1/(x*math.log(2))


def f_x_second(x):
    return -1/(x**2 * math.log(2))


def bisection_method(f, a, b, eps):
    count = 0
    while b - a > eps:
        count += 1
        m = f((a + b) / 2)
        if m == 0:
            return (a + b) / 2
        elif m * f(a) < 0:
            b = (a + b) / 2
        else:
            a = (a + b) / 2
    return (a + b) / 2, count


def chord_method(f, f__, x_prev, b, eps):
    count = 1
    x = x_prev - f(x_prev)/(f(b) - f(x_prev))*(b - x_prev)

    if f(b)*f__(b) < 0:
        f = reverse(f)
    if f(b) < 0:
        x_prev, b = b, x_prev
    while abs(x - x_prev) > eps:
        count += 1
        x_prev = x
        x = x_prev - f(x_prev) / (f(b) - f(x_prev)) * (b - x_prev)
    return x, count


def newton_method(f, f_, f__, a, b, eps):
    count = 1
    if f(a)*f__(a) > 0:
        x_prev = a
    else:
        x_prev = b
    x = x_prev - f(x_prev)/f_(x_prev)
    while abs(x - x_prev) > eps:
        count += 1
        x_prev = x
        x = x_prev - f(x_prev) / f_(x_prev)
    return x, count


def secant_method(f, f_, f__, a, b, eps):
    count = 1
    if f(a) > 0:
        x_prev = b
        x = a
    else:
        x_prev = a
        x = b

    x_new = x - f(x)*(x - x_prev) / (f(x) - f(x_prev))
    print(x_new)
    while abs(x_new - x) > eps:
        count += 1
        x_prev = x
        x = x_new

        x_new = max(x - f(x)*(x - x_prev)/(f(x) - f(x_prev)), eps)

    return x_new, count






def main():
    a = 1 / 64
    b = 16
    eps = 0.0001
    print('Метод половинного деления')
    x, k = bisection_method(f_x, a, b, eps)
    print('Значение x:', x)
    print('Значение f(x):', f_x(x))
    print('Число итераций:', k)
    print()
    print('Метод хорд')
    x, k = chord_method(f_x, f_x_second, a, b, eps)
    print('Значение x:', x)
    print('Значение f(x):', f_x(x))
    print('Число итераций:', k)
    print()
    print('Метод Ньютона')
    x, k = newton_method(f_x, f_x_first, f_x_second, a, b, eps)
    print('Значение x:', x)
    print('Значение f(x):', f_x(x))
    print('Число итераций:', k)
    print()
    print('Метод Ньютона')
    x, k = secant_method(f_x, f_x_first, f_x_second, a, b, eps)
    print('Значение x:', x)
    print('Значение f(x):', f_x(x))
    print('Число итераций:', k)
    print()


main()
