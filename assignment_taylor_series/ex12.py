from math import factorial
from math import pi, sin


def my_sin_order5(x):
    sign = -1
    p = 1  # starting power (x^1)
    i = 0  # term index
    sinx = 0.0  # accumulated sum

    # Calculate up to the x^5 term
    while p <= 5:
        term = (x**p) / float(factorial(p))
        sinx += (sign**i) * term
        i += 1
        p += 2  # move to the next odd power

    return sinx


# Testing the function
test_list = [0, pi / 2, pi, 1, 0.7]

print("inbuilt sin(x)\tapprox_sin_order5(x)")
for t in test_list:
    print("%.8f\t%.8f" % (sin(t), my_sin_order5(t)))
