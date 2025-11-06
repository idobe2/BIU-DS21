p = 1.0
f = 1.0

def e(x, n):
    global p, f

    if (n == 0):
        return 1

    r = e(x, n - 1)

    p = p * x

    f = f * n

    return (r + p / f)

x = 4
n = 15
print(e(x, n))
