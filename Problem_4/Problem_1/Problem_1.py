
def golden_section_method(
        func, x_l: float, x_r: float, tol: float, maxit: int):
    phi = (3 - 5**0.5) / 2  # Golden ratio
    iter_count = 0

    while (abs(x_r - x_l) >= tol and iter_count < maxit):
        x_l_pr = phi * x_r + (1-phi) * x_l
        x_r_pr = (1-phi) * x_r + phi * x_l
        if (func(x_l_pr) < func(x_r_pr)):
            x_r = x_r_pr
        else:
            x_l = x_l_pr
        iter_count += 1

    x_res = (x_l + x_r) / 2
    func_val = func(x_res)

    print("The result of x is: {0}".format(x_res))
    print("The result of f(x) is: {0}".format(func_val))

    return ((x_res, func_val))


def f(x):
    a = (1/4) * (x**2 - 1)**2
    b = (1/2) * (x - 2)**2
    res = a + b
    return res


# Main
if __name__ == "__main__":
    x_left = 0
    x_right = 2
    tol = 1e-5
    maxit = 100
    x_res, func_val = golden_section_method(f, x_left, x_right, tol, maxit)

'''
The result of x is: 1.2599213332006465
The result of f(x) is: 0.36011842515788134
'''
