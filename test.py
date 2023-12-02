import matplotlib.pyplot as plt
import numpy as np
import sympy as sp



def grad(main_func, x_symbol, x_point) -> np.array:
    '''
    func: the function to be calculated (a sympy f expression)
    x_array: a list of symbols (an np.array)
    x_point: a list of values (an np.array)
    the dim of x_array and x_point are the same, consistent with the func
    '''
    gradient = [sp.diff(main_func, var) for var in x_symbol]
    point = [(x_symbol[i], x_point[i]) for i in range(len(x_symbol))]
    result = [float(grad.subs(point)) for grad in gradient]
    return (np.array(result))


def hess(main_func, x_symbol, x_point) -> np.array:
    gradient = [sp.diff(main_func, var) for var in x_symbol]
    hessian = [[sp.diff(gradient[i], var)
                for i in range(len(x_symbol))]for var in x_symbol]
    point = [(x_symbol[i], x_point[i]) for i in range(len(x_symbol))]
    
    result = [ [float(i.subs(point)) for i in j] for j in hessian]

    return np.array(result)


x1, x2 = sp.symbols('x1 x2')

# main method parameters
main_func = x1**2 + x2**2
x_symbol = np.array([x1, x2])

x_point = np.array([1, 1])

print(hess(main_func, x_symbol, x_point))