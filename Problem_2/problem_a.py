
from algorithms import *

# Main
if __name__ == "__main__":

    x1, x2 = sp.symbols('x1 x2')

    # main method parameters
    main_func = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)
    x_symbol = np.array([x1, x2])
    x0 = np.array([1, 1])
    tol = 1e-5

    # golden section method parameters
    maxit_GSM = 100
    tol_GSM = 1e-6
    a_GSM = 2
    golden_section_method_para = [maxit_GSM, tol_GSM, a_GSM]

    x_res, func_val, x_record, y_record = gradient_method_exact_line_search(
        main_func, grad, x_symbol, x0, tol, golden_section_method_para)

    # backtracking method parameters
    theta_BTM = 0.5
    gamma_BTM = 0.1
    backtracking_method_para = [theta_BTM, gamma_BTM]

    x_res, func_val, x_record, y_record = gradient_method_backtracking(
        main_func, grad, x_symbol, x0, tol, backtracking_method_para)

'''

[Ziyu] : Gradient method with exact line search
[Ziyu] : maxit_alpha = 100, tol_alpha = 1e-06, a_alpha = 2
[Ziyu] : INIT_POINT -> [1 1]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 8
[Ziyu] : The result of x is: [-1.89668870e-08 -1.18514685e-09]
[Ziyu] : The result of f(x) is: 3.61615563043383E-16


[Ziyu] : Gradient method with backtracking
[Ziyu] : theta = 0.5, gamma = 0.1
[Ziyu] : INIT_POINT -> [1 1]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 14
[Ziyu] : The result of x is: [ 0.00000000e+00 -5.74956685e-07]
[Ziyu] : The result of f(x) is: 4.40766919277909E-13


'''