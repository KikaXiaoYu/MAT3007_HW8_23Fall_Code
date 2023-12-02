
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

    # backtracking method parameters
    theta_BTM = 0.5
    gamma_BTM = 0.1
    backtracking_method_para = [theta_BTM, gamma_BTM]

    # more init point
    x_point_list = [[-3, -3], [3, -3], [-3, 3], [3, 3]]
    for i in range(len(x_point_list)):
        x0 = np.array(x_point_list[i])
        x_res, func_val, x_record, y_record = gradient_method_exact_line_search(
            main_func, grad, x_symbol, x0, tol, golden_section_method_para)
        x_res, func_val, x_record, y_record = gradient_method_backtracking(
            main_func, grad, x_symbol, x0, tol, backtracking_method_para)

'''


[Ziyu] : Gradient method with exact line search
[Ziyu] : maxit_alpha = 100, tol_alpha = 1e-06, a_alpha = 2
[Ziyu] : INIT_POINT -> [-3 -3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 7
[Ziyu] : The result of x is: [-9.53869984e-09  6.60611206e-08]
[Ziyu] : The result of f(x) is: 5.90974899413149E-15


[Ziyu] : Gradient method with backtracking
[Ziyu] : theta = 0.5, gamma = 0.1
[Ziyu] : INIT_POINT -> [-3 -3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 15
[Ziyu] : The result of x is: [2.05557806e-33 8.90919175e-07]
[Ziyu] : The result of f(x) is: 1.05831596954784E-12        


[Ziyu] : Gradient method with exact line search
[Ziyu] : maxit_alpha = 100, tol_alpha = 1e-06, a_alpha = 2
[Ziyu] : INIT_POINT -> [ 3 -3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 7
[Ziyu] : The result of x is: [2.58640235e-09 2.05793890e-08]
[Ziyu] : The result of f(x) is: 5.71371147284376E-16


[Ziyu] : Gradient method with backtracking
[Ziyu] : theta = 0.5, gamma = 0.1
[Ziyu] : INIT_POINT -> [ 3 -3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 16
[Ziyu] : The result of x is: [4.19127256e-37 4.43532186e-07]
[Ziyu] : The result of f(x) is: 2.62294399797590E-13


[Ziyu] : Gradient method with exact line search
[Ziyu] : maxit_alpha = 100, tol_alpha = 1e-06, a_alpha = 2
[Ziyu] : INIT_POINT -> [-3  3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 10
[Ziyu] : The result of x is: [-3.44151754e-09 -1.53544791e-10]
[Ziyu] : The result of f(x) is: 1.18754776083173E-17


[Ziyu] : Gradient method with backtracking
[Ziyu] : theta = 0.5, gamma = 0.1
[Ziyu] : INIT_POINT -> [-3  3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 16
[Ziyu] : The result of x is: [-2.92366987e-36  5.75128197e-07]
[Ziyu] : The result of f(x) is: 4.41029923984161E-13


[Ziyu] : Gradient method with exact line search
[Ziyu] : maxit_alpha = 100, tol_alpha = 1e-06, a_alpha = 2
[Ziyu] : INIT_POINT -> [3 3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 8
[Ziyu] : The result of x is: [ 4.57570346e-08 -1.68020952e-09]
[Ziyu] : The result of f(x) is: 2.09747042749425E-15


[Ziyu] : Gradient method with backtracking
[Ziyu] : theta = 0.5, gamma = 0.1
[Ziyu] : INIT_POINT -> [3 3]
[Ziyu] : TOLERANCE -> 1e-05
[Ziyu] : The iteration count is: 15
[Ziyu] : The result of x is: [-4.07758731e-32 -1.19141105e-06]
[Ziyu] : The result of f(x) is: 1.89261372220044E-12


'''