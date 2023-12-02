from algorithms import *

# Main
if __name__ == "__main__":

    x1, x2 = sp.symbols('x1 x2')

    # main method parameters
    main_func = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)
    x_symbol = np.array([x1, x2])
    x0 = np.array([1, 1])
    tol = 1e-5

    # backtracking Armijo method parameters
    theta_BTM = 0.5
    gamma_BTM = 0.1
    Armijo_method_para = [theta_BTM, gamma_BTM]

    # backtracking Adagrad method parameters
    theta_BTM = 0.5
    gamma_BTM = 0.1
    eps_BTM = 10-6
    m_BTM = 25
    Adagrad_method_para = [theta_BTM, gamma_BTM, eps_BTM, m_BTM]

    # more init point
    x_point_list = [[-3, -3], [3, -3], [-3, 3], [3, 3]]
    for i in range(len(x_point_list)):
        x0 = np.array(x_point_list[i])
        x_res, func_val, x_record, y_record = gradient_method_backtracking_Armijo(
            main_func, grad, x_symbol, x0, tol, Armijo_method_para)
        x_res, func_val, x_record, y_record = gradient_method_backtracking_Adagrad(
            main_func, grad, x_symbol, x0, tol, Adagrad_method_para)
        