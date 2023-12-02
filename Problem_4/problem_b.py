from algorithms import *

# Main
if __name__ == "__main__":

    x1, x2 = sp.symbols('x1 x2')

    # main method parameters
    main_func = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)
    x_symbol = np.array([x1, x2])
    x0 = np.array([3, 3])
    tol = 1e-5

    # parameters
    theta_BTM = 0.5
    gamma_BTM = 0.1
    gamma1 = 1e-6
    gamma2 = 0.1

    backtracking_para = [theta_BTM, gamma_BTM]
    GNT_method_para = [theta_BTM, gamma_BTM, gamma1, gamma2]

    x_res, func_val, x_record, y_record = newton_glob(
        main_func, grad, x_symbol, x0, tol, GNT_method_para)

    # more init point
    x_point_list = [[-3, -3], [3, -3], [-3, 3], [3, 3]]
    for i in range(len(x_point_list)):
        x0 = np.array(x_point_list[i])
        x_res, func_val, x_record, y_record = newton_glob(
            main_func, grad, x_symbol, x0, tol, GNT_method_para)

