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

    result = [[float(i.subs(point)) for i in j] for j in hessian]

    return np.array(result)


def contourPlot(x_record, title_info):
    # 准备数据
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    x1, x2 = np.meshgrid(x, y)
    Z = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)
    # Z = 100 * (x2 - x1**2)**2 + (1-x1)**2

    level_lst = [0.1 * 2**i for i in range(20)]
    # level_lst = [0.1 * 2**i for i in range(10)]

    contour_plot = plt.contour(x1, x2, Z, level_lst, alpha=0.5)

    plt.clabel(contour_plot, fontsize=10, colors=('k', 'r'))

    data = np.array(x_record)

    size = len(data)
    for i in range(len(data)):
        if i+1 != len(data):
            plt.plot([data[i, 0], data[i+1, 0]],
                     [data[i, 1], data[i+1, 1]], c='black', linewidth=1)
    plt.scatter(data[1:size-2, 0], data[1:size-2, 1], c='blue', s=10)
    plt.scatter(data[0, 0], data[0, 1], c='red', s=50)
    plt.scatter(data[size-1, 0], data[size-1, 1], c='red', s=50)

    plt.title(title_info)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('P4_figures/' + title_info + '.jpg')
    plt.show()

# Gradient method with backtracking Armijo


def gradient_method_backtracking_Armijo(
        main_func, grad, x_symbol, x0, tol, local_method_para):

    x_k = x0
    gradient = grad(main_func, x_symbol, x_k)
    theta = local_method_para[0]
    gamma = local_method_para[1]
    iter_count = 0
    x_record = [x_k]
    y_record = [main_func.subs([(x_symbol[i], x_k[i])
                                for i in range(len(x_symbol))])]

    print()
    print("[Ziyu] : Gradient method with backtracking Armijo")
    # print("[Ziyu] : theta = {0}, gamma = {1}".format(
    # theta, gamma))
    print("[Ziyu] : INIT_POINT -> {0}".format(x0))
    print("[Ziyu] : TOLERANCE -> {0}".format(tol))

    # Backtracking method
    def backtrackingMethod(main_func, x_k, d_k, theta, gamma):
        alpha_k = 1
        a = main_func.subs([(x_symbol[i], (x_k + alpha_k * d_k)[i])
                            for i in range(len(x_symbol))])
        b = main_func.subs([(x_symbol[i], x_k[i])
                           for i in range(len(x_symbol))])
        c = gamma * np.dot(grad(main_func, x_symbol, x_k), d_k)

        while (a > b + c):
            alpha_k = theta * alpha_k
            a = main_func.subs([(x_symbol[i], (x_k + alpha_k * d_k)[i])
                                for i in range(len(x_symbol))])
            b = main_func.subs([(x_symbol[i], x_k[i])
                                for i in range(len(x_symbol))])
            c = gamma * alpha_k * np.dot(grad(main_func, x_symbol, x_k), d_k)
        return alpha_k
    # end of backtracking method

    while (np.linalg.norm(gradient) > tol):
        gradient = grad(main_func, x_symbol, x_k)
        hessian = hess(main_func, x_symbol, x_k)
        d = -gradient
        s_k = np.dot(np.linalg.inv(hessian), -gradient)

        print()
        print(- np.dot(gradient, s_k))
        print(gamma1 * min(1, np.linalg.norm(s_k)**gamma2)
              * np.linalg.norm(s_k)**2)

        alpha_k = backtrackingMethod(main_func, x_k, d, theta, gamma)
        # print("current alpha:", alpha_k)
        x_k = x_k + alpha_k * d
        iter_count += 1
        x_record.append(x_k)
        y_record.append(main_func.subs([(x_symbol[i], x_k[i])
                                        for i in range(len(x_symbol))]))

    x_res = x_k
    func_val = main_func.subs([(x_symbol[i], x_k[i])
                              for i in range(len(x_symbol))])

    print("[Ziyu] : The iteration count is: {0}".format(iter_count))
    print("[Ziyu] : The result of x is: {0}".format(x_res))
    print("[Ziyu] : The result of f(x) is: {0}".format(func_val))
    print()
    title_info = "[Ziyu]_Backtracking_Armijo_with_init{0}_iter{1}".format(
        str(x0), iter_count)

    contourPlot(x_record, title_info)
    return (x_res, func_val, x_record, y_record)

# Gradient method with backtracking Adagrad


# global newton's method
def newton_glob(
        main_func, grad, x_symbol, x0, tol, local_method_para):

    x_k = x0
    gradient = grad(main_func, x_symbol, x_k)
    theta = local_method_para[0]
    gamma = local_method_para[1]
    gamma1 = local_method_para[2]
    gamma2 = local_method_para[3]

    iter_count = 0
    x_record = [x_k]
    y_record = [main_func.subs([(x_symbol[i], x_k[i])
                                for i in range(len(x_symbol))])]

    print()
    print("[Ziyu] : Global Newton's Method")
    print("[Ziyu] : INIT_POINT -> {0}".format(x0))
    print("[Ziyu] : TOLERANCE -> {0}".format(tol))

    # Backtracking method
    def backtrackingMethod(main_func, x_k, d_k, theta, gamma):
        alpha_k = 1
        mor1 = x_k + np.dot(alpha_k, d_k)
        a = main_func.subs([(x_symbol[i], mor1[i])
                            for i in range(len(x_symbol))])
        b = main_func.subs([(x_symbol[i], x_k[i])
                           for i in range(len(x_symbol))])
        c = gamma * alpha_k * np.dot(grad(main_func, x_symbol, x_k), d_k)
        while (a > b + c):
            alpha_k = theta * alpha_k
            mor1 = x_k + np.dot(alpha_k, d_k)

            a = main_func.subs([(x_symbol[i], mor1[i])
                                for i in range(len(x_symbol))])
            b = main_func.subs([(x_symbol[i], x_k[i])
                                for i in range(len(x_symbol))])
            c = gamma * alpha_k * np.dot(grad(main_func, x_symbol, x_k), d_k)
        return alpha_k
    # end of backtracking method

    while (np.linalg.norm(gradient) > tol and iter_count < 1000):
        gradient = grad(main_func, x_symbol, x_k)
        hessian = hess(main_func, x_symbol, x_k)
        s_k = np.linalg.solve(hessian, -gradient)
        com = gamma1 * min(1, np.linalg.norm(s_k)**gamma2) *  np.linalg.norm(s_k)**2
        if (-np.dot(gradient, s_k) >= com):
            d_k = s_k
            # print("GA")
        else:
            d_k = -gradient

        alpha_k = backtrackingMethod(main_func, x_k, d_k, theta, gamma)
        x_k = x_k + alpha_k * d_k
        iter_count += 1
        x_record.append(x_k)
        y_record.append(main_func.subs([(x_symbol[i], x_k[i])
                                        for i in range(len(x_symbol))]))

    x_res = x_k
    func_val = main_func.subs([(x_symbol[i], x_k[i])
                              for i in range(len(x_symbol))])

    print("[Ziyu] : The iteration count is: {0}".format(iter_count))
    print("[Ziyu] : The result of x is: {0}".format(x_res))
    print("[Ziyu] : The result of f(x) is: {0}".format(func_val))
    print()
    title_info = "[Ziyu]_newton_glob_with_init{0}_iter{1}".format(
        str(x0), iter_count)

    contourPlot(x_record, title_info)
    return (x_res, func_val, x_record, y_record)


# Main
if __name__ == "__main__":

    x1, x2 = sp.symbols('x1 x2')

    # main method parameters
    main_func = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)
    # main_func = 100 * (x2 - x1**2)**2 + (1-x1)**2
    x_symbol = np.array([x1, x2])
    x0 = np.array([3, 3])
    tol = 1e-7

    # parameters
    theta_BTM = 0.5
    gamma_BTM = 1e-4
    gamma1 = 1e-6
    gamma2 = 0.1

    backtracking_para = [theta_BTM, gamma_BTM]
    GNT_method_para = [theta_BTM, gamma_BTM, gamma1, gamma2]

    x_res, func_val, x_record, y_record = gradient_method_backtracking_Armijo(
            main_func, grad, x_symbol, x0, tol, backtracking_para)

    x_res, func_val, x_record, y_record = newton_glob(
        main_func, grad, x_symbol, x0, tol, GNT_method_para)

    # more init point
    # x_point_list = [[-3, -3], [3, -3], [-3, 3], [3, 3]]
    # for i in range(len(x_point_list)):
    #     x0 = np.array(x_point_list[i])
    #     x_res, func_val, x_record, y_record = newton_glob(
    #         main_func, grad, x_symbol, x0, tol, GNT_method_para)

    # x0 = [-3, -3]
    # # m_lst = [5, 10, 15, 25]
    # for i in range(len(m_lst)):
    #     gamma_BTM = m_lst[i]
    #     GNT_method_para = [theta_BTM, gamma_BTM, gamma1, gamma2]
    #     x_res, func_val, x_record, y_record = newton_glob(
    #         main_func, grad, x_symbol, x0, tol, GNT_method_para)
