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


# Gradient method with exact line search
def gradient_method_exact_line_search(
        main_func, grad_func, x_symbol, x0, tol, local_method_para):

    x_k = x0
    gradient = grad_func(main_func, x_symbol, x_k)
    maxit_alpha = local_method_para[0]
    tol_alpha = local_method_para[1]
    a_alpha = local_method_para[2]
    iter_count = 0

    x_record = [x_k]
    y_record = [main_func.subs([(x_symbol[i], x_k[i])
                                for i in range(len(x_symbol))])]

    print()
    print("[Ziyu] : Gradient method with exact line search")
    print("[Ziyu] : maxit_alpha = {0}, tol_alpha = {1}, a_alpha = {2}".format(
        maxit_alpha, tol_alpha, a_alpha))
    print("[Ziyu] : INIT_POINT -> {0}".format(x0))
    print("[Ziyu] : TOLERANCE -> {0}".format(tol))

    # Golden section method
    def goldenSectionMethod(func, x_l, x_r, tol, maxit):
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
        return ((x_res, func_val))
    # end of golden section method

    while (np.linalg.norm(gradient) > tol):
        gradient = grad_func(main_func, x_symbol, x_k)
        d = -gradient

        def func_alpha(alpha):
            x_new = x_k + alpha * d
            x1 = x_new[0]
            x2 = x_new[1]
            return 2*(x1**4) + (2/3)*(x1**3) + (x1**2) - 2*(x1**2)*x2 + (4/3)*(x2**2)

        alpha_k = goldenSectionMethod(
            func_alpha, 0, a_alpha, tol_alpha, maxit_alpha)[0]
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
    
    title_info = "[Ziyu]_ExactLine_with_init{0}_iter{1}".format(str(x0), iter_count)
    
    contourPlot(x_record, title_info)
    return (x_res, func_val, x_record, y_record)


# Gradient method with backtracking
def gradient_method_backtracking(
        main_func, grad_func, x_symbol, x0, tol, local_method_para):

    x_k = x0
    gradient = grad_func(main_func, x_symbol, x_k)
    theta = local_method_para[0]
    gamma = local_method_para[1]
    iter_count = 0
    x_record = [x_k]
    y_record = [main_func.subs([(x_symbol[i], x_k[i])
                                for i in range(len(x_symbol))])]

    print()
    print("[Ziyu] : Gradient method with backtracking")
    print("[Ziyu] : theta = {0}, gamma = {1}".format(
        theta, gamma))
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
        gradient = grad_func(main_func, x_symbol, x_k)
        d = -gradient

        alpha_k = backtrackingMethod(main_func, x_k, d, theta, gamma)
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
    title_info = "[Ziyu]_Backtracking_Armijo_with_init{0}_iter{1}".format(str(x0), iter_count)
    
    contourPlot(x_record, title_info)
    return (x_res, func_val, x_record, y_record)


def contourPlot(x_record, title_info):
    # 准备数据
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    x1, x2 = np.meshgrid(x, y)
    Z = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)

    level_lst = [0.1 * 2**i for i in range(20)]

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
    plt.savefig('P2_figures/' + title_info + '.jpg')
    plt.show()  


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

    # more init point
    x_point_list = [[-3, -3], [3, -3], [-3, 3], [3, 3]]
    for i in range(len(x_point_list)):
        x0 = np.array(x_point_list[i])
        x_res, func_val, x_record, y_record = gradient_method_exact_line_search(
            main_func, grad, x_symbol, x0, tol, golden_section_method_para)
        x_res, func_val, x_record, y_record = gradient_method_backtracking(
            main_func, grad, x_symbol, x0, tol, backtracking_method_para)
