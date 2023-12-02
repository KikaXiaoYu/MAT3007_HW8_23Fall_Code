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


def contourPlot(x_record, title_info):
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
    plt.savefig('Problem_3/P3_figures/' + title_info + '.jpg')
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

# Gradient method with backtracking Adagrad


def gradient_method_backtracking_Adagrad(
        main_func, grad, x_symbol, x0, tol, local_method_para):

    x_k = x0
    gradient = grad(main_func, x_symbol, x_k)
    theta = local_method_para[0]
    gamma = local_method_para[1]
    eps = local_method_para[2]
    m = local_method_para[3]

    iter_count = 0
    x_record = [x_k]
    y_record = [main_func.subs([(x_symbol[i], x_k[i])
                                for i in range(len(x_symbol))])]

    print()
    print("[Ziyu] : Gradient method with backtracking Adagrad")
    print("[Ziyu] : m = {0}".format(m))
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
        
        v_lst = []
        size = len(x_k)
        for i in range(size):
            start = max(0, iter_count-m)
            end = iter_count
            sums = 0
            for j in range(start, end+1, 1):
                sums += (grad(main_func, x_symbol, x_record[j])[i])**2
            sums += eps
            v_lst.append(np.sqrt(sums))
        
        D_k = np.diag(v_lst)
        
        d = -np.dot(np.linalg.inv(D_k), gradient)

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
    title_info = "[Ziyu]_Backtracking_Adagrad_with_init{0}_iter{1}_m{2}".format(str(x0), iter_count, m)

    contourPlot(x_record, title_info)
    return (x_res, func_val, x_record, y_record)


def new_contourPlot(x_records, title_info):

    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    x1, x2 = np.meshgrid(x, y)
    Z = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)

    level_lst = [0.1 * 2**i for i in range(20)]

    contour_plot = plt.contour(x1, x2, Z, level_lst, alpha=0.5)

    plt.clabel(contour_plot, fontsize=10, colors=('k', 'r'))

    colors = ['red', 'blue', 'green', 'purple']

    for i in range(len(x_records)):
        x_record = x_records[i]
        data = np.array(x_record)
        the_color = colors[i]
        size = len(data)

        for i in range(len(data)):
            if i+1 != len(data):
                plt.plot([data[i, 0], data[i+1, 0]],
                         [data[i, 1], data[i+1, 1]], c=the_color, linewidth=1)
        plt.scatter(data[1:size-2, 0], data[1:size-2, 1], c=the_color, s=10)
        plt.scatter(data[0, 0], data[0, 1], c=the_color, s=50)
        plt.scatter(data[size-1, 0], data[size-1, 1], c=the_color, s=50)

    plt.title(title_info)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('Problem_3/P3_figures/' + title_info + '.jpg')
    plt.show()
