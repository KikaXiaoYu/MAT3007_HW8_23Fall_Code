from algorithms import *

# Main
if __name__ == "__main__":

    x1, x2 = sp.symbols('x1 x2')

    # main method parameters
    main_func = 2*x1**4 + (2/3)*x1**3 + x1**2 - 2*(x1**2)*x2 + (4/3)*(x2**2)
    x_symbol = np.array([x1, x2])
    x0 = np.array([1, 1])
    tol = 1e-5

    # backtracking Adagrad method parameters
    theta_BTM = 0.5
    gamma_BTM = 0.1
    eps_BTM = 10-6
    m_BTM = 25
    Adagrad_method_para = [theta_BTM, gamma_BTM, eps_BTM, m_BTM]
        
    x0 = [-3, -3]
    m_lst = [5, 10, 15, 25]
    for i in range(len(m_lst)):
        m_BTM = m_lst[i]
        Adagrad_method_para = [theta_BTM, gamma_BTM, eps_BTM, m_BTM]
        x_res, func_val, x_record, y_record = gradient_method_backtracking_Adagrad(
                main_func, grad, x_symbol, x0, tol, Adagrad_method_para)
