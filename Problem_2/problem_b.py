
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
    
    x_records = []
    for i in range(len(x_point_list)):
        x0 = np.array(x_point_list[i])
        x_res, func_val, x_record, y_record = gradient_method_exact_line_search(
            main_func, grad, x_symbol, x0, tol, golden_section_method_para)
        x_records.append(x_record)
    title_info = "[Ziyu]_exactline_all_in_one"
    new_contourPlot(x_records, title_info)


    x_records = []
    for i in range(len(x_point_list)):
        x0 = np.array(x_point_list[i])
        x_res, func_val, x_record, y_record = gradient_method_backtracking(
            main_func, grad, x_symbol, x0, tol, backtracking_method_para)
        x_records.append(x_record)
    title_info = "[Ziyu]_backtracking_all_in_one"
    new_contourPlot(x_records, title_info)