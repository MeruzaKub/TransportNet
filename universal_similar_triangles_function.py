from math import sqrt
import numpy as np

def universal_similar_triangles_function(phi_big_oracle, phi_small_solver, dual_func_calculator,
                                         t_start, L_init = 1.0, max_iter = 1000,
                                         epsilon = 1e-5, verbose = False):
    L_value = L_init
    A_previous = 0.0
    A_current = None
    u_parameter = np.copy(t_start)
    t_parameter = np.zeros(len(t_start))
    
    dual_function_history = np.zeros(max_iter)
    dual_function_history[0] = dual_func_calculator.compute_value(t_parameter)
    epsilon_inner = epsilon
    
    for counter in range(0, max_iter):
        alpha = 0.5 / L_value + sqrt(0.25 / L_value**2 + A_previous / L_value)
        A_current = A_previous + alpha

        y_parameter = (alpha * u_parameter + A_previous * t_parameter) / A_current
        phi_small_solver.update(alpha, y_parameter)

        u_parameter = phi_small_solver.argmin_function(u_start = u_parameter)
        t_parameter = (alpha * u_parameter + A_previous * t_parameter) / A_current
        #print('_t_param_univers = ' + str(t_parameter))

        left_value = (phi_big_oracle.func(y_parameter) + 
                      np.dot(phi_big_oracle.grad(y_parameter), t_parameter - y_parameter) + 
                      0.5 * alpha / A_current * epsilon_inner) - phi_big_oracle.func(t_parameter)
        right_value = - 0.5 * L_value * np.sum(np.square(t_parameter - y_parameter))
        
        while (left_value < right_value):
            L_value = 2.0 * L_value
            right_value = 2.0 * right_value
            
        A_previous = A_current
        L_value = L_value / 2.0
        dual_function_history[counter] = dual_func_calculator.compute_value(t_parameter)
        
        if counter >= 2 and abs(dual_function_history[counter / 2] - dual_function_history[counter]) <= \
               0.5 * epsilon * abs(dual_function_history[0] - dual_function_history[counter / 2]):
            if verbose:
                print('Success!  Iterations number: ' + str(counter + 1))
            return t_parameter, counter, 'success'
            
        if verbose:
            if counter % 10 == 9:
                print('Iterations number: ' + str(counter + 1))
                #print('Epsilon inner = ' + str(epsilon_inner))
                if counter >= 2:
                    print('Criterion / epsilon = ' + \
                          str(abs(dual_function_history[counter] - dual_function_history[counter / 2]) / \
                              (0.5 * epsilon * abs(dual_function_history[0] - dual_function_history[counter / 2])))) 
                    
    return t_parameter, counter, 'iterations_exceeded'
