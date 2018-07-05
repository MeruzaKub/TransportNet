from math import sqrt
import numpy as np
from numba import jit

@jit
def universal_similar_triangles_function(phi_big_oracle, phi_small_solver, primal_dual_oracle,
                                         t_start, L_init = 1.0, max_iter = 1000,
                                         epsilon = 1e-5, verbose = False):
    iter_step = 5
    L_value = L_init
    A_previous = 0.0
    A_current = 0.0
    u_parameter = np.copy(t_start)
    t_parameter = np.copy(t_start)
    y_parameter = np.copy(t_parameter)
    
    flows_weighted = np.zeros(len(t_start))
    entropy_weighted = 0.0
    
    duality_gap_init = 0.0
    epsilon_absolute = 0.0
    
    if verbose:
        duality_gap_history = []
    
    for counter in range(max_iter):
        
        alpha = 0.5 / L_value + sqrt(0.25 / L_value**2 + A_previous / L_value)
        A_current = A_previous + alpha

        y_parameter = (alpha * u_parameter + A_previous * t_parameter) / A_current
        phi_small_solver.update(alpha, phi_big_oracle.grad(y_parameter))

        u_parameter = phi_small_solver.argmin_function(u_start = u_parameter)
        t_parameter = (alpha * u_parameter + A_previous * t_parameter) / A_current
        
        flows_weighted = (A_previous * flows_weighted - 
                          alpha * phi_big_oracle.grad(y_parameter)) / A_current
        entropy_weighted = (A_previous * entropy_weighted + 
                            alpha * phi_big_oracle.entropy(y_parameter)) / A_current           
            
        left_value = (phi_big_oracle.func(y_parameter) + 
                      np.dot(phi_big_oracle.grad(y_parameter), t_parameter - y_parameter) + 
                      0.5 * alpha / A_current * epsilon_absolute) - phi_big_oracle.func(t_parameter)
        right_value = - 0.5 * L_value * np.sum(np.square(t_parameter - y_parameter))
        
        while (left_value < right_value):
            L_value *= 2
            right_value *= 2
                    
        A_previous = A_current
        L_value /= 2
        
        duality_gap = primal_dual_oracle.sigma_sum_func(flows_weighted) - \
                      entropy_weighted + \
                      primal_dual_oracle.dual_func_value(t_parameter)
                
        if counter == 0:
            duality_gap_init = duality_gap
            epsilon_absolute = epsilon * duality_gap_init
            
        if verbose:
            duality_gap_history.append(duality_gap)
        
        if duality_gap < epsilon_absolute:
            result = {'times': t_parameter,
                      'flows': flows_weighted,
                      'iter_num' :counter + 1,
                      'duality_gap': duality_gap,
                      'res_msg': 'success'}
            if verbose:
                result['dual_gap_history'] = duality_gap_history
                print('Success!  Iterations number: ' + str(counter + 1))
                print('Duality_gap / Duality_gap_init = ' + str(duality_gap / duality_gap_init))
                print('Phi big oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
                
            return result  

        
        if verbose and ((counter + 1) % iter_step == 0 or counter == 0):
            print('Iterations number: ' + str(counter + 1))
            print('Duality_gap / Duality_gap_init = ' + str(duality_gap / duality_gap_init))
            #print('Duality_gap = ' + str(duality_gap))
            
    result = {'times': t_parameter,
              'flows': flows_weighted,
              'iter_num' :counter + 1,
              'duality_gap': duality_gap,
              'res_msg': 'iterations number exceeded'}
    if verbose:
        result['dual_gap_history'] = duality_gap_history
        print('Iterations number exceeded!')
    return result
