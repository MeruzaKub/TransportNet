from math import sqrt
import numpy as np
from numba import jit

@jit
def universal_gradient_descent_function(phi_big_oracle, phi_small_solver, primal_dual_oracle,
                                         t_start, L_init = 1.0, max_iter = 1000,
                                         epsilon = 1e-5, verbose = False):
    iter_step = 5
    L_value = L_init
    t_next = np.copy(t_start)
    t_prev = np.copy(t_start)
    
    flows_mean = np.zeros(len(t_start))
    t_mean = np.zeros(len(t_start))
    entropy_mean = 0.0
    
    duality_gap_init = 0.0
    epsilon_absolute = 0.0
    #epsilon_absolute = epsilon
    
    primal_func_history = []
    inner_iters_history = []
    
    if verbose:
        duality_gap_history = []
    
    for counter in range(max_iter):
        inner_iters_num = 1
        L_value /= 2
        grad_prev = phi_big_oracle.grad(t_prev)
        func_prev = phi_big_oracle.func(t_prev)
        entropy_prev = phi_big_oracle.entropy(t_prev)
        
        flows_mean = (counter * flows_mean - grad_prev) / (counter + 1)
        t_mean = (counter * t_mean + t_prev) / (counter + 1)
        entropy_mean = (counter * entropy_mean + entropy_prev) / (counter + 1)   
        
   
        primal_val = primal_dual_oracle.sigma_sum_func(flows_mean) - entropy_mean
        duality_gap = primal_val + \
                      primal_dual_oracle.dual_func_value(t_mean)
        primal_func_history.append(primal_val)
        if counter == 0:
            duality_gap_init = duality_gap
            epsilon_absolute = epsilon * duality_gap_init
                
        while True:
            phi_small_solver.update(L_value, grad_prev, t_prev)
            t_next = phi_small_solver.argmin_function(u_start = t_prev)
        
            left_value = phi_big_oracle.func(t_next) - \
                         (func_prev + 
                             np.dot(grad_prev, t_next - t_prev) + 
                                 0.5 * epsilon_absolute)
            right_value = 0.5 * L_value * np.sum(np.square(t_next - t_prev))
            if left_value <= right_value:
                t_prev = np.copy(t_next)
                break
            else:
                L_value *= 2
                print('iteration_num = ' + str(counter+ 1) + ': L_value = ' + str(L_value))
                inner_iters_num += 1

                
        
        if verbose:
            duality_gap_history.append(duality_gap)
            inner_iters_history.append(inner_iters_num)

        
        if duality_gap < epsilon_absolute:
            result = {'times': t_mean,
                      'flows': flows_mean,
                      'iter_num' :counter + 1,
                      'duality_gap_history': duality_gap_history,
                      'primal_func_history': primal_func_history,
                      'inner_iters_history': inner_iters_history,
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
            
    result = {'times': t_mean,
              'flows': flows_mean,
              'iter_num' :counter + 1,
              'duality_gap_history': duality_gap_history,
              'primal_func_history': primal_func_history,
              'inner_iters_history': inner_iters_history,
              'res_msg': 'iterations number exceeded'}
    if verbose:
        result['dual_gap_history'] = duality_gap_history
        print('Iterations number exceeded!')
    return result