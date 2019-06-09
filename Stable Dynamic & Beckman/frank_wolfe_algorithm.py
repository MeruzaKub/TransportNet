from math import sqrt
import numpy as np
from numba import jit

@jit
def frank_wolfe_algorithm(phi_big_oracle, primal_dual_oracle,
                                         t_start, L_init = 1.0, max_iter = 1000,
                                         eps = 1e-5, eps_abs = None, verbose = False):
    iter_step = 5
    times = primal_dual_oracle.freeflowtimes
    flows = - phi_big_oracle.grad(times)
    
    #eps_abs = 0.0
    eps_abs = eps
    print('L_init = ', L_init)
    
    primal_func_history = []
    
    for counter in range(1, max_iter):
        times = primal_dual_oracle.times_function(flows)
        y_parameter = - phi_big_oracle.grad(times)
        
        dist_solution = np.dot(times, flows - y_parameter)
        
        primal_func_value = primal_dual_oracle.primal_func_value(flows)
        primal_func_history.append(primal_func_value)
        
        if counter == 1:
            dist_solution_init = dist_solution
            #eps_absolute = eps * dist_solution_init
            
        '''
        if dist_solution < eps_absolute:
            result = {'times': times,
                      'flows': flows,
                      'iter_num':counter,
                      'res_msg': 'success',
                      'primal_func_history': primal_func_history}

            if verbose:
                print('Primal_func_value = ' + str(primal_func_value))
                print('Success!  Iterations number: ' + str(counter))
                print('Dist_solution / Dist_solution_init = ' + str(dist_solution / dist_solution_init))
                print('Phi big oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
                
            return result  
        '''

        if verbose and ((counter) % iter_step == 0 or counter == 1):
            print('Iterations number: ' + str(counter))
            print('Primal_func_value = ' + str(primal_func_value))
            print('Dist_solution / Dist_solution_init = ' + str(dist_solution / dist_solution_init))
        
        gamma = 2.0 / (counter + 2)
        flows = (1.0 - gamma) * flows + gamma * y_parameter
            
    result = {'times': times,
              'flows': flows,
              'iter_num' :counter,
              'res_msg': 'iterations number exceeded',
              'primal_func_history': primal_func_history}

    if verbose:
        print('Iterations number exceeded!')
    return result
