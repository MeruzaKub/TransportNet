from math import sqrt
import numpy as np
from history import History

def frank_wolfe_algorithm(phi_big_oracle, primal_dual_oracle,
                          t_start, L_init = 1.0, max_iter = 1000,
                          eps = 1e-5, eps_abs = None, verbose = False, save_history = False):
    """
    TODO: duality gap, 
    res_msg: not only 'Iterations exceeded!'
    """
    iter_step = 5
    times = primal_dual_oracle.freeflowtimes
    flows = - phi_big_oracle.grad(times)
    
    #eps_abs = 0.0
    eps_abs = eps
    print('L_init = ', L_init)
    
    if save_history:
        history = History('iter', 'primal_func')
    
    for it_counter in range(1, max_iter+1):
        times = primal_dual_oracle.times_function(flows)
        y_parameter = - phi_big_oracle.grad(times)
        
        dist_solution = np.dot(times, flows - y_parameter)
        
        primal_func_value = primal_dual_oracle.primal_func_value(flows)
        if save_history:
            history.update(it_counter, primal_func_value)
        
        if it_counter == 1:
            dist_solution_init = dist_solution
            #eps_absolute = eps * dist_solution_init
            
        '''
        if dist_solution < eps_absolute:
            result = {'times': times,
                      'flows': flows,
                      'iter_num':it_counter,
                      'res_msg': 'success',
                      'primal_func_history': primal_func_history}

            if verbose:
                print('Primal_func_value = ' + str(primal_func_value))
                print('Success!  Iterations number: ' + str(it_counter))
                print('Dist_solution / Dist_solution_init = ' + str(dist_solution / dist_solution_init))
                print('Phi big oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
                
            return result  
        '''

        if verbose and ((it_counter) % iter_step == 0 or it_counter == 1):
            print('Iterations number: ' + str(it_counter))
            print('Primal_func_value = {:g}'.format(primal_func_value))
            print('Dist_solution / Dist_solution_init = ' + str(dist_solution / dist_solution_init))
        
        gamma = 2.0 / (it_counter + 2)
        flows = (1.0 - gamma) * flows + gamma * y_parameter
            
    result = {'times': times,
              'flows': flows,
              'iter_num': it_counter,
              'res_msg': 'iterations number exceeded'}
    
    if save_history:
        result['history'] = history.dict

    if verbose:
        print('Iterations number exceeded!')
    return result
