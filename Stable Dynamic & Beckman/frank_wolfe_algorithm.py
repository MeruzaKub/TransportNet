from math import sqrt
import numpy as np
from history import History

def frank_wolfe_algorithm(phi_big_oracle, primal_dual_oracle,
                          t_start, max_iter = 10000,
                          eps = 1e-5, eps_abs = None, verbose = False, save_history = False):
    iter_step = 100
    t = np.copy(t_start)
    flows = - phi_big_oracle.grad(t)
    
    duality_gap_init = primal_dual_oracle.duality_gap(t, flows)
    primal_func_value = primal_dual_oracle.primal_func_value(flows)
    dual_func_value = primal_dual_oracle.dual_func_value(t)
    if eps_abs is None:
        eps_abs = eps * duality_gap_init
    if verbose:
        print('Primal_init = {:g}'.format(primal_func_value))
        print('Dual_init = {:g}'.format(dual_func_value))
        print('Duality_gap_init = {:g}'.format(duality_gap_init))    
    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap')
        history.update(0, primal_func_value, dual_func_value, duality_gap_init)
    
    success = False
    for it_counter in range(1, max_iter+1):
        t = primal_dual_oracle.times_function(flows)
        y_parameter = - phi_big_oracle.grad(t)
        gamma = 2.0 / (it_counter + 2)
        flows = (1.0 - gamma) * flows + gamma * y_parameter
        
        primal_func_value = primal_dual_oracle.primal_func_value(flows)
        dual_func_value = primal_dual_oracle.dual_func_value(t)
        duality_gap = primal_dual_oracle.duality_gap(t, flows)
        if save_history:
            history.update(it_counter, primal_func_value, dual_func_value, duality_gap)
        #if duality_gap < eps_abs:
        #    success = True
        #    break
        if verbose and (it_counter % iter_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print('Primal_func_value = {:g}'.format(primal_func_value))
            print('Dual_func_value = {:g}'.format(dual_func_value))
            print('Duality_gap = {:g}'.format(duality_gap))
            print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init), flush=True)
     
    result = {'times': t,
              'flows': flows,
              'iter_num': it_counter,
              'res_msg' : 'success' if success else 'iterations number exceeded'}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print(result['res_msg'], 'total iters: ' + str(it_counter))
        print('Primal_func_value = {:g}'.format(primal_func_value))
        print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init))
        print('Phi_big_oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
    return result