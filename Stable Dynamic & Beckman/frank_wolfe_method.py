from math import sqrt
import numpy as np
from history import History

def frank_wolfe_method(oracle, primal_dual_oracle,
                       t_start, max_iter = 1000,
                       eps = 1e-5, eps_abs = None, verbose_step = 100, verbose = False, save_history = False):
    iter_step = verbose_step
    t = None
    flows = - oracle.grad(t_start)
    t_weighted = np.copy(t_start)
    
    primal, dual, duality_gap_init, state_msg = primal_dual_oracle(flows, t_weighted) 
    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap')
        history.update(0, primal, dual, duality_gap_init)
    if verbose:
        print(state_msg) 
    if eps_abs is None:
        eps_abs = eps * duality_gap_init
    
    success = False
    for it_counter in range(1, max_iter+1):
        t = primal_dual_oracle.times_function(flows)
        y_parameter = primal_dual_oracle.get_flows(t) 
        gamma = 2.0 / (it_counter + 1)
        flows = (1.0 - gamma) * flows + gamma * y_parameter
        t_weighted = (1.0 - gamma) * t_weighted + gamma * t
        
        primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows, t_weighted)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap)
        if verbose and (it_counter % iter_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print(state_msg, flush = True)
        if duality_gap < eps_abs:
            success = True
            break
     
    result = {'times': t_weighted, 'flows': flows,
              'iter_num': it_counter,
              'res_msg' : 'success' if success else 'iterations number exceeded'}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('Result: ' + result['res_msg'], 'Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result