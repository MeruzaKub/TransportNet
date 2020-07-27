from math import sqrt
import numpy as np
from history import History

def weighted_dual_averages_method(oracle, prox, primal_dual_oracle,
                                  t_start, max_iter = 1000,
                                  eps = 1e-5, eps_abs = None, stop_crit = 'dual_gap_rel',
                                  verbose_step = 100, verbose = False, save_history = False):
    if stop_crit == 'dual_gap_rel':
        def crit():
            return duality_gap <= eps * duality_gap_init
    elif stop_crit == 'dual_gap':
        def crit():
            return duality_gap <= eps_abs
    elif stop_crit == 'max_iter':
        def crit():
            return it_counter == max_iter
    elif callable(stop_crit):
        crit = stop_crit
    else:
        raise ValueError("stop_crit should be callable or one of the following names: \
                         'dual_gap', 'dual_gap_rel', 'max iter'")
    
    A = 0.0
    t = np.copy(t_start)
    grad_sum = np.zeros(len(t_start))
    beta_seq = 1.0
    rho_wda = np.sqrt(2) * np.linalg.norm(t_start)

    flows_weighted = primal_dual_oracle.get_flows(t_start)
    t_weighted = np.copy(t_start)
    primal, dual, duality_gap_init, state_msg = primal_dual_oracle(flows_weighted, t_weighted)
    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap')
        history.update(0, primal, dual, duality_gap_init)
    if verbose:
        print(state_msg)
    
    success = False
    
    for it_counter in range(1, max_iter+1):
        grad_t = oracle.grad(t)
        flows = primal_dual_oracle.get_flows(t) #grad() is called here
        alpha = 1 / np.linalg.norm(grad_t)
        A += alpha
        grad_sum += alpha * grad_t
        
        beta_seq = 1 if it_counter == 1 else beta_seq + 1.0 / beta_seq
        beta = beta_seq / rho_wda
        t = prox(grad_sum / A, t_start, beta / A)

        t_weighted = (t_weighted * (A - alpha) + t * alpha) / A
        flows_weighted = (flows_weighted * (A - alpha) + flows * alpha) / A
        
        primal, dual, duality_gap, state_msg = primal_dual_oracle(flows_weighted, t_weighted)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap)
        if verbose and (it_counter % verbose_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print(state_msg, flush = True)
        if crit():
            success = True
            break
            
    result = {'times': t_weighted, 'flows': flows_weighted,
              'iter_num': it_counter,
              'res_msg': 'success' if success else 'iterations number exceeded'}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('\nResult: ' + result['res_msg'])
        print('Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result

