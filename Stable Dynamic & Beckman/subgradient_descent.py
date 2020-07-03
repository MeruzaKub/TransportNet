import numpy as np
from history import History

def subgradient_descent_function(phi_big_oracle, prox_h, primal_dual_oracle,
                                        t_start, max_iter = 1000,
                                        eps = 1e-5, eps_abs = None, verbose_step = 100,
                                        verbose = False, save_history = False):    
    iter_step = verbose_step
    A = 0.0
    t = np.copy(t_start)
    t_weighted = np.zeros(len(t_start))
    grad_sum = np.zeros(len(t_start))

    flows_weighted = - phi_big_oracle.grad(t_start)
    primal, dual, duality_gap_init, state_msg = primal_dual_oracle(flows_weighted, t_start)
    if eps_abs is None:
        eps_abs = eps * duality_gap_init
    if verbose:
        print(state_msg)
    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap')
        history.update(0, primal, dual, duality_gap_init)
    
    success = False
    for it_counter in range(1, max_iter+1):
        phi_grad_t = phi_big_oracle.grad(t)
        alpha = eps_abs / np.linalg.norm(phi_grad_t)**2
        t = prox_h(t - alpha * phi_grad_t, alpha)

        t_weighted = (A * t_weighted + alpha * t) / (A + alpha)
        A += alpha
        grad_sum += alpha * phi_grad_t
        flows_weighted = - grad_sum / A
        
        primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows_weighted, t_weighted)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap)
        if verbose and (it_counter % iter_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print(state_msg, flush = True)
        if duality_gap < eps_abs:
            success = True
            break
            
    result = {'times': t_weighted, 'flows': flows_weighted,
              'iter_num': it_counter,
              'res_msg' : 'success' if success else 'iterations number exceeded'}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('Result: ' + result['res_msg'], 'Total iters: ' + str(it_counter))
        print(state_msg)
        print('Phi_big_oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
    return result
