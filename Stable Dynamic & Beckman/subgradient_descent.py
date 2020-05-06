from math import sqrt
import numpy as np

def subgradient_descent_function(phi_big_oracle, prox_h, primal_dual_oracle,
                                        t_start, max_iter = 10000,
                                        eps = 1e-5, eps_abs = None, 
                                        verbose = False, save_history = False):    
    iter_step = 100
    
    A = 0.0
    t = np.copy(t_start)
    t_weighted = np.zeros(len(t_start))
    grad_sum = np.zeros(len(t_start))

    flows_weighted = - phi_big_oracle.grad(t_start)
    duality_gap_init = primal_dual_oracle.duality_gap(t_start, flows_weighted)
    primal_func_value = primal_dual_oracle.primal_func_value(flows_weighted)
    dual_func_value = primal_dual_oracle.dual_func_value(t_start)
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
        phi_grad_t = phi_big_oracle.grad(t)
        alpha = eps_abs / np.linalg.norm(phi_grad_t)**2
        t = prox_h(t - alpha * phi_grad_t, alpha)

        t_weighted = (A * t_weighted + alpha * t) / (A + alpha)
        A += alpha
        grad_sum += alpha * phi_grad_t
        flows_weighted = - grad_sum / A
        
        primal_func_value = primal_dual_oracle.primal_func_value(flows_weighted)
        dual_func_value = primal_dual_oracle.dual_func_value(t_weighted)
        duality_gap = primal_dual_oracle.duality_gap(t, flows_weighted)
        if save_history:
            history.update(it_counter, primal_func_value, dual_func_value, duality_gap)
        if duality_gap < eps_abs:
            success = True
            break
        if verbose and (it_counter % iter_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print('Primal_func_value = {:g}'.format(primal_func_value))
            print('Dual_func_value = {:g}'.format(dual_func_value))
            print('Duality_gap = {:g}'.format(duality_gap))
            print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init), flush=True)
            
    result = {'times': t_weighted,
              'flows': flows_weighted,
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
