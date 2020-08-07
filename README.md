# [Stable Dynamic & Beckmann models](https://github.com/MeruzaKub/TransportNet/tree/master/Stable%20Dynamic%20%26%20Beckman)

The project contains implementations of several primal-dual subgradient methods for searching traffic equilibria in the Stable Dynamic model and the Beckmann model. 
Results of experiments on the [Anaheim transportation network](https://github.com/bstabler/TransportationNetworks) are included.

The following methods are implemented:
1.	Universal gradient method [[ref](http://www.optimization-online.org/DB_FILE/2013/04/3833.pdf)]
2.	Universal method of similar triangles [[arXiv:1701.02473](https://arxiv.org/ftp/arxiv/papers/1701/1701.02473.pdf)]
3.  Method of Weighted Dual Averages [[ref](https://ium.mccme.ru/postscript/s12/GS-Nesterov%20Primal-dual.pdf)]
4.	Subgradient method with adaptive step size [[arXiv:1604.08183](https://arxiv.org/ftp/arxiv/papers/1604/1604.08183.pdf)].

Convergence rates of UMST, UGM, composite and non-composite WDA-methods for the Stable Dynamics model:

<img src="https://github.com/MeruzaKub/TransportNet/blob/master/Stable%20Dynamic%20%26%20Beckman/pics/sd_convergence_rel_eps.jpg" width="500">

Convergence rates of UMST, UGM, composite and non-composite WDA-methods, and the Frank–Wolfe method for the Beckmann model:

<img src="https://github.com/MeruzaKub/TransportNet/blob/master/Stable%20Dynamic%20%26%20Beckman/pics/beckmann_convergence_rel_eps.jpg" width="500">


## Installing graph-tool
Native installation of [graph-tool](https://graph-tool.skewed.de/) on Windows isn't supported. But if you have Docker installed, you can easily download the following container image with all the packages required to run the project:
https://hub.docker.com/r/ziggerzz/graph-tool-extra 

## How to Cite
1. Article: [arXiv:2008.02418](https://arxiv.org/abs/2008.02418)
2. The source code: Kubentayeva M. TransportNet. https://github.com/MeruzaKub/TransportNet. Accessed Month, Day, Year.

## More Resources
More information about the models can be found in [[Nesterov-de Palma](https://link.springer.com/article/10.1023/A:1025350419398)] and [[Beckmann](https://cowles.yale.edu/sites/default/files/files/pub/misc/specpub-beckmann-mcguire-winsten.pdf)].

# [Stochastic Nash-Wardrop Equilibria in the Beckmann model](https://github.com/MeruzaKub/TransportNet/tree/master/Stochastic%20Nash-Wardrop%20equilibrium)
Agents’ behavior is not completely rational, what is described by the introduction of Markov logit dynamics: any driver selects a route randomly according to the Gibbs’ distribution taking into account current time costs on the edges of the graph.
<img src="https://render.githubusercontent.com/render/math?math=\gamma > 0"> is a stochasticity parameter (when <img src="https://render.githubusercontent.com/render/math?math=\gamma \rightarrow 0"> the model boils down to the ordinary Beckmann model). The figure below shows convergence of flows in stochastic equilibrium to equilibrium flows in non-stochastic case as  <img src="https://render.githubusercontent.com/render/math?math=\gamma"> tends to zero.

<img src="https://github.com/MeruzaKub/TransportNet/blob/master/Stochastic%20Nash-Wardrop%20equilibrium/pics/anaheim_error_vs_gamma_eps_1e-3.png" width="500">

## How to Cite
1. [Article](http://crm.ics.org.ru/uploads/crmissues/crm_2018_3/2018_01_07.pdf): Gasnikov A.V., Kubentayeva M.B. Searching stochastic equilibria in transport networks by universal primal-dual gradient method // Computer Research and Modeling, 2018, vol. 10, no. 3, pp. 335-345. DOI: 10.20537/2076-7633-2018-10-3-335-345.
2. The source code: Kubentayeva M. TransportNet. https://github.com/MeruzaKub/TransportNet. Accessed Month, Day, Year.

<!--- Convergence process on 10 000 iterations for Stable Dynamic model:--->
<!---![](methods_stable_dynamic.png)--->

<!---Convergence process on 8000 iterations for Beckmann model (+ Frank-Wolfe algorithm):--->
<!---![](methods_beckmann.png)--->

<!--[Anaheim_Experiments.ipynb](https://github.com/MeruzaKub/TransportNet/blob/master/Stable%20Dynamic%20%26%20Beckman/Anaheim_Experiments.ipynb) contains code of experiments on comparison of the above methods and Frank-Wolfe algorithm (only for the Beckmann model).-->
