# Stable Dynamic & Beckmann models

The project contains implementations of several primal-dual subgradient methods for searching traffic equilibria in Stable Dynamic and Beckmann models. 
Results of experiments on [Anaheim transport network](https://github.com/bstabler/TransportationNetworks) are included.

The following methods are implemented:
1.	Subgradient method with adaptive step size [[arXiv:1604.08183](https://arxiv.org/ftp/arxiv/papers/1604/1604.08183.pdf)]
2.	Universal gradient method [[ref](http://www.optimization-online.org/DB_FILE/2013/04/3833.pdf)]
3.	Universal method of similar triangles [[arXiv:1701.02473](https://arxiv.org/ftp/arxiv/papers/1701/1701.02473.pdf)].

More information about models can be found in [[Nesterov-de Palma](https://link.springer.com/article/10.1023/A:1025350419398)] and [[Beckmann](https://cowles.yale.edu/sites/default/files/files/pub/misc/specpub-beckmann-mcguire-winsten.pdf)].
[Anaheim_Experiments.ipynb](https://github.com/MeruzaKub/TransportNet/blob/master/Stable%20Dynamic%20%26%20Beckman/Anaheim_Experiments.ipynb) contains code of experiments on comparison of the above methods and Frank-Wolfe algorithm (only for Beckmann model).

Convergence process on 10 000 iterations for Stable Dynamic model:
![](methods_stable_dynamic.png)

Convergence process on 8000 iterations for Beckmann model (+ Frank-Wolfe algorithm):
![](methods_beckmann.png)
