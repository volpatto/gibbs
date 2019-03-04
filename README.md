# gibbs: an open source python library for equilibrium calculation based in global minimization of Gibbs free energy

## What you will find here? 

Simply put, a library for calculating equilibrium and related properties, like phase diagrams, by means of 
formulating the equilibrium as a optimization problem instead of non-linear flash calculations systems.

The main ideas are based in this paper written by [Nichita et al](https://www.sciencedirect.com/science/article/pii/S0098135402001448). However, instead Tunneling method proposed there, here a derivative-free and stocasthic global optimization method named [Differential Evolution](https://link.springer.com/article/10.1023/A:1008202821328) (DE for short) is employed.

## So what are the advantages? 

There are two main advantages compared with the Tunneling method:

  * No need to compute derivatives such as Jacobians and Hessians, which can add computational cost;
  * No need to provide initial estimates for components molar fraction due to the stochastic characteristic of DE method, which is
  population-based.

Nonetheless, the price is paid by means of computational demand because several computations are performed for each solution candidate over the population.

## Contributions

Contributions are not allowed right now, but I plan to allow it soon! If you want to contribute, just wait a little more! The project needs to mature a bit!

## About me

My name is Diego Volpatto, I'm a Numerical Dev at [ESSS](https://www.esss.co/). Check it out some of our public repos clicking [here](https://github.com/ESSS). Also, I'm currently DSc. student in Computational Modeling at [brazilian National Laboratory for Scientific Computing](https://www.lncc.br), where I do research in Mixed Hybrid and Discontinuous Galerkin Finite Element Methods. One can contact me though the email: dtvolpatto@gmail.com