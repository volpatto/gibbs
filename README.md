# gibbs

[![Build Status](https://travis-ci.com/volpatto/gibbs.svg?branch=master)](https://travis-ci.com/volpatto/gibbs)
[![Build status](https://ci.appveyor.com/api/projects/status/gkl9lve28byp60jr/branch/master?svg=true)](https://ci.appveyor.com/project/volpatto/gibbs/branch/master)
[![Build Status](https://dev.azure.com/volpatto/volpatto/_apis/build/status/gibbs?branchName=master)](https://dev.azure.com/volpatto/volpatto/_build/latest?definitionId=2&branchName=master)
[![codecov](https://codecov.io/gh/volpatto/gibbs/branch/master/graph/badge.svg)](https://codecov.io/gh/volpatto/gibbs)

**READ HERE FIRST: I don't have available time for further developments or improvements to `gibbs` lib at the moment. If you have interest in contributions, feel free to submit Pull Requests. Any question please feel free to text me, see the Contact section below. Sorry for the trouble!**

An open source python library for equilibrium calculation based on global minimization of Gibbs free energy.

## What you will find here? 

Simply put, a library for calculating equilibrium and related properties, like phase diagrams, by means of 
formulating the equilibrium as a optimization problem instead of non-linear flash calculations systems.

The main ideas are based in this paper written by [Nichita et al](https://www.sciencedirect.com/science/article/pii/S0098135402001448). However, instead Tunneling method proposed there, here a derivative-free and stochastic global optimization method named [Differential Evolution](https://link.springer.com/article/10.1023/A:1008202821328) (DE for short) is employed.

## So what are the advantages? 

There are two main advantages when compared with the Tunneling method:

  * No need to compute derivatives such as Jacobians and Hessians, which can add computational cost. Also, such quantities can be very hard to calculate exactly or even numerically;
  * No need to provide initial estimates for components molar fraction due to the stochastic characteristic of DE method, which is population-based.

Nonetheless, the price is paid by means of computational demand because several computations are performed for each solution candidate over the population.

## Current features

By now, `gibbs` can perform the following features:

* Classic cubic Equations of State:
  - Soave-Redlich-Kwong;
  - Peng-Robinson;
  - Peng-Robinson 78.
* Equation of States for both mixtures and pure components;
* Perform fugacities computations from the cubic EoS, as well as compressibility factor;
* Reduced TPD stability analysis with DE;
* Equilibrium calculations: phase component compositions and phase molar fractions.

## Contributions

Contributions are not allowed right now, but I plan to allow it soon! If you want to contribute, just wait a little more! The project needs to mature a bit!

* Golden rule: master is always passing. `gibbs` is under Test Driven Development strategy. New features must be implemented
in a proper branch, then it can be merged in master if it pass in all the proposed tests.

## About me

My name is Diego Volpatto, I'm a Numerical Dev at [ESSS](https://www.esss.co/). Check it out some of our public repos clicking [here](https://github.com/ESSS). Also, I'm currently DSc. student in Computational Modeling at [brazilian National Laboratory for Scientific Computing](https://www.lncc.br), where I do research in Mixed Hybrid and Discontinuous Galerkin Finite Element Methods. One can contact me through the email: dtvolpatto@gmail.com
