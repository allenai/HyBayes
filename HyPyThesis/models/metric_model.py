import logging
import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import numpy as np
from scipy import stats

logger = logging.getLogger('root')


def add_exp_uniform_normal_t_model(hierarchical_model):
    """
    A student-t model with normal, uniform, exp priors for mu, sigma, nu parameters, respectively.
    Credits of the implementation of this model in pymc3 belongs to
      http://nbviewer.jupyter.org/github/JWarmenhoven/DBDA-python/blob/master/Notebooks/Chapter%2016.ipynb
      For a discussion on this model and implementation on R refer to Chapter 16 in the book
        'Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan', Second Edition, by John Kruschke (2015).
    """
    mean_y = np.mean([hierarchical_model.stats_y[i].mean for i in range(hierarchical_model.n_groups)])
    sd_y = np.mean([hierarchical_model.stats_y[i].variance for i in range(hierarchical_model.n_groups)]) ** (0.5)
    with pm.Model() as hierarchical_model.pymc_model:
        nu = pm.Exponential("nu", 1 / 30)  # mean = sd = 30
        sigma = pm.Uniform("sigma", sd_y / 100, sd_y * 100, shape=hierarchical_model.n_groups)
        mu = pm.Normal("mu", mean_y, (100 * sd_y), shape=hierarchical_model.n_groups)
        observations = []
        hierarchical_model.mu_parameter = "mu"
        hierarchical_model.sigma_parameter = "sigma"
        hierarchical_model.outlierness_parameter = "nu"

        def add_observations():
            with hierarchical_model.pymc_model:
                for i in range(hierarchical_model.n_groups):
                    observations.append(pm.StudentT(f'y_{i}', nu=nu, mu=mu[i], sd=sigma[i], observed=hierarchical_model.y[i]))

        hierarchical_model.add_observations_function = add_observations
