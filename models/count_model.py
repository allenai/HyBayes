import logging
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from scipy import stats

logger = logging.getLogger('root')


def add_count_model(hierarchical_model):
  '''
  Adding a model that estimates decisions on "count" data. In particular, "count" variables
  are observations that take only the non-negative integer values {0, 1, 2, 3, ...}, and they
  arise from counting rather than ranking.
  Credits of the implementation of this model in pymc3 belongs to
  http://nbviewer.jupyter.org/github/JWarmenhoven/DBDA-python/blob/master/Notebooks/Chapter%2024.ipynb
  For a discussion on this model and implementation on R refer to Chapter 24 in the book
    'Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan', Second Edition, by John Kruschke (2015).
  '''
  mean_y = np.mean([hierarchical_model.stats_y[i].mean for i in range(hierarchical_model.n_groups)])
  sd_y = np.mean([hierarchical_model.stats_y[i].variance for i in range(hierarchical_model.n_groups)]) ** (0.5)
  logger.debug(f"sd_y={sd_y}")
  with pm.Model() as hierarchical_model.pymc_model:
    log_mu = pm.Normal("logMu", mu=np.log(mean_y), sd=sd_y, shape=hierarchical_model.n_groups)
    alpha = pm.Exponential("alpha", 1 / 30, shape=hierarchical_model.n_groups)

    mu = pm.Deterministic("mu", tt.exp(log_mu))
    sigma = pm.Deterministic("sigma", tt.sqrt(mu + alpha * mu ** 2))
    skewness = pm.Deterministic("skewness", 2 / tt.sqrt(alpha))  # double check
    observations = []
    hierarchical_model.mu_parameter = "mu"
    hierarchical_model.sigma_parameter = "sigma"
    hierarchical_model.skewness = "skewness"

    def add_observations():
      with hierarchical_model.pymc_model:
        for i in range(hierarchical_model.n_groups):
          observations.append(pm.NegativeBinomial(f'y_{i}', mu=mu[i], alpha=alpha[i], observed=hierarchical_model.y[i]))

    hierarchical_model.add_observations_function = add_observations


