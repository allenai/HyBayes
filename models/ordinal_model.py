import logging
import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import numpy as np
from scipy import stats

logger = logging.getLogger('root')


@as_op(itypes=[tt.dvector, tt.dvector, tt.dvector], otypes=[tt.dmatrix])
def outcome_probabilities(theta, mu, sigma):
  out = np.empty((theta.shape[0], mu.shape[0]), dtype=np.float)
  # out = np.empty((n_y_levels, hierarchical_model.n_groups), dtype=np.float)
  normal_dist = stats.norm(loc=mu, scale=sigma)
  out[0, :] = normal_dist.cdf(theta[0])
  for i in range(1, theta.shape[0] - 1):
    out[i, :] = np.max([[0, 0], normal_dist.cdf(theta[i]) - normal_dist.cdf(theta[i - 1])], axis=0)
  out[-1, :] = 1 - normal_dist.cdf(theta[-2])
  return out


def add_ordinal_model(hierarchical_model, min_level=0, max_level=2):
  """
  Adding a model that estimates decisions on "ordinal" data. In particular, "ordinal" data
  is a categorical where the variables have *ordered categories*, however the distances
  between the categories is not known. Each of the categories are represented by "levels"
  (the range of [min_level, max_level].)
  Credits of the implementation of this model in pymc3 belongs to
  http://nbviewer.jupyter.org/github/JWarmenhoven/DBDA-python/blob/master/Notebooks/Chapter%2023.ipynb
  For a discussion on this model and implementation on R refer to Chapter 23 in the book 
    'Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan', Second Edition, by John Kruschke (2015).
  """
  mean_y = np.mean([hierarchical_model.stats_y[i].mean for i in range(hierarchical_model.n_groups)])
  sd_y = np.mean([hierarchical_model.stats_y[i].variance for i in range(hierarchical_model.n_groups)]) ** (0.5)
  logger.debug(f"sd_y={sd_y}")
  logger.debug(f"mean_y={mean_y}")
  n_y_levels = max_level - min_level + 1

  thresh = np.arange(n_y_levels, dtype=np.float) + min_level + 0.5
  thresh_obs = np.ma.asarray(thresh)
  thresh_obs[1:-1] = np.ma.masked


  with pm.Model() as hierarchical_model.pymc_model:
    theta = pm.Normal('theta', mu=thresh, tau=np.repeat(.5 ** 2, len(thresh)),
                      shape=len(thresh), observed=thresh_obs)
    mu = pm.Normal('mu', mu=n_y_levels / 2.0, tau=1.0 / (n_y_levels ** 2), shape=hierarchical_model.n_groups)
    sigma = pm.Uniform('sigma', n_y_levels / 1000.0, n_y_levels * 10.0, shape=hierarchical_model.n_groups)
    logger.debug((mu.shape[0], n_y_levels, theta.shape[0]))
    levelProbs = pm.Deterministic("levelProbs", outcome_probabilities(theta, mu, sigma))

  observations = []
  hierarchical_model.mu_parameter = "mu"
  hierarchical_model.sigma_parameter = "sigma"

  def add_observations():
    with hierarchical_model.pymc_model:
      for i in range(hierarchical_model.n_groups):
        observations.append(pm.Categorical(f'y_{i}', levelProbs[:, i], observed=hierarchical_model.y[i]))

  hierarchical_model.add_observations_function = add_observations
