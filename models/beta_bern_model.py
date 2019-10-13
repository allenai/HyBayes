import logging
import pymc3 as pm

logger = logging.getLogger('root')


def add_beta_bern_model(hierarchical_model, a=1, b=1):
  '''
  A model for binary observations via a Bernoulli variable, and a Beta prior.
  :param hierarchical_model:
  :param a: the first parameter of the Beta prior
  :param b: the second parameter of the Beta prior
  '''
  with pm.Model() as hierarchical_model.pymc_model:
    theta = pm.Beta("theta", a, b, shape=hierarchical_model.n_groups)
    observations = []
    hierarchical_model.mu_parameter = "theta"

    def add_observations():
      with hierarchical_model.pymc_model:
        for i in range(hierarchical_model.n_groups):
          observations.append(pm.Bernoulli(f'y_{i}', theta[i], observed=hierarchical_model.y[i]))

    hierarchical_model.add_observations_function = add_observations

