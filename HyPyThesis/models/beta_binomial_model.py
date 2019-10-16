import logging
import pymc3 as pm

logger = logging.getLogger('root')


def add_beta_binomial_model(hierarchical_model, a=1, b=1):
  '''
  A model for binomial observations (number of successes in a sequence of n independent experiments)
  via a Binomial variable, and a Beta prior.
  :param a:
  :param b:
  :return:
  '''
  with pm.Model() as hierarchical_model.pymc_model:
    theta = pm.Beta("theta", a, b, shape=hierarchical_model.n_groups)
    observations = []
    hierarchical_model.mu_parameter = "theta"

    def add_observations():
      with hierarchical_model.pymc_model:
        for i in range(hierarchical_model.n_groups):
          observations.append(
            pm.Binomial(f'y_{i}', n=hierarchical_model.y[i][:, 0], p=theta[i], observed=hierarchical_model.y[i][:, 1]))

    hierarchical_model.add_observations_function = add_observations
