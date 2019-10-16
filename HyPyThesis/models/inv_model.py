import logging
import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import numpy as np
from scipy import stats

logger = logging.getLogger('root')

# TODO has to be tested with sample data, to make sure that it works properly.
def add_inv_logit_normal_model(hierarchical_model):
  raise NotImplementedError("work in progress . . . ")
  with pm.Model() as hierarchical_model.pymc_model:
    mu = pm.Normal('mu', mu=0, sd=2)
    theta = pm.invlogit("p", mu)
    observations = []

    def addObservations():
      with hierarchical_model.pymc_model:
        for i in range(hierarchical_model.n_groups):
          observations.append(pm.Bernoulli(f'y_{i}', theta[i], observed=y[i]))

    hierarchical_model.add_observations_function = addObservations

