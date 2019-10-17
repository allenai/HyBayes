import logging
import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import numpy as np
from scipy import stats

logger = logging.getLogger('root')



# TODO has to be tested with sample data, to make sure that it works properly.
def addBetaContingencyModel(hierarchicalModel, a=1, b=1, c=1):
  assert hierarchicalModel.nGroups == 1
  #https: // onlinelibrary.wiley.com / doi / pdf / 10.1002 / sim.6875
  raise NotImplementedError("work in progress . . . ")
  with pm.Model() as hierarchicalModel.pymcModel:
    theta = pm.Dirichlet("theta", np.array((a, b, c)))
    # theta_0 -> 01
    # theta_1 -> 10
    # theta_2 -> 00 + 11



    observations = []
    hierarchicalModel.MuParameter = "theta"

    def addObservations():
      with hierarchicalModel.pymcModel:
        for i in range(hierarchicalModel.nGroups):
          y = hierarchicalModel.y[i]
          y_obs = np.zeros((y.shape[0], 3))
          y_obs[:, :2] = y[:, 1:3]
          y_obs[:, 2] = y[:, 0] + y[:, 3]
          n_obs = np.sum(y_obs, axis=1)
          observations.append(pm.Multinomial(f'y_{i}', n=n_obs, p=theta, observed=y_obs)) # todo theta for several groups

    hierarchicalModel.addObservationsFunction = addObservations


if __name__ == '__main__':
  print("starts here")