import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt
from scipy import stats
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from scipy.special import expit
from matplotlib import gridspec
plt.style.use('seaborn-white')

color = '#87ceeb'

parser = argparse.ArgumentParser(description='Bayesian Analysis')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument('--describe', action='store_true')
parser.add_argument('--onlyPrior', action='store_true')

args = parser.parse_args([
  "--describe",
  # "--onlyPrior"
])

if __name__ == '__main__':

  y = []
  y.append(np.genfromtxt("./data/heads3of4.csv"))
  y.append(np.genfromtxt("./data/heads2of4.csv"))

  # y[0] = np.array([])
  # y[1] = np.array([])
  # y.append(np.genfromtxt("./data/heads30of40.csv"))
  # y.append(np.genfromtxt("./data/heads20of40.csv"))

  desc = []
  for i in range(len(y)):
    desc.append(stats.describe(y[i]))

  if args.describe:
    for i in range(len(y)):
      print(f"Alg_{i}:", desc[i])

  with pm.Model() as hierarchical_model:
    if False:
      mu = pm.Normal('mu', mu=0, sd=2)
      theta = pm.invlogit("p", mu)
    else:
      theta = pm.Beta("theta", 1, 1, shape=len(y))
      likelihood = []
      if not args.onlyPrior:
        for i in range(len(y)):
          likelihood.append(pm.Bernoulli(f'y_{i}', theta[i], observed=y[i]))

  graph = pm.model_to_graphviz(hierarchical_model)
  graph.format = "png"
  graph.view("output/test")

  with hierarchical_model:
    trace1 = pm.sample(3000, cores=4, tune=500)


  fig, axes = plt.subplots(1, 2)
  traceAxes = axes.reshape(1, 2)
  pm.traceplot(trace1, ax=traceAxes)
  plt.savefig("output/diag.png")
  plt.clf()

  plt.figure(figsize=(10, 10))
  # Define gridspec
  gs = gridspec.GridSpec(3, 4)
  ax1 = plt.subplot(gs[:2, :4])
  ax2 = plt.subplot(gs[2, :2])
  ax3 = plt.subplot(gs[2, 2:])
  var = "theta"

  # exit(0)
  diff = trace1[var][:, 0]-trace1[var][:, 1]
  pm.plot_posterior(diff,
                    # varnames=var,
                    alpha_level=0.05,
                    rope=(-0.1, 0.1),
                    point_estimate='mode',
                    ax=ax1,
                    color=color,
                    round_to=3,
                    ref_val=0,
                    )

  pm.plot_posterior(trace1[var][:,0],
                    varnames=var,
                    alpha_level=0.05,
                    rope=(0.49, 0.51),
                    point_estimate='mode',
                    ax=ax2,
                    color=color,
                    round_to=3,
                    ref_val=desc[0].mean,
                    )
  pm.plot_posterior(trace1[var][:,1],
                    varnames=var,
                    alpha_level=0.05,
                    rope=(0.49, 0.51),
                    point_estimate='mode',
                    ax=ax3,
                    color=color,
                    round_to=3,
                    ref_val=desc[1].mean,
                    )

  plt.savefig("output/post.png")
  plt.clf()