import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pymc3 as pm
import copy
from matplotlib import gridspec

class HierarchicalModel:

  def __init__(self, y) -> None:
    super().__init__()
    self.nGroups = len(y)
    self.statsY = [stats.describe(yi) for yi in y]
    self.y = y  # nGroups list of numpy arrays or it s None
    self.pymcModel = None
    self.addObservations = None

  def __str__(self) -> str:
    return f"{self.onlyPrior}_{self.nGroups}_{super().__str__()}"

  def addBetaBernModel(self):
    with pm.Model() as self.pymcModel:
      theta = pm.Beta("theta", 1, 1, shape=len(y))
      likelihood = []
      if not self.onlyPrior:
        for i in range(self.nGroups):
          likelihood.append(pm.Bernoulli(f'y_{i}', theta[i], observed=y[i]))

  def addInvLogitNormalModel(self):
    with pm.Model() as self.pymcModel:
        mu = pm.Normal('mu', mu=0, sd=2)
        theta = pm.invlogit("p", mu)
        likelihood = []
        if not self.onlyPrior:
          for i in range(len(y)):
            likelihood.append(pm.Bernoulli(f'y_{i}', theta[i], observed=y[i]))

  def addInvLogitNormalModel(self):
    with pm.Model() as self.pymcModel:
        mu = pm.Normal('mu', mu=0, sd=2)
        theta = pm.invlogit("p", mu)
        observations = []

        def addObservations():
          for i in range(self.nGroups):
            observations.append(pm.Bernoulli(f'y_{i}', theta[i], observed=y[i]))
        self.addObservations = addObservations

  def addExpUniformNormalTModel(self):
    meanY = np.mean( [self.statsY[i].mean for i in range(self.nGroups)])
    sdY = np.mean( [self.statsY[i].variance for i in range(self.nGroups)])**(0.5)
    with pm.Model() as self.pymcModel:
      nu = pm.Exponential("nu", 1/30)  # mean = sd = 30
      sigma = pm.Uniform("sigma", sdY/100, sdY * 100, shape=self.nGroups)
      mu = pm.Normal("mu", meanY, (100*sdY), shape=self.nGroups)
      observations = []

      def addObservations():
        with self.pymcModel:
          for i in range(self.nGroups):
            observations.append(pm.StudentT(f'y_{i}', nu= nu, mu = mu[i],sd = sigma[i], observed=self.y[i]))

      self.addObservations = addObservations

  def getGraphViz(self, filePrefix, saveDot = True, savePng = True):
    graph = pm.model_to_graphviz(self.pymcModel)
    graph.format = "png"
    if saveDot:
      txtFileName = f"{filePrefix}_heirarchicalGraph.txt"
      graph.save(txtFileName)
      print(f"Graph's source saved to {txtFileName}")
    if savePng:
      pngFileName = f"{filePrefix}_heirarchicalGraph"
      graph.render(pngFileName, view=False, cleanup=True)
      print(f"Graph picture saved to {pngFileName}")
    return graph



class Experiment:

  def __init__(self, y, runPrior=False, runPost=True, postPredict=True,
               filePrefix="defaultFolder/noName") -> None:
    super().__init__()
    self.y = y
    self.runPrior = runPrior
    self.runPost = runPost
    self.postPredict = postPredict
    self.filePrefix = filePrefix

  def __str__(self) -> str:
    return super().__str__()

  def runModel(self, model , y = None,
               filePrefix = "experiment",
               draws=500, chains=None, cores=None, tune=500,
               progressbar=True):
    color = '#87ceeb' # TODO: this can become a parameter
    trace = pm.sample(model=model.pymcModel,
                      draws = draws, chaines = chains, cores= cores, tune=tune)
    print(f"Effective Sample Size (ESS) = {pm.diagnostics.effective_n(trace)}")
    # TODO: save the trace here
    # Plot autocor
    #pm.autocorrplot()
    pm.traceplot(trace)
    diagFileName = f"{filePrefix}_diagnostics.png"
    plt.savefig(diagFileName)
    print(f"{diagFileName} is saved!")
    plt.clf()

    if model.nGroups == 2:
      plt.figure(figsize=(10, 10))
      # Define gridspec
      gs = gridspec.GridSpec(3, 4)
      ax1 = plt.subplot(gs[:2, :4])
      ax2 = plt.subplot(gs[2, :2])
      ax3 = plt.subplot(gs[2, 2:])
      var = "mu"
      diff = trace[var][:, 0] - trace[var][:, 1]
      pm.plot_posterior(diff,
                        # varnames=var,
                        alpha_level=0.05,
                        rope=(-0.1, 0.1), #TODO: calculate ROPE
                        point_estimate='mode',
                        ax=ax1,
                        color=color,
                        round_to=3,
                        ref_val=0,
                        )

      pm.plot_posterior(trace[var][:, 0],
                        varnames=var,
                        alpha_level=0.05,
                        # rope=(0.49, 0.51),
                        point_estimate='mode',
                        ax=ax2,
                        color=color,
                        round_to=3,
                        ref_val=model.statsY[0].mean,
                        )
      pm.plot_posterior(trace[var][:, 1],
                        varnames=var,
                        alpha_level=0.05,
                        # rope=(0.49, 0.51),
                        point_estimate='mode',
                        ax=ax3,
                        color=color,
                        round_to=3,
                        ref_val=model.statsY[1].mean,
                        )
      # TODO: add Effect size comparison and sigma comparison
      distFileName =f"{filePrefix}.png"
      plt.savefig(distFileName)
      print(f"{distFileName} is saved!")
      plt.clf()
      return trace



  def run(self):
    y = self.y
    priorModel = HierarchicalModel(y=y)
    print(priorModel.nGroups, priorModel.statsY)
    priorModel.addExpUniformNormalTModel()
    if self.runPrior:
      priorModel.getGraphViz(self.filePrefix+"_prior", True, True)
      self.runModel(priorModel, filePrefix=self.filePrefix+"_prior", draws=2000, chains=4, cores=1, tune=1500)
    if self.runPost:
      postModel = copy.copy(priorModel)
      postModel.addObservations()
      priorModel.getGraphViz(self.filePrefix+"_posterior", True, True)
      # input("Enter to cont")
      trace = self.runModel(postModel, y, self.filePrefix + "_posterior", draws=4000, chains=4, cores=1 , tune=2000)
      if self.postPredict:
        self.drawPPC(trace, model=postModel)

  def drawPPC(self, trace, model):
    # print(model.pymcModel.observed_RVs)
    # print(len(model.pymcModel.observed_RVs))
    # print(model.pymcModel.observed_RVs[0])
    # print(type(model.pymcModel.observed_RVs[0]))
    # print(type(model.pymcModel.observed_RVs))
    # print(model.pymcModel.mu)
    ppc = pm.sample_posterior_predictive(trace, samples=500, model=model.pymcModel,
                                         vars=[model.pymcModel.mu,
                                                 model.pymcModel.nu,
                                                 model.pymcModel.sigma])

    _, ax = plt.subplots(figsize=(12, 6))
    # print(type(ppc['y_0']))
    # print(ppc['y_0'].shape)
    # ax.hist([y0.mean() for y0 in ppc['y_0']], bins=19, alpha=0.5)
    ax.hist(self.y[0], bins=19, alpha=0.5, histtype ='bar', color = "red", rwidth=0.3)
    # pm.densityplot(trace, varnames=["y_0",],ax=ax)
    # ax.axvline(data.mean())
    MLmu = np.mean(ppc["mu"][0])
    MLsd = np.mean(ppc["sigma"][0])
    MLnu = np.mean(ppc["nu"])
    print(MLmu)

    xp = np.linspace(MLmu - 4 * MLsd, MLmu + 4 * MLsd, 100)
    yp = MLsd*stats.t(nu=MLnu).pdf(xp)+MLmu
    ax.scatter(x=xp,
               y=yp)
    ax.scatter(x=self.y[0],
               y=np.zeros(self.y[0].shape), marker='x', color = "black")
    ax.set(title='Posterior predictive of the mean',
           xlabel='mean(x)',
           ylabel='Frequency');
    plt.savefig("ppc.png")
    plt.clf()




if __name__ == '__main__':
  print("Not this file")
  # exp1 = Experiment(runPrior=True, runPost=True, filePrefix="newOutput/normalTest")# old
  # exp1.run()
