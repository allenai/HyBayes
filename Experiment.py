import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc3 as pm
import theano.tensor as tt
import copy
import logging
import os
from matplotlib import gridspec
from theano.compile.ops import as_op
from dataVizualisation import differencePlots
import pandas as pd
import pickle #
logger = logging.getLogger('root')


class HierarchicalModel:
  def __init__(self, y) -> None:
    super().__init__()
    self.nGroups = len(y)
    self.statsY = [stats.describe(yi) for yi in y]
    self.y = y  # nGroups list of numpy arrays or it s None
    self.pymcModel = None
    self.addObservationsFunction = None
    self.MuParameter = None
    self.SigmaParameter = None
    self.OutliernessParameter = None
    self.Skewness = None
    self.trace = None

  def __str__(self) -> str:
    return f"{self.nGroups}_{super().__str__()}"

  def gerIntervalProb(self, a, b):
    muDiff = self.trace[self.MuParameter][:, 0] - self.trace[self.MuParameter][:, 1]
    numerator = np.logical_and(muDiff > a, muDiff < b).sum()
    denominator = muDiff.size
    return numerator, denominator

  def addOrdinalModel(self):
    # TODO finish up the ordinal
    meanY = np.mean([self.statsY[i].mean for i in range(self.nGroups)])
    sdY = np.mean([self.statsY[i].variance for i in range(self.nGroups)]) ** (0.5)
    logger.debug("sdY=" + str(sdY))
    logger.debug("meanY=" + str(meanY))

    @as_op(itypes=[tt.fvector, tt.fvector, tt.fvector], otypes=[tt.fmatrix])
    def outcome_probabilities(theta, mu, sigma): # TODO working here
      out = np.empty((nYlevels2, self.nGroups), dtype=np.float32)
      n = tt.norm(loc=mu, scale=sigma)
      out[0, :] = n.cdf(theta[0])
      out[1, :] = np.max([[0, 0], n.cdf(theta[1]) - n.cdf(theta[0])], axis=0)
      out[2, :] = np.max([[0, 0], n.cdf(theta[2]) - n.cdf(theta[1])], axis=0)
      out[3, :] = np.max([[0, 0], n.cdf(theta[3]) - n.cdf(theta[2])], axis=0)
      out[4, :] = 1 - n.cdf(theta[3])
      return out

    with pm.Model() as self.pymcModel:
      theta = pm.Normal('theta', mu=thresh2, tau=np.repeat(.5 ** 2, len(thresh2)),
                        shape=len(thresh2), observed=thresh_obs2)

    mu = pm.Normal('mu', mu=nYlevels2 / 2.0, tau=1.0 / (nYlevels2 ** 2), shape=n_grps)
    sigma = pm.Uniform('sigma', nYlevels2 / 1000.0, nYlevels2 * 10.0, shape=n_grps)

    pr = outcome_probabilities(theta, mu, sigma)

    y = pm.Categorical('y', pr[:, grp_idx].T, observed=df2.Y.cat.codes.as_matrix())

  def addCountModel(self):
    meanY = np.mean([self.statsY[i].mean for i in range(self.nGroups)])
    sdY = np.mean([self.statsY[i].variance for i in range(self.nGroups)]) ** (0.5)
    logger.debug("sdY="+str(sdY))
    with pm.Model() as self.pymcModel:
      logMu = pm.Normal("logMu", mu=np.log(meanY), sd = sdY, shape=self.nGroups)
      alpha = pm.Exponential("alpha", 1/30, shape=self.nGroups)

      mu = pm.Deterministic("mu", tt.exp(logMu))
      sigma = pm.Deterministic("sigma", tt.sqrt(mu+alpha * mu**2))
      skewness = pm.Deterministic("skewness", 2/tt.sqrt(alpha))# double check
      observations = []
      self.MuParameter = "mu"
      self.SigmaParameter = "sigma"
      self.Skewness = "skewness"

      def addObservations():
        with self.pymcModel:
          for i in range(self.nGroups):
            observations.append(pm.NegativeBinomial(f'y_{i}', mu=mu[i], alpha=alpha[i], observed=self.y[i]))

      self.addObservationsFunction = addObservations

  def addBetaBernModel(self, a=1, b=1):
    with pm.Model() as self.pymcModel:
      theta = pm.Beta("theta", a, b, shape=self.nGroups)
      observations = []
      self.MuParameter = "theta"

      def addObservations():
        with self.pymcModel:
          for i in range(self.nGroups):
            observations.append(pm.Bernoulli(f'y_{i}', theta[i], observed=self.y[i]))
      self.addObservationsFunction = addObservations

  def addBetaBinomialModel(self, a=1, b=1):
    with pm.Model() as self.pymcModel:
      theta = pm.Beta("theta", a, b, shape=self.nGroups)
      observations = []
      self.MuParameter = "theta"

      def addObservations():
        with self.pymcModel:
          for i in range(self.nGroups):
            observations.append(pm.Binomial(f'y_{i}', n=self.y[i][:, 0], p=theta[i], observed=self.y[i][:, 1]))
      self.addObservationsFunction = addObservations


  def addInvLogitNormalModel(self):
    with pm.Model() as self.pymcModel:
        mu = pm.Normal('mu', mu=0, sd=2)
        theta = pm.invlogit("p", mu)
        observations = []

        def addObservations():
          with self.pymcModel:
            for i in range(self.nGroups):
              observations.append(pm.Bernoulli(f'y_{i}', theta[i], observed=y[i]))
        self.addObservationsFunction = addObservations

  def addExpUniformNormalTModel(self):
    meanY = np.mean( [self.statsY[i].mean for i in range(self.nGroups)])
    sdY = np.mean( [self.statsY[i].variance for i in range(self.nGroups)])**(0.5)
    with pm.Model() as self.pymcModel:
      nu = pm.Exponential("nu", 1/30)  # mean = sd = 30
      sigma = pm.Uniform("sigma", sdY/100, sdY * 100, shape=self.nGroups)
      mu = pm.Normal("mu", meanY, (100*sdY), shape=self.nGroups)
      observations = []
      self.MuParameter = "mu"
      self.SigmaParameter = "sigma"
      self.OutliernessParameter = "nu"

      def addObservations():
        with self.pymcModel:
          for i in range(self.nGroups):
            observations.append(pm.StudentT(f'y_{i}', nu=nu, mu=mu[i], sd=sigma[i], observed=self.y[i]))

      self.addObservationsFunction = addObservations

  def getGraphViz(self, filePrefix: str, saveDot: bool = True, savePng: bool = True,
                  extension="png",
                  config=None):
    # logger.debug(saveDot)
    # logger.debug(savePng)
    graph = pm.model_to_graphviz(self.pymcModel)
    graph.format = extension
    # graph.format = "png"
    if saveDot:
      txtFileName = f"{filePrefix}_hierarchicalGraph.txt"
      graph.save(txtFileName)
      logger.info(f"Graph's source saved to {txtFileName}")
    if savePng:
      pngFileName = f"{filePrefix}_hierarchicalGraph"
      graph.render(pngFileName, view=False, cleanup=True)
      logger.info(f"Graph picture saved to {pngFileName}")
    return graph


class Experiment:

  def __init__(self, y, config) -> None:
    super().__init__()
    self.y = y
    self.runPrior = config["Prior"].getboolean("Analyze")
    self.runPost = config["Posterior"].getboolean("Analyze")
    self.filePrefix = config["Files"].get("OutputPrefix")
    self.configModel = config["Model"]
    self.configPrior = config["Prior"]
    self.configPost = config["Posterior"]
    self.configPlots = config["Plots"]
    self.rope = (self.configModel.getfloat("ROPE0"), self.configModel.getfloat("ROPE1"))
    self.extension = self.configPlots.get("Extension")


  def __str__(self) -> str:
    return super().__str__()

  def runModel(self, hierarchicalModel,
               filePrefix="experiment",
               draws=500, chains=None, cores=None, tune=500,
               progressbar=True,
               modelConfig=None,
               plotsConfig=None):
    hierarchicalModel.trace = pm.sample(model=hierarchicalModel.pymcModel,
                                        draws=draws, chaines=chains, cores=cores, tune=tune)
    logger.info(f"Effective Sample Size (ESS) = {pm.diagnostics.effective_n(hierarchicalModel.trace)}")
    if modelConfig.getboolean("SaveTrace"):
      traceFolderName = f"{filePrefix}_trace"
      if os.path.exists(traceFolderName):
        ind = 0
        while os.path.exists(f"{traceFolderName}_{ind}"):
          ind += 1
        traceFolderName = f"{traceFolderName}_{ind}"
      pm.save_trace(hierarchicalModel.trace, directory=traceFolderName)
      with open(os.path.join(traceFolderName, "pickeledTrace.pkl"), 'wb') as buff:
        pickle.dump({'model': hierarchicalModel.pymcModel, 'trace': hierarchicalModel.trace}, buff)
      logger.info(f"{traceFolderName} is saved!")
    # Plot autocor
    #pm.autocorrplot()
    if modelConfig.getboolean("DiagnosticPlots"):
      pm.traceplot(hierarchicalModel.trace)
      diagFileName = f"{filePrefix}_diagnostics.{self.extension}"
      plt.savefig(diagFileName)
      logger.info(f"{diagFileName} is saved!")
      plt.clf()

    if hierarchicalModel.nGroups == 2:
      differencePlots(hierarchicalModel=hierarchicalModel,
                      modelConfig=modelConfig,
                      filePrefix=filePrefix,
                      rope=self.rope,
                      config=self.configPlots)


  def addModel(self, modelObj):
    Error = False
    modelName = self.configModel.get("VariableType")
    if modelName == "Binary":
      if self.configModel.get("PriorModel") == "Beta":
        modelObj.addBetaBernModel()
      else:
        logger.error(f'The given prior model {self.configModel.get("PriorModel")} is not recognized')
    elif modelName == "Metric":
      if self.configModel.getboolean("UnitInterval"):
        modelObj.addInvLogitNormalModel()
      else:
        modelObj.addExpUniformNormalTModel()
    elif modelName == "Count":
      modelObj.addCountModel()
    elif modelName == "Ordinal":
      modelObj.addOrdinalModel()
    elif modelName == "Binomial":
      modelObj.addBetaBinomialModel()
    else:
      Error = False
    if Error:
      logger.error("The model in config file not found. Exiting the program!")
      exit(0)

  def run(self):
    y = self.y
    priorModel = HierarchicalModel(y=y)
    logger.info(f"nGroups: {priorModel.nGroups}")
    for x in priorModel.statsY:
      logger.info(x)
    self.addModel(priorModel)
    if self.runPrior:
      priorModel.getGraphViz(
        self.filePrefix+"_prior",
        self.configPrior.getboolean("SaveHierarchicalTXT"),
        self.configPrior.getboolean("SaveHierarchicalPNG"),
        extension=self.extension,
        config=self.configPlots,
      )
      self.runModel(
        priorModel,
        filePrefix=self.filePrefix+"_prior",
        draws=self.configPrior.getint("Draws"),
        chains=self.configPrior.getint("Chains"),
        cores=1,
        tune=self.configPrior.getint("Tune"),
        modelConfig=self.configPrior,
        plotsConfig=self.configPlots,
      )
      # logger.debug("Success!")
      # exit(0)

    if self.runPost:
      postModel = copy.copy(priorModel)
      postModel.addObservationsFunction()
      postModel.getGraphViz(
        self.filePrefix + "_posterior",
        self.configPost.getboolean("SaveHierarchicalTXT"),
        self.configPost.getboolean("SaveHierarchicalPNG"),
        extension=self.extension,
        config=self.configPlots,
      )
      self.runModel(
        postModel,
        filePrefix=self.filePrefix + "_posterior",
        draws=self.configPost.getint("Draws"),
        chains=self.configPost.getint("Chains"),
        cores=1,
        tune=self.configPost.getint("Tune"),
        modelConfig=self.configPost,
        plotsConfig=self.configPlots,
      )
      if self.runPrior and self.runPost and self.configModel.getboolean("BayesFactor"):
        BFDataFrame = self.bayesFactorAnalysis(priorModel, postModel, initRope= self.rope)
        BFfileName = self.filePrefix+"_BayesFactor.csv"
        BFDataFrame.to_csv(BFfileName)
        logger.info(f"Bayes Factor DataFrame is saved at {BFfileName}")
      # if self.postPredict: #TODO impose data
      #   self.drawPPC(trace, model=postModel)
  def bayesFactorAnalysis(self, priorModel, postModel, initRope = (-0.1, 0.1)):
    columnNames = ["ROPE", "priorProb", "postProb",
                   "BF", "BF_Savage_Dickey",
                   "prioNSample", "postNSample"]
    df = pd.DataFrame(columns=columnNames)
    rope = np.array(initRope)
    n = 100 if self.configModel.getboolean("TrySmallerROPEs") else 1
    for i in range(n):
      priorRopeProbFrac = priorModel.gerIntervalProb(rope[0], rope[1])
      postRopeProbFrac = postModel.gerIntervalProb(rope[0], rope[1])
      if priorRopeProbFrac[0] <= 0 or postRopeProbFrac[0] <= 0:
        break
      priorRopeProb = priorRopeProbFrac[0] / priorRopeProbFrac[1]
      postRopeProb = postRopeProbFrac[0] / postRopeProbFrac[1]
      # logger.debug(priorRopeProb)
      # logger.debug(postRopeProb)
      # bfsv = postRopeProbFrac[0] * priorRopeProbFrac[1] / (postRopeProbFrac[1] * priorRopeProbFrac[0])
      bfsv = postRopeProb/priorRopeProb
      bf = bfsv * (1-priorRopeProb)/(1-postRopeProb)
      row ={
        columnNames[0]: rope,
        columnNames[1]: priorRopeProb,
        columnNames[2]: postRopeProb,
        columnNames[3]: bfsv,
        columnNames[4]: bf,
        columnNames[5]: priorRopeProbFrac[0],
        columnNames[6]: postRopeProbFrac[0],
      }
      logger.debug(row)
      df = df.append(row, ignore_index=True)
      # logger.info(f"For ROPE={rope}:")
      # logger.info(f"  ROPE probibility in prior= {priorRopeProb}")
      # logger.info(f"  ROPE probibility in posteirour= {postRopeProb}")
      # logger.info(f"    Bayes Factor = {bf}")
      # logger.info(f"    Bayes Factor = {bf}")
      rope = rope/1.2
    logger.info(df["BF"])
    return df

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

    xp = np.linspace(MLmu - 4 * MLsd, MLmu + 4 * MLsd, 100)
    yp = MLsd*stats.t(nu=MLnu).pdf(xp)+MLmu
    ax.scatter(x=xp,
               y=yp)
    ax.scatter(x=self.y[0],
               y=np.zeros(self.y[0].shape), marker='x', color = "black")
    ax.set(title='Posterior predictive of the mean',
           xlabel='mean(x)',
           ylabel='Frequency')
    plt.savefig("ppc.png")
    plt.clf()




if __name__ == '__main__':
  print("Not this file")
  # exp1 = Experiment(runPrior=True, runPost=True, filePrefix="newOutput/normalTest")# old
  # exp1.run()
