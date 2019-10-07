import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pymc3 as pm
import theano.tensor as tt

from dataVizualisation import difference_plots
from scipy import stats
from theano.compile.ops import as_op

logger = logging.getLogger('root')


class HierarchicalModel:
    """
    Keeps the configuration of different models.
    """

    def __init__(self, y) -> None:
        """
        :param y: the list of observations
        """
        super().__init__()
        self.n_groups = len(y)  # the number of experiments (often set to 2)
        self.stats_y = [stats.describe(yi) for yi in y]  # statistics describing the results
        self.y = y  # n_groups list of numpy arrays or it s None
        self.pymc_model = None  # the internal model to communicate with PyMC3
        self.add_observations_function = None
        self.mu_parameter = None
        self.sigma_parameter = None
        self.outlierness_parameter = None
        self.skewness = None
        self.trace = None

    def __str__(self) -> str:
        return f"{self.n_groups}_{super().__str__()}"

    def estimate_interval_prob(self, interval_begin, interval_end):
        """
        Estimating probability of an interval, used in calculation of BayesFactor  
        """
        mu_diff = self.trace[self.mu_parameter][:, 0] - self.trace[self.mu_parameter][:, 1]
        numerator = np.logical_and(mu_diff > interval_begin, mu_diff < interval_end).sum()
        denominator = mu_diff.size
        return numerator, denominator

    def add_ordinal_model(self, min_level=0, max_level=2):
        '''
        Adding a model that estimates decisions on "ordinal" data. In particular, "ordinal" data
        is a categorical where the variables have *ordered categories*, however the distances
        between the categories is not known. Each of the categories are represented by "levels"
        (the range of [min_level, max_level].)
        '''
        mean_y = np.mean([self.stats_y[i].mean for i in range(self.n_groups)])
        sd_y = np.mean([self.stats_y[i].variance for i in range(self.n_groups)]) ** (0.5)
        logger.debug(f"sd_y={sd_y}")
        logger.debug(f"mean_y={mean_y}")
        n_y_levels = max_level - min_level + 1

        thresh = np.arange(n_y_levels, dtype=np.float) + min_level + 0.5
        thresh_obs = np.ma.asarray(thresh)
        thresh_obs[1:-1] = np.ma.masked

        @as_op(itypes=[tt.dvector, tt.dvector, tt.dvector], otypes=[tt.dmatrix])
        def outcome_probabilities(theta, mu, sigma):
            out = np.empty((n_y_levels, self.n_groups), dtype=np.float)
            normal_dist = stats.norm(loc=mu, scale=sigma)
            out[0, :] = normal_dist.cdf(theta[0])
            for i in range(1, n_y_levels - 1):
                out[i, :] = np.max([[0, 0], normal_dist.cdf(theta[i]) - normal_dist.cdf(theta[i - 1])], axis=0)
            out[-1, :] = 1 - normal_dist.cdf(theta[-2])
            return out

        with pm.Model() as self.pymc_model:
            theta = pm.Normal('theta', mu=thresh, tau=np.repeat(.5 ** 2, len(thresh)),
                              shape=len(thresh), observed=thresh_obs)
            mu = pm.Normal('mu', mu=n_y_levels / 2.0, tau=1.0 / (n_y_levels ** 2), shape=self.n_groups)
            sigma = pm.Uniform('sigma', n_y_levels / 1000.0, n_y_levels * 10.0, shape=self.n_groups)

            levelProbs = pm.Deterministic("levelProbs", outcome_probabilities(theta, mu, sigma))

        observations = []
        self.mu_parameter = "mu"
        self.sigma_parameter = "sigma"

        def add_observations():
            with self.pymc_model:
                for i in range(self.n_groups):
                    observations.append(pm.Categorical(f'y_{i}', levelProbs[:, i], observed=self.y[i]))

        self.add_observations_function = add_observations

    def add_count_model(self):
        '''
        Adding a model that estimates decisions on "count" data. In particular, "count" variables
        are observations that take only the non-negative integer values {0, 1, 2, 3, ...}, and they
        arise from counting rather than ranking.
        '''
        mean_y = np.mean([self.stats_y[i].mean for i in range(self.n_groups)])
        sd_y = np.mean([self.stats_y[i].variance for i in range(self.n_groups)]) ** (0.5)
        logger.debug(f"sd_y={sd_y}")
        with pm.Model() as self.pymc_model:
            log_mu = pm.Normal("logMu", mu=np.log(mean_y), sd=sd_y, shape=self.n_groups)
            alpha = pm.Exponential("alpha", 1 / 30, shape=self.n_groups)

            mu = pm.Deterministic("mu", tt.exp(log_mu))
            sigma = pm.Deterministic("sigma", tt.sqrt(mu + alpha * mu ** 2))
            skewness = pm.Deterministic("skewness", 2 / tt.sqrt(alpha))  # double check
            observations = []
            self.mu_parameter = "mu"
            self.sigma_parameter = "sigma"
            self.skewness = "skewness"

            def add_observations():
                with self.pymc_model:
                    for i in range(self.n_groups):
                        observations.append(pm.NegativeBinomial(f'y_{i}', mu=mu[i], alpha=alpha[i], observed=self.y[i]))

            self.add_observations_function = add_observations

    def add_beta_bern_model(self, a=1, b=1):
        '''
        A model for binary observations via a Bernoulli variable, and a Beta prior.
        :param a: the first parameter of the Beta prior
        :param b: the second parameter of the Beta prior
        '''
        with pm.Model() as self.pymc_model:
            theta = pm.Beta("theta", a, b, shape=self.n_groups)
            observations = []
            self.mu_parameter = "theta"

            def add_observations():
                with self.pymc_model:
                    for i in range(self.n_groups):
                        observations.append(pm.Bernoulli(f'y_{i}', theta[i], observed=self.y[i]))

            self.add_observations_function = add_observations

    def add_beta_binomial_model(self, a=1, b=1):
        '''
        A model for binomial observations (number of successes in a sequence of n independent experiments)
        via a Binomial variable, and a Beta prior.
        :param a:
        :param b:
        :return:
        '''
        with pm.Model() as self.pymc_model:
            theta = pm.Beta("theta", a, b, shape=self.n_groups)
            observations = []
            self.mu_parameter = "theta"

            def add_observations():
                with self.pymc_model:
                    for i in range(self.n_groups):
                        observations.append(
                            pm.Binomial(f'y_{i}', n=self.y[i][:, 0], p=theta[i], observed=self.y[i][:, 1]))

            self.add_observations_function = add_observations

    # TODO has to be tested with sample data, to make sure that it works properly.
    def add_inv_logit_normal_model(self):
        raise NotImplementedError("work in progress . . . ")
        with pm.Model() as self.pymc_model:
            mu = pm.Normal('mu', mu=0, sd=2)
            theta = pm.invlogit("p", mu)
            observations = []

            def addObservations():
                with self.pymc_model:
                    for i in range(self.n_groups):
                        observations.append(pm.Bernoulli(f'y_{i}', theta[i], observed=y[i]))

            self.add_observations_function = addObservations

    def add_exp_uniform_normal_t_model(self):
        '''
        A student-t model with normal, uniform, exp priors for mu, sigma, nu parameters, respectively.
        '''
        mean_y = np.mean([self.stats_y[i].mean for i in range(self.n_groups)])
        sd_y = np.mean([self.stats_y[i].variance for i in range(self.n_groups)]) ** (0.5)
        with pm.Model() as self.pymc_model:
            nu = pm.Exponential("nu", 1 / 30)  # mean = sd = 30
            sigma = pm.Uniform("sigma", sd_y / 100, sd_y * 100, shape=self.n_groups)
            mu = pm.Normal("mu", mean_y, (100 * sd_y), shape=self.n_groups)
            observations = []
            self.mu_parameter = "mu"
            self.sigma_parameter = "sigma"
            self.outlierness_parameter = "nu"

            def add_observations():
                with self.pymc_model:
                    for i in range(self.n_groups):
                        observations.append(pm.StudentT(f'y_{i}', nu=nu, mu=mu[i], sd=sigma[i], observed=self.y[i]))

            self.add_observations_function = add_observations

    def get_GraphViz_object(self, filePrefix: str, saveDot: bool = True, savePng: bool = True, extension="png"):
        '''
        Returns the GraphViz object corresponding to the underlying hierarchical model.
        '''
        graph = pm.model_to_graphviz(self.pymc_model)
        graph.format = extension
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
        '''
        :param y: observations
        :param config: configuration of the experiments.
        '''
        super().__init__()
        self.y = y
        self.run_prior = config["Prior"].getboolean("Analyze")
        self.run_post = config["Posterior"].getboolean("Analyze")
        self.file_prefix = config["Files"].get("OutputPrefix")
        self.config_model = config["Model"]
        self.config_prior = config["Prior"]
        self.config_post = config["Posterior"]
        self.config_plots = config["Plots"]
        self.rope = (self.config_model.getfloat("ROPE0"), self.config_model.getfloat("ROPE1"))
        self.extension = self.config_plots.get("Extension")

    def __str__(self) -> str:
        return super().__str__()

    # TODO two inputs are not used
    def run_model(self, hierarchical_model,
                  file_prefix="experiment",
                  draws=500, chains=None, cores=None, tune=500,
                  progressbar=True,
                  model_config=None,
                  plots_config=None):
        with hierarchical_model.pymcModel:
            hierarchical_model.trace = pm.sample(model=hierarchical_model.pymcModel,
                                                 draws=draws, chains=chains, cores=cores, tune=tune)
        logger.info(f"Effective Sample Size (ESS) = {pm.diagnostics.effective_n(hierarchical_model.trace)}")
        if model_config.getboolean("SaveTrace"):
            traceFolderName = f"{file_prefix}_trace"
            if os.path.exists(traceFolderName):
                ind = 0
                while os.path.exists(f"{traceFolderName}_{ind}"):
                    ind += 1
                traceFolderName = f"{traceFolderName}_{ind}"
            pm.save_trace(hierarchical_model.trace, directory=traceFolderName)
            with open(os.path.join(traceFolderName, "pickeledTrace.pkl"), 'wb') as buff:
                pickle.dump({'model': hierarchical_model.pymcModel, 'trace': hierarchical_model.trace}, buff)
            logger.info(f"{traceFolderName} is saved!")
        # TODO drop these?
        # Plot autocor
        # pm.autocorrplot()
        if model_config.getboolean("DiagnosticPlots"):
            pm.traceplot(hierarchical_model.trace)
            diag_file_name = f"{file_prefix}_diagnostics.{self.extension}"
            plt.savefig(diag_file_name)
            logger.info(f"{diag_file_name} is saved!")
            plt.clf()

        if hierarchical_model.nGroups == 2:
            difference_plots(hierarchicalModel=hierarchical_model,
                             modelConfig=model_config,
                             filePrefix=file_prefix,
                             rope=self.rope,
                             config=self.config_plots)

    def add_model(self, model_object):
        '''
        Constructing the appropriate model based on the specifications in the config file.
        :param model_object: the default model
        '''
        error = False
        model_name = self.config_model.get("VariableType")
        if model_name == "Binary":
            if self.config_model.get("PriorModel") == "Beta":
                model_object.add_beta_bern_model()
            else:
                logger.error(f'The given prior model {self.config_model.get("PriorModel")} is not recognized')
        elif model_name == "Metric":
            if self.config_model.getboolean("UnitInterval"):
                model_object.add_inv_logit_normal_model()
            else:
                model_object.add_exp_uniform_normal_t_model()
        elif model_name == "Count":
            model_object.add_count_model()
        elif model_name == "Ordinal":
            model_object.add_ordinal_model()
        elif model_name == "Binomial":
            model_object.add_beta_binomial_model()
        else:
            error = False
        if error:
            logger.error("The model in config file not found. Exiting the program!")
            exit(0)

    # TODO add comments for this function
    def run(self):
        y = self.y
        prior_model = HierarchicalModel(y=y)
        logger.info(f"nGroups: {prior_model.n_groups}")
        for x in prior_model.stats_y:
            logger.info(x)
        self.add_model(prior_model)
        if self.run_prior:
            prior_model.get_GraphViz_object(
                self.file_prefix + "_prior",
                self.config_prior.getboolean("SaveHierarchicalTXT"),
                self.config_prior.getboolean("SaveHierarchicalPNG"),
                extension=self.extension,
                config=self.config_plots,
            )

            logger.info("Sampling From Prior ...")
            self.run_model(
                prior_model,
                file_prefix=self.file_prefix + "_prior",
                draws=self.config_prior.getint("Draws"),
                chains=self.config_prior.getint("Chains"),
                cores=1,
                tune=self.config_prior.getint("Tune"),
                model_config=self.config_prior,
                plots_config=self.config_plots,
            )

        if self.run_post:
            post_model = copy.copy(prior_model)
            post_model.add_observations_function()
            post_model.get_GraphViz_object(
                self.file_prefix + "_posterior",
                self.config_post.getboolean("SaveHierarchicalTXT"),
                self.config_post.getboolean("SaveHierarchicalPNG"),
                extension=self.extension,
                config=self.config_plots,
            )

            logger.info("Sampling From Posterior ...")
            self.run_model(
                post_model,
                file_prefix=self.file_prefix + "_posterior",
                draws=self.config_post.getint("Draws"),
                chains=self.config_post.getint("Chains"),
                cores=1,
                tune=self.config_post.getint("Tune"),
                model_config=self.config_post,
                plots_config=self.config_plots,
            )
            if self.run_prior and self.run_post and self.config_model.getboolean("BayesFactor"):
                bayes_factor_data_frame = self.bayes_factor_analysis(prior_model, post_model, initRope=self.rope)
                bayes_factor_file_name = self.file_prefix + "_BayesFactor.csv"
                bayes_factor_data_frame.to_csv(bayes_factor_file_name)
                logger.info(f"Bayes Factor DataFrame is saved at {bayes_factor_file_name}")
            # if self.postPredict: #TODO impose data
            #   self.drawPPC(trace, model=postModel)

    def bayes_factor_analysis(self, priorModel, postModel, initRope=(-0.1, 0.1)):
        column_names = ["ROPE", "priorProb", "postProb",
                       "BF", "BF_Savage_Dickey",
                       "prioNSample", "postNSample"]
        df = pd.DataFrame(columns=column_names)
        rope = np.array(initRope)
        n = 100 if self.config_model.getboolean("TrySmallerROPEs") else 1
        for i in range(n):
            prior_rope_prob_frac = priorModel.estimate_interval_prob(rope[0], rope[1])
            post_rope_prob_frac = postModel.estimate_interval_prob(rope[0], rope[1])
            if prior_rope_prob_frac[0] <= 0 or post_rope_prob_frac[0] <= 0:
                break
            prior_rope_prob = prior_rope_prob_frac[0] / prior_rope_prob_frac[1]
            post_rope_prob = post_rope_prob_frac[0] / post_rope_prob_frac[1]
            # logger.debug(priorRopeProb)
            # logger.debug(postRopeProb)
            # bfsv = postRopeProbFrac[0] * priorRopeProbFrac[1] / (postRopeProbFrac[1] * priorRopeProbFrac[0])
            # TODO `bfsv` stands for?
            bfsv = post_rope_prob / prior_rope_prob
            bf = bfsv * (1 - prior_rope_prob) / (1 - post_rope_prob)
            row = {
                column_names[0]: rope,
                column_names[1]: prior_rope_prob,
                column_names[2]: post_rope_prob,
                column_names[3]: bfsv,
                column_names[4]: bf,
                column_names[5]: prior_rope_prob_frac[0],
                column_names[6]: post_rope_prob_frac[0],
            }
            logger.debug(row)
            df = df.append(row, ignore_index=True)
            # logger.info(f"For ROPE={rope}:")
            # logger.info(f"  ROPE probibility in prior= {priorRopeProb}")
            # logger.info(f"  ROPE probibility in posteirour= {postRopeProb}")
            # logger.info(f"    Bayes Factor = {bf}")
            # logger.info(f"    Bayes Factor = {bf}")
            rope = rope / 1.2
        logger.info(df["BF"])
        return df

    # TODO PPC stands for?
    def draw_ppc(self, trace, model):
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
        ax.hist(self.y[0], bins=19, alpha=0.5, histtype='bar', color="red", rwidth=0.3)
        # pm.densityplot(trace, varnames=["y_0",],ax=ax)
        # ax.axvline(data.mean())
        MLmu = np.mean(ppc["mu"][0])
        MLsd = np.mean(ppc["sigma"][0])
        MLnu = np.mean(ppc["nu"])

        xp = np.linspace(MLmu - 4 * MLsd, MLmu + 4 * MLsd, 100)
        yp = MLsd * stats.t(nu=MLnu).pdf(xp) + MLmu
        ax.scatter(x=xp,
                   y=yp)
        ax.scatter(x=self.y[0],
                   y=np.zeros(self.y[0].shape), marker='x', color="black")
        ax.set(title='Posterior predictive of the mean',
               xlabel='mean(x)',
               ylabel='Frequency')
        plt.savefig("ppc.png")
        plt.clf()


if __name__ == '__main__':
    print("Not this file")
    # exp1 = Experiment(runPrior=True, runPost=True, filePrefix="newOutput/normalTest")# old
    # exp1.run()
