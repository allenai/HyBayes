import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pymc3 as pm
import theano.tensor as tt

from models.beta_bern_model import add_beta_bern_model
from models.beta_binomial_model import add_beta_binomial_model
from models.metric_model import add_exp_uniform_normal_t_model
from models.count_model import add_count_model
from models.ordinal_model import add_ordinal_model

from visualization import difference_plots
from scipy import stats
from theano.compile.ops import as_op
from Bayes_factor_analysis import bayes_factor_analysis
logger = logging.getLogger('root')


def get_rope(config, parameter):
    """
    Read ROPE (corresponding to the parameter) information from config
    :param config:
    :param parameter:
    :return:
    """
    return config.getfloat(f"{parameter}_ROPE_begin"), config.getfloat(f"{parameter}_ROPE_end")


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

    def get_GraphViz_object(self, file_prefix: str, save_dot: bool = True, save_png: bool = True, extension="png"):
        '''
        Returns the GraphViz object corresponding to the underlying hierarchical model.
        '''
        graph = pm.model_to_graphviz(self.pymc_model)
        graph.format = extension
        if save_dot:
            txtFileName = f"{file_prefix}_hierarchicalGraph.txt"
            graph.save(txtFileName)
            logger.info(f"Graph's source saved to {txtFileName}")
        if save_png:
            pngFileName = f"{file_prefix}_hierarchicalGraph"
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
        self.file_prefix = config["Files"].get("Output_prefix")
        self.config_model = config["Model"]
        self.config_prior = config["Prior"]
        self.config_post = config["Posterior"]
        self.config_plots = config["Plots"]
        self.config_Bayes_factor = config["Bayes_factor"]
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
        with hierarchical_model.pymc_model:
            hierarchical_model.trace = pm.sample(model=hierarchical_model.pymc_model,
                                                 draws=draws, chains=chains, cores=cores, tune=tune)
        logger.info(f"Effective Sample Size (ESS) = {pm.diagnostics.effective_n(hierarchical_model.trace)}")
        if model_config.getboolean("Save_trace"):
            traceFolderName = f"{file_prefix}_trace"
            if os.path.exists(traceFolderName):
                ind = 0
                while os.path.exists(f"{traceFolderName}_{ind}"):
                    ind += 1
                traceFolderName = f"{traceFolderName}_{ind}"
            pm.save_trace(hierarchical_model.trace, directory=traceFolderName)
            with open(os.path.join(traceFolderName, "pickeledTrace.pkl"), 'wb') as buff:
                pickle.dump({'model': hierarchical_model.pymc_model, 'trace': hierarchical_model.trace}, buff)
            logger.info(f"{traceFolderName} is saved!")
        # TODO drop these?
        # Plot autocor
        # pm.autocorrplot()
        if model_config.getboolean("Diagnostic_plots"):
            pm.traceplot(hierarchical_model.trace)
            diag_file_name = f"{file_prefix}_diagnostics.{self.extension}"
            plt.savefig(diag_file_name)
            logger.info(f"{diag_file_name} is saved!")
            plt.clf()

        if hierarchical_model.n_groups == 2:
            difference_plots(hierarchical_model=hierarchical_model,
                             model_config=model_config,
                             file_prefix=file_prefix,
                             rope=get_rope(model_config, hierarchical_model.mu_parameter),
                             config=self.config_plots)

    def add_model(self, model_object):
        '''
        Constructing the appropriate model based on the specifications in the config file.
        :param model_object: the default model
        '''
        error = False
        model_name = self.config_model.get("Variable_type")
        if model_name == "Binary":
            if self.config_model.get("Prior_model") == "Beta":
                add_beta_bern_model(model_object)
            else:
                logger.error(f'The given prior model {self.config_model.get("Prior_model")} is not recognized')
        elif model_name == "Metric":
            if self.config_model.getboolean("UnitInterval"):
                add_inv_logit_normal_model(model_object)
            else:
                add_exp_uniform_normal_t_model(model_object)
        elif model_name == "Count":
            add_count_model(model_object)
        elif model_name == "Ordinal":
            add_ordinal_model(model_object)
        elif model_name == "Binomial":
            add_beta_binomial_model(model_object)
        else:
            error = False
        if error:
            logger.error("The model in config file not found. Exiting the program!")
            exit(0)

    # TODO add comments for this function
    def run(self):
        y = self.y
        prior_model = HierarchicalModel(y=y)
        logger.info("Summary of statistics for the given data")
        logger.info(f"n_groups: {prior_model.n_groups}")
        for ind, x in enumerate(prior_model.stats_y):
            logger.info(f"Group index = {ind}:")
            logger.info(x)
        self.add_model(prior_model)
        if self.run_prior:
            prior_model.get_GraphViz_object(
                self.file_prefix + "_prior",
                self.config_prior.getboolean("Save_hierarchical_TXT"),
                self.config_prior.getboolean("Save_hierarchical_PNG"),
                extension=self.extension,
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
                self.config_post.getboolean("Save_hierarchical_TXT"),
                self.config_post.getboolean("Save_hierarchical_PNG"),
                extension=self.extension,
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
            if self.config_Bayes_factor.get("analyze"):

                if self.run_prior and self.run_post:
                    rope = get_rope(self.config_Bayes_factor, prior_model.mu_parameter)
                    if None in rope:
                        rope = get_rope(self.config_model, prior_model.mu_parameter)
                    if None in rope:
                        # TODO infer the rope from input data if not given in config
                        rope = (-0.1, 0.1)
                    bayes_factor_data_frame = bayes_factor_analysis(self.config_Bayes_factor, prior_model, post_model, init_rope=rope)
                    bayes_factor_file_name = self.file_prefix + "_Bayes_factor.csv"
                    bayes_factor_data_frame.to_csv(bayes_factor_file_name)
                    logger.info(f"Bayes Factor DataFrame is saved at {bayes_factor_file_name}")
                else:
                    logger.info("For running Bayes factor analysis, "
                                "flags for both prior and posterior analysis should be on.")
            # if self.postPredict: #TODO impose data
            #   self.drawPPC(trace, model=postModel)

    def draw_ppc(self, trace, model):
        """
        Makes Posterior Predictive Checks (PPC). Posterior predictive checks are, in simple words, "simulating replicated data under the fitted model and then comparing these to the observed data" (Gelman and Hill, 2007, p. 158). So, you use posterior predictive to "look for systematic discrepancies between real and simulated data" (Gelman et al. 2004, p. 169).
        :param trace:
        :param model:
        :return:
        """
        # print(model.pymc_model.observed_RVs)
        # print(len(model.pymc_model.observed_RVs))
        # print(model.pymc_model.observed_RVs[0])
        # print(type(model.pymc_model.observed_RVs[0]))
        # print(type(model.pymc_model.observed_RVs))
        # print(model.pymc_model.mu)
        ppc = pm.sample_posterior_predictive(trace, samples=500, model=model.pymc_model,
                                             vars=[model.pymc_model.mu,
                                                   model.pymc_model.nu,
                                                   model.pymc_model.sigma])

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

