import configparser
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

from matplotlib import gridspec
from .utils import *

logger = logging.getLogger('root')
color_list = [plt.cm.get_cmap("tab10").colors[0], plt.cm.get_cmap("tab10").colors[1]]


def get_rope(config: configparser.ConfigParser, parameter: str):
    """
    Read ROPE (corresponding to the parameter) information from config
    :param config:
    :param parameter:
    :return: two floats indicating two ends of the ROPE
    """
    return config.getfloat(f"{parameter}_ROPE_begin"), config.getfloat(f"{parameter}_ROPE_end")


def fix_hdp_labels(texts) -> None:
    """
    Goes over all texts in a matplotlib axis and changes 'HPD's to 'HDI's.
    HPD: "Highest Posterior Density" which is a default name in Pymc3
    HDI: "Highest Density Interval" which is a term we used in our paper
    :param texts:
    :return:
    """
    for tx in texts:
        if "HPD" in tx.get_text():
            tx.set_text("HDI")


def pre_analysis_plots(y, config: configparser.ConfigParser):
    """
    Depending on the Variable_type (indicated in config), puts the raw data in a few informative plots
    :param y: A length-2 list of 1-d numpy arrays
    :param config: a config object that indicate Variable_type and desired output file info
    :return None
    """
    # Preparation
    count_df = None
    proportions = []
    dfs = []
    if config["Model"].get("Variable_type") in ["Count", "Ordinal", "Binary", "Metric"]:
        for i in range(len(y)):
            dfs.append(pd.DataFrame(columns=["value", "Group"]))
            dfs[-1].value = y[i]
            dfs[-1].Group = i  # or class or category
            dfs[-1] = dfs[-1].astype(np.int)
        count_df = pd.concat(dfs)

    # Plotting:
    if config["Model"].get("Variable_type") in ["Contingency", ]:
        mat = y[0]
        # print(np.sum(mat, axis=1))
        mat = mat / np.sum(mat, axis=1)[:, np.newaxis]
        if config["Plots"].get("Histogram_plot"):
            plt.hist(mat[:,1]-mat[:,2])
            file_name = f"{config['Files'].get('OutputPrefix')}_Histogram_plot.{config['Plots'].get('Extension')}"
            plt.savefig(file_name)
            logger.info(f"Histogram Plot is saved to {file_name}.")
            plt.clf()
        if config["Plots"].get("Avg_confusion_heat_map"):
            avg_table = np.average(mat, axis=0).reshape(2, 2)
            sns.heatmap(avg_table, annot=True)
            file_name = f"{config['Files'].get('Output_prefix')}_Heat_map_plot.{config['Plots'].get('Extension')}"
            plt.savefig(file_name)
            logger.info(f"Histogram Plot is saved to {file_name}.")
            plt.clf()

    if config["Model"].get("Variable_type") in ["Metric", ] and config["Plots"].get("Histogram_plot"):
        fig, axes = plt.subplots(1, 2, sharey='row')
        for i in range(len(y)):
            axes[i].hist(y[i], bins=20, color=color_list[i])
            axes[i].set_title(f"Group {i}")
        file_name = f"{config['Files'].get('Output_prefix')}_Histogram_plot.{config['Plots'].get('Extension')}"
        plt.savefig(file_name)
        logger.info(f"Histogram Plot is saved to {file_name}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Binary",] and config["Plots"].get("Bar_plot"):
        for i in range(len(y)):
            n_ones = np.count_nonzero(y[i])
            proportions.append(n_ones/y[i].shape[0])
        plt.bar([0, 1], proportions, color = color_list)
        plt.xticks([0, 1], ["Group 0", "Group 1"])
        plt.ylabel("Portion of value 1")
        file_name = f"{config['Files'].get('Output_prefix')}_bar_plot.{config['Plots'].get('Extension')}"
        file_name = f"{config['Files'].get('Output_prefix')}_bar_plot.{config['Plots'].get('Extension')}"
        plt.savefig(file_name)
        logger.info(f"Bar Plot is saved to {file_name}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Binomial",] and config["Plots"].get("Histogram_plot"):
        fig, axes = plt.subplots(1, 2, sharey='row')
        for i in range(len(y)):
            a = y[i][:, 1] / (y[i][:, 0] + y[i][:, 1])
            axes[i].hist(a, bins=20, color=color_list[i])
            axes[i].set_title(f"Group {i}")
        file_name = f"{config['Files'].get('Output_prefix')}_histogram_plot.{config['Plots'].get('Extension')}"
        plt.savefig(file_name)
        logger.info(f"Histogram Plot is saved to {file_name}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Count", "Ordinal", "Binary", "Metric"]:
        if config["Plots"].getboolean("Count_plot"):
            sns_count_plot = sns.countplot(x=count_df["value"], hue=count_df["Group"])
            file_name = f"{config['Files'].get('Output_prefix')}_count_plot.{config['Plots'].get('Extension')}"
            plt.savefig(file_name)
            logger.info(f"Count Plot is saved to {file_name}.")
            plt.clf()

        if config["Plots"].getboolean("Scatter_plot"):
            plt.ylim(-0.9, 1.9)
            plt.yticks([0, 1], ["Group 0", "Group 1", ])
            colors = [color_list[x] for x in count_df["Group"]]
            plt.scatter(count_df["value"], count_df["Group"], color=colors)
            file_name = f"{config['Files'].get('Output_prefix')}_scatter_plot.{config['Plots'].get('Extension')}"
            plt.savefig(file_name)
            logger.info(f"Scatter Plot is saved to {file_name}.")
            plt.clf()


def one_parameter_plot(hierarchical_model, var, file_prefix, config_plot=None, show= False, rope=(-0.1, 0.1)):
    """
    Draws three plots corresponding to one parameter in the model.
    There will be one plot for each group in the bottom row and one difference plot at the top row.
    :param show: whether to matplotlib shows the plot in an interactive mode.
    :param hierarchical_model: the model object to get the samples(trace)
    :param var: the variable of interest to plot
    :param file_prefix: the string used for all the files
    :param rope: the value of ROPE used in plot
    :param config_plot: Several configuration properties e.g., file extension.
    :return:
    """
    extension = config_plot.get("Extension")
    plot_kind = config_plot.get("Kind")
    text_ratio = config_plot.getfloat("text_size_ratio")
    point_estimate = config_plot.get("point_estimate")
    round_to = config_plot.getint("round_to")
    credible_interval = config_plot.getfloat("credible_interval")
    color = "#" + config_plot.get("color")
    # How best to put defaults
    #  color = '#87ceeb'
    printLine()
    trace = hierarchical_model.trace
    plt.figure(figsize=(17, 17))
    gs = gridspec.GridSpec(3, 4)
    ax1 = plt.subplot(gs[:2, :4])
    ax2 = plt.subplot(gs[2, :2])
    ax3 = plt.subplot(gs[2, 2:])
    printLine()
    diff_values = trace[var][:, 0] - trace[var][:, 1]
    printLine()
    diff_var_name = f"{var}_1-{var}_2"
    # trace.add_values({diff_var_name: diff})
    printLine()
    logger.debug(diff_var_name)
    pm.plot_posterior(diff_values,
                      # var_names=[diff_var_name, ],
                      figsize=(4, 4),
                      textsize=text_ratio,
                      credible_interval = credible_interval,
                      round_to=round_to,
                      point_estimate=point_estimate,
                      rope=rope,
                      ref_val=0,
                      kind=plot_kind,
                      ax=ax1,
                      color=color,
                      )
    printLine()
    ax1.set_title(diff_var_name)
    printLine()
    for ind, ax in enumerate([ax2, ax3]):
        printLine()
        pm.plot_posterior(trace[var][:, ind],
                          # varnames=var,
                          figsize=(4, 4),
                          textsize=text_ratio,
                          credible_interval = credible_interval,
                          round_to=round_to,
                          point_estimate=point_estimate,
                          # no rope, ref_val
                          kind = plot_kind,
                          ax=ax,
                          color=color,
                          )
        printLine()
        ax.set_title("")
        ax.set_xlabel(f"{var}_{ind + 1}", fontdict={"size": text_ratio})
    printLine()
    if config_plot.getboolean("HPDtoHDI"):
        for ax in [ax1, ax2, ax3]:
            fix_hdp_labels(list(filter(lambda x: isinstance(x, matplotlib.text.Text), ax.get_children())))

    dist_file_name = f"{file_prefix}_{var}.{extension}"
    plt.savefig(dist_file_name)
    logger.info(f"{dist_file_name} is saved!")
    if show:
        plt.show()
    plt.clf()
    printLine()


def difference_plots(hierarchical_model, corresponding_config, file_prefix, config_plot, config_model):
    """
    Handles the plot requests indicated in the config object
    :param config_model:
    :param hierarchical_model:
    :param corresponding_config: Configuration of the either prior or post that is running here.
    :param file_prefix:
    :param config_plot:
    :return:
    """
    trace = hierarchical_model.trace
    mu_var = hierarchical_model.mu_parameter
    sigma_var = hierarchical_model.sigma_parameter
    printLine()
    if corresponding_config.getboolean("mean_plot"):
        logger.info(f"Plots corresponding to the mean will be under parameter name: {mu_var}")
        corresponding_config[f"{mu_var}_plot"] = "True"
        corresponding_config[f"show_{mu_var}_plot"] = str(corresponding_config.getboolean("Show_mean_plot"))

    if corresponding_config.getboolean("SD_plot") and sigma_var is not None:
        logger.info(f"Plots corresponding to the standard deviation will be under parameter name:{sigma_var}")
        corresponding_config[f"{sigma_var}_plot"] = "True"
        corresponding_config[f"show_{sigma_var}_plot"] = str(corresponding_config.getboolean("Show_SD_plot"))

    parameters_to_plot = []
    for var in trace.varnames:
        var = var.split("_")[0]
        if var not in parameters_to_plot and corresponding_config.getboolean(f"{var}_plot"):
            parameters_to_plot.append(var)
        printLine()
    printLine()
    logger.debug(parameters_to_plot)

    for param in parameters_to_plot:
        printLine()
        one_parameter_plot(hierarchical_model, param, file_prefix, config_plot,
                           corresponding_config.getboolean(f"Show_{param}_plot"), get_rope(config_model, param))
        printLine()
    if corresponding_config.getboolean("Compare_all_parameters_plot"):
        compare_all_parameters_plot(hierarchical_model, config_plot,
                                    [mu_var, sigma_var, "effect_size", "nu"],
                                    file_prefix)


def compare_all_parameters_plot(hierarchical_model, config_plot, vars, file_prefix):
    """
    Draws a difference plot for all variables in vars.
    :param hierarchical_model:
    :param config_plot:
    :param vars: on top of parameters in model. This accepts "effect_size" too.
    :param file_prefix:
    :return:
    """
    printLine()
    extension = config_plot.get("Extension")
    plot_kind = config_plot.get("Kind")
    text_ratio = config_plot.getfloat("text_size_ratio")
    point_estimate = config_plot.get("point_estimate")
    round_to = config_plot.getint("round_to")
    credible_interval = config_plot.getfloat("credible_interval")
    color = "#" + config_plot.get("color")
    # How best to put defaults
    #  color = '#87ceeb'

    trace = hierarchical_model.trace
    mu_var = hierarchical_model.mu_parameter
    sigma_var = hierarchical_model.sigma_parameter
    variables = [var for var in vars if var is not None and var in trace.varnames]
    if "effect_size" in variables and (mu_var is None or sigma_var is None):
        variables.remove("effect_size")

    logger.info(f"Parameters included in compare_all_plot: {variables}")
    fig, axes = plt.subplots(len(variables), 1, figsize=(30, len(variables) * 30), squeeze = False)
    axes = axes.reshape(-1)
    printLine()
    for ind, var in enumerate(variables):
        if var is "effect_size":
            array = (trace[mu_var][:, 0] - trace[mu_var][:, 1]) / np.sqrt(
                (trace[sigma_var][:, 0] ** 2 + trace[sigma_var][:, 1] ** 2) / 2)
        elif var in trace.varnames and len(trace[var].shape) == 1:
            array = trace[var]
        else:
            array = trace[var][:, 0] - trace[var][:, 1]
        pm.plot_posterior(array,
                          textsize=text_ratio,
                          credible_interval=credible_interval,
                          round_to=round_to,
                          point_estimate=point_estimate,
                          ref_val=0,
                          # Get a rope here
                          kind=plot_kind,
                          ax=axes[ind],
                          color=color,
                          )
        if var is "effect_size":
            axes[ind].set_title(f"Effect size", fontdict={"size": text_ratio})
        elif var in trace.varnames and len(trace[var].shape) == 1:
            axes[ind].set_title(f"{var}", fontdict={"size": text_ratio})
        else:
            axes[ind].set_title(f"{var} difference", fontdict={"size": text_ratio})

    dist_file_name = f"{file_prefix}_compare_all_parameters.{extension}"
    plt.savefig(dist_file_name)
    logger.info(f"{dist_file_name} is saved!")
    plt.clf()
    printLine()
