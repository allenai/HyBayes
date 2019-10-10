import logging
import pandas as pd
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from matplotlib import gridspec

logger = logging.getLogger('root')
color = '#87ceeb'  # TODO: this can become a parameter
color_list = [plt.cm.get_cmap("tab10").colors[0], plt.cm.get_cmap("tab10").colors[1]]


def fix_hdp_labels(texts):
    for tx in texts:
        if "HPD" in tx.get_text():
            tx.set_text("HDI")


def pre_analysis_plots(y, config):
    """
    Depending on the Variable_type (indicated in config), puts the raw data in a few informative plots
    :param y: A length-2 list of 1-d numpy arrays
    :param config: configparser.ConfigParser that indicate Variable_type and desired output file info
    :return None
    """
    # for binomial make histogram of y[.][:,1]/y[.][:,0]
    # Preparation
    count_df = None
    hist_df = None
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
            avgTable = np.average(mat, axis=0).reshape(2, 2)
            sns.heatmap(avgTable, annot=True)
            file_name = f"{config['Files'].get('Output_prefix')}_Heat_map_plot.{config['Plots'].get('Extension')}"
            plt.savefig(file_name)
            logger.info(f"Histogram Plot is saved to {file_name}.")
            plt.clf()




    if config["Model"].get("Variable_type") in ["Metric", ] and config["Plots"].get("Histogram_plot"):
        fig, axes = plt.subplots(1, 2, sharey='row')
        for i in range(len(y)):
            axes[i].hist(y[i], bins=20, color=color_list[i])
            axes[i].set_title(f"Group {i}")
        file_name = f"{config['Files'].get('Output_prefix')}_HistogramPlot.{config['Plots'].get('Extension')}"
        plt.savefig(file_name)
        logger.info(f"Histogram Plot is saved to {file_name}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Binary",] and config["Plots"].get("Bar_plot"):
        for i in range(len(y)):
            nOnes = np.count_nonzero(y[i])
            proportions.append(nOnes/y[i].shape[0])
        plt.bar([0, 1], proportions, color = color_list)
        plt.xticks([0, 1], ["Group 0", "Group 1"])
        plt.ylabel("Portion of value 1")
        file_name = f"{config['Files'].get('Output_prefix')}_barPlot.{config['Plots'].get('Extension')}"
        plt.savefig(file_name)
        logger.info(f"Bar Plot is saved to {file_name}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Binomial",] and config["Plots"].get("Histogram_plot"):
        fig, axes = plt.subplots(1, 2, sharey='row')
        for i in range(len(y)):
            a = y[i][:, 1] / (y[i][:, 0] + y[i][:, 1])
            axes[i].hist(a, bins=20, color=color_list[i])
            axes[i].set_title(f"Group {i}")
        file_name = f"{config['Files'].get('Output_prefix')}_HistogramPlot.{config['Plots'].get('Extension')}"
        plt.savefig(file_name)
        logger.info(f"Histogram Plot is saved to {file_name}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Count", "Ordinal", "Binary", "Metric"]:
        if config["Plots"].getboolean("Count_plot"):
            sns_count_plot = sns.countplot(x=count_df["value"], hue=count_df["Group"])
            file_name = f"{config['Files'].get('Output_prefix')}_countPlot.{config['Plots'].get('Extension')}"
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


def one_parameter_plot(hierarchical_model, var, file_prefix, rope, improvements=False,
                       config=None):
    """
    Draws three plots corresponding to one parameter in the model.
    There will be one plot for each group in the bottom row and one difference plot at the top row.
    :param hierarchical_model: the model object to get the samples(trace)
    :param var: the variable of interest to plot
    :param file_prefix: the string used for all the files
    :param rope: the value of ROPE used in plot
    :param improvements: Not used
    :param config: Several configuration properties e.g., file extension.
    :return:
    """
    extension = config.get("Extension")
    trace = hierarchical_model.trace
    plt.figure(figsize=(17, 17))
    gs = gridspec.GridSpec(3, 4)
    ax1 = plt.subplot(gs[:2, :4])
    ax2 = plt.subplot(gs[2, :2])
    ax3 = plt.subplot(gs[2, 2:])
    diff = trace[var][:, 0] - trace[var][:, 1]
    diff_var_name = f"{var}_1-{var}_2"
    trace.add_values({diff_var_name: diff})
    # print(trace.varnames, diffVarName)
    # TODO check the interface to give varname and text_size
    pm.plot_posterior(diff,
                      figsize=(4, 4),
                      # varnames=diffVarName,
                      # alpha_level=0.05,
                      rope=rope,
                      point_estimate='mode',
                      ax=ax1,
                      color=color,
                      round_to=3,
                      ref_val=0,
                      # text_size=config.getint("Font_size"),
                      )
    ax1.set_xlabel(r"$\theta_1-\theta_2$", fontdict={"size": int(config.getint("Font_size") * 0.5)})

    list_of_children = ax1.get_children()
    texts = list(filter(lambda x: isinstance(x, matplotlib.text.Text), list_of_children))
    lines = list(filter(lambda x: isinstance(x, matplotlib.lines.Line2D), list_of_children))
    # TODO: optional CI

    #   for tx in texts:
    #     if "HPD" in tx.get_text():
    #       tx.set_text("HDI,")
    #       ax1.text(1.1*tx._x, tx._y, "CI", fontsize = 20, color="b")
    #
    #
    #
    #
    # CI = (0.0136, 0.057)
    # CIline = None
    # for i in range(len(lines)):
    # lin: matplotlib.lines.Line2D = lines[i]
    # if lin.get_markerfacecolor() == "k":
    #   CIline = copy.copy(lin)
    #   yy = 0.5 * CIline.get_ydata()[0]
    #   CIline.set_ydata((yy, yy))
    #   CIline.set_xdata(CI)
    #   CIline.set_color("b")
    #   ax1.add_line(CIline)

    pm.plot_posterior(trace[var][:, 0],
                      figsize=(4, 4),
                      # varnames=var,
                      # alpha_level=0.05,
                      point_estimate='mode',
                      ax=ax2,
                      color=color,
                      round_to=3,
                      # ref_val=hierarchical_model.statsY[0].mean,
                      # text_size=config.getint("Font_size"),
                      )
    ax2.set_xlabel(r"$\theta_1$", fontdict={"size": int(config.getint("Font_size"))})
    pm.plot_posterior(trace[var][:, 1],
                      figsize=(4, 4),
                      # varnames=var,
                      # alpha_level=0.05,
                      point_estimate='mode',
                      ax=ax3,
                      color=color,
                      round_to=3,
                      # text_size=config.getint("Font_size"),
                      # ref_val=hierarchical_model.statsY[1].mean,
                      )
    # todo use name from varname
    ax3.set_xlabel(r"$\theta_2$", fontdict={"size": int(config.getint("Font_size"))})
    if config.getboolean("HPDtoHDI"):
        for ax in [ax1, ax2, ax3]:
            fix_hdp_labels(list(filter(lambda x: isinstance(x, matplotlib.text.Text), ax.get_children())))

    dist_file_name = f"{file_prefix}_{var}.{extension}"
    plt.savefig(dist_file_name)
    logger.info(f"{dist_file_name} is saved!")
    plt.clf()


def difference_plots(hierarchical_model, model_config, file_prefix, rope, config):
    trace = hierarchical_model.trace
    muVar = hierarchical_model.mu_parameter
    sigma_var = hierarchical_model.sigma_parameter
    if model_config.getboolean("Mean_plot"):
        one_parameter_plot(hierarchical_model, muVar, file_prefix, rope,
                           model_config.getboolean("Plot_improvements"),
                           config,
                           )

    if model_config.getboolean("SD_plot"):
        one_parameter_plot(hierarchical_model, sigma_var, file_prefix, rope,
                           model_config.getboolean("Plot_improvements"), config)

    if model_config.getboolean("Compare_all_plot"):
        fig, axes = plt.subplots(3, 1, figsize=(20, 60))

        pm.plot_posterior(trace[muVar][:, 0] - trace[muVar][:, 1],
                          # varnames=var,
                          # alpha_level=0.05,
                          rope=rope,
                          point_estimate='mode',
                          ax=axes[0],
                          color=color,
                          round_to=3,
                          ref_val=0,
                          )
        axes[0].set_title("Mu difference")

        pm.plot_posterior(trace[sigma_var][:, 0] - trace[sigma_var][:, 1],
                          # varnames=var,
                          # alpha_level=0.05,
                          rope=(-0.1, 0.1),  # TODO: calculate ROPE
                          point_estimate='mode',
                          ax=axes[1],
                          color=color,
                          round_to=3,
                          ref_val=0,
                          )
        axes[1].set_title("Sigma difference")

        es = (trace[muVar][:, 0] - trace[muVar][:, 1]) / np.sqrt(
            (trace[sigma_var][:, 0] ** 2 + trace[sigma_var][:, 1] ** 2) / 2)
        pm.plot_posterior(es,
                          # varnames=var,
                          # alpha_level=0.05,
                          rope=(-0.1, 0.1),  # TODO: calculate ROPE
                          point_estimate='mode',
                          ax=axes[2],
                          color=color,
                          round_to=3,
                          ref_val=0,
                          )
        axes[2].set_title("Effect size")
        dist_file_name = f"{file_prefix}_allComp.{config.get('Extension')}"
        plt.savefig(dist_file_name)
        logger.info(f"{dist_file_name} is saved!")
        plt.clf()

