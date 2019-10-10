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
colorList = [plt.cm.get_cmap("tab10").colors[0], plt.cm.get_cmap("tab10").colors[1]]


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
    countDf = None
    histDf = None
    proportions = []
    dfs = []
    if config["Model"].get("Variable_type") in ["Count", "Ordinal", "Binary", "Metric"]:
        for i in range(len(y)):
            dfs.append(pd.DataFrame(columns=["value", "Group"]))
            dfs[-1].value = y[i]
            dfs[-1].Group = i  # or class or category
            dfs[-1] = dfs[-1].astype(np.int)
        countDf = pd.concat(dfs)

    # Plotting:
    if config["Model"].get("Variable_type") in ["Metric", ] and config["Plots"].get("HistogramPlot"):
        fig, axes = plt.subplots(1, 2, sharey='row')
        for i in range(len(y)):
            axes[i].hist(y[i], bins=20, color=colorList[i])
            axes[i].set_title(f"Group {i}")
        fileName = f"{config['Files'].get('Output_prefix')}_HistogramPlot.{config['Plots'].get('Extension')}"
        plt.savefig(fileName)
        logger.info(f"Histogram Plot is saved to {fileName}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Binary", ] and config["Plots"].get("BarPlot"):
        for i in range(len(y)):
            nOnes = np.count_nonzero(y[i])
            proportions.append(nOnes / y[i].shape[0])
        plt.bar([0, 1], proportions, color=colorList)
        plt.xticks([0, 1], ["Group 0", "Group 1"])
        plt.ylabel("Portion of value 1")
        fileName = f"{config['Files'].get('Output_prefix')}_barPlot.{config['Plots'].get('Extension')}"
        plt.savefig(fileName)
        logger.info(f"Bar Plot is saved to {fileName}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Binomial", ] and config["Plots"].get("HistogramPlot"):
        fig, axes = plt.subplots(1, 2, sharey='row')
        for i in range(len(y)):
            a = y[i][:, 1] / (y[i][:, 0] + y[i][:, 1])
            axes[i].hist(a, bins=20, color=colorList[i])
            axes[i].set_title(f"Group {i}")
        fileName = f"{config['Files'].get('Output_prefix')}_HistogramPlot.{config['Plots'].get('Extension')}"
        plt.savefig(fileName)
        logger.info(f"Histogram Plot is saved to {fileName}.")
        plt.clf()

    if config["Model"].get("Variable_type") in ["Count", "Ordinal", "Binary", "Metric"]:
        if config["Plots"].getboolean("CountPlot"):
            sns_count_plot = sns.countplot(x=countDf["value"], hue=countDf["Group"])
            fileName = f"{config['Files'].get('Output_prefix')}_countPlot.{config['Plots'].get('Extension')}"
            plt.savefig(fileName)
            logger.info(f"Count Plot is saved to {fileName}.")
            plt.clf()

        if config["Plots"].getboolean("ScatterPlot"):
            plt.ylim(-0.9, 1.9)
            plt.yticks([0, 1], ["Group 0", "Group 1", ])
            colors = [colorList[x] for x in countDf["Group"]]
            plt.scatter(countDf["value"], countDf["Group"], color=colors)
            fileName = f"{config['Files'].get('Output_prefix')}_scatterPlot.{config['Plots'].get('Extension')}"
            plt.savefig(fileName)
            logger.info(f"Scatter Plot is saved to {fileName}.")
            plt.clf()


def one_parameter_plot(hierarchicalModel, var, filePrefix, rope, improvements=False,
                       config=None):
    extension = config.get("Extension")
    trace = hierarchicalModel.trace
    plt.figure(figsize=(17, 17))
    gs = gridspec.GridSpec(3, 4)
    ax1 = plt.subplot(gs[:2, :4])
    ax2 = plt.subplot(gs[2, :2])
    ax3 = plt.subplot(gs[2, 2:])
    diff = trace[var][:, 0] - trace[var][:, 1]
    diffVarName = f"{var}_1-{var}_2"
    trace.add_values({diffVarName: diff})
    # print(trace.varnames, diffVarName)
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

    listOfChildren = ax1.get_children()
    texts = list(filter(lambda x: isinstance(x, matplotlib.text.Text), listOfChildren))
    lines = list(filter(lambda x: isinstance(x, matplotlib.lines.Line2D), listOfChildren))
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
                      # ref_val=hierarchicalModel.statsY[0].mean,
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
                      # ref_val=hierarchicalModel.statsY[1].mean,
                      )
    ax3.set_xlabel(r"$\theta_2$", fontdict={"size": int(config.getint("Font_size"))})
    if config.getboolean("HPDtoHDI"):
        for ax in [ax1, ax2, ax3]:
            fix_hdp_labels(list(filter(lambda x: isinstance(x, matplotlib.text.Text), ax.get_children())))

    distFileName = f"{filePrefix}_{var}.{extension}"
    plt.savefig(distFileName)
    logger.info(f"{distFileName} is saved!")
    plt.clf()


def difference_plots(hierarchicalModel, modelConfig, filePrefix, rope, config):
    trace = hierarchicalModel.trace
    muVar = hierarchicalModel.mu_parameter
    sigmaVar = hierarchicalModel.sigma_parameter
    if modelConfig.getboolean("Mean_plot"):
        one_parameter_plot(hierarchicalModel, muVar, filePrefix, rope,
                           modelConfig.getboolean("Plot_improvements"),
                           config,
                           )

    if modelConfig.getboolean("SD_plot"):
        one_parameter_plot(hierarchicalModel, sigmaVar, filePrefix, rope,
                           modelConfig.getboolean("Plot_improvements"), config)

    if modelConfig.getboolean("Compare_all_plot"):
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

        pm.plot_posterior(trace[sigmaVar][:, 0] - trace[sigmaVar][:, 1],
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
            (trace[sigmaVar][:, 0] ** 2 + trace[sigmaVar][:, 1] ** 2) / 2)
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
        distFileName = f"{filePrefix}_allComp.{config.get('Extension')}"
        plt.savefig(distFileName)
        logger.info(f"{distFileName} is saved!")
        plt.clf()


if __name__ == '__main__':
    print("starts here")
