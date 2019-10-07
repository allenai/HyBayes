import pickle
import pymc3 as pm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import numpy as np

# @as_op(itypes=[tt.fvector, tt.fvector, tt.fvector], otypes=[tt.fmatrix])
# def outcome_probabilities(theta, mu, sigma):
#     out = np.empty((nYlevels2, n_grps), dtype=np.float32)
#     n = norm(loc=mu, scale=sigma)
#     out[0,:] = n.cdf(theta[0])
#     out[1,:] = np.max([[0,0], n.cdf(theta[1]) - n.cdf(theta[0])], axis=0)
#     out[2,:] = np.max([[0,0], n.cdf(theta[2]) - n.cdf(theta[1])], axis=0)
#     out[3,:] = np.max([[0,0], n.cdf(theta[3]) - n.cdf(theta[2])], axis=0)
#     out[4,:] = 1 - n.cdf(theta[3])
#     return out

lightBlueColor = np.array((219, 232, 255)) / 255
KruscheColor = '#87ceeb'


def posterior_plot(varTrace, HDI=(0.2, 0.85), rope=(0.05, 0.11), CI=(0.4, 0.9), meanVal=0.3, modeVal=0.2,
                   color=lightBlueColor):
    ax = plt.gca()
    ax.hist(varTrace, bins=40, color=lightBlueColor, density=True, )
    yHDI = 0.08
    yrope = 0.03
    yCI = 0.03
    ll = [(HDI, yHDI, "95% HDI", "black", 8, 0.6),
          (rope, yrope, "ROPE", "red", 8, 0.5),
          (CI, yCI, "95% CI", "green", 8, 0.6), ]

    ind = 0
    for interval, y, text, textColor, lineWidth, alpha in ll:
        ax.text(1.25, 0.9 - 0.1 * ind,
                text + " " + str(interval),
                fontsize=15,
                # horizontalalignment='center',
                transform=ax.transAxes,
                color=textColor)
        ind += 1
        line = mlines.Line2D(interval, [y, y], color=textColor, linewidth=lineWidth, alpha=alpha)
        ax.add_line(line)


if __name__ == '__main__':
    print("running utils")
    pklAddress = "experimentFiles2/BinomialQACompare_posterior_trace/pickeledTrace.pkl"
    with open(pklAddress, 'rb') as buff:
        data = pickle.load(buff)

    basic_model, trace = data['model'], data['trace']
    ax = plt.gca()
    fig = plt.figure()
    newAx = pm.plot_posterior(trace["theta"][0] - trace["theta"][1], text_size=20, rope=(-0.01, 0.01), ax=ax)
    listOfChildren = ax.get_children()
    texts = list(filter(lambda x: isinstance(x, matplotlib.text.Text), listOfChildren))
    lines = list(filter(lambda x: isinstance(x, matplotlib.lines.Line2D), listOfChildren))
    # texts[2].set_visible(False)
    # texts[2].set_text("HDI")
    # print(dir(texts[2]))
    # print(texts[2].x)
    # ax.text(texts[2])
    # trace = pm.load_trace(r".\experimentFiles\BinomialQACompare_posterior_trace")
    # print(trace)
    #
    # posteriorPlot(trace["theta"][0]-trace["theta"][1])
    # ax.remove(texts[2])
    # ci = (0.0136, 0.057)
    for i in range(len(lines)):
        print(lines[i].get_xdata())
        print(lines[i].get_ydata())
        print(lines[i].get_markerfacecolor())
        print()

    # ax.add_line(matplotlib.lines.Line2D((ci[0], ci[1]), (0.2, 0.2)))
    # mlines.Line2D(interval, [y, y], color=textColor, linewidth=lineWidth, alpha=alpha)
    # ax.axhline(y=2*lines[i].get_ydata()[0], xmin=0.1, xmax=0.4, )
    # yy =2*lines[i].get_ydata()[0]
    # newLine = copy.copy(lines[0])
    # newLine.set_ydata((yy, yy))
    # newLine.set_xdata(ci)
    # ax.add_line(newLine)
    # plt.show()
