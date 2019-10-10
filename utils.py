import pickle
import pymc3 as pm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import numpy as np
import os

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

light_blue_color = np.array((219, 232, 255)) / 255
KruscheColor = '#87ceeb'


def posterior_plot(var_trace, HDI=(0.2, 0.85), rope=(0.05, 0.11), CI=(0.4, 0.9), mean_val=0.3, mode_val=0.2,
                   color=light_blue_color):
    ax = plt.gca()
    ax.hist(var_trace, bins=40, color=light_blue_color, density=True, )
    yHDI = 0.08
    yrope = 0.03
    yCI = 0.03
    ll = [(HDI, yHDI, "95% HDI", "black", 8, 0.6),
          (rope, yrope, "ROPE", "red", 8, 0.5),
          (CI, yCI, "95% CI", "green", 8, 0.6), ]

    ind = 0
    for interval, y, text, text_color, line_width, alpha in ll:
        ax.text(1.25, 0.9 - 0.1 * ind,
                text + " " + str(interval),
                fontsize=15,
                # horizontalalignment='center',
                transform=ax.transAxes,
                color=text_color)
        ind += 1
        line = mlines.Line2D(interval, [y, y], color=text_color, linewidth=line_width, alpha=alpha)
        ax.add_line(line)


def mk_dir_if_not_exists(folder_address):
    chain = folder_address.split("/")
    folder_address = ""
    changed = False
    for newPart in chain:
        folder_address = os.path.join(folder_address, newPart)
        if not os.path.exists(folder_address):
          os.mkdir(folder_address)
          changed = True
    return changed

