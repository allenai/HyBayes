import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from Experiment import Experiment


#parsing arguments
parser = argparse.ArgumentParser(description="Run Bayesian Stat")
parser.add_argument("-f1", "--file1", help="Performances of first Algorithm", required=True)
parser.add_argument("-f2", "--file2", help="Performances of second Algorithm", required=True)
parser.add_argument("-o", "--output-prefix", help="Where to put output", required=True)

args = parser.parse_args()
if __name__ == '__main__':
  y = list()
  y.append(np.loadtxt(args.file1))
  y.append(np.loadtxt(args.file2))
  # print(args.output_prefix)
  exp1 = Experiment(y=y, runPrior=True, runPost=True, filePrefix=args.output_prefix, postPredict=False)
  exp1.run()
