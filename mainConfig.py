import argparse, configparser
import matplotlib as mpl
import numpy as np
from Experiment import Experiment
import logging
import time
from dataVizualisation import preAnalysisPlots
mpl.use('Agg')

loggingTotalFileName = f"logs/allLogs.log"
logging.basicConfig(filename=loggingTotalFileName, filemode="a", level=logging.DEBUG)
logger = logging.getLogger('root')

parser = argparse.ArgumentParser(description="Run Bayesian Stat")
parser.add_argument("-c", "--config", help="address of Configfile", default="configs/configQABinomial.ini")
parser.add_argument("-v", "--verbose", help="prints the report of the steps", action="store_true", default=True)
args = parser.parse_args()

if __name__ == '__main__':
  loggingOneFileName = f"logs/{time.time()}.log"
  print(f"Logs for this run will be stored at {loggingOneFileName}", flush=True)
  shortFormatter = logging.Formatter('%(levelname)s: %(message)s')
  # longFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  consoleHandler = logging.StreamHandler()
  if args.verbose:
    consoleHandler.setLevel(logging.DEBUG)
  else:
    consoleHandler.setLevel(logging.ERROR)
  consoleHandler.setFormatter(shortFormatter)
  logger.addHandler(consoleHandler)

  fho = logging.FileHandler(filename=loggingOneFileName, mode='w')
  fho.setLevel(logging.DEBUG)
  logger.addHandler(fho)

  try:
    config = configparser.ConfigParser()
    config.read(args.config)
    nCol = np.int(config["Files"]["NumberOfColumns"])
    y = list()
    for fileName in [config["Files"]["File1"], config["Files"]["File2"]]:
      y.append(np.loadtxt(fileName))
      if nCol > 1:
        y[-1] = y[-1].reshape(-1, nCol)
      logger.info(f"File {fileName} is loaded.")


    config.write(open(config["Files"]["OutputPrefix"]+"_config.ini", 'w'))

    preAnalysisPlots(y=y, config=config)
    experiment = Experiment(y=y, config=config)
    experiment.run()
  except Exception as e: #TODO: make seperate messages for Exceptions (config file not found, import not working, theano...)
    logger.exception("An exception occurred. Halting the execution!")
  # TODO: Copy log file in the folder
  # TODO: draw post on top of prior
  # TODO: change use getBoolean() for config
