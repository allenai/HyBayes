import argparse, configparser
import matplotlib as mpl
import numpy as np
from Experiment import Experiment
import logging
import time
from visualization import pre_analysis_plots
from shutil import copyfile
import os
from utils import *
import traceback
mpl.use('Agg')


if __name__ == '__main__':
  logs_folder_made = mk_dir_if_not_exists("logs")
  loggingTotalFileName = f"logs/allLogs.log"
  logging.basicConfig(filename=loggingTotalFileName, filemode="a", level=logging.DEBUG)
  logger = logging.getLogger('root')

  parser = argparse.ArgumentParser(description="Run Bayesian Statistics Tailored towards analysing the experiment results specially in NLP area. Email @.com for comments.")
  parser.add_argument("-c", "--config", help="address of Config file", default="configs/configMetric.ini")
  parser.add_argument("-v", "--verbose", help="prints the report of the steps", action="store_true", default=False)
  args = parser.parse_args()
  loggingOneFileName = f"{time.time()}.log"
  loggingOneFileAddress = f"logs/{loggingOneFileName}"
  print(f"Logs for this run will be stored at {loggingOneFileAddress}", flush=True)
  shortFormatter = logging.Formatter('%(levelname)s: %(message)s')
  # longFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  consoleHandler = logging.StreamHandler()
  if args.verbose:
    consoleHandler.setLevel(logging.DEBUG)
  else:
    consoleHandler.setLevel(logging.ERROR)
  consoleHandler.setFormatter(shortFormatter)
  logger.addHandler(consoleHandler)

  fho = logging.FileHandler(filename=loggingOneFileAddress, mode='w')
  fho.setLevel(logging.DEBUG)
  logger.addHandler(fho)
  config = configparser.ConfigParser()
  logger.info(f"Reading config file: {args.config}")
  config.read(args.config)

  try:
    if logs_folder_made:
      logger.info(f"logs folder is made!")

    output_prefix = config["Files"].get("Output_prefix")
    last_slash_index = output_prefix.rfind("/")

    if last_slash_index != -1:
      mk_dir_if_not_exists(output_prefix[:last_slash_index])

    nCol = config["Files"].getint("Number_of_columns")
    nFile = config["Files"].getint("Number_of_files")
    if nFile is None:
      nFile = 2
      
    y = list()
    fileNameList = [config["Files"].get(x) for x in [f"File{ind}" for ind in range(1, nFile+1)]]
    for fileName in fileNameList:
      y.append(np.loadtxt(fileName))
      if nCol > 1:
        y[-1] = y[-1].reshape(-1, nCol)
      logger.info(f"File {fileName} is loaded.")

    destConfigFileName = config["Files"].get("Output_prefix") + "_config.ini"
    config.write(open(destConfigFileName, 'w'))
    logger.info(f"Copying the Config file of this run to {destConfigFileName}.")
    
    if nFile > 0:
      pre_analysis_plots(y=y, config=config)

      experiment = Experiment(y=y, config=config)
      experiment.run()
  except Exception as e: #TODO: make seperate messages for Exceptions (config file not found, import not working, theano...)
    logger.exception("An exception occurred. Halting the execution!")
    traceback.print_exc()
  finally:
    userLogFileName = f'{config["Files"].get("Output_prefix")}_log.log'
    logger.info(f"Copying the log file of this run to {userLogFileName}.")
    copyfile(loggingOneFileAddress, userLogFileName)
