import argparse, configparser
import matplotlib as mpl
import numpy as np
from experiment import Experiment
import logging
import time
from visualization import pre_analysis_plots
from shutil import copyfile
import os
from utils import *
import traceback



if __name__ == '__main__':
  logs_folder_made = mk_dir_if_not_exists("logs")
  logging_total_file_name = f"logs/allLogs.log"
  logging.basicConfig(filename=logging_total_file_name, filemode="a", level=logging.DEBUG)
  logger = logging.getLogger('root')

  parser = argparse.ArgumentParser(description="Run Bayesian Statistics Tailored towards analysing the experiment results specially in NLP area. Email @.com for comments.")
  parser.add_argument("-c", "--config", help="address of Config file", default="configs/configMetric.ini")
  parser.add_argument("-v", "--verbose", help="prints the report of the steps", action="store_true", default=False)
  args = parser.parse_args()
  logging_one_file_name = f"{time.time()}.log"
  logging_one_file_address = f"logs/{logging_one_file_name}"
  print(f"Logs for this run will be stored at {logging_one_file_address}", flush=True)
  short_formatter = logging.Formatter('%(levelname)s: %(message)s')
  # long_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  console_handler = logging.StreamHandler()
  if args.verbose:
    console_handler.setLevel(logging.DEBUG)
  else:
    console_handler.setLevel(logging.ERROR)
  console_handler.setFormatter(short_formatter)
  logger.addHandler(console_handler)

  fho = logging.FileHandler(filename=logging_one_file_address, mode='w')
  fho.setLevel(logging.DEBUG)
  logger.addHandler(fho)
  config = configparser.ConfigParser()
  logger.info(f"Reading config file: {args.config}")
  config.read(args.config)

  # If all show_ configs are off, the interactive mode for mpl will be closed.
  interactive_mode = False
  for plot_config in [config["Prior"], config["Posterior"]]:
    for key in plot_config.keys():
      if "show_" in key and plot_config[key]:
        interactive_mode = True
  logger.debug(interactive_mode)
  if not interactive_mode:
    mpl.use('Agg')
  try:
    if logs_folder_made:
      logger.info(f"logs folder is made!")

    output_prefix = config["Files"].get("Output_prefix")
    last_slash_index = output_prefix.rfind("/")

    if last_slash_index != -1:
      mk_dir_if_not_exists(output_prefix[:last_slash_index])

    number_of_columns = config["Files"].getint("Number_of_columns")
    number_of_file = config["Files"].getint("Number_of_files")
    if number_of_file is None:
      number_of_file = 2
      
    y = list()
    file_name_list = [config["Files"].get(x) for x in [f"File{ind}" for ind in range(1, number_of_file + 1)]]
    for file_name in file_name_list:
      y.append(np.loadtxt(file_name))
      if number_of_columns > 1:
        y[-1] = y[-1].reshape(-1, number_of_columns)
      logger.info(f"File {file_name} is loaded.")

    dest_config_file_name = config["Files"].get("Output_prefix") + "_config.ini"
    config.write(open(dest_config_file_name, 'w'))
    logger.info(f"Copying the Config file of this run to {dest_config_file_name}.")
    
    if number_of_file > 0:
      pre_analysis_plots(y=y, config=config)

      experiment = Experiment(y=y, config=config)
      experiment.run()
  except Exception as e: #TODO: make seperate messages for Exceptions (config file not found, import not working, theano...)
    logger.exception("An exception occurred. Halting the execution!")
    traceback.print_exc()
  finally:
    user_log_file_name = f'{config["Files"].get("Output_prefix")}_log.log'
    logger.info(f"Copying the log file of this run to {user_log_file_name}.")
    copyfile(logging_one_file_address, user_log_file_name)
