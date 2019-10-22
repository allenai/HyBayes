import argparse
import configparser
import copy
import os

from .utils import *

script_address = os.path.abspath(__file__)
config_folder_name = "configs"
config_template_address = script_address.replace("make_configs.py", "config_template.ini")
artificial_data_folder_name = "artificial_data"


def read_template_config(quick: bool = False) -> configparser.ConfigParser:
    """
    Returns config object with default values
    :param quick: if True only posterior analysis with short sample length will be on.
    :return:
    """
    config = configparser.ConfigParser()
    config.read(config_template_address)
    if quick:
        plot_stage = "Prior"
        config[plot_stage]["analyze"] = "False"

        plot_stage = "Posterior"
        config[plot_stage]["draws"] = "100"
        config[plot_stage]["chains"] = "2"
        config[plot_stage]["tune"] = "10"

    return config


def make_metric_config(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """
    Returns a config object corresponding to the metric observations in artificial data folder
    :param config:
    :return:
    """
    config = copy.copy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Positive_real_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Positive_real_data_1.csv"
    config["Files"]["Output_prefix"] = "metric_experiment_files/Metric"

    config["Plots"]["Scatter_plot"] = "True"
    config["Plots"]["Histogram_plot"] = "True"

    config["Model"]["Variable_type"] = "Metric"
    config["Model"]["mu_ROPE_begin"] = "-0.1"
    config["Model"]["mu_ROPE_end"] = "0.1"

    # config["Model"]["UnitInterval"] = "False" # UnitInterval is not implemented yet

    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["SD_plot"] = "True"
        config[plot_stage]["Show_SD_plot"] = "False"
        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Show_Mean_plot"] = "False"
        config[plot_stage]["Effective_sample_size_plot"] = "True"
        config[plot_stage]["Show_Effective_sample_size_plot"] = "False"

    return config


def make_count_config(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """
    Returns a config object corresponding to the count observations in artificial data folder
    :param config: a input config object with default values
    :return:
    """
    config = copy.copy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Count_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Count_data_1.csv"
    config["Files"]["Output_prefix"] = "count_experiment_files/Count"

    config["Plots"]["Scatter_plot"] = "True"
    config["Plots"]["Count_plot"] = "True"
    config["Plots"]["kind"] = "kde"

    config["Model"]["Variable_type"] = "Count"
    config["Model"]["mu_ROPE_begin"] = "-0.1"
    config["Model"]["mu_ROPE_end"] = "0.1"

    # config["Model"]["Zero_inflation"] = "False" # Zero_inflation is not implemented yet

    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["sigma_plot"] = "True"
        config[plot_stage]["Show_sigma_plot"] = "False"

        config[plot_stage]["mu_plot"] = "True"
        config[plot_stage]["Show_mu_plot"] = "False"

        config[plot_stage]["alpha_plot"] = "True"
        config[plot_stage]["Show_alpha_plot"] = "False"

        config[plot_stage]["skewness_plot"] = "True"
        config[plot_stage]["Show_skewness_plot"] = "False"

    return config


def make_ordinal_config(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """
    Returns a config object corresponding to the ordinal observations in artificial data folder
    :param config: a input config object with default values
    :return:
    """
    config = copy.copy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Ordinal_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Ordinal_data_1.csv"
    config["Files"]["Output_prefix"] = "ordinal_experiment_files/Ordinal"

    config["Plots"]["Scatter_plot"] = "True"
    config["Plots"]["Count_plot"] = "True"
    config["Plots"]["kind"] = "kde"
    
    config["Model"]["Variable_type"] = "Ordinal"
    config["Model"]["Min_level"] = "0"
    config["Model"]["Max_level"] = "2"
    config["Model"]["mu_ROPE_begin"] = "-0.1"
    config["Model"]["mu_ROPE_end"] = "0.1"

    # config["Model"]["Zero_inflation"] = "False" # Zero_inflation is not implemented yet

    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["sigma_plot"] = "True"
        config[plot_stage]["Show_sigma_plot"] = "False"

        config[plot_stage]["mu_plot"] = "True"
        config[plot_stage]["Show_mu_plot"] = "False"

    return config


def make_binary_config(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """
    Returns a config object corresponding to the binary observations in artificial data folder
    :param config: a input config object with default values
    :return:
    """
    config = copy.copy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Binary_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Binary_data_1.csv"
    config["Files"]["Output_prefix"] = "binary_experiment_files/Binary"

    config["Plots"]["Count_plot"] = "True"
    config["Plots"]["Bar_plot"] = "True"

    config["Model"]["Variable_type"] = "Binary"
    config["Model"]["Prior_model"] = "Beta"
    config["Model"]["Beta_a"] = "1"
    config["Model"]["Beta_b"] = "1"
    config["Model"]["theta_ROPE_begin"] = "-0.1"
    config["Model"]["theta_ROPE_end"] = "0.1"


    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Show_Mean_plot"] = "False"

    return config


def make_binomial_config(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """
    Returns a config object corresponding to the binomial observations in artificial data folder
    :param config: a input config object with default values
    :return:
    """
    config = copy.copy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Binomial_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Binomial_data_1.csv"
    config["Files"]["Output_prefix"] = "binomial_experiment_files/Binomial"

    config["Plots"]["Histogram_plot"] = "True"

    config["Model"]["Variable_type"] = "Binomial"
    config["Model"]["Prior_model"] = "Beta"
    config["Model"]["Beta_a"] = "1"
    config["Model"]["Beta_b"] = "1"
    config["Model"]["theta_ROPE_begin"] = "-0.1"
    config["Model"]["theta_ROPE_end"] = "0.1"

    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Show_Mean_plot"] = "False"

    return config


def write_config(config: configparser.ConfigParser, file_name: str) -> None:
    """
    Stores the information in config object in file_name.
    :param config:
    :param file_name:
    :return:
    """
    address = os.path.join(config_folder_name, file_name)
    with open(address, "w") as write_file:
        config.write(write_file)
        print(f"Writing config file: {address}")

def run_all(quick=False):
    logs_folder_made = mk_dir_if_not_exists(config_folder_name)
    if logs_folder_made:
        print(f"The directory {config_folder_name} is created.")
    write_config(make_binary_config(read_template_config(quick)), "config_binary.ini")
    write_config(make_binomial_config(read_template_config(quick)), "config_binomial.ini")
    write_config(make_count_config(read_template_config(quick)), "config_count.ini")
    write_config(make_metric_config(read_template_config(quick)), "config_metric.ini")
    write_config(make_ordinal_config(read_template_config(quick)), "config_ordinal.ini")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Make example config files for running artificial files")
    parser.add_argument("-q", "--quick", help="With this flag the configs only include posterior and with low values"
                                              " for draw, cains, and tune", action="store_true", default=False)
    args = parser.parse_args()
    run_all(args.quick)



