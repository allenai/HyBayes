import os
import configparser
import copy

config_folder_name = "configs"
config_template_file_name = "config_template.ini"
artificial_data_folder_name = "artificial_data"


def make_metric_config(config: configparser.ConfigParser):
    config = copy.deepcopy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Positive_real_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Positive_real_data_1.csv"
    config["Files"]["Output_prefix"] = "metric_experiment_files/Metric"

    config["Plots"]["Scatter_plot"] = "True"
    config["Plots"]["Histogram_plot"] = "True"

    config["Model"]["Variable_type"] = "Metric"
    # config["Model"]["UnitInterval"] = "False" # UnitInterval is not implemented yet

    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["SD_plot"] = "True"
        config[plot_stage]["SD_plot_kind"] = "hist"
        config[plot_stage]["Show_SD_plot"] = "False"
        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Mean_plot_kind"] = "hist"
        config[plot_stage]["Show_Mean_plot"] = "False"

    return config


def make_count_config(config: configparser.ConfigParser):
    config = copy.deepcopy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Count_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Count_data_1.csv"
    config["Files"]["Output_prefix"] = "count_experiment_files/Count"

    config["Plots"]["Scatter_plot"] = "True"
    config["Plots"]["Count_plot"] = "True"

    config["Model"]["Variable_type"] = "Count"
    # config["Model"]["Zero_inflation"] = "False" # Zero_inflation is not implemented yet

    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["SD_plot"] = "True"
        config[plot_stage]["SD_plot_kind"] = "hist"
        config[plot_stage]["Show_SD_plot"] = "False"

        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Mean_plot_kind"] = "hist"
        config[plot_stage]["Show_Mean_plot"] = "False"

        config[plot_stage]["Effect_size_plot "] = "True"
        config[plot_stage]["Effect_size_plot_kind"] = "hist"
        config[plot_stage]["Show_Effect_size_plot"] = "False"

    return config


def make_ordinal_config(config: configparser.ConfigParser):
    config = copy.deepcopy(config)
    config["Files"]["Number_of_columns"] = "1"
    config["Files"]["Number_of_files"] = "2"
    config["Files"]["File1"] = f"{artificial_data_folder_name}/Ordinal_data_0.csv"
    config["Files"]["File2"] = f"{artificial_data_folder_name}/Ordinal_data_1.csv"
    config["Files"]["Output_prefix"] = "ordinal_experiment_files/Ordinal"

    config["Plots"]["Scatter_plot"] = "True"
    config["Plots"]["Count_plot"] = "True"

    config["Model"]["Variable_type"] = "Ordinal"
    config["Model"]["Min_level"] = "0"
    config["Model"]["Max_level"] = "2"
    # config["Model"]["Zero_inflation"] = "False" # Zero_inflation is not implemented yet

    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["SD_plot"] = "True"
        config[plot_stage]["SD_plot_kind"] = "hist"
        config[plot_stage]["Show_SD_plot"] = "False"

        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Mean_plot_kind"] = "hist"
        config[plot_stage]["Show_Mean_plot"] = "False"

        config[plot_stage]["Effect_size_plot "] = "True"
        config[plot_stage]["Effect_size_plot_kind"] = "hist"
        config[plot_stage]["Show_Effect_size_plot"] = "False"

    return config


def make_binary_config(config: configparser.ConfigParser):
    config = copy.deepcopy(config)
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


    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["Effect_size_plot "] = "True"
        config[plot_stage]["Effect_size_plot t_kind"] = "hist"
        config[plot_stage]["Show_Effect_size_plot "] = "False"
        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Mean_plot_kind"] = "hist"
        config[plot_stage]["Show_Mean_plot"] = "False"

    return config


def make_binomial_config(config: configparser.ConfigParser):
    config = copy.deepcopy(config)
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


    for plot_stage in ["Prior", "Posterior"]:
        config[plot_stage]["Effect_size_plot "] = "True"
        config[plot_stage]["Effect_size_plot t_kind"] = "hist"
        config[plot_stage]["Show_Effect_size_plot "] = "False"
        config[plot_stage]["Mean_plot"] = "True"
        config[plot_stage]["Mean_plot_kind"] = "hist"
        config[plot_stage]["Show_Mean_plot"] = "False"

    return config



def write_config(config, file_name):
    address = os.path.join(config_folder_name, file_name)
    with open(address, "w") as write_file:
        config.write(write_file)
        print(f"Writing config file: {address}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join(config_folder_name, config_template_file_name))
    print(f"Reading config file: {config_template_file_name}")

    write_config(make_binary_config(config), "new_binary.ini")
    write_config(make_binomial_config(config), "new_binomial.ini")
    write_config(make_count_config(config), "new_count.ini")
    write_config(make_metric_config(config), "new_metric.ini")
    write_config(make_ordinal_config(config), "new_ordinal.ini")


