# User Manual for BayesianEstimationInNLP
A Bayesian hypothesis assessment framework, tailored to comparing the performances of algorithm pairs.

## Getting started
For running this code, you will need python at least 3.6 and the probabilistic programming package `pymc3/3.6`.
Before running the code, make sure you have all the requirements by running the following line:
```bash
pip install -r requirements.txt
``` 

## Preparing Artificial Data
For purposes of this manual, we make some artificial data. If you already have your data (as observations) from your experiments, you can skip this step.

Use the following command to  generate the artificial data: 

```bash
python makeData.py
```

## Prepare configuration file
To use this package to analyze your data, you need to prepare the a configuration file that indicates the information needed for the analysis, including the type of plots and files you want to be stored.

The following shows the general framework of a configuration file. The main sections are Files, Plots, Model, Prior, and Posterior. You can find examples of complete configurations in 'configs' folder.

Here we explain the first section 'Files': For now, all the experiments use one column observation files except when the observations are assumed to follow Binomial distribution.
Second and third lines indicate the names and the addresses of two files. The last line indicates a prefix of the output file names. One can make a directory beforehand include in this prefix.  
```bash
[Files]
NumberOfColumns = 1
File1 = ArtificialData/BinaryData_0.csv
File2 = ArtificialData/BinaryData_1.csv
OutputPrefix = experimentFiles/Bern

[Plots]
...

[Model]
VariableType = Binary
...

[Prior]
Analyze = True
...

[Posterior]
Analyze = True
...
```

## Running the analysis 
We assume you have already put the observations of the performances two algorithms in two separate files and indicate the information in a config file. To run the analysis, you pass the name of the config file as argument. For example:
```bash
python main.py --config configs/configBinary.ini --verbose
```

When the flag '--verbose' is one, the more detailed information of the steps of the analysis will be printed in standard output. This flag does not affect the log file that is stored along side other outputs.

And here is the general usage template which can be accessed using '--help' flag at any time:
```bash
usage: main.py [-h] [-c CONFIG] [-v]

Run Bayesian Stat

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        address of Config file
  -v, --verbose         prints the report of the steps

``` 

## Output files
Depending on the types of analyses the user asks, via configuration file, a subset of following files or folders might be stored as the output of the analysis. Note that all the files names below will come after a prefix for the whole analysis.

- **posterior_trace** : A directory where all the samples taken using pymc3 engine for the posterior distribution is stored for further analysis. In literature these samples can be refered to as 'trace', 'chain', or 'samples'. 
For regular usages, you don't need to touch these files. However for long chains, this might save time for future analyses not to run the sampling process again.
It also highly recommended that the files in this directory be made available for public to investigate and run their own analysis. 

- **prior_trace** : A directory where all the samples taken using pymc3 engine for the [prior] distribution is stored for further analysis. See [posterior_trace] for more info.

- **config.ini** : The user's desired configuration file is also copied alongside other output files, for reproducibility and ease of future references.

- **log.log** : The log of the analysis is stored in this file for debugging purposes.

- **posterior_hierarchicalGraph.png** : A pictorial representation of the hierarchical model used in the analysis. This file is made using the  'model_to_graphviz' in 'pymc3' directly. To understand the notation refer to their docs.