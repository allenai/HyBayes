# User Manual for BayesianEstimationInNLP
A Bayesian hypothesis assessment framework, tailored to comparing the performances of algorithm pairs. This package is preliminary implementation statistical analyses described in [OUR Paper].
Also this package can be seen as a complimentary package to [https://github.com/rtmdrr/testSignificanceNLP] where the frequentest approaches are implemented. We followed the simple file format that the previous package required.

Similar to the previous package we focus on the setup where the performance of two algorithms, A and B, applied on a dataset X, is compared based on an evaluation measure M. Although here, when making such comparison, we study the posterior probabilities instead of performing a significance test.   
In the paper, we provide a comparison these two categories of approaches and advocate for including the Bayesian analysis (weather along with or in substitution to frequentest approaches).
Both these approaches help a researcher claim superiority of one algorithm over another from different aspects.

If you used this package in your research feel free to cite: [XX our paper]


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
    - https://graphviz.readthedocs.io/en/stable/manual.html
    - https://github.com/pymc-devs/pymc3/blob/master/pymc3/model_graph.py
- **posterior_hierarchicalGraph.txt** : A textual representation of the hierarchical model corresponding to [posterior_hierarchicalGraph.png].
- **posterior_diagnostics.png** and **prior_diagnostics.png** : The diagnostics plots drawn using [https://docs.pymc.io/api/diagnostics.html]. It is recommended to include these plots in the supplementary material section of the paper to convince the reader that the sampling process was successful.
- **countPlot.png**, **barPlot**, **histogramPlot**, or **scatterPlot**: These plots some basic visualizations of the raw data. Often these plots help match the result of Bayesian analysis to the given data.  
- 
 
 
