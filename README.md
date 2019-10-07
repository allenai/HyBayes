# Bayesian Hypothesis Assessment
A Bayesian hypothesis assessment framework, tailored to comparing pairs of experiments. This package is preliminary 
implementation statistical analyses described in [PAPER LINK](?). This can also be seen as a complimentary to  
[testSignificanceNLP](https://github.com/rtmdrr/testSignificanceNLP) package which contains popular Frequentist tests. 

We focus on the setup where the performances of two algorithms, applied on a certain dataset, are provided based on an 
evaluation measure. The comparison is done by studying the posterior probabilities (rather than binary commonly-used binary decisions in other tests.)

For further discussion of the approaches implemented here (and their comparison to other techniques), refer to [PAPER LINK](?). 


## Getting started
### Installation
For running this code, you will need Python (version >=3.6).
You have two options for installation: 
 - If you don't want to modify any code, you can just install the package and use it in commandline: 
```
 > pip install ????
```

 - Alternatively, you can clone this project (by running `git clone git@github.com:turkerfan/BayesianEstimationInNLP.git`.) 
 Before running the code, make sure that you have all the requirements by 
 running the following line to install all the necessary dependencies: 
```bash
pip install -r requirements.txt
``` 


If you're using MacOS, install GraphViz via `brew` command: 
```
> brew install graphviz
```

### Preparing configuration file
To use this package for analyzing your data, you need to prepare a configuration file that indicates the information 
needed for the analysis, including the type of plots and files you want to be stored.

The following snippet shows the general structure of a configuration file. The main sections are `Files`, `Plots`, `Model`, 
`Prior`, and `Posterior`. You can find examples of complete configurations in [`configs`](configs) folder.

The first section of the configuration file is `Files` and it indicates the details of the files which contain 
your experimental result. All the experiments use single column observation files (i.e., `NumberOfColumns = 1`) except 
when the observations are assumed to follow Binomial distribution ([read more here]() **<-- correct the link**).
Second and third lines indicate the names and the location of two files (which inlucde the result of your experiments). 
The results of the analaysis will be stored in the address specified by `OutputPrefix`.   
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

## Running the Analysis 
Let's assume that you have followed the above instructions and:  
 - (1) you have included the performances observations in two separate files,  
 - (2) and you have prepared a config file. 

To run the analysis, you pass the name of the config file as argument: 
```bash
python main.py --config myConfigFile.ini --verbose
```

When the flag `--verbose` is on, the more detailed information of the steps of the analysis will be printed in standard output. This flag does not affect the log file that is stored along side other outputs.

And here is the general usage template which can be accessed using `--help` flag at any time:
```bash
usage: main.py [-h] [-c CONFIG] [-v]

Run Bayesian Stat

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        address of Config file
  -v, --verbose         prints the report of the steps

```

### A Quick Example 
To demonstrate everything with a quick example, run the following command to generate toy data: 
```bash
python makeData.py
```

And then execute the analysis on that: 
```bash
python main.py --config configs/configBinary.ini --verbose
```

## Further Reading
If you want to learn more about this package, please refer to the [extended manual](docs/MANUAL.md). 
If you want to learn about the concepts discusssed here, refer to the paper below. 

## Citation: 
If you used this package in your research feel free to cite the following paper: 
```
TODO
```
