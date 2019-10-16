# Bayesian Hypothesis Assessment
A Bayesian hypothesis assessment framework, tailored to comparing pairs of experiments. This package is preliminary 
implementation statistical analyses described in [this paper](#citation). This can also be seen as a complimentary to  
[testSignificanceNLP](https://github.com/rtmdrr/testSignificanceNLP) package which contains popular Frequentist tests. 

We focus on the setup where the performances of two algorithms, applied on a certain dataset, are provided based on an 
evaluation measure. The comparison is done by studying the posterior probabilities (rather than binary commonly-used binary decisions in other tests.)

For further discussion of the approaches implemented here (and their comparison to other techniques), refer to [the paper](#citation) or the extended [manual](docs/MANUAL.md). 


## Getting started
### Installation
For running this code, you will need **Python (version >=3.6)**.
You have two options for installation: 
 - If you don't want to modify any code, you can just install the package and use it in commandline: 
```bash
 > pip install HyPyThesis
```

 - Alternatively, if you'd like to make modifications to the code (e.g., the underlying model) you can clone this project. Before running the code, make sure that you have all the requirements by 
 running the following line to install all the necessary dependencies: 
```bash
 > pip install -r requirements.txt
``` 


If you're using MacOS, install GraphViz via `brew` command: 
```bash
 > brew install graphviz
```

### Preparing configuration file
To analyze your data, you need to prepare the a configuration file that specifies the information needed for the analysis (e.g., the address to your observation files, the type of plots and files you want to be stored).

You can find more details on this in the [extended manual](docs/MANUAL.md). Additionally, there are examples of complete configurations in [`configs`](configs) folder, after running `make_configs.py`.

## Running the Analysis 
Let's assume that you have followed the above instructions and:  
 - (1) you have included the performances observations in two separate files,  
 - (2) and you have prepared a config file. 

To run the analysis, you pass the name of the config file as argument: 
```bash
 > python main.py --config myConfigFile.ini --verbose
```

When the flag `--verbose` is on, the details of the analysis will be printed in standard output. 
This flag does not affect the log file that is stored along side other outputs.

And here is the general usage template which can be accessed using `--help` flag at any time:
```bash
usage: main.py [-h] [-c CONFIG] [-v]

Run Bayesian Statistics Tailored towards analysing the experiment results
specially in NLP area. Email @.com for comments.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        address of Config file
  -v, --verbose         prints the report of the steps

```

### Examples 
To demonstrate everything with a quick example, run the following command to generate toy data: 
```bash
 > python make_data.py
 > python make_configs.py
```

And then execute the analysis on any of the newly made config files. For example: 
```bash
 > python main.py --config configs/config_metric.ini --verbose
```

## Further Reading
If you want to learn more about this package, please refer to the [extended manual](docs/MANUAL.md). 
If you want to learn about the concepts discusssed here, refer to the paper below. 

## License 
This code is published under [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.  


## Citation 
If you used this package in your research feel free to cite the following paper: 
```
@article{peng2019solving,
  title={??},
  author={Sadeqi Azer, Erfan and Khashabi, Daniel and Sabharwal, Ashish and Roth, Dan},
  journal={??},
  year={2019}
}
```
