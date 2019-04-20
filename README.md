# BayesianEstimationInNLP
A frame work to do Bayesian hypothesis assessment for comparing the performences of two algorithms.

Before running the codes, make sure you have all the requirements by running the following line:
```bash
pip install -r requirements.txt
``` 

Prepare your experiment results to run the analysis on. To make our fake inputs, you can run the following:

```bash
python makeData.py
```

To run the main analysis you need to give the addresses of two input files and an output folder. For example:
```bashmodule 'pymc3' has no attribute 'model_to_graphviz'
python main.py --file1 ./fakeExperimentResults/PositiveInp_0.csv --file2 ./fakeExperimentResults/PositiveInp_1.csv --output-prefix ./posOutput/Positive
```

And here is the command template
```bash
usage: main.py [-h] -f1 FILE1 -f2 FILE2 -o OUTPUT_PREFIX

Run Bayesian Stat

optional arguments:
  -h, --help            show this help message and exit
  -f1 FILE1, --file1 FILE1
                        Performances of first Algorithm
  -f2 FILE2, --file2 FILE2
                        Performances of second Algorithm
  -o OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
                        Where to put output
``` 