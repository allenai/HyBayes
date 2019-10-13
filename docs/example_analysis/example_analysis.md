## Artificial Examples
Current versioin of the package provides 5 pre-implimented models:
- Binary observations: Bernoulli distribution with Beta prior
- Multiple Binary observations: Binomial distribution with Beta prior
- Metric observations: T-Student distribution with muliple priors
- Count observations: Negative Binomial distribution with Normal prior
- Ordinal observations: Normal distribution with variable tresholds

These models capture a lot of common assumption on observed data. Note that if you have specific information on your observation or other assumptions, it is highly recommended that you add your costum model. If you have model in mind that is a general model and other researchers (expecially within NLP community) are likely to need, feel free to ask us to add to the pakcage.


### Metric observations: T-Student distribution with multiple choices for priors

For this model, you can indicate the "HistogramPlot" and "scatterPlot" to view a visualisation of the input. For our contrived data, we get the Figures 1 and 2.


![Histogram plot](outputs/Metric_HistogramPlot.png)

![Scatter plot](outputsForManual/Metric/Metric_scatterPlot.png)

#### Model
To indicate this model in the config file, it is enough to set the arguments "VariableType" to "Metric" in the "[Model]" section of the config file.

By indicating this model, the observation of two groups are assumed to follow two separate T-student distributions with parameters $$(mu, sigma)$$, corresponding to each group, and a share parameter nu. Each mu parameter is the indicator of overall performance of corresponding group, whereas sigma is how dispersed the value of the groups are and nu indicates how close the given distribution is to a normal distribution.  Thus the distribution of $$mu_1-mu_2$$ indicates how superior group one is over group two. 

***
In a higher level, each mu is assumed to follow a wide normal distribution with identical parameters. Each sigma follows a wide uniform distribution and the shared parameter nu follows an exponential distribution. 

See a visualization of this model in Following Figure:
![Hierarchical model.](outputsForManual/Metric/Metric_posterior_hierarchicalGraph.png)

To check the effect of this model. One can see the Prior of distribution of each five parameters and their differences in the following Figure: 
![Mu Prior.](outputsForManual/Metric/Metric_prior_mu.png)
![Sigma Prior.](outputsForManual/Metric/Metric_prior_sigma.png)



Notice that before taking the observed data into account, our prior knowledge, in this case, is formalized as any reasonable value for each mu is almost equally likely. Since the domain of mu is the whole R, we can not assume an exactly uniform prior on it.

The package also combines the difference plots with the plot for nu in the following Figure:
![All prior.](outputsForManual/Metric/Metric_prior_allComp.png)
#### Posterior Plots
The main output of the analysis is the following Figure.

![Mu Posterior.](outputsForManual/Metric/Metric_posterior_mu.png)


The information that can be read from this plot includes:
- XX


Similar to prior, we get more plots for the posterior too:
![Sigma posterior.](outputsForManual/Metric/Metric_posterior_sigma.png)
![All posterior.](outputsForManual/Metric/Metric_posterior_allComp.png)

#### Diagnostic Plots
Since this package is based on MCMC sampling methods for infering the posteriour, it is important to make sure the sampling process has been done with sufficient granularity. For this purpose you can investigate the diagnastic plots produced by Pymc3:

![Prior Diagnostic Plot.](outputsForManual/Metric/Metric_prior_diagnostics.png)
![Posterior Diagnostic Plot.](outputsForManual/Metric/Metric_posterior_diagnostics.png)

Notice that different chains for each parameter as converged to one distribution.

The parameters affecting granularity of the analysis can be indicated in sections "[Prior]" and "[Posteriour]" in the config file. Especially, the following three parameters are the most important ones:
- "Tune": number of samples to throw away in the beginning. A value of at least 500 is recommended.
- "Chains": number independent chains. Four chains seem to be sufficient to confirm successful convergence of the chains.
- "Draws": This is the number of samples used to plot the distribution. The higher this value, the smooth the plot will look like. Also you can consult the Effective Sample Size (ESS) printed in the log.

#### Bayes Factor (BF)
The package also outputs the Bayes Factor for several lengths of intervals around the ROPE.



### Binary observations: Bernoulli distribution with Beta prior
For this model, you can indicate the "countPlot" and "barPlot" to view a visualisation of the input. For our contrived data, we get the Figures 1 and 2.

![Bar plot](outputsForManual/Binary/Bern_barPlot.png)

![Count plot](outputsForManual/Binary/Bern_countPlot.png)

#### Model
To indicate this model in the config file, it is enough to set the arguments "VariableType" and "PriorModel" to "Binary" and "Beta" in the "[Model]" section of the config file.

By indicating this model, the observations of two groups are assumed to be follow two separate Bernoulli distributions with parameters $$theta_1$$ and $$theta_2$$, corresponding to each group. Each parameter is the indicator of overall performance of corresponding group. Thus the distribution of $$theta_1-theta_2$$ indicates how superior group one is over group two. 


In a higher level, these two parameters are assumed to follow two Beta distribution with identical parameters. The parameters of this Beta distribution, i.e., priors of thetas, can be indicated with "Beta_a" and "Beta_b" in the "[Model]" section of the config file. Note setting both these parameters two one will result in a uniform prior.

See a visualization of this model in Following Figure:
![Hierarchical model.](outputsForManual/Binary/Bern_posterior_hierarchicalGraph.png)

To check the effect of this model. One can see the Prior of distribution of each theta and their difference in the following Figure: 
![Theta prior.](outputsForManual/Binary/Bern_prior_theta.png)

Notice that before taking the observed data into account, our prior knowledge, in this case, is formalized as any value between zero and one for each theta is equally likely.
#### Posterior Plots
The main output of the analysis is the following Figure.

![Posterior Theta.](outputsForManual/Binary/Bern_posterior_theta.png)

Notice that the mode of the the difference of the distribution is at $$-0.238$$, also known as Maximum A priori Estimate, the %95-HDI quantifies the uncertainty around this mode, which is the main goal of this analysis.

Other information that can be read from this plot includes:
- The probability that group 2 is superior to group 1, i.e., $$Theta_2>Theta_1$$ is at least $$95%$$.
- 
#### Diagnostic Plots
Since this package is based on MCMC sampling methods for inferring the posterior, it is important to make sure the sampling process has been done with sufficient granularity. For this purpose you can investigate the diagnostic plots produced by Pymc3:

![Prior Diagnostic plot.](outputsForManual/Binary/Bern_prior_diagnostics.png)
![Posterior Diagnostic plot.](outputsForManual/Binary/Bern_posterior_diagnostics.png)

Notice that different chains for each parameter as converged to one distribution.

The parameters affecting granularity of the analysis can be indicated in sections "[Prior]" and "[Posteriour]" in the config file. Especially, the following three parameters are the most important ones:
- "Tune": number of samples to throw away in the beginning. A value of at least 500 is recommended.
- "Chains": number independent chains. Four chains seem to be sufficient to confirm successful convergence of the chains.
- "Draws": This is the number of samples used to plot the distribution. The higher this value, the smooth the plot will look like. Also you can consult the Effective Sample Size (ESS) printed in the log.

#### Bayes Factor (BF)
The package also outputs the Bayes Factor for several lengths of intervals around the ROPE.

### Multiple Binary observations: Binomial distribution with Beta prior
Ungoing work!


### Count observations: Negative Binomial distribution with Normal prior
Ungoing work!

### Ordinal observations: Normal distribution with variable tresholds
Ungoing work!

## Adding a new Model
Ungoing work!


