import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('root')


def estimate_interval_prob(trace, parameter: str, interval_begin: float, interval_end: float):
    """
    Estimating probability of an interval, used in calculation of Bayes_factor for a specific parameter
    :param trace: an object containing the samples, i.e., output of pymc3's sampling
    :param parameter: (str) the parameter of interest for calculating Bayes Factor,
                        most commonly mu or any centrality parameter.
    :param interval_begin: (float)
    :param interval_end: (float)
    """
    diff = trace[parameter][:, 0] - trace[parameter][:, 1]
    numerator = np.logical_and(diff > interval_begin, diff < interval_end).sum()
    denominator = diff.size
    return numerator, denominator


def bayes_factor_analysis(config_bayes_factor, prior_model, post_model, init_rope=(-0.1, 0.1)):
    """
    Returns the number of samples needed for calculating Bayes Factor in a pandas dataframe form.
        If config_bayes_factor["number_of_smaller_ropes"] is greater than 1, several values of ROPE
        will be considered starting with init_rope, otherwise only init_rope.
    :param config_bayes_factor: config object with number_of_smaller_ropes, rope_begin, and rope_end.
    :param prior_model:
    :param post_model:
    :param init_rope:
    :return:
    """
    column_names = ["ROPE", "priorProb", "postProb",
                    "BF", "BF_Savage_Dickey",
                    "prioNSample", "postNSample"]

    df = pd.DataFrame(columns=column_names)
    rope = np.array(init_rope)
    n = config_bayes_factor.getint("number_of_smaller_ropes")
    if n is None:
        n = 1
    for i in range(n):
        prior_rope_prob_frac = estimate_interval_prob(prior_model.trace, prior_model.mu_parameter, rope[0], rope[1])
        post_rope_prob_frac = estimate_interval_prob(post_model.trace, post_model.mu_parameter, rope[0], rope[1])
        if prior_rope_prob_frac[0] <= 0 or post_rope_prob_frac[0] <= 0:
            break
        prior_rope_prob = prior_rope_prob_frac[0] / prior_rope_prob_frac[1]
        post_rope_prob = post_rope_prob_frac[0] / post_rope_prob_frac[1]
        # BF_Savage_Dickey = postRopeProbFrac[0] * priorRopeProbFrac[1] / (postRopeProbFrac[1] * priorRopeProbFrac[0])

        # Savage_Dickey estimate of Bayes Factor:
        BF_Savage_Dickey: float = post_rope_prob / prior_rope_prob

        # True value (modulo sampling) of Bayes Factor
        bf = BF_Savage_Dickey * (1 - prior_rope_prob) / (1 - post_rope_prob)
        row = {
            column_names[0]: rope,
            column_names[1]: prior_rope_prob,
            column_names[2]: post_rope_prob,
            column_names[3]: BF_Savage_Dickey,
            column_names[4]: bf,
            column_names[5]: prior_rope_prob_frac[0],
            column_names[6]: post_rope_prob_frac[0],
        }
        # logger.debug(row)
        df = df.append(row, ignore_index=True)
        rope = rope / 1.2
    logger.info("************************* Bayes Factor Analysis *********************************")
    logger.info(f"Estimated Bayes factor corresponding to ROPE = {df['ROPE'][0]} is equal to {df['BF'][0]}")
    logger.info("To get a trusted estimation, make sure the number of draws is high enough"
                " and multiple runs of the software give similar estimate")
    logger.info("To interpret the Bayes factor refer to the table in "
                "https://www.r-bloggers.com/what-does-a-bayes-factor-feel-like/")
    if df.shape[0] > 1:
        logger.info("Here are values for Bayes factors for other smaller ROPE intervals:")
        logger.info(df[["ROPE", "BF"]])
    logger.info("*********************************************************************************")
    return df
