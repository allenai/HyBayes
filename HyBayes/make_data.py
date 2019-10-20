import numpy as np

from .utils import *


def get_positive_input(n1=40, n2=30, mean_distance=0.8):
    """
    Constructs two groups with n1 observations in first and n2 observations in second.
    :param n1:
    :param n2:
    :param mean_distance: distance of means in two gorups
    :return:
    """
    y = [2 * np.random.normal(0, 1, size=n1) + 100 - mean_distance,
         2 * np.random.normal(0, 1, size=n2) + 100 + mean_distance]
    return y


def get_binary_input(n1=25, a1=12, n2=28, a2=20):
    x1 = np.zeros(shape=n1, dtype=np.int)
    x1[:a1] = 1

    x2 = np.zeros(shape=n2, dtype=np.int)
    x2[:a2] = 1

    return [x1, x2]


def save_input(input_function, file_name_prefix, folder_name):
    y = input_function()
    form = "%d" if y[0].dtype == np.int else "%f"
    for i in range(2):
        np.savetxt(f"./{folder_name}/{file_name_prefix}_{i}.csv", y[i], fmt=form)


def get_count_input(n1=67, lambda1=11.1, n2=76, lambda2=9.9):
    x1 = np.random.poisson(lam=lambda1, size=n1)
    x2 = np.random.poisson(lam=lambda2, size=n2)
    return [x1, x2]


def get_ordinal_input(f1=(10, 9, 20), f2=(18, 9, 7)):
    x1 = np.zeros(shape=sum(f1), dtype=np.int)
    for i in range(1, len(f1)):
        begin = sum(f1[:i])
        x1[begin: (begin + f1[i])] = i

    x2 = np.zeros(shape=sum(f2), dtype=np.int)
    for i in range(1, len(f2)):
        begin = sum(f2[:i])
        x2[begin: (begin + f2[i])] = i
    return [x1, x2]


def get_binomial_input(n1=20, p1=0.44, n2=25, p2=0.52):
    x1 = np.zeros(shape=(n1, 2), dtype=np.int)
    x1[:, 0] = np.random.poisson(lam=4, size=n1)
    x1[:, 1] = np.random.binomial(n=x1[:, 0], p=p1)

    x2 = np.zeros(shape=(n2, 2), dtype=np.int)
    x2[:, 0] = np.random.poisson(lam=4, size=n2)
    x2[:, 1] = np.random.binomial(n=x2[:, 0], p=p2, )
    return [x1, x2]

def run_all():
    folder_name = "artificial_data"
    mk_dir_if_not_exists(folder_name)

    save_input(get_binary_input, file_name_prefix="Binary_data", folder_name=folder_name)
    save_input(get_positive_input, file_name_prefix="Positive_real_data", folder_name=folder_name)
    save_input(get_count_input, file_name_prefix="Count_data", folder_name=folder_name)
    save_input(get_ordinal_input, file_name_prefix="Ordinal_data", folder_name=folder_name)
    save_input(get_binomial_input, file_name_prefix="Binomial_data", folder_name=folder_name)

if __name__ == '__main__':
    run_all()
