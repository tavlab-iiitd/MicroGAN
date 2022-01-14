from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import torch
import os
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import random
from time import time


def main(filename, doPca, plot_AIC_BIC):
    if not plot_AIC_BIC:
        df = pd.read_csv(filename).iloc[:, 1:]
        cols = df.columns
        data = df.values
        if doPca:
            pca = PCA(0.99)
            data = pca.fit_transform(data)
        model = GaussianMixture(5, covariance_type="full", random_state=0).fit(data)
        model.fit(data)
        print("Model Covergence", model.converged_)
        for s_size in [len(data), 250, 550]:
            data_new = model.sample(s_size)
            if doPca:
                digits_new = pca.inverse_transform(data_new[0])
            else:
                digits_new = data_new[0]
            df_to_save = pd.DataFrame(digits_new, columns=cols, index=None)
            df_to_save.to_csv(filename[:-4] + "_gen_{}.csv".format(s_size))
    else:
        df = pd.read_csv(filename).iloc[:, 1:]
        cols = df.columns
        data = df.values
        if doPca:
            pca = PCA(0.99)
            data = pca.fit_transform(data)
        n_components_range = np.arange(1, 21)
        models = [
            GaussianMixture(n_components=n, covariance_type="full", random_state=0).fit(
                data
            )
            for n in n_components_range
        ]
        plt.plot(n_components_range, [m.bic(data) for m in models], label="BIC")
        plt.plot(n_components_range, [m.aic(data) for m in models], label="AIC")
        plt.legend(loc="best")
        plt.xlabel("n_components")
        plt.savefig("AIC_BIC_plot_{}.png".format(filename[:-4]))


if __name__ == "__main__":
    allfiles = sorted(glob.glob(os.path.join("../..", "saad18409", "Group*Data.csv")))
    allfiles = [allfiles[0], allfiles[1]]
    for root in allfiles:
        print("-------- ", root, " --------")
        main(root, True, False)  # doPCA is true for Non-Axes data and vice-versa
        print("-" * (len(root) + 18))
