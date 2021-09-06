from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import glob
from torch import autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import torch
from scipy.stats import entropy
import torch
import os
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import random
from time import time


def prep(data):
    if "Unnamed: 0" in data.columns.values:
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    if "Unnamed: 0.1" in data.columns.values:
        data.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    if "Unnamed: 0.1.1" in data.columns.values:
        data.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def calculate_wstd(org, gen):
    prep(org)
    prep(gen)
    wsd = 0
    orig_data = org.values
    gen_data = gen.values
    # if orig_data.shape[1] != gen_data.shape[1]:
    #    pca = PCA(n_components=gen_data.shape[1])
    #    pca = pca.fit(orig_data)
    #    gen_data = pca.inverse_transform(gen_data)
    columns = orig_data.shape[1]
    w = 0
    print(orig_data.shape, gen_data.shape, " Test WSTD Function")
    for i in range(columns):
        orig_data_ = orig_data[:, i].reshape(-1, 1)
        gen_data_ = gen_data[:, i].reshape(-1, 1)
        orig_data_ = orig_data_.reshape(
            len(orig_data),
        )
        gen_data_ = gen_data_.reshape(
            len(gen_data),
        )
        # orig_data_ += 1
        # gen_data_ += 1
        w += wasserstein_distance(orig_data_, gen_data_)
        # break
    w /= columns
    wsd += w
    return wsd


def calculate_jsd(org, gen):
    prep(org)
    prep(gen)
    KLD = 0
    orig_data = org.values
    gen_data = gen.values
    # if orig_data.shape[1] != gen_data.shape[1]:
    #    pca = PCA(n_components=gen_data.shape[1])
    #    pca = pca.fit(orig_data)
    #    gen_data = pca.inverse_transform(gen_data)
    columns = orig_data.shape[1]
    scaler = MinMaxScaler()
    softmax = torch.nn.Softmax(0)
    kld = 0
    print(orig_data.shape, gen_data.shape, " Test JSD Function")
    for i in range(columns):
        orig_data_ = scaler.fit_transform(orig_data[:, i].reshape(-1, 1))
        gen_data_ = scaler.fit_transform(gen_data[:, i].reshape(-1, 1))
        orig_data_ = orig_data_.reshape(
            len(orig_data),
        )
        gen_data_ = gen_data_.reshape(
            len(gen_data),
        )
        orig_data_ = softmax(torch.from_numpy(orig_data_)).numpy()
        gen_data_ = softmax(torch.from_numpy(gen_data_)).numpy()
        # orig_data_ += 1
        # gen_data_ += 1
        kld += js_divergence(orig_data_, gen_data_)
        # break
    kld /= columns
    KLD += kld
    return KLD


# gmm = GaussianMixture(5, covariance_type="full")

def main(root, name, doPca, doMinMax):
    os.makedirs("../OutputFiles/GMM/", exist_ok=True)
    df_t = pd.read_csv(root)
    prep(df_t)
    orig_data = df_t
    print(df_t)
    df = df_t.values
    if doPca:
        pca = PCA(0.99)
        df = pca.fit_transform(df)
        print("Pca fitted")
    model = GaussianMixture(5, covariance_type="full", random_state=0).fit(df)
    if(doPca):
        gendatax = pd.DataFrame(
            pca.inverse_transform(model.sample(len(orig_data))[0]),
            columns=orig_data.columns,
        )
    else:
        gendatax = pd.DataFrame(
            model.sample(len(orig_data))[0],
            columns=orig_data.columns,
        )
    gendatax.to_csv(f"../OutputFiles/GMM/{name}.csv", index=False)
    jsd = calculate_jsd(orig_data, gendatax)
    wsd = calculate_wstd(orig_data, gendatax)
    print("wsd, jsd", wsd, jsd)
#
# plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
# plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('n_components')
# plt.savefig('AIC_BIC_plot_gp2ax.png')
#
# gmm.fit(data)
## filename = "GMM_log2_GP2.sav"
# print(gmm.converged_)
## pickle.dump(gmm, open(filename, "wb"))
# for i in [len(data), 250, 550]:
#    data_new = gmm.sample(i)
#    digits_new = pca.inverse_transform(data_new[0])
#    df2 = pd.DataFrame(digits_new, columns=df_x.columns, index=None)
#    df2.to_csv("./Group2Data/GMM2_{}.csv".format(i))
#


if __name__=="__main__":
    allfiles = sorted(glob.glob("../InputFiles/Group*Data.csv"))
    print(allfiles)
    for root in allfiles:
        print(f"---- {root} ----")
        main(root, root.split("/")[-1][5], True, False)
        print("-"*(len(root) + 10))


    allfiles = sorted(glob.glob("../InputFiles/Group*Axes.csv"))
    print(allfiles)
    for root in allfiles:
        print(f"---- {root} ----")
        main(root, root.split("/")[-1][5], False, False)
        print("-"*(len(root) + 10))