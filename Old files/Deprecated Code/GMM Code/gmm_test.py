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
    return data


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def calculate_wstd(org, gen):
    org = prep(org)
    gen = prep(gen)
    wsd = 0
    orig_data = org.values
    gen_data = gen.values
    # Uncomment for Non-Axes plots
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
        # Uncomment for non-Axes plots
        # orig_data_ += 1
        # gen_data_ += 1
        w += wasserstein_distance(orig_data_, gen_data_)
        # break
    w /= columns
    wsd += w
    return wsd


def calculate_jsd(org, gen):
    org = prep(org)
    gen = prep(gen)
    KLD = 0
    orig_data = org.values
    gen_data = gen.values
    # Uncomment for Non-Axes Plots
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
        # Uncomment for Non-Axes plots
        # orig_data_ += 1
        # gen_data_ += 1
        kld += js_divergence(orig_data_, gen_data_)
        # break
    kld /= columns
    KLD += kld
    return KLD


sample_sizes = [25, 35, 50, 65, 75, 80]

for sample_size in sample_sizes:
    df_t = pd.read_csv("/home/ayushig/saad18409/Group2Data.csv").iloc[:, 1:]
    orig_data = df_t
    df_t = df_t.sample(sample_size)
    df = df_t.values
    pca = PCA(0.99)
    data = pca.fit_transform(df)
    model = GaussianMixture(5, covariance_type="full", random_state=0).fit(data)
    # Uncomment this for calculate sample_size vs metrics
    # models = [
    #     GaussianMixture(n, covariance_type="full", random_state=0).fit(data.sample(n))
    #     for n in sample_sizes
    # ]
    gendatax = pd.DataFrame(
        pca.inverse_transform(model.sample(len(orig_data))[0]),
        columns=orig_data.columns,
    )
    gendatax.to_csv(f"./Group2Data_{sample_size}.csv", index=False)
    jsd = calculate_jsd(orig_data, gendatax)
    wsd = calculate_wstd(orig_data, gendatax)
    print("Sample_Size, wsd, jsd", sample_size, wsd, jsd)
