from os import name
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import glob
from scipy.stats import wasserstein_distance as ssw
import scipy.stats as sst
import seaborn as sns
from sklearn.datasets import make_moons
import os
import numpy as np
import pandas as pd


class ReadDatasets:
    def __init__(self, org, gen):
        self.org = pd.read_csv(org)
        self.gen = pd.read_csv(gen)
        self.preprocess(self.org)
        self.preprocess(self.gen)
        if self.org.shape[0] != self.gen.shape[0]:
            if self.org.shape[0] > self.gen.shape[0]:
                self.org = self.org.iloc[: self.gen.shape[0], :]
            else:
                self.gen = self.gen.iloc[: self.org.shape[0], :]
        print(self.org.shape, self.gen.shape)

    def preprocess(self, df):
        if "Unnamed: 0" in list(df.columns.values):
            df.drop(["Unnamed: 0"], axis=1, inplace=True)
        if "Unnamed: 0.1" in list(df.columns.values):
            df.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        if "Unnamed: 0.1.1" in list(df.columns.values):
            df.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)

    def get_distances(self, X, g1, k):
        org = X.iloc[:g1, :]
        gen = X.iloc[g1:, :]
        org = org.reset_index()
        gen = gen.reset_index()
        dist_final = []
        for i in range(len(gen)):
            x1, y1 = gen.loc[i, "dim1"], gen.loc[i, "dim2"]
            dist_ll = []
            for j in range(len(org)):
                x2, y2 = org.loc[j, "dim1"], org.loc[j, "dim2"]
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                dist_ll.append(dist)
            dist_ll.sort()
            dist_final.append(np.average(dist_ll[:k]))
        return np.average(dist_final)

    def plot_tsne(self, model_name, category, plotpath):
        org = self.org
        gen = self.gen
        norg = org.shape[0]
        ngen = gen.shape[0]
        label1 = ["ORG Data-{}".format(category)] * norg
        label2 = ["{} Data-{}".format(model_name, category)] * ngen
        labels = pd.Series(label1 + label2).to_frame()
        dfeatures = pd.concat([org, gen], ignore_index=True, axis=0, sort=False)
        t_sne = TSNE(n_components=2, random_state=0, perplexity=100)
        t_sne.fit(dfeatures)
        X_embedded = t_sne.fit_transform(dfeatures)
        print(t_sne.kl_divergence_)
        # .csv store
        X_embedded = pd.DataFrame(X_embedded, columns=["dim1", "dim2"])
        avg_dist = self.get_distances(X_embedded, norg, 2)
        print(avg_dist, model_name, category, plotpath)
        X_embedded = pd.DataFrame(np.hstack([np.array(X_embedded), np.array(labels)]))
        X_embedded.columns = ["dim1", "dim2", "label"]
        sns.set(rc={"figure.figsize": (20, 80)})
        sns_fig = sns.lmplot(
            x="dim1", y="dim2", data=X_embedded, fit_reg=False, hue="label"
        )
        filename = "./"
        X_embedded.to_csv(filename + plotpath + ".csv")
        filename = filename + plotpath + ".png"
        plt.title(model_name + str(avg_dist))
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":

    random = [
        "/home/ayushig/TB/WGAN/Group1Data/nonormal/gendata/Random_org.csv",
        "/home/ayushig/TB/WGAN/Group2Data/nonormal/gendata/Random_org.csv",
        "/home/ayushig/TB/WGAN/Group1Axes/nonormal/gendata/Random_org.csv",
        "/home/ayushig/TB/WGAN/Group2Axes/nonormal/gendata/Random_org.csv",
    ]
    gen_wgang1 = [
        "/home/ayushig/TB/WGAN/Group1Data/gendata/WGAN_org.csv",
        "/home/ayushig/TB/WGAN/Group2Data/gendata/WGAN_org.csv",
        "/home/ayushig/TB/WGAN/Group1Axes/gendata/WGAN_org.csv",
        "/home/ayushig/TB/WGAN/Group2Axes/gendata/WGAN_org.csv",
    ]
    gen_ctgang1 = [
        "/home/ayushig/TB/CTGAN/Group1Data/gendata/CTGAN_org.csv",
        "/home/ayushig/TB/CTGAN/Group2Data/gendata/CTGAN_org.csv",
        "/home/ayushig/TB/CTGAN/Group1Axes/gendata/CTGAN_Axes_org.csv",
        "/home/ayushig/TB/CTGAN/Group2Axes/gendata/CTGAN_Axes_org.csv",
    ]
    gen_gmmg1 = [
        "/home/ayushig/TB/GMMGAN/Group1Data/GMM1_83.csv",
        "/home/ayushig/TB/GMMGAN/Group2Data/GMM2_97.csv",
        "/home/ayushig/TB/GMMGAN/Group1Axes/GMM1_83.csv",
        "/home/ayushig/TB/GMMGAN/Group2Axes/GMM2_97.csv",
    ]
    gen_vae1 = [
        "/home/ayushig/TB/VAE/gen_final_org1/VAE_80.csv",
        "/home/ayushig/TB/VAE/gen_final_org2/VAE_80.csv",
        "/home/ayushig/TB/VAE/gen_final_org1/VAE_83.csv",
        "/home/ayushig/TB/VAE/gen_final_org2/VAE_83.csv",
    ]
    orgg1 = [
        "/home/ayushig/saad18409/Group1Data.csv",
        "/home/ayushig/saad18409/Group2Data.csv",
        "/home/ayushig/saad18409/Group1Axes.csv",
        "/home/ayushig/saad18409/Group2Axes.csv",
    ]  # , "/home/ayushig/saad18409/Group1Axes.csv", "/home/ayushig/saad18409/Group2Axes.csv"]
    nm = ["Healthy", "TB", "Axis-Healthy", "Axis-TB"]
    iii = ["1", "2", "1", "2"]
    ii = ["", "", "Axes", "Axes"]

    for i in range(2, 4):
        rd = ReadDatasets(orgg1[i], gen_vae1[i])
        rd.plot_tsne("VAE_Axes", nm[i], "VAE_" + ii[i] + iii[i])
    for i in range(2, 4):
        rd = ReadDatasets(orgg1[i], gen_wgang1[i])
        rd.plot_tsne("WGAN_Axes", nm[i], "WGAN_Axes_" + ii[i] + iii[i])
    for i in range(2, 4):
        rd = ReadDatasets(orgg1[i], gen_ctgang1[i])
        rd.plot_tsne("CTGAN_Axes", nm[i], "CTGAN_Axes_" + ii[i] + iii[i])
    for i in range(2, 4):
        rd = ReadDatasets(orgg1[i], gen_gmmg1[i])
        rd.plot_tsne("GMM_Axes", nm[i], "GMM_Axes_" + ii[i] + iii[i])
    # for i in range(4):
    #     rd = ReadDatasets(random[i], gen_wgang1[i])
    #     rd.plot_tsne("WGAN", nm[i], "WGAN_Random_" + ii[i] + iii[i])
