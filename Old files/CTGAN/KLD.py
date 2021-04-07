import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import glob
import torch
from math import sqrt
import os
from sklearn.decomposition import PCA


def load_data(path):
    data = pd.read_csv(path, engine='python')
    return data


def kl_divergence(p, q):
    return entropy(p, q)


def main(root):
    origpath = os.path.join(root, 'origdata')
    genpath = os.path.join(root, 'gendata')

    origfiles = glob.glob(origpath + '/data*')
    origfiles.sort()
    origfiles.remove(os.path.join(root, "origdata", "data.csv"))
    genfiles = glob.glob(genpath + "/*")
    genfiles.sort()
    gendict = {}
    origdict = {}

    for file in origfiles:
        origdict[file[-5]] = pd.read_csv(file)
        for col in ['Unnamed: 0', 'host_name', 'label']:
            if col in origdict[file[-5]].columns:
                origdict[file[-5]].drop(col, axis=1, inplace=True)

    for file in genfiles:
        num = file.split("/")[-1][-5]
        if num in gendict.keys():
            df = pd.read_csv(file)
            df.reset_index(drop=True, inplace=True)
            for col in ['Unnamed: 0', 'host_name', 'label']:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

            gendict[num] = pd.concat(
                [gendict[num], df], axis=0, ignore_index=True)
            gendict[num].reset_index(drop=True, inplace=True)
        else:
            df = pd.read_csv(file)
            df.reset_index(drop=True, inplace=True)
            for col in ['Unnamed: 0', 'host_name', 'label']:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

            gendict[num] = df

    KLD = 0
    for i in origdict.keys():
        orig_data = origdict[i]
        gen_data = gendict[i]

        if len(orig_data) < len(gen_data):
            gen_data = gen_data.sample(len(orig_data))
        else:
            orig_data = orig_data.sample(len(gen_data))

        orig_data = orig_data.values
        gen_data = gen_data.values

        print(orig_data.shape)
        print(gen_data.shape)

        if orig_data.shape[1] != gen_data.shape[1]:
            pca = PCA(n_components=gen_data.shape[1])
            pca = pca.fit(orig_data)
            gen_data = pca.inverse_transform(gen_data)

        columns = orig_data.shape[1]

        scaler = MinMaxScaler()
        softmax = torch.nn.Softmax(0)
        kld = 0

        for i in range(columns):
            orig_data_ = scaler.fit_transform(orig_data[:, i].reshape(-1, 1))
            gen_data_ = scaler.fit_transform(gen_data[:, i].reshape(-1, 1))
            orig_data_ = orig_data_.reshape(len(orig_data), )
            gen_data_ = gen_data_.reshape(len(gen_data), )

            orig_data_ = softmax(torch.from_numpy(orig_data_)).numpy()
            gen_data_ = softmax(torch.from_numpy(gen_data_)).numpy()

            orig_data_ += 1
            gen_data_ += 1

            kld += kl_divergence(orig_data_, gen_data_)
            # break

        kld /= columns
        KLD += kld

    return KLD


if __name__ == "__main__":
    allfiles = glob.glob("Chromosome_*")
    allfiles.sort()
    KLD = []
    for root in allfiles:
        print(f"---- {root} ----")
        KLD.append(main(root))

    final = []
    for val in range(len(KLD)):
        final.append([allfiles[val], KLD[val]])

    final = np.array(final)
    final = pd.DataFrame(final, columns=["Chromosome", "Similarity"])
    final.to_csv("./CTGAN_KLD_c.csv")
