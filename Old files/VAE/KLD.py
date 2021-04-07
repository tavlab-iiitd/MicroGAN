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
    try:
        origfiles.remove(os.path.join(root, "origdata", "data.csv"))
    except:
        pass
    
    genfiles = glob.glob(genpath + "/*")
    genfiles.sort()
    gendict = {}
    origdict = {}

    for file in origfiles:
        origdict[file[-5]] = pd.read_csv(file)
        for col in ['Unnamed: 0', 'host_name', 'label', 'tuberculosis', 'hiv']:
            if col in origdict[file[-5]].columns:
                origdict[file[-5]].drop(col, axis=1, inplace=True)

    labels = [0, 1, 2, 3, 4, 5]
    df = pd.read_csv(genfiles[0])
    for col in ["Unnamed: 0", "host_name"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    for i in labels:
        gendict[str(i)] = df[df['class'] == i]
        gendict[str(i)].drop('class', axis=1, inplace=True)

    KLD = 0
    for i in origdict.keys():
        orig_data = origdict[i]
        gen_data = gendict[i]
        #print(orig_data.columns)
        #print(gen_data.columns)
        if len(orig_data) < len(gen_data):
            gen_data = gen_data.sample(len(orig_data))
        else:
            orig_data = orig_data.sample(len(gen_data))

        orig_data = orig_data.values
        gen_data = gen_data.values

        #print(orig_data.columns)
        #print(gen_data.columns)

        #if orig_data.shape[1] != gen_data.shape[1]:
        #    pca = PCA(n_components=gen_data.shape[1])
        #    pca = pca.fit(orig_data)
        #    gen_data = pca.inverse_transform(gen_data)

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
    # allfiles.remove("Chromosome_20")
    KLD = []
    for root in allfiles:
        print(f"---- {root} ----")
        KLD.append(main(root))

    final = []
    for val in range(len(KLD)):
        final.append([allfiles[val], KLD[val]])

    final = np.array(final)
    final = pd.DataFrame(final, columns=["Chromosome", "Similarity"])
    final.to_csv("./VAE_KLD_c.csv")
