from sklearn.preprocessing import MinMaxScaler
import ctgan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import glob
import os
import sys
from time import time


# global plot_path
original = sys.stdout


def main(root, i, dopca=False, minmax=False):

    print("[INFO] Make nescessary directories")
    os.makedirs("../OutputFiles/CTGAN/plots/", exist_ok=True)
    os.makedirs("../OutputFiles/CTGAN/gendata/", exist_ok=True)
    os.makedirs("../OutputFiles/CTGAN/LossDetails/", exist_ok=True)

    model_path = root
    dest_path_gen = os.path.join("../OutputFiles/CTGAN/", "gendata/")
    plot_path = os.path.join("../OutputFiles/CTGAN/", "plots/")
    loss_path = os.path.join("../OutputFiles/CTGAN/", "LossDetails/")
    name = f"{dest_path_gen}/gen_sample_model{i}"
    print(model_path, dest_path_gen, plot_path, i, name, sep="\n", end="\n*****\n")
    print("[INFO] Preprocessing the data")
    data = pd.read_csv(model_path, engine='python')
    data = data.drop("Unnamed: 0", axis=1)
    column_names = data.columns

    if minmax:
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data),
                            columns=list(column_names))

    if dopca:
        pca = PCA(0.9)
        data = pca.fit_transform(data)
        data = pd.DataFrame(data)

    print("[INFO] Training the model")
    model = ctgan.CTGANSynthesizer(
        batch_size=10, generator_dim=(128, 128), discriminator_dim=(128, 128))
    sys.stdout = open(f"{loss_path}/loss.txt", "w+")
    start = time()
    model.fit(data, epochs=1000)

    print("[INFO] Generate Samples")
    gen_sample4 = model.sample(400)
    
    if dopca:
        gen_sample4 = pca.inverse_transform(gen_sample4)


    sys.stdout = original
    print("Time required = ", time()-start)
    gen_sample = pd.DataFrame(gen_sample4, columns=list(column_names))
    gen_sample.to_csv(name+"four.csv")



if __name__ == "__main__":

    # input data
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