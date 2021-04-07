import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
import warnings

warnings.filterwarnings("ignore")


print(ctgan)

# global plot_path
original = sys.stdout


def main(root, path_to_store, dopca=False, minmax=False):

    print("[INFO] Make nescessary directories")
    os.makedirs(os.path.join(path_to_store, "plots"), exist_ok=True)
    os.makedirs(os.path.join(path_to_store, "gendata"), exist_ok=True)
    os.makedirs(os.path.join(path_to_store, "LossDetails"), exist_ok=True)

    model_path = root
    dest_path_gen = path_to_store + "/gendata"
    plot_path = path_to_store + "/plots"
    name = f"{dest_path_gen}/CTGAN_Axes_"
    print(
        model_path,
        dest_path_gen,
        plot_path,
        path_to_store,
        name,
        sep="\n",
        end="\n*****\n",
    )
    print("[INFO] Preprocessing the data")
    data = pd.read_csv(model_path, engine="python")
    if "Unnamed: 0" in data.columns.values:
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    if "Unnamed: 0.1" in data.columns.values:
        data.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    if "Unnamed: 0.1.1" in data.columns.values:
        data.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)
    print(data.head(1))
    column_names = data.columns

    # if dopca:
    #     pca = PCA(0.99)
    #     data = pca.fit_transform(data)
    #     data = pd.DataFrame(data)

    print(data.shape, "data shape")
    print("[INFO] Training the model")
    model = ctgan.CTGANSynthesizer(
        batch_size=10, generator_dim=(128, 128), discriminator_dim=(128, 128)
    )
    sys.stdout = open(f"{path_to_store}/LossDetails/model{path_to_store}.txt", "w+")
    start = time()
    model.fit(data, epochs=1000)

    print("[INFO] Generate Samples")
    gen_sample4 = model.sample(len(data))
    # gen_sample6 = model.sample(250)
    # gen_sample10 = model.sample(550)

    # if dopca:
    #     gen_sample10 = pca.inverse_transform(gen_sample10)
    #     gen_sample4 = pca.inverse_transform(gen_sample4)
    #     gen_sample6 = pca.inverse_transform(gen_sample6)

    sys.stdout = original
    print("Time required = ", time() - start)
    # gen_final = pd.DataFrame(gen_sample10, columns=list(column_names))
    # gen_final.to_csv(name + "550" + ".csv")
    # gen_final = pd.DataFrame(gen_sample6, columns=list(column_names))
    # gen_final.to_csv(name + "250" + ".csv")
    gen_final = pd.DataFrame(gen_sample4, columns=list(column_names))
    gen_final.to_csv(name + "org" + ".csv")
    # gen_final_log2 = pd.DataFrame(np.log2(gen_sample10), columns=list(column_names))
    # gen_final_log2.to_csv(name+"_log2"+".csv")


if __name__ == "__main__":

    # input data
    allfiles = sorted(glob.glob("../../saad18409/Group*Axes.csv"))
    print(allfiles)
    allfiles = allfiles[:2]
    for root in allfiles:
        print("bro wassup")
        print(f"---- {root} ----")
        main(root, root.split("/")[-1][:-4], False, False)
        print("-" * (len(root) + 10))
