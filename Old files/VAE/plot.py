# Plot VAE Class Wise
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def plot_tsne(gendata, actual, name, labels=6):
    n_patients, n_genes = gendata.shape
    train_marker = ['x'] * labels
    test_marker = ['o'] * labels
    marker = [x for A in [train_marker, test_marker] for x in A]

    if isinstance(actual, str):
        df = pd.read_csv(actual)

    else:
        df = actual

    df.reset_index(drop=True, inplace=True)

    train_map = {0: "train_0", 1: "train_1", 2: "train_2",
                 3: "train_3", 4: "train_4", 5: "train_5"}
    df['label'] = df['label'].map(train_map)

    gen_map = {0: "gen_0", 1: "gen_1", 2: "gen_2",
               3: "gen_3", 4: "gen_4", 5: "gen_5"}
    gendata['label'] = gendata['label'].map(gen_map)

    df.reset_index(drop=True, inplace=True)
    for col in ["host_name", "Unnamed: 0"]:
        if col in list(df.columns.values):
            df = df.drop(col, axis=1)

    df.columns = gendata.columns

    dfeatures = pd.concat([df, gendata], ignore_index=True,
                          axis=0)
    labels = dfeatures[['label']]
    dfeatures.drop(['label'], axis=1, inplace=True)
    if 'Unnamed: 0' in list(dfeatures.columns.values):
        dfeatures.drop(['Unnamed: 0'], axis=1, inplace=True)

    df.drop(['label'], axis=1, inplace=True)
    gendata.drop(['label'], axis=1, inplace=True)

    X_embedded = TSNE(n_components=2, random_state=0,
                      perplexity=100).fit_transform(dfeatures)
    X_embedded = pd.DataFrame(X_embedded, columns=['dim1', 'dim2'])
    X_embedded = pd.DataFrame(
        np.hstack([np.array(X_embedded), np.array(labels)]))
    X_embedded.columns = ['dim1', 'dim2', 'label']

    sns_fig = sns.lmplot(x='dim1', y='dim2', data=X_embedded, fit_reg=False, hue='label', markers=marker,
                         palette=dict(gen_0=(0.219, 0.568, 0.050), train_0=(0.325, 0.843, 0.078),
                                      gen_1=(0.917, 0.223, 0.266), train_1=(0.933, 0.525, 0.549),
                                      gen_2=(0.874, 0.164, 0.654), train_2=(0.905, 0.431, 0.760),
                                      gen_3=(0.407, 0.086, 0.890), train_3=(0.662, 0.482, 0.937),
                                      gen_4=(0.176, 0.270, 0.882), train_4=(0.427, 0.494, 0.909),
                                      gen_5=(0.086, 0.635, 0.627), train_5=(0.215, 0.882, 0.874)))

    plt.savefig(name)


allfiles = glob.glob("Chromosome_*")
allfiles.sort()
#allfiles.remove("Chromosome_20")

for file in allfiles:
    genpath = os.path.join(file, "gendata", "gendata.csv")
    origpath = os.path.join(file, "origdata", "data.csv")
    os.makedirs(os.path.join(file, "figures"), exist_ok=True)
    labels = [0, 1, 2, 3, 4, 5]

    gendata = pd.read_csv(genpath)
    origdata = pd.read_csv(origpath)
    gendata.rename(columns = {"class":"label"}, inplace=True)
    origdata.rename(columns = {"class":"label"}, inplace=True)
    for la in labels:
        gen = gendata[gendata["label"] == la]
        orig = origdata[origdata["label"] == la]

        if len(orig) > len(gen):
            orig = orig.sample(len(gen)-10)
        else:
            gen = gen.sample(len(orig))

        if "Unnamed: 0" in gen.columns:
            gen = gen.drop("Unnamed: 0", axis=1)

        if "Unnamed: 0" in orig.columns:
            orig = orig.drop("Unnamed: 0", axis=1)
        
        for col in ["tuberculosis", "hiv", "host_name"]:
            if col in orig.columns:
               orig.drop(col, axis=1, inplace=True)

        name = f"{file}/figures/tsne_plot_class_{la}_orig_{orig.shape[0]}_gen_{gen.shape[0]}.png"

        plot_tsne(gen, orig, name, 1)
