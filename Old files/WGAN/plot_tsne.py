import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import seaborn as sns
import sys
from tempfile import TemporaryFile

scaler = MinMaxScaler()
original = sys.stdout


def plot_tsne(gendata, actual, name, labels=6, plotpath=None):
    n_patients, n_genes = gendata.shape
    train_marker = ['x'] * labels
    test_marker = ['o'] * labels
    marker = [x for A in [train_marker, test_marker] for x in A]

    if isinstance(actual, str):
        df = pd.read_csv(actual)
    else:
        df = actual

    df.reset_index(drop=True, inplace=True)
    train_map = {0: "Orig_0", 1: "Orig_1", 2: "Orig_2",
                 3: "Orig_3", 4: "Orig_4", 5: "Orig_5"}
    df['label'] = df['label'].map(train_map)

    gen_map = {0: "Gen_0", 1: "Gen_1", 2: "Gen_2",
               3: "Gen_3", 4: "Gen_4", 5: "Gen_5"}
    gendata['label'] = gendata['label'].map(gen_map)

    gendata.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dfeatures = pd.concat([gendata, df], ignore_index=True,
                          axis=0, sort=False)

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
                         palette=dict(Gen_0=(0.219, 0.568, 0.050), Orig_0=(0.325, 0.843, 0.078),
                                      Gen_1=(0.917, 0.223, 0.266), Orig_1=(0.933, 0.525, 0.549),
                                      Gen_2=(0.874, 0.164, 0.654), Orig_2=(0.905, 0.431, 0.760),
                                      Gen_3=(0.407, 0.086, 0.890), Orig_3=(0.662, 0.482, 0.937),
                                      Gen_4=(0.176, 0.270, 0.882), Orig_4=(0.427, 0.494, 0.909),
                                      Gen_5=(0.086, 0.635, 0.627), Orig_5=(0.215, 0.882, 0.874)))

    filename = f"{plotpath}/tsne_plot_"
    filename = filename + name + ".png"
    plt.savefig(filename)


def main(root):
    genpath = os.path.join(root, 'gendata')
    plotpath = os.path.join(root, 'plots')
    origpath = os.path.join(root, 'origdata', 'data.csv')

    print("[INFO] Making Directories")
    os.makedirs(plotpath, exist_ok=True)

    print("[INFO] Loading Data")
    genfiles = sorted(glob.glob(genpath+'/data*'))
    gendata = None
    for file in genfiles:
        df = pd.read_csv(file, engine='python')
        for col in ['Unnamed: 0']:
            df.drop(col, axis=1, inplace=True)
        if gendata is None:
            gendata = df
        else:
            gendata.reset_index(drop=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            gendata = pd.concat([gendata, df], axis=0, ignore_index=True)
            gendata.reset_index(drop=True, inplace=True)

    origdata = pd.read_csv(origpath, engine='python')
    origdata.drop(["Unnamed: 0", "host_name", "tuberculosis", "hiv"], axis=1, inplace=True)
    origdata.rename(columns={"class":"label"}, inplace=True)
    origdata.reset_index(drop=True, inplace=True)
    label_orig = origdata[['label']]
    origdata = origdata.drop('label', axis=1)
    origdata = scaler.fit_transform(origdata)
    origdata = pd.DataFrame(origdata)
    origdata = pd.concat([origdata, label_orig], axis=1)
    gendata['label'] = gendata['label'].astype('int')
    label_gen = gendata[['label']]
    gendata.columns = origdata.columns

    print("[INFO] Plotting All Classes Together")
    plot_tsne(gendata, origdata, f"combine_plot", plotpath=plotpath)

    gendata = pd.concat([gendata, label_gen], axis=1)
    origdata = pd.concat([origdata, label_orig], axis=1)

    print("[INFO] Plotting class wise")
    f = TemporaryFile("w+")
    sys.stdout = f
    for i in [0, 1, 2, 3, 4, 5]:
        for j in range(2):
            if j == 0:
                gen = gendata[gendata['label'] == i]
                orig = origdata[origdata['label'] == i]
                plot_tsne(
                    gen, orig, f"class_{i}_gen_{len(gen)}_orig_{len(orig)}", labels=1, plotpath=plotpath
                )
            elif j == 1:
                gen = gendata[gendata['label'] == i]
                orig = origdata[origdata['label'] == i]
                if len(orig) > len(gen):
                    orig = orig.sample(len(gen))
                    plot_tsne(
                        gen, orig, f"class_{i}_gen_{len(gen)}_orig_{len(orig)}", labels=1, plotpath=plotpath
                    )
                else:
                    gen = gen.sample(len(orig))
                    plot_tsne(
                        gen, orig, f"class_{i}_gen_{len(gen)}_orig_{len(orig)}", labels=1, plotpath=plotpath
                    )

    sys.stdout = original
    f.close()
    print("[INFO] Done")


allfiles = glob.glob("Chromosome_*")
allfiles.sort()
for root in allfiles:
    print(f"---- {root} ----")
    main(root)
    print("-"*(10 + len(root)))
