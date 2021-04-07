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
    class_map = {0: "Class_0", 1: "Class_1", 2: "Class_2",
                 3: "Class_3", 4: "Class_4", 5: "Class_5"}
    df['label'] = df['label'].map(class_map)

    gendata['label'] = gendata['label'].map(class_map)

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
                         palette=dict(Class_0=(0.325, 0.843, 0.078),
                                      Class_1=(0.933, 0.525, 0.549),
                                      Class_2=(0.905, 0.431, 0.760),
                                      Class_3=(0.662, 0.482, 0.937),
                                      Class_4=(0.427, 0.494, 0.909),
                                      Class_5=(0.215, 0.882, 0.874)))

    filename = f"{plotpath}/tsne_plot_"
    filename = filename + name + ".png"
    plt.savefig(filename)
    plt.close()


def genmain(root):
    path = os.path.join(root, 'gendata')
    plotpath = os.path.join(root, 'plots')
    
    allfiles = glob.glob(path + "/*.csv")
    dicti = {}
    for file in allfiles:
        num = int(file.split("\\")[-1][-5])
        df = pd.read_csv(file)

        if num in dicti.keys():
            for col in ["Unnamed: 0", "host_name"]:
                if col in df.columns:
                    df.drop(col, inplace=True, axis=1)
            
            df.reset_index(drop=True, inplace=True)
            df = pd.concat([dicti[num], df], ignore_index=True, axis=0)
            df.reset_index(drop=True, inplace=True)
        
        dicti[num] = df

    labels = [0, 1, 2, 3, 4, 5]    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            print(f"Class{i}/{j}")
            modelA = dicti[i]
            modelB = dicti[j]

            modelA.reset_index(drop=True, inplace=True)
            modelB.reset_index(drop=True, inplace=True)

            for col in ["Unnamed: 0", "host_name"]:
                try:
                    modelA.drop(col, axis=1, inplace=True)
                except:
                    pass

                try:
                    modelB.drop(col, axis=1, inplace=True)
                except:
                    pass

            if 'label' not in modelA.columns:
                a = np.ones((len(modelA), 1))
                a = a * i
                a = pd.DataFrame(a, columns=["label"])
                modelA = pd.concat([modelA, a], axis=1)

            if 'label' not in modelB.columns:
                a = np.ones((len(modelB), 1))
                a = a * j
                a = pd.DataFrame(a, columns=["label"])
                modelB = pd.concat([modelB, a], axis=1)

            assert len(modelA.columns) == len(modelB.columns), "Wrongggg !!!"
            modelA.columns = modelB.columns
            plot_tsne(modelA, modelB,
                      f"generated_comparison_class_{i}_class{j}", 1, plotpath)


def main(root):
    path = os.path.join(root, 'origdata')
    plotpath = os.path.join(root, 'plots')
    allfiles = glob.glob(path + "/*.csv")
    allfiles.sort()
    allfiles.remove(os.path.join(root, "origdata", "data.csv"))

    for i in range(0, len(allfiles)):
        for j in range(i + 1, len(allfiles)):
            print(f'Class {allfiles[i][-5]}/{allfiles[j][-5]}')
            modelA = pd.read_csv(allfiles[i])
            modelB = pd.read_csv(allfiles[j])

            modelA.reset_index(drop=True, inplace=True)
            modelB.reset_index(drop=True, inplace=True)

            for col in ["Unnamed: 0", "host_name"]:
                try:
                    modelA.drop(col, axis=1, inplace=True)
                except:
                    pass

                try:
                    modelB.drop(col, axis=1, inplace=True)
                except:
                    pass

            if 'label' not in modelA.columns:
                a = np.ones((len(modelA), 1))
                a = a * (int(allfiles[i][-5]))
                a = pd.DataFrame(a, columns=["label"])
                modelA = pd.concat([modelA, a], axis=1)

            if 'label' not in modelB.columns:
                a = np.ones((len(modelB), 1))
                a = a * (int(allfiles[j][-5]))
                a = pd.DataFrame(a, columns=["label"])
                modelA = pd.concat([modelB, a], axis=1)

            assert len(modelA.columns) == len(modelB.columns), "Wrongggg !!!"
            modelA.columns = modelB.columns
            plot_tsne(modelA, modelB,
                      f"original_comparison_class_{i}_class{j}", 1, plotpath)


allfiles = glob.glob("Chromosome_*")
allfiles.sort()

for root in allfiles:
    print(f"---- {root} ----")
    genmain(root)
