import glob
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def plot_tsne(gendata, actual, name, labels=6, epoch=None):
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

    filename = f"{actual[:-9]}/tsne_plot_"

    if epoch is not None:
        filename = filename + f"_{epoch}"
    filename = filename + name + ".png"
    plt.savefig(filename)


def get_data(path):
    files = glob.glob(path)
    files.sort()

    model_dict = dict()
    for file in files:
        model_num = file[-5]
        if model_num not in model_dict.keys():
            model_dict[model_num] = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
            model_dict[model_num] = pd.concat(
                [model_dict[model_num], df], axis=0)
            model_dict[model_num].reset_index(drop=True, inplace=True)

    for key in model_dict:
        df = model_dict[key]
        for i in ["level_0", "index", "Unnamed: 0"]:
            try:
                df = df.drop(i, axis=1)
            except:
                continue

    return model_dict


def plot_complete(path, model_dict):
    df = None
    for key in model_dict:
        if df is not None:
            df = pd.concat([df, model_dict[key]], axis=0)
            df = shuffle(df)
        else:
            df = model_dict[key]

    n_samples = [50, 100, 150, 300, 500, len(df)]
    for i in n_samples:
        data = df.sample(i)
        data.reset_index(drop=True, inplace=True)
        plot_tsne(data, path,
                  f"split_before_combined_plot_{i}")


def plot_pairs(A, B):
    modelA = model_dict[A]
    modelB = model_dict[B]
    plot_tsne(modelA, modelB, f"combined_model_{A}_model_{B}", labels=1)

    root = "../Original Samples After PCA/data"
    orig = dict()
    for model in [A, B]:
        df = pd.read_csv(root+model+".csv")
        label = [int(model)]*len(df)
        label = pd.DataFrame(label, columns=['label'])
        df = pd.concat([df, label], axis=1)
        orig[model] = df
        orig[model].reset_index(drop=True, inplace=True)

    plot_tsne(orig[A], orig[B],
              f"compare_original_data_class_{A}_class_{B}", labels=1)


if __name__ == "__main__":
    import glob
    allfiles = glob.glob("Chromosome_*")
    allfiles.sort()

    from sklearn.decomposition import PCA

    for dire in allfiles:
        if dire[-1] >= '1' and dire[-1] <= '7':
            continue
        genpath = f"{dire}/gendata/*"
        origpath = f"{dire}/origdata/data.csv"

        model_dict = get_data(genpath)
        pca = PCA(n_components=model_dict['0'].shape[1])
        df = pd.read_csv(origpath)
        df = df.drop("host_name", axis=1)
        try:
            df = df.drop("Unnamed: 0", axis=1)
        except:
            pass
        pca = pca.fit(df.iloc[:, :-1])
        for key in model_dict:
            df = pd.DataFrame(pca.inverse_transform(model_dict[key]))
            length = df.shape[0]
            a = []
            a.append(int(key))
            a = a*length
            a = np.array(a)
            a = np.reshape(a, (length, 1))
            a = pd.DataFrame(a, columns=["label"])
            model_dict[key] = pd.concat([df, a], axis=1)
            del a
            # print(model_dict[key].columns)

        plot_complete(origpath, model_dict)
