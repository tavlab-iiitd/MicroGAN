from sklearn.manifold import TSNE
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras

from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

sns.set(rc={'figure.figsize': (16, 10)})
plt.style.use('seaborn-notebook')

sns.set(style="white", color_codes=True)
sns.set_context("paper", rc={"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 20,
                             'xtick.labelsize': 14, 'ytick.labelsize': 14})


latent_dim = 50

batch_size = 50
epochs = 2000
learning_rate = 0.0005

epsilon_std = 1.0
beta = K.variable(0)
kappa = 1


def sampling(args):

    import tensorflow as tf
    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)

    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    # Behavior on each epoch

    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


def label_def(row):
    if row['tuberculosis'] == 0 and row['hiv'] == 0:
        return 0
    elif row['tuberculosis'] == 1 and row['hiv'] == 0:
        return 1
    elif row['tuberculosis'] == 2 and row['hiv'] == 0:
        return 2
    elif row['tuberculosis'] == 0 and row['hiv'] == 1:
        return 3
    elif row['tuberculosis'] == 1 and row['hiv'] == 1:
        return 4
    elif row['tuberculosis'] == 2 and row['hiv'] == 1:
        return 5


np.random.seed(123)


def main(root):
    with tf.device("/gpu:1"):
        modelpath = os.path.join(root, "origdata", "data.csv")
        destpath = os.path.join(root, "gendata")

        input_ltpm_matrix = pd.read_csv(modelpath)

        if "Unnamed: 0" in list(input_ltpm_matrix.columns.values):
            input_ltpm_matrix = input_ltpm_matrix.drop("Unnamed: 0", axis=1)

        # input_ltpm_matrix.reset_index(drop=True, inplace=True)
        label = input_ltpm_matrix[['class']]
        host_names = input_ltpm_matrix[['host_name', 'class']]

        input_ltpm_matrix.drop(['host_name', 'class'], axis=1, inplace=True)
        input_ltpm_matrix = input_ltpm_matrix.reset_index(drop=True)

        rnaseq_df = input_ltpm_matrix
        rnaseq_df = pd.concat([rnaseq_df, label], axis=1)

        # Split 10% test set randomly
        test_set_percent = 0.1
        rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
        rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

        label_train = rnaseq_train_df['class']
        label_test = rnaseq_test_df['class']
        rnaseq_train_df.drop(['class'], axis=1, inplace=True)
        rnaseq_test_df.drop(['class'], axis=1, inplace=True)

        label_total = rnaseq_df['class']
        rnaseq_df.drop(['class'], axis=1, inplace=True)

        # ## Encoder
        # ### 6 layer
        original_dim = rnaseq_df.shape[1]

        class CustomVariationalLayer(Layer):
            """
            Define a custom layer that learns and performs the training
            This function is borrowed from:
            https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
            """

            def __init__(self, **kwargs):
                # https://keras.io/layers/writing-your-own-keras-layers/
                self.is_placeholder = True
                super(CustomVariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, x_input, x_decoded):
                reconstruction_loss = original_dim * \
                    metrics.binary_crossentropy(x_input, x_decoded)
                kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) -
                                        K.exp(z_log_var_encoded), axis=-1)
                return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

            def call(self, inputs):
                x = inputs[0]
                x_decoded = inputs[1]
                loss = self.vae_loss(x, x_decoded)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        # Input place holder for RNAseq data with specific input size
        rnaseq_input = Input(shape=(original_dim, ))
        rnaseq_cond = Input(shape=(1, ))

        rnaseq_inp = Concatenate(axis=1)([rnaseq_input, rnaseq_cond])

        # Input layer is compressed into a mean and log variance vector of size `latent_dim`
        # Each layer is initialized with glorot uniform weights and each step (dense connections,
        # batch norm, and relu activation) are funneled separately
        # Each vector of length `latent_dim` are connected to the rnaseq input tensor
        z_mean_dense_linear1 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(rnaseq_inp)
        z_mean_dense_batchnorm1 = BatchNormalization()(z_mean_dense_linear1)
        z_mean_encoded1 = Activation('relu')(z_mean_dense_batchnorm1)

        z_mean_dense_linear2 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_mean_encoded1)
        z_mean_dense_batchnorm2 = BatchNormalization()(z_mean_dense_linear2)
        z_mean_encoded2 = Activation('relu')(z_mean_dense_batchnorm2)

        z_mean_dense_linear3 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_mean_encoded2)
        z_mean_dense_batchnorm3 = BatchNormalization()(z_mean_dense_linear3)
        z_mean_encoded3 = Activation('relu')(z_mean_dense_batchnorm3)

        z_mean_dense_linear4 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_mean_encoded3)
        z_mean_dense_batchnorm4 = BatchNormalization()(z_mean_dense_linear4)
        z_mean_encoded4 = Activation('relu')(z_mean_dense_batchnorm4)

        z_mean_dense_linear5 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_mean_encoded4)
        z_mean_dense_batchnorm5 = BatchNormalization()(z_mean_dense_linear5)
        z_mean_encoded5 = Activation('relu')(z_mean_dense_batchnorm5)

        z_mean_dense_linear = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_mean_encoded5)
        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
        z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

        z_log_var_dense_linear1 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
        z_log_var_dense_batchnorm1 = BatchNormalization()(z_log_var_dense_linear1)
        z_log_var_encoded1 = Activation('relu')(z_log_var_dense_batchnorm1)

        z_log_var_dense_linear2 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_log_var_encoded1)
        z_log_var_dense_batchnorm2 = BatchNormalization()(z_log_var_dense_linear2)
        z_log_var_encoded2 = Activation('relu')(z_log_var_dense_batchnorm2)

        z_log_var_dense_linear3 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_log_var_encoded2)
        z_log_var_dense_batchnorm3 = BatchNormalization()(z_log_var_dense_linear3)
        z_log_var_encoded3 = Activation('relu')(z_log_var_dense_batchnorm3)

        z_log_var_dense_linear4 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_log_var_encoded3)
        z_log_var_dense_batchnorm4 = BatchNormalization()(z_log_var_dense_linear4)
        z_log_var_encoded4 = Activation('relu')(z_log_var_dense_batchnorm4)

        z_log_var_dense_linear5 = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_log_var_encoded4)
        z_log_var_dense_batchnorm5 = BatchNormalization()(z_log_var_dense_linear5)
        z_log_var_encoded5 = Activation('relu')(z_log_var_dense_batchnorm5)

        z_log_var_dense_linear = Dense(
            latent_dim, kernel_initializer='glorot_uniform')(z_log_var_encoded5)
        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
        z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

        # return the encoded and randomly sampled z vector
        # Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
        z = Lambda(sampling, output_shape=(latent_dim, ))(
            [z_mean_encoded, z_log_var_encoded])
        z = Concatenate(axis=1)([z, rnaseq_cond])

        # ## Decoder
        # ### 6 layer

        # The decoding layer is much simpler with a single layer and sigmoid activation
        decoder_to_reconstruct = Dense(
            latent_dim, kernel_initializer='glorot_uniform', activation='relu')
        rnaseq_reconstruct1 = decoder_to_reconstruct(z)
        decoder_to_reconstruct2 = Dense(
            latent_dim, kernel_initializer='glorot_uniform', activation='relu')
        rnaseq_reconstruct2 = decoder_to_reconstruct2(rnaseq_reconstruct1)
        decoder_to_reconstruct3 = Dense(
            latent_dim, kernel_initializer='glorot_uniform', activation='relu')
        rnaseq_reconstruct3 = decoder_to_reconstruct3(rnaseq_reconstruct2)
        decoder_to_reconstruct4 = Dense(
            latent_dim, kernel_initializer='glorot_uniform', activation='relu')
        rnaseq_reconstruct4 = decoder_to_reconstruct4(rnaseq_reconstruct3)
        decoder_to_reconstruct5 = Dense(
            latent_dim, kernel_initializer='glorot_uniform', activation='relu')
        rnaseq_reconstruct5 = decoder_to_reconstruct5(rnaseq_reconstruct4)
        decoder_to_reconstruct6 = Dense(
            original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
        # rnaseq_reconstruct1 = decoder_to_reconstruct()
        rnaseq_reconstruct = decoder_to_reconstruct6(rnaseq_reconstruct5)

        # ## Connect the encoder and decoder to make the VAE
        #
        # The `CustomVariationalLayer()` includes the VAE loss function (reconstruction + (beta * KL)), which is what will drive our model to learn an interpretable representation of gene expression space.
        #
        # The VAE is compiled with an Adam optimizer and built-in custom loss function. The `loss_weights` parameter ensures beta is updated at each epoch end callback

        adam = optimizers.Adam(lr=learning_rate)
        vae_layer = CustomVariationalLayer()(
            [rnaseq_input, rnaseq_reconstruct])
        vae = Model([rnaseq_input, rnaseq_cond], vae_layer)
        vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

        vae.summary()

        # ## Train the model
        #
        # The training data is shuffled after every epoch and 10% of the data is heldout for calculating validation loss.

        hist = vae.fit([np.array(rnaseq_train_df), np.array(label_train).reshape(-1, 1)],
                       shuffle=True, epochs=epochs, verbose=1,
                       batch_size=batch_size,
                       validation_data=([np.array(rnaseq_test_df), np.array(
                           label_test).reshape(-1, 1)], None),
                       callbacks=[WarmUpCallback(beta, kappa)])

        history_df = pd.DataFrame(hist.history)
        os.makedirs(os.path.join(root, 'figures'), exist_ok=True)
        hist_plot_file = os.path.join(
            root, 'figures')+'/sixhidden_2000_vae_training_50.pdf'
        ax = history_df.plot()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('VAE Loss')
        fig = ax.get_figure()
        fig.savefig(hist_plot_file)

        # ## Compile and output trained models
        #
        # We are interested in:
        #
        # 1. The model to encode/compress the input gene expression data
        #   * Can be possibly used to compress other tumors
        # 2. The model to decode/decompress the latent space back into gene expression space
        #   * This is our generative model
        # 3. The latent space compression of all pan cancer TCGA samples
        #   * Non-linear reduced dimension representation of tumors can be used as features for various tasks
        #     * Supervised learning tasks predicting specific gene inactivation events
        #     * Interpolating across this space to observe how gene expression changes between two cancer states
        # 4. The weights used to compress each latent node
        #   * Potentially indicate learned biology differentially activating tumors

        # ### Encoder model
        # Model to compress input
        encoder = Model([rnaseq_input, rnaseq_cond], z_mean_encoded)

        # Encode rnaseq into the hidden/latent representation - and save output
        encoded_rnaseq_df = encoder.predict_on_batch([rnaseq_df, label_total])
        encoded_rnaseq_df = pd.DataFrame(
            encoded_rnaseq_df, index=rnaseq_df.index)

        encoded_rnaseq_df.columns.name = 'sample_id'
        encoded_rnaseq_df.columns = encoded_rnaseq_df.columns + 1
        # encoded_file = os.path.join('data', 'encoded_rnaseq_onehidden_warmup_batchnorm.tsv')
        # encoded_rnaseq_df.to_csv(encoded_file, sep='\t')

        # ### Decoder (generative) model
        # ### 6 layer

        # build a generator that can sample from the learned distribution
        # can generate from any sampled z vector
        decoder_input = Input(shape=(latent_dim, ))
        decoder_cond = Input(shape=(1, ))
        decoder_inp = Concatenate(axis=1)([decoder_input, decoder_cond])
        _x_decoded_mean1 = decoder_to_reconstruct(decoder_inp)
        _x_decoded_mean2 = decoder_to_reconstruct2(_x_decoded_mean1)
        _x_decoded_mean3 = decoder_to_reconstruct3(_x_decoded_mean2)
        _x_decoded_mean4 = decoder_to_reconstruct4(_x_decoded_mean3)
        _x_decoded_mean5 = decoder_to_reconstruct5(_x_decoded_mean4)
        _x_decoded_mean = decoder_to_reconstruct6(_x_decoded_mean5)
        decoder = Model([decoder_input, decoder_cond], _x_decoded_mean)

        # ## Save the encoder/decoder models for future investigation

        os.makedirs(root + "/models", exist_ok=True)
        encoder_model_file = os.path.join(
            root, 'models') + '/encoder_sixhidden_vae_2000_goodplot.hdf5'
        decoder_model_file = os.path.join(
            root, 'models') + '/decoder_sixhidden_vae_2000_goodplot.hdf5'

        encoder.save(encoder_model_file)
        decoder.save(decoder_model_file)

        encoder = keras.models.load_model(
            root + "/models/encoder_sixhidden_vae_2000_goodplot.hdf5")
        decoder = keras.models.load_model(
            root + "/models/decoder_sixhidden_vae_2000_goodplot.hdf5")

        # _____
        # ## Generate samples

        labels = [0, 1, 2, 3, 4, 5]*100

        gendata = decoder.predict(
            [np.abs(3 * np.random.randn(600, 50)), np.array(labels).reshape(-1, 1)])

        genlabel_np = np.array(labels).reshape(-1, 1)

        genlabel_np.shape

        genlabel_np = pd.DataFrame(genlabel_np, columns=['class'])

        # ### 6 layer - 2000 - again

        gendata = pd.DataFrame(gendata)

        rnaseq_train_df = pd.concat([rnaseq_train_df, label_train], axis=1)
        # rnaseq_test_df = pd.concat([rnaseq_test_df, label_test], axis = 1)
        gendata = pd.concat([gendata, genlabel_np], axis=1)

        gen_mapping = {0: 'Gen_No_disease', 1: 'Gen_Active_tub', 2: 'Gen_latent_tub',
                       3: 'Gen_HIV', 4: 'Gen_Active_tub_HIV', 5: 'Gen_Latent_tub_HIV'}
        train_mapping = {0: 'Train_No_disease', 1: 'Train_Active_tub', 2: 'Train_latent_tub',
                         3: 'Train_HIV', 4: 'Train_Active_tub_HIV', 5: 'Train_Latent_tub_HIV'}

        gendata['class'] = gendata['class'].map(gen_mapping)
        rnaseq_train_df['class'] = rnaseq_train_df['class'].map(train_mapping)

        dfeatures = pd.concat([rnaseq_train_df, gendata],
                              ignore_index=True, axis=0)
        labels = dfeatures[['class']]
        dfeatures.drop(['class'], axis=1, inplace=True)
        rnaseq_train_df.drop(['class'], axis=1, inplace=True)
        # rnaseq_test_df.drop(['class'], axis=1, inplace=True)
        gendata.drop(['class'], axis=1, inplace=True)
        X_embedded = TSNE(n_components=2, random_state=0,
                          perplexity=100).fit_transform(dfeatures)
        X_embedded = pd.DataFrame(X_embedded, columns=['dim1', 'dim2'])
        X_embedded = pd.DataFrame(
            np.hstack([np.array(X_embedded), np.array(labels)]))
        X_embedded.columns = ['dim1', 'dim2', 'label']

        fig = plt.figure(figsize=(800, 600))
        sns_fig = sns.lmplot(x='dim1', y='dim2', data=X_embedded, fit_reg=False, hue='label', markers=["x", "x", "x", "x", "x", "x", "o", "o", "o", "o", "o", "o"],
                             palette=dict(Gen_No_disease=(0.219, 0.568, 0.050), Train_No_disease=(0.325, 0.843, 0.078),
                                          Gen_Active_tub=(0.917, 0.223, 0.266), Train_Active_tub=(0.933, 0.525, 0.549),
                                          Gen_latent_tub=(0.874, 0.164, 0.654), Train_latent_tub=(0.905, 0.431, 0.760),
                                          Gen_HIV=(0.407, 0.086, 0.890), Train_HIV=(0.662, 0.482, 0.937),
                                          Gen_Active_tub_HIV=(0.176, 0.270, 0.882), Train_Active_tub_HIV=(0.427, 0.494, 0.909),
                                          Gen_Latent_tub_HIV=(0.086, 0.635, 0.627), Train_Latent_tub_HIV=(0.215, 0.882, 0.874)))

        gendata = pd.DataFrame(gendata)
        # os.makedirs(os.path.join(root, "images"), exist_ok=True)
        sns_fig.savefig(root + "/figures/TSNE_Plot.png")
        gendata['class'] = genlabel_np.values
        os.makedirs(os.path.join(root, "gendata"), exist_ok=True)
        gendata.to_csv(f"{root}/gendata/VAE Data 300.csv")


if __name__ == "__main__":

    alldirs = glob.glob("Chromosome_*")
    alldirs.remove("VAE_script.py")
    for root in alldirs:
        print("-" * 100)
        print(f"{root}...")
        main(root)
        print("-" * 100)
        print()
