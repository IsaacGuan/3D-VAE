import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

from VAE import *
from utils import npytar

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9
batch_size = 10
epoch_num = 150

def data_loader(fname):
    reader = npytar.NpyTarReader(fname)
    xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
    reader.reopen()
    for ix, (x, name) in enumerate(reader):
        xc[ix] = x.astype(np.float32)
    return 3.0 * xc - 1.0

def weighted_binary_crossentropy(target, output):
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss

def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = learning_rate_2
    return lr

if __name__ == '__main__':
    model = get_model()

    inputs = model['inputs']
    outputs = model['outputs']
    mu = model['mu']
    sigma = model['sigma']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']

    plot_model(encoder, to_file = 'vae_encoder.pdf', show_shapes = True)
    plot_model(decoder, to_file = 'vae_decoder.pdf', show_shapes = True)

    vae = model['vae']

    # kl_div = -0.5 * K.mean(1 + 2 * sigma - K.square(mu) - K.exp(2 * sigma))
    voxel_loss = K.cast(K.mean(weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32') # + kl_div
    vae.add_loss(voxel_loss)

    sgd = SGD(lr = learning_rate_1, momentum = momentum, nesterov = True)
    vae.compile(optimizer = sgd, metrics = ['accuracy'])

    plot_model(vae, to_file = 'vae.pdf', show_shapes = True)

    data_train = data_loader('datasets/shapenet10_chairs_nr.tar')

    vae.fit(
        data_train,
        epochs = epoch_num,
        batch_size = batch_size,
        validation_data = (data_train, None),
        callbacks = [LearningRateScheduler(learning_rate_scheduler)]
    )

    vae.save_weights('vae.h5')
