# 3D-VAE

This is the [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) implementation of the volumetric [variational autoencoder (VAE)](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_%28VAE%29) described in the paper ["Generative and Discriminative Voxel Modeling with Convolutional Neural Networks"](https://arxiv.org/abs/1608.04236).

## Preparing the Data

Some experimental shapes from the [ModelNet10](http://modelnet.cs.princeton.edu/) dataset are saved in the `datasets` folder. The model consumes volumetric shapes compressed in the [TAR](https://www.gnu.org/software/tar/) file format. For details about the structure and preparation of the TAR files, please refer to [voxnet](https://github.com/dimatura/voxnet).

## Training

```bash
python train.py
```

## Testing

```bash
python test.py
```
