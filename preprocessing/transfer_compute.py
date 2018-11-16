# coding=utf-8
import tensorflow as tf
import numpy as np
from settings import *
import scipy.io as sio
from PIL import Image

_vgg_params = None


def generate_noize(content_image, noise_ratio=NOISE_RATIO):
    if noise_ratio == 1:
        return tf.Variable(tf.random_normal(content_image.shape), dtype=tf.float32, name='gen_image')
    else:
        noise_image = np.random.uniform(-20, 20, content_image.shape)
        random_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
        gen_image = tf.Variable(
            random_image,
            dtype=tf.float32,
            name='gen_image'
        )
        return gen_image * noise_ratio + content_image * (1 - noise_ratio)


def read_image_as_tf_constant(path):
    return tf.constant(np.array(Image.open(path)).astype(np.float32), dtype=tf.float32)


def vgg_params():
    global _vgg_params
    if _vgg_params is None:
        _vgg_params = sio.loadmat(PARAMETERS_PATH)
    return _vgg_params


def vgg19(input_image):
    '''
        重建vgg19正向传播的网络,并用训练好的参数固化
        network[name] = net  # net为存储数据流的一个变量
        使用方法:
             features = vgg19([input_image])
             layername_feature_map = features['layername']
             数据结构是 tf.constant 或者 tf.Variable
        重建网络代码源于网络
    '''
    layers = LAYERS

    weights = vgg_params()['layers'][0]
    net = input_image
    network = {}
    for i, name in enumerate(layers):
        layer_type = name[:4]
        # 若是卷积层
        if layer_type == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # 由于 imagenet-vgg-verydeep-19.mat 中的参数矩阵和我们定义的长宽位置颠倒了，所以需要交换
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            conv = tf.nn.conv2d(net, tf.constant(kernels), strides=(1, 1, 1, 1), padding='SAME', name=name)
            net = tf.nn.bias_add(conv, bias.reshape(-1))
            net = tf.nn.relu(net)
        # 若是池化层
        elif layer_type == 'pool':
            net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # 将隐藏层加入到集合中
        # 若为`激活函数`直接加入集合
        network[name] = net
    return network


if __name__ == '__main__':
    params = vgg_params()
    for key, value in params.items():
        print(key, type(value))
