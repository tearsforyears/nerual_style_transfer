# coding=utf-8
import tensorflow as tf
from functools import reduce
from preprocessing.transfer_compute import vgg19
from settings import *


def gram(vector):
    channels = vector.shape.as_list()[-1]
    vector = tf.reshape(vector, (-1, channels))
    return tf.matmul(tf.transpose(vector), vector)


def get_content_loss(gen_conv, content_conv):
    norm_param = 1 / reduce(lambda x, y: x * y, gen_conv.shape.as_list())
    # return norm_param * tf.nn.l2_loss(gen_conv - content_conv)
    return norm_param * tf.norm(gen_conv - content_conv)


def get_style_loss(gen_conv, style_conv):
    # norm_param = 1 / reduce(lambda x, y: x * y, gen_conv.shape.as_list())
    # return norm_param * tf.nn.l2_loss(gram(style_conv) * norm_param - gram(gen_conv) * norm_param)
    norm_param = 1 / (4 * reduce(lambda x, y: x * y, gen_conv.shape.as_list()) ** 2)
    return norm_param * (tf.norm(gram(style_conv) - gram(gen_conv)) ** 2)


def get_loss(gen_image, style_image, content_image):
    gen_features_dict = vgg19([gen_image])  # generation graph
    style_features_dict = vgg19([style_image])  # constant flow
    content_features_dict = vgg19([content_image])  # constant flow

    # only one layer for content loss
    content_loss = 0.0
    for layername in CONTENT_LAYER:
        content_loss = content_loss + get_content_loss(gen_features_dict[layername], content_features_dict[layername])

    style_loss = 0.0
    for layername in STYLE_LAYERS:
        style_loss = style_loss + get_style_loss(gen_features_dict[layername], style_features_dict[layername])
    # print(style_loss, content_loss)
    return ALPHA * style_loss + BETA * content_loss


def main():
    pass


if __name__ == '__main__':
    main()
