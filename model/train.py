# coding=utf-8
import tensorflow as tf
import numpy as np
from settings import *
from model.loss import get_loss
from PIL import Image
from preprocessing.transfer_compute import read_image_as_tf_constant, generate_noize


def backward_prop():
    pass


def train_single_image(content_image=None, style_image=None):
    if content_image == None:
        content_image = read_image_as_tf_constant(CONTENT_IMAGE_PATH) - 128.0
    if style_image == None:
        style_image = read_image_as_tf_constant(STYLE_IMAGE_PATH) - 128.0
    gen_image = generate_noize(content_image)
    loss = get_loss(gen_image, style_image, content_image)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        # 我用global init 老给我报错我也很绝望啊
        tf.initialize_all_variables().run()
        for i in range(STEPS):
            _, cost, image = sess.run([train_op, loss, gen_image])
            print('loss=%f,iter=%d' % (cost, i))
            if i % 50 == 0:
                save_image = np.clip(image + 128.0, 0, 255).astype(np.uint8)
                Image.fromarray(save_image).save(TRANSFER_STYLE_IMAGE_SAVE + r"/challenger_%d.jpg" % (i,))
        save_image = np.clip(image + 128.0, 0, 255).astype(np.uint8)
        Image.fromarray(save_image).save(TRANSFER_STYLE_IMAGE_SAVE + r"/challenger_%d.jpg" % (i,))


def main():
    train_single_image()


if __name__ == '__main__':
    main()
