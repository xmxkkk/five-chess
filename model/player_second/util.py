import tensorflow as tf


def board1_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=1, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=1, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=1, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv3)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board2_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=1, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=1, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=1, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=1, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv4)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board3_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=1, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=1, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)

    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=1, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=1, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=1, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv5)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board4_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=1, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=1, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=1, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=1, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=1, strides=1, padding='same', activation='relu')
    conv6 = tf.layers.conv2d(conv5, 128, kernel_size=1, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv6)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board5_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=1, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=1, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 63, kernel_size=1, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=1, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=1, strides=1, padding='same', activation='relu')
    conv6 = tf.layers.conv2d(conv5, 128, kernel_size=1, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv6)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1