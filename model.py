import tensorflow as tf
import numpy as np
import scipy.io


VGG_MODEL = 'models/imagenet-vgg-verydeep-19.mat'


def build_net(ntype, nin, rwb=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, rwb[0], strides=[1, 1, 1, 1],
                                       padding='SAME') + rwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def vgg19_pretrained(input_shape):
    net = {}
    vgg_rawnet = scipy.io.loadmat(VGG_MODEL)
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = tf.Variable(
        np.zeros((1, input_shape[0], input_shape[1], 3)).astype(np.float32))
    net['conv1_1'] = build_net(
        'conv', net['input'], get_weight_bias(vgg_layers, 0))
    net['conv1_2'] = build_net(
        'conv', net['conv1_1'], get_weight_bias(vgg_layers, 2))
    net['pool1'] = build_net('pool', net['conv1_2'])
    net['conv2_1'] = build_net(
        'conv', net['pool1'], get_weight_bias(vgg_layers, 5))
    net['conv2_2'] = build_net(
        'conv', net['conv2_1'], get_weight_bias(vgg_layers, 7))
    net['pool2'] = build_net('pool', net['conv2_2'])
    net['conv3_1'] = build_net(
        'conv', net['pool2'], get_weight_bias(vgg_layers, 10))
    net['conv3_2'] = build_net(
        'conv', net['conv3_1'], get_weight_bias(vgg_layers, 12))
    net['conv3_3'] = build_net(
        'conv', net['conv3_2'], get_weight_bias(vgg_layers, 14))
    net['conv3_4'] = build_net(
        'conv', net['conv3_3'], get_weight_bias(vgg_layers, 16))
    net['pool3'] = build_net('pool', net['conv3_4'])
    net['conv4_1'] = build_net(
        'conv', net['pool3'], get_weight_bias(vgg_layers, 19))
    net['conv4_2'] = build_net(
        'conv', net['conv4_1'], get_weight_bias(vgg_layers, 21))
    net['conv4_3'] = build_net(
        'conv', net['conv4_2'], get_weight_bias(vgg_layers, 23))
    net['conv4_4'] = build_net(
        'conv', net['conv4_3'], get_weight_bias(vgg_layers, 25))
    net['pool4'] = build_net('pool', net['conv4_4'])
    net['conv5_1'] = build_net(
        'conv', net['pool4'], get_weight_bias(vgg_layers, 28))
    net['conv5_2'] = build_net(
        'conv', net['conv5_1'], get_weight_bias(vgg_layers, 30))
    net['conv5_3'] = build_net(
        'conv', net['conv5_2'], get_weight_bias(vgg_layers, 32))
    net['conv5_4'] = build_net(
        'conv', net['conv5_3'], get_weight_bias(vgg_layers, 34))
    net['pool5'] = build_net('pool', net['conv5_4'])
    return net
