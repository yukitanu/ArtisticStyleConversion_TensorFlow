import tensorflow as tf
import numpy as np
import scipy.misc
import os
from model import vgg19_pretrained

CONTENT_IMG = 'images/eiffel.jpg'
STYLE_IMG = 'images/starrynight.jpg'

OUTPUT_DIR = 'results'
OUTPUT_IMG = 'result.png'

width = 400
height = 400

MEAN_VALUES = np.array([123, 117, 104]).reshape((1, 1, 1, 3))


def build_content_loss(p, x):
    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1. / (2 * N ** 0.5 * M ** 0.5)) * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def gram_matrix(x, area, depth):
    x1 = tf.reshape(x,(area,depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g


def gram_matrix_val(x, area, depth):
    x1 = x.reshape(area,depth)
    g = np.dot(x1.T, x1)
    return g


def build_style_loss(a, x):
    M = a.shape[1] * a.shape[2]
    N = a.shape[3]
    A = gram_matrix_val(a, M, N )
    G = gram_matrix(x, M, N )
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


def read_image(path):
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (height, width))
    image = image[np.newaxis, : , :, :]
    image = image - MEAN_VALUES
    return image


def write_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


#net = vgg19(height, width)
net = vgg19_pretrained(height, width)
noise_img = np.random.uniform(-20, 20, (1, height, width, 3)).astype('float32')
content_img = read_image(CONTENT_IMG)
style_img = read_image(STYLE_IMG)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

INIT_NOISE_RATIO = 0.7
STYLE_STRENGTH = 1000
ITERATION = 4000

CONTENT_LAYERS =[('conv4_2',1.)]
STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]

sess.run([net['input'].assign(content_img)])
cost_content = sum(map(lambda l: l[1] * build_content_loss(sess.run(net[l[0]]), net[l[0]]), CONTENT_LAYERS))

sess.run([net['input'].assign(style_img)])
cost_style = sum(map(lambda l: l[1] * build_style_loss(sess.run(net[l[0]]), net[l[0]]), STYLE_LAYERS))

cost_total = cost_content + STYLE_STRENGTH * cost_style
optimizer = tf.train.AdamOptimizer(2.0)

train = optimizer.minimize(cost_total)
sess.run( tf.global_variables_initializer())
sess.run(net['input'].assign(INIT_NOISE_RATIO * noise_img + (1.-INIT_NOISE_RATIO) * content_img))

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

for i in range(ITERATION):
    sess.run(train)
    if i % 100 == 0:
        result_img = sess.run(net['input'])
        print ('ITERATION: ', i, ', ', sess.run(cost_total))
        write_image(os.path.join(OUTPUT_DIR, '%s.png' % (str(i).zfill(4))), result_img)

write_image(os.path.join(OUTPUT_DIR, OUTPUT_IMG), result_img)