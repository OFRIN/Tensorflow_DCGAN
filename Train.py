import cv2
import numpy as np
import tensorflow as tf

from DCGAN import *
from Utils import *
from Define import *

# 1. load dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(MNIST_DB_DIR, one_hot = True, reshape = [])

print('[i] mnist train data : {}'.format(mnist.train.images.shape))
print('[i] mnist test data : {}'.format(mnist.test.images.shape))

# 2. build model
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
z_var = tf.placeholder(tf.float32, [None, 1, 1, HIDDEN_VECTOR_SIZE])

is_training = tf.placeholder(tf.bool)

G_z = Generator(z_var, is_training, reuse = False)

D_real, D_real_logits = Discriminator(input_var, is_training, reuse = False)
D_fake, D_fake_logits = Discriminator(G_z, is_training, reuse = True)

print('[i] Generator : {}'.format(G_z))
print('[i] Discriminator : {}'.format(D_real))

# 3. loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logits, labels = tf.ones([BATCH_SIZE, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.zeros([BATCH_SIZE, 1, 1, 1])))
D_loss_op = D_loss_real + D_loss_fake

G_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.ones([BATCH_SIZE, 1, 1, 1])))

# 4. select variables
vars = tf.trainable_variables()
D_vars = [var for var in vars if var.name.startswith('Discriminator')]
G_vars = [var for var in vars if var.name.startswith('Generator')]

# 5. optimizer
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_train_op = tf.train.AdamOptimizer(LEARNING_RATE, beta1 = 0.5).minimize(D_loss_op, var_list = D_vars)
    G_train_op = tf.train.AdamOptimizer(LEARNING_RATE, beta1 = 0.5).minimize(G_loss_op, var_list = G_vars)

# 6. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

real_images = (mnist.train.images - 0.5) / 0.5 # -1 ~ 1
fixed_z = np.random.normal(0, 1, (SAVE_WIDTH * SAVE_HEIGHT, 1, 1, HIDDEN_VECTOR_SIZE)) # 0 ~ 1

train_iteration = len(real_images) // BATCH_SIZE

for epoch in range(1, MAX_EPOCH + 1):
    
    G_loss_list = []
    D_loss_list = []

    # train
    for iter in range(train_iteration):
        batch_x = real_images[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]
        batch_z = np.random.normal(0, 1, (BATCH_SIZE, 1, 1, 100))

        _, D_loss = sess.run([D_train_op, D_loss_op], feed_dict = {input_var : batch_x, z_var : batch_z, is_training : True})
        D_loss_list.append(D_loss)
        
        batch_z = np.random.normal(0, 1, (BATCH_SIZE, 1, 1, 100))
        _, G_loss = sess.run([G_train_op, G_loss_op], feed_dict = {input_var : batch_x, z_var : batch_z, is_training : True})
        G_loss_list.append(G_loss)
    
    G_loss = np.mean(G_loss_list)
    D_loss = np.mean(D_loss_list)

    print('[i] epoch : {}, G_loss : {:.5f}, D_loss : {:.5f}'.format(epoch, G_loss, D_loss))

    # test
    fake_images = sess.run(G_z, feed_dict = {z_var : fixed_z, is_training : False})
    Save(fake_images, './results/{}.jpg'.format(epoch))

'''
[i] mnist train data : (55000, 28, 28, 1)
[i] mnist test data : (10000, 28, 28, 1)
[i] Generator : Tensor("Generator/Tanh:0", shape=(?, 28, 28, 1), dtype=float32)
[i] Discriminator : Tensor("Discriminator/conv2d_2/BiasAdd:0", shape=(?, 1, 1, 1), dtype=float32)
[i] epoch : 0, G_loss : 0.65148, D_loss : 1.14092
[i] epoch : 1, G_loss : 0.65081, D_loss : 1.12089
[i] epoch : 2, G_loss : 0.64525, D_loss : 1.14597
[i] epoch : 3, G_loss : 0.64455, D_loss : 1.14715
[i] epoch : 4, G_loss : 0.64825, D_loss : 1.13305
[i] epoch : 5, G_loss : 0.65157, D_loss : 1.12158
[i] epoch : 6, G_loss : 0.65143, D_loss : 1.12256
[i] epoch : 7, G_loss : 0.65296, D_loss : 1.12234
[i] epoch : 8, G_loss : 0.65504, D_loss : 1.11843
[i] epoch : 9, G_loss : 0.65405, D_loss : 1.11460
[i] epoch : 10, G_loss : 0.65420, D_loss : 1.11801
[i] epoch : 11, G_loss : 0.65582, D_loss : 1.10998
[i] epoch : 12, G_loss : 0.65601, D_loss : 1.11006
[i] epoch : 13, G_loss : 0.65727, D_loss : 1.10695
[i] epoch : 14, G_loss : 0.65835, D_loss : 1.10730
[i] epoch : 15, G_loss : 0.65815, D_loss : 1.10332
[i] epoch : 16, G_loss : 0.65880, D_loss : 1.10284
[i] epoch : 17, G_loss : 0.66001, D_loss : 1.10050
[i] epoch : 18, G_loss : 0.65901, D_loss : 1.10166
[i] epoch : 19, G_loss : 0.66140, D_loss : 1.09691
'''