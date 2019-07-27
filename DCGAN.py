
import tensorflow as tf

def Generator(x, isTraining, reuse = False, name = 'Generator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        x = tf.layers.conv2d_transpose(x, 256, [7, 7], strides=1, padding='valid')
        x = tf.layers.batch_normalization(x, training=isTraining)
        x = tf.nn.leaky_relu(x, 0.2)

        x = tf.layers.conv2d_transpose(x, 128, [5, 5], strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTraining)
        x = tf.nn.leaky_relu(x, 0.2)
        
        x = tf.layers.conv2d_transpose(x, 1, [5, 5], strides=2, padding='same')
        x = tf.nn.tanh(x)

        return x

def Discriminator(x, isTraining, reuse = False, name = 'Discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        x = tf.layers.conv2d(x, 128, [5, 5], strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTraining)
        x = tf.nn.leaky_relu(x, 0.2)

        x = tf.layers.conv2d(x, 256, [5, 5], strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTraining)
        x = tf.nn.leaky_relu(x, 0.2)

        logits = tf.layers.conv2d(x, 1, [7, 7], strides=1, padding='valid')
        predictions = tf.nn.sigmoid(logits)

        return logits, predictions
