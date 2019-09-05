import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Conv2D, BatchNormalization, Lambda
import numpy as np

#ref: https://github.com/tetrachrome/subpixel
class Subpixel(Layer):
    def __init__(self, r, **kwargs):
        self.r = r
        super(Subpixel, self).__init__(**kwargs)
    
    def _phase_shift(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, [bsize, a, b, c//(r*r), r, r]) # bsize, a, b, c/(r*r), r, r
        X = tf.transpose(X, (0, 1, 2, 5, 4, 3)) # bsize, a, b, r, r, c/(r*r)
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r, c/(r*r)]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r, c/(r*r)
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r, c/(r*r)]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r, c/(r*r)
        return X
    
    def call(self, x):
        return self._phase_shift(x, self.r)
    
    def compute_output_shape(self, input_shape):
        bsize, a, b, c = input_shape
        return (bsize, a*self.r, b*self.r, c//(self.r*self.r))

class Model():
    def __init__(self, C, height, width, gradient_checkpointing=False):
        self.C = C
        self.height = height
        self.width = width
        self.x_batch = tf.placeholder(tf.float32, [None, height, width, 3], name = "x_batch")
        self.y_batch = tf.placeholder(tf.float32, [None, height, width, 3], name = "y_batch")
        self.output = self.build_net()
        self.loss = tf.losses.mean_squared_error(self.y_batch, self.output)
        #By using the straigh-through estimator,
        #the gradient of Hardtanh layer can be replaced by the gradient of Discretization layer
        encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
        decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        self.g_Hardtanh = tf.gradients(self.loss, self.discrete)[0]
        encoder_grads = tf.gradients(self.htanh, encoder_params, grad_ys=self.g_Hardtanh)
        decoder_grads = tf.gradients(self.loss, decoder_params)
        if not gradient_checkpointing:
            grads_and_vars = list(zip(
                encoder_grads+decoder_grads, encoder_params+decoder_params))
            self.train_op = tf.train.AdamOptimizer().apply_gradients(grads_and_vars)
        else:
            #To be memory efficient, we use gradient checkpoint
            decoder_grads_and_vars = list(zip(decoder_grads, decoder_params))
            encoder_grads_and_vars = list(zip(encoder_grads, encoder_params))
            self.train_decoder = tf.train.AdamOptimizer().apply_gradients(decoder_grads_and_vars)
            self.train_encoder = tf.train.AdamOptimizer().apply_gradients(encoder_grads_and_vars)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
    
    def build_net(self):
        def Hardtanh(x):
            return tf.clip_by_value(x, -1, +1)
        
        def Discretization(x):
            zeros = tf.zeros(tf.shape(x))
            ones = tf.ones(tf.shape(x))
            minus_ones = tf.fill(tf.shape(x), -1.)
            cond = tf.greater_equal(x, zeros)
            return tf.where(cond, x=ones, y=minus_ones)
        
        # encoder layers
        # We use L convolutional layers in our encoder, in which each
        # layer has the same channel number C and a stride of two
        # that down-samples feature maps.
        # model.add(BatchNormalization())   
        C = self.C
        height = self.height
        width = self.width     
        with tf.variable_scope('encoder'):
            conv_1 = Conv2D(C, (3, 3), strides=(2, 2), input_shape=(height, width, 3), padding='same')(self.x_batch)
            bnorm_1 = BatchNormalization()(conv_1)
            relu_1 = Activation('relu')(bnorm_1)
            
            conv_2 = Conv2D(C, (3, 3), strides=(2, 2), padding='same')(relu_1)
            bnorm_2 = BatchNormalization()(conv_2)
            relu_2 = Activation('relu')(bnorm_2)
            
            conv_3 = Conv2D(C, (3, 3), strides=(2, 2), padding='same')(relu_2)
            bnorm_3 = BatchNormalization()(conv_3)
        
        # binarization
        with tf.variable_scope('binarizer'):
            self.htanh = Lambda(Hardtanh, name="Hardtanh")(bnorm_3)
            self.discrete = Lambda(Discretization, name="Discretization")(self.htanh)
        
        # decoder layers
        with tf.variable_scope('decoder'):
            r = 2
            conv_4 = Conv2D(r*r*C, (3, 3), strides=(1, 1), padding='same')(self.discrete)
            sub_4 = Subpixel(r=r)(conv_4)
            bnorm_4 = BatchNormalization()(sub_4)
            relu_4 = Activation('relu')(bnorm_4)
            
            conv_5 = Conv2D(r*r*C, (3, 3), strides=(1, 1), padding='same')(relu_4)
            sub_5 = Subpixel(r=r)(conv_5)
            bnorm_5 = BatchNormalization()(sub_5)
            relu_5 = Activation('relu')(bnorm_5)
            
            conv_6 = Conv2D(r*r*3, (3, 3), strides=(1, 1), padding='same')(relu_5)
            output = Subpixel(r=r)(conv_6)
        return output
