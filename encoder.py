import tensorflow as tf
from keras.layers import Activation, Conv2D, BatchNormalization, Lambda
from keras import backend as K
import numpy as np
import cv2
import pickle
from utils import DataGenerator
from utils import Huffman_Encoder
K.set_learning_phase(1) #set learning phase

class Encoder():
    def __init__(self, C, height, width):
        self.C = C
        self.height = height
        self.width = width
        self.x_batch = tf.placeholder(tf.float32, [None, height, width, 3], name = "x_batch")
        self.output = self.build_net()
        self.encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
    
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
            htanh = Lambda(Hardtanh, name="Hardtanh")(bnorm_3)
            output = Lambda(Discretization, name="Discretization")(htanh)
        return output
    
    def load_weights(self, sess, weight_file):
        weights = np.load(weight_file)
        assert len(weights) == len(self.encoder_params)
        for i in range(len(self.encoder_params)):
            sess.run(self.encoder_params[i].assign(weights[i]))

height = 720
width = 1280
bits_group_num = 16
path = "./data/NBA"
weight_file = "./saved_model/encoder_weights.npy"
batch_size = 10

model = Encoder(C=32, height=height, width=width)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.load_weights(sess, weight_file)

datagen = DataGenerator(path, resize=False, height=height, width=width)
data_size = datagen.data_size
batch_num = int(np.ceil(data_size/batch_size))

#residual encoder
x_binary=[]
for i in range(batch_num):
    if (i == batch_num-1 and data_size % batch_size != 0):
        residual_batch, _ = datagen.next_batch(int(data_size % batch_size))
    else:
        residual_batch, _ = datagen.next_batch(batch_size)
    residual_batch /= 255.
    x_binary = np.append(x_binary, sess.run(model.output, {model.x_batch: residual_batch}))

print("---Residual Encoded---")
#huffman encoder
x_encoded, huffman_codec = Huffman_Encoder(x_binary, bits_group_num)
with open("./saved_model/huffman_codec", 'wb') as f:
    pickle.dump(huffman_codec, f)
with open(path+"_residual.npy", 'wb') as f:
    f.write(x_encoded)
print("---Huffman Encoded---")