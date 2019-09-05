import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from autoencoder import Model
from utils import DataGenerator
from keras import backend as K
from utils import Huffman_Encoder, Huffman_Decoder
import pickle
K.set_learning_phase(1) #set learning phase

def PSNR(X, X_ref):
    mse = np.mean(np.square(X - X_ref))
    psnr = 10.*np.log10((255.**2)/mse)
    return psnr

def info(path, write_video=False):
    datagen = DataGenerator(path, resize=True, height=height, width=width)
    data_size = datagen.data_size
    image_dim = datagen.height * datagen.width * 3
    batch_num = int(np.ceil(data_size/batch_size))
    mse1 = 0.
    mse2 = 0.
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path+"_out.avi", fourcc, 30.0, (width, height))
    for i in range(batch_num):
        if (i == batch_num-1 and data_size % batch_size != 0):
            residual_batch, compressed_batch = datagen.next_batch(int(data_size % batch_size))
        else:
            residual_batch, compressed_batch = datagen.next_batch(batch_size)
        residual_batch /= 255.
        compressed_batch /= 255.
        x_bin, residual_pred = sess.run([model.discrete, model.output], 
            {model.x_batch: residual_batch})
        #reconstructed residual
        output_batch = residual_pred + compressed_batch
        original_batch = residual_batch + compressed_batch
        mse1 += np.sum(np.square(compressed_batch*255. - original_batch*255.))
        mse2 += np.sum(np.square(output_batch*255. - original_batch*255.))
        if write_video:
            for i in range(len(output_batch)):
                frame = (np.clip(output_batch, 0, 1)*255.).astype(np.uint8)
                out.write(frame)
    if write_video:
        out.release()
    mse1 /= data_size*image_dim
    mse2 /= data_size*image_dim
    psnr1 = 10.*np.log10((255.**2)/mse1)
    psnr2 = 10.*np.log10((255.**2)/mse2)
    return psnr1, psnr2

#original 720P
height = 720
width = 1280
set_gradient_checkpointing = False

#DataGenerator return: 
#residual: float, range [0, 255], compressed_video: float, range [0, 255]
path = "./data/NBA"
datagen = DataGenerator(path, resize=False, height=height, width=width)
data_size = datagen.data_size
epochs = 2000
batch_size = 10

model = Model(C=32, 
              height=height, 
              width=width, 
              gradient_checkpointing=set_gradient_checkpointing)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    loss = 0
    batch_num = int(np.ceil(data_size/batch_size))
    for i in range(batch_num):
        x_batch, _ = datagen.next_batch(batch_size)
        x_batch /= 255.
        x_batch = x_batch.astype(np.float32)
        if set_gradient_checkpointing:
            #split training process
            #1. store mid-layer output (Check point)
            discrete = sess.run(model.discrete, {
                model.x_batch: x_batch})
            #2. train decoder, and get the gradient of the mid-layer to 
            #   apply backprop to previous layers
            loss_, g_Hardtanh, _ = sess.run([model.loss, model.g_Hardtanh, model.train_decoder], {
                model.discrete: discrete, model.y_batch: x_batch})
            #3. train encoder by the given gradient
            sess.run(model.train_encoder, {
                model.x_batch: x_batch, model.g_Hardtanh: g_Hardtanh})
        else:
            loss_, _ = sess.run([model.loss, model.train_op], {
                model.x_batch: x_batch, model.y_batch: x_batch})
        loss += loss_
    print("epoch %d, loss = %f" % (epoch+1, loss/batch_num))
    
    if epoch % 10 == 0:
        #write_video = True if epoch % 20 == 0 else False
        #psnr1, psnr2 = info(path, write_video=write_video)
        #print("Training PSNR: Compressed Video = %f, Output Video = %f" % (psnr1, psnr2))
        #psnr1, psnr2 = info(path2, write_video=write_video)
        #print("Testing PSNR: Compressed Video = %f, Output Video = %f" % (psnr1, psnr2))
        
        #Save weights
        encoder_weights = sess.run(model.encoder_params)
        decoder_weights = sess.run(model.decoder_params)
        np.save("./saved_model/encoder_weights.npy", encoder_weights)
        np.save("./saved_model/decoder_weights.npy", decoder_weights)
        print("model saved")