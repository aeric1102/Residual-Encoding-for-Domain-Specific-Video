import numpy as np
np.random.seed(0)
import cv2
from dahuffman import HuffmanCodec

class DataGenerator(object):
    def __init__(self, path, shuffle=False, resize=False, height=None, width=None):
        if (shuffle == True):
            raise Exception("Shuffle not defined")
        self.raw_file = path + "_raw.mp4"
        self.com_file = path + "_com.mp4"
        self.raw = cv2.VideoCapture(self.raw_file)
        self.com = cv2.VideoCapture(self.com_file)
        if(not self.raw.isOpened()):
            raise Exception(self.raw_file + ": read error")
        if(not self.com.isOpened()):
            raise Exception(self.com_file + ": read error")
        self._epochs_completed = 0
        self.data_size = self.raw.get(cv2.CAP_PROP_FRAME_COUNT)
        self.shuffle = shuffle
        self.resize = resize
        if(resize):
            self.height = height
            self.width = width
        else:
            self.height = self.raw.get(cv2.CAP_PROP_FRAME_HEIGHT) 
            self.width = self.raw.get(cv2.CAP_PROP_FRAME_WIDTH) 
    
    def next_batch(self, batch_size):
        """Return the next batch_size examples from this data set."""
        compressed_data = []
        residual_data = []
        index = 0
        assert batch_size <= self.data_size
        while index < batch_size:
            r_ret, r_frame = self.raw.read()
            c_ret, c_frame = self.com.read()
            if(r_ret and c_ret):
                if(self.resize):
                    r_frame = cv2.resize(r_frame, (self.width, self.height), 
                                         interpolation=cv2.INTER_CUBIC)
                    c_frame = cv2.resize(c_frame, (self.width, self.height), 
                                         interpolation=cv2.INTER_CUBIC) 
                #cast to float, avoid overflow
                residual_frame = r_frame.astype(float) - c_frame.astype(float)
                compressed_data.append(c_frame.astype(float))
                residual_data.append(residual_frame)   
                index += 1
            else:
                self._epochs_completed += 1
                self.raw.release()
                self.com.release()
                self.raw = cv2.VideoCapture(self.raw_file)
                self.com = cv2.VideoCapture(self.com_file)
                if(not self.raw.isOpened()):
                    raise Exception(self.raw_file + ": read error")
                if(not self.com.isOpened()):
                    raise Exception(self.com_file + ": read error")
        residual_data = np.asarray(residual_data)
        compressed_data = np.asarray(compressed_data)
        return residual_data, compressed_data
"""
def write_residual(path, resize=False, height=None, width=None):
    raw_file = path + ".avi"
    com_file = path + ".mp4"
    raw = cv2.VideoCapture(raw_file)
    com = cv2.VideoCapture(com_file)
    if(not raw.isOpened()):
        raise Exception(raw_file + ": read error")
    if(not com.isOpened()):
        raise Exception(com_file + ": read error")
    data_size = raw.get(cv2.CAP_PROP_FRAME_COUNT)
    if(resize):
        height = height
        width = width
    else:
        height = int(raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(raw.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path+"_residual.avi", fourcc, 30.0, (width, height))
    residual_data=[]
    while(raw.isOpened() and com.isOpened()):
        r_ret, r_frame = raw.read()
        c_ret, c_frame = com.read()
        if(r_ret and c_ret):
            if(resize):
                r_frame = cv2.resize(r_frame, (width, height), 
                                     interpolation=cv2.INTER_CUBIC)
                c_frame = cv2.resize(c_frame, (width, height), 
                                     interpolation=cv2.INTER_CUBIC) 
            #cast to float, avoid overflow
            residual_frame = (r_frame ^ c_frame)
            out.write(residual_frame)
            residual_data.append(residual_frame)
        else:
            break
    return np.asarray(residual_data)
"""
def float2bin(X):
    fbin = np.asarray(X, dtype=int)
    cond = (fbin==-1)
    fbin[cond] = 0
    rbin = np.asarray(fbin, dtype=bool)
    return rbin

def bin2float(X):
    rbin = np.asarray(X, dtype=int)
    cond = (rbin==0)
    rbin[cond] = -1
    fbin = np.asarray(rbin, dtype=float)
    return fbin

#input binary_map is flattened np array, type float
#output Python bytes
def Huffman_Encoder(binary_map, bits_group_num = 64):
    binary_map = float2bin(binary_map)
    map_length = len(binary_map)
    if map_length%bits_group_num:
        bools_list = list(binary_map[:-(map_length%bits_group_num)].reshape(-1, bits_group_num))
        bools_list.append(binary_map[-(map_length%bits_group_num):])
    else:
        bools_list = list(binary_map.reshape(-1, bits_group_num))
    bits_string = [b.tobytes() for b in bools_list]
    codec = HuffmanCodec.from_data(bits_string)
    output = codec.encode(bits_string)
    return output, codec

#input Python bytes
#output binary_map is flattened np array
def Huffman_Decoder(input, codec):
    bits_string = codec.decode(input)
    bools_list = [np.frombuffer(b, dtype=np.bool) for b in bits_string]
    binary_map = [bool for bools in bools_list for bool in bools]
    return bin2float(np.asarray(binary_map))
