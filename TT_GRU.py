_author__ = "Yinchong Yang"
__copyright__ = "Siemens AG, 2017"
__licencse__ = "MIT"
__version__ = "0.1"

"""
MIT License
Copyright (c) 2017 Siemens AG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import numpy as np
import pickle
import datetime
import random
import sys

from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Masking, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

# Custom Functions -----------------------------------------------------------------------------------------------------
from TTRNN import TT_GRU, TT_LSTM


def Readdata(file_address,tot_time_points, height, width, true_label,class_num,in_channel):
    data = np.load(file_address)
    data=data['arr_0']
    s = data.shape
    assert s[0] == tot_time_points and s[2]==height and s[3]==width
    num_points = s[0]-in_channel+1
    data_split = (np.expand_dims(data[0:in_channel,...],axis=4)).swapaxes(0,4)
    data = np.delete(data,[0],axis=0)
    for i in range(1,num_points):
        temp_point = (np.expand_dims(data[0:in_channel,...],axis=4)).swapaxes(0,4)
        data = np.delete(data,[0],axis=0)
        data_split = np.concatenate([data_split, temp_point],axis=0)
    data_split = data_split.swapaxes(0,1).reshape(-1,20,64*64)
    label = np.zeros([s[1],class_num])
    label[:,true_label] = 1
    return data_split,label

def shuffle_to_batch(data,label):
    batch_data = []
    batch_label = []
    total_num = (label.shape)[0]
    list_ = random.sample(range(total_num),total_num)  # shuffle the data and label with the same index
    data = data[list_,...]
    label = label[list_,...]
    
    return data,label

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

batch_size = 50
height = 64
width = 64
tot_time_points = 20
class_num = len(sys.argv)-1
epoch_num = 400
in_channel = 1


GLOBAL_MAX_LEN = tot_time_points

#########################Change the below lines for more classes###########################################################################################################
for i in range(class_num):
    data0,label0 = Readdata(file_address='../'+sys.argv[i+1],tot_time_points=tot_time_points,height=height,width=width,true_label=i,class_num=class_num, in_channel=in_channel)
    if i==0:
       data = data0
       label = label0
    else:
       data = np.concatenate((data,data0),axis = 0)
       label = np.concatenate((label,label0),axis = 0)
############################################################################################################################################################################


data, label = shuffle_to_batch(data, label)
tr_idx = int(data.shape[0]*0.9)
X_train = data[0:tr_idx,...]
Y_train = label[0:tr_idx,...]

X_test = data[tr_idx:,...]
Y_test = label[tr_idx:,...]


# Define the model -----------------------------------------------------------------------------------------------------
tt_input_shape = [4, 8, 8, 16]
tt_output_shape = [4, 4, 4, 4]
tt_ranks = [1, 4, 4, 4, 1]
alpha = 1e-2

input = Input(shape=(GLOBAL_MAX_LEN, 64*64))
masked_input = Masking(mask_value=0, input_shape=(GLOBAL_MAX_LEN, 64*64))(input)
rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                           tt_ranks=tt_ranks,
                            return_sequences=False,
                            dropout=0.25, recurrent_dropout=0.25, activation='tanh')
h = rnn_layer(masked_input)
output = Dense(output_dim=class_num, activation='softmax', kernel_regularizer=l2(alpha))(h)
model = Model(input, output)
model.summary()

# Start training -------------------------------------------------------------------------------------------------------

batch_num_idx = range(data.shape[0])
k_fold = KFold(n_splits=10)
final_acc_fold = np.zeros((10,1))

start = datetime.datetime.now()
co = 0
final_acc = 0.
for tr_indices, ts_indices in k_fold.split(batch_num_idx):
    X_train = data[tr_indices,...]
    Y_train = label[tr_indices,...]
    X_test = data[ts_indices,...]
    Y_test = label[ts_indices,...]
    reset_weights(model)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=epoch_num, batch_size=50, verbose=1)

    # if l % 10 == 0:
    #     save_name = str(CV_setting) + '_' + str(model_type) + '_' + str(use_TT)
    #     write_out = open(write_out_path + save_name +'.pkl', 'wb')
    #     pickle.dump(model.get_weights(), write_out)
    #     write_out.close()

    final_acc_fold[co] = model.evaluate(X_test, Y_test)[1]
    print('After kth fold' , final_acc_fold[co])
    final_acc = final_acc + final_acc_fold[co]*1.0/10
    co += 1
np.save('TT_GRU_10_15.npy',final_acc_fold)

