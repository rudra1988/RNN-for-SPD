import tensorflow as tf
import numpy as np
import random
import pdb
import math, time, sys
from sklearn.model_selection import KFold


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
    data_split = data_split.swapaxes(0,1)
    label = np.zeros([s[1],class_num])
    label[:,true_label] = 1
    return data_split,label

def shuffle_to_batch(data,label,batch_size):
    batch_data = []
    batch_label = []
    total_num = (label.shape)[0]
    list_ = random.sample(range(total_num),total_num)  # shuffle the data and label with the same index
    data = data[list_,:,:,:]
    label = label[list_,:]
    for i in range(0,total_num,batch_size):
        temp_data = data[i:i+batch_size,:,:,:]
        temp_label = label[i:i+batch_size,:]
        batch_data.append(temp_data)
        batch_label.append(temp_label)
    return batch_data,batch_label




batch_size = 50
height = 64
width = 64
in_channel = 1 #5
out_channel = 15
tot_time_points = 20
class_num = len(sys.argv)-1 #change this
#matrix_size = out_channel+1
epoch_num = 200 #240
depth = 5
reduced_spatial_dim = 256
beta = 0.3
units = 10

eps = 1e-10
#n = matrix_size
a = [0.01, 0.25, 0.5, 0.9, 0.99]
a_num = len(a)

lr = 0.9
decay_steps = 1000
decay_rate = 0.99

matrix_length = tot_time_points - in_channel + 1
global_steps = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(lr, global_step = global_steps, decay_steps = decay_steps, decay_rate = decay_rate)
add_global = global_steps.assign_add(1)




Weights_rnn = []
Bias_rnn = []


X = tf.placeholder(np.float32,shape = (batch_size,matrix_length,height,width,in_channel)) 
y = tf.placeholder(np.float32,shape = (batch_size,class_num)) 

Weights_cnn = {
            'W1':tf.Variable(tf.random_normal([5,5,in_channel,10],stddev=1e-4)),
            'W2':tf.Variable(tf.random_normal([5,5,10,out_channel],stddev=1e-4))
            #'W3':tf.Variable(tf.random_normal([3,3,15,out_channel],stddev=1e-4))
            #'W4':tf.Variable(tf.random_normal([3,3,20,25],stddev=1e-4)),
            #'W5':tf.Variable(tf.random_normal([3,3,25,30],stddev=1e-4))
            }


W2_1 = tf.Variable(tf.random_normal([units, class_num],stddev=np.sqrt(2./(class_num*units))))
b2_1 = tf.Variable(tf.random_normal([1, class_num],stddev=np.sqrt(2./class_num)))

inputs_series = tf.unstack(tf.transpose(X,[1,0,2,3,4]))
Fl_out = None
tf.keras.backend.set_learning_phase(True)

for current_X in inputs_series:
    ### CNN
    C1_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(current_X,Weights_cnn['W1'],[1,1,1,1],'SAME'))
    C1 = tf.nn.relu(C1_bn)
    P1 = tf.nn.max_pool(C1,[1,2,2,1],[1,2,2,1],'SAME')
    C2_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(P1,Weights_cnn['W2'],[1,1,1,1],'SAME'))
    C2 = tf.nn.relu(C2_bn)
    P2 = tf.nn.max_pool(C2,[1,2,2,1],[1,2,2,1],'SAME')
    #C3_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(P2,Weights_cnn['W3'],[1,1,1,1],'SAME'))
    #C3 = tf.nn.relu(C3_bn)
    #P3 = tf.nn.max_pool(C3,[1,2,2,1],[1,2,2,1],'SAME')
    #C4_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(P3,Weights_cnn['W4'],[1,1,1,1],'SAME'))
    #C4 = tf.nn.relu(C4_bn)
    ##P4 = tf.nn.max_pool(C4,[1,2,2,1],[1,2,2,1],'SAME')
    #C5_bn = tf.keras.layers.BatchNormalization()(tf.nn.conv2d(C4,Weights_cnn['W5'],[1,1,1,1],'SAME'))
    #C5 = tf.nn.relu(C5_bn)
    #P5 = tf.nn.max_pool(C5,[1,2,2,1],[1,2,2,1],'SAME')
    
    P2 = tf.transpose(P2,[0,3,2,1])
    Fl = tf.reshape(P2,[batch_size,-1])
    ##End of CNN block
    if Fl_out is None:
       Fl_out = tf.expand_dims(Fl,1)
    else:
       Fl_out = tf.concat([Fl_out, tf.expand_dims(Fl,1)],axis=1)
    
  
lstm_out = tf.keras.layers.LSTM(units,return_sequences=False)(Fl_out)  
predict_label = tf.add( tf.matmul ( lstm_out, W2_1 ), b2_1 )
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
     logits = predict_label,
     labels = y
))

correct_prediction = tf.equal(tf.argmax(predict_label, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.control_dependencies([add_global]):
    #opt = tf.train.AdagradOptimizer(learning_rate)
    #opt = tf.train.RMSPropOptimizer(learning_rate,momentum=0.9)
    opt = tf.train.AdadeltaOptimizer(learning_rate)
    train_step = opt.minimize(loss)

'''
load data and label here
and make them randomly and group into batch
'''

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

batch_data,batch_label = shuffle_to_batch(data,label,batch_size)

batch_num = len(batch_data)
batch_num_idx = range(batch_num)
k_fold = KFold(n_splits=10)
final_acc_fold = np.zeros((10,1))
with tf.Session() as sess:
    final_acc = 0.
    co = 0
    for tr_indices, ts_indices in k_fold.split(batch_num_idx):
        sess.run(tf.global_variables_initializer())
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        start_time = time.time()
        for epoch in range(epoch_num):
            for batch_idx in tr_indices:
                data_batch_in = np.reshape(batch_data[batch_idx],[batch_size,matrix_length,height,width,in_channel])
                label_batch_in = np.reshape(batch_label[batch_idx],[batch_size,class_num])
                #pdb.set_trace()
                #print batch_idx
                #CL_,CC_ ,current_X_= sess.run([CL,CC,current_X],
                #_, loss_ , predict_label_,Weights_,Rt_,Yt_,tt_,Phit_,Mt_,St_,yt_,ot_,Bias_,W2_,b2_,grad_= sess.run([train_step,loss,predict_label,Weights,Rt,Yt,tt,Phit,Mt,St,yt,ot,Bias,W2,b2,grad],
                _, loss_, predict_, W2_1_,y_, acc_ = sess.run([train_step,loss,predict_label,W2_1,y,accuracy],
                #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                         feed_dict={
                               X:data_batch_in,
                               y:label_batch_in,
                                })
                #pdb.set_trace()
                # if not batch_idx%100:
                #pdb.set_trace()
                if math.isnan(loss_):
                   print(grad_)
                   pdb.set_trace()
                else:
                   print(loss_,acc_,epoch)
                # print predict_label_
        print(time.time()-start_time)
        final_acc_fold[co] = 0.
        for batch_idx in ts_indices:
            data_batch_in = np.reshape(batch_data[batch_idx],[batch_size,matrix_length,height,width,in_channel])
            label_batch_in = np.reshape(batch_label[batch_idx],[batch_size,class_num])
            loss_, acc_ = sess.run([loss,accuracy],
                #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                         feed_dict={
                               X:data_batch_in,
                               y:label_batch_in,
                                })
            final_acc_fold[co] = final_acc_fold[co] + 1.0*acc_/len(ts_indices)
            print(loss_,acc_)
        print('After kth fold' , final_acc_fold[co])
        final_acc = final_acc + final_acc_fold[co]*1.0/10
        co += 1
    print(final_acc)
    np.save('lstm_30_60.npy',final_acc_fold)
