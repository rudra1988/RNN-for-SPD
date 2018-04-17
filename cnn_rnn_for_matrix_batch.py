import tensorflow as tf
import numpy as np
import random
import pdb
import math

from tensorflow.python.ops.distributions.util import fill_triangular

def f(x):
    return x
    return tf.nn.relu(x)

def myeigvalue(x):
    return list(np.linalg.eig(x))[0]

def myeigvector(x):
    return list(np.linalg.eig(x))[1]



def Eig(x):
    #return [tf.py_func(myeigvalue,[x],tf.float32),tf.py_func(myeigvector,[x],tf.float32)]
    return tf.self_adjoint_eig(x)

#def FM(A,B,a,n):
#     '''
#     input matrix A and matrix B and scalar a(lpha) and scalar n
#     n is the size of A (and B)
#     return the form of FM(A,B,a) in the paper
#     A * sqrt (A^-1 * B + (2a-1)/4*(I-A^-1 *B)^2) - (2a-1)/2*(I-A^-1 * B)
#     '''

#     AB = tf.matmul(tf.matrix_inverse(A),B) # A^-1 * B
#     IAB =  tf.subtract( tf.tile ( tf.expand_dims( tf.eye(n),0),[A.shape[0],1,1] )  , AB)         # I - A^-1 * B
#     eta = (2 * a - 1)/2                    # (2a-1)/2
#     before_root = tf.add( AB , eta * eta * tf.matmul(IAB,IAB)) # (A^-1 * B + (2a-1)/4*(I-A^-1 *B)^2)
#     before_root = tf.add (before_root , 1e-5 * tf.diag(tf.random_uniform([n])) )
#     S, U, V = tf.svd(before_root)
#     Sigma_root = tf.matrix_diag(tf.sqrt(S))
#     after_root = tf.matmul(tf.matmul(U,Sigma_root),V, transpose_b=True) # calculate the square root by eig-decomponent
#     after_root = tf.add(after_root,1e-5 * tf.eye(n))
#     result = tf.subtract(after_root, eta*IAB)
#     result = tf.add( tf.matmul(A,result), 1e-5 * tf.eye(n) )
#     # result = tf.add(result,tf.transpose(result,[0,2,1]))/2
#     return result

def FM(A,B,a,n):
    return tf.add((1.-a)*A,a*B)


def NUS(W_root, A, a_num, tot, n=1):
    W = tf.pow(W_root,2)
    if a_num==1:
       return (W[0]/tot)*A
    else:
        #result = tf.squeeze(tf.slice(A,[0,0,0,0],[-1,1,-1,-1]))
        result = tf.squeeze(tf.slice(A,[0,0,0,0],[-1,1,-1,-1]))*(W[0]/ tot)
        for i in range(1, A.shape[1]):
            #t = tf.reduce_sum(tf.slice(W,[0],[i+1]))
            result = result + tf.squeeze(tf.slice(A,[0,i,0,0],[-1,1,-1,-1]))*(W[i]/tot)
            #result = FM(result, tf.squeeze(tf.slice(A,[0,i,0,0],[-1,1,-1,-1])),W[i]/t,n)
        return result



def MatrixExp(B,l,n):
    '''
    input a matrix B, and the total length to be calculated, n is the size of B
    output the somehow exp(B) = I + B + B^2 / 2! + B^3 / 3! + ... + B^l / l!
    '''
    
    Result = tf.eye(n)
    # temp_result = tf.eye(n)
    # factorial = 1.
    # for i in range(1,l+1):
    #     temp_result = tf.matmul(temp_result,B)
    #     factorial = factorial * (i)
    #     Result = tf.add(Result,temp_result/factorial)
    # return tf.matrix_inverse ( tf.matrix_inverse ( Result ) )

    return tf.matmul( tf.matrix_inverse(tf.subtract(Result , B)) , tf.add( Result , B) )

def Translation(A,B,n, batch_size):

    '''
    input the matrix A and vector B
    change B to be SO 
    like [[0 ,  1, 2]
          [-1,  0, 3]
          [-2, -3, 0]]
    return B * A * B.T
    '''
    power_matrix = 5
    B = tf.reshape(B,[1,-1])

    #lower_triangel = fill_triangular(B)
    line_B = [tf.zeros([1,n])]
    for i in range (n-1):
        temp_line = tf.concat([ tf.slice(B,[0,i],[1,i+1]) , tf.zeros([1,n-i-1]) ] ,axis = 1)
        line_B.append(temp_line)

    lower_triangel = tf.concat(line_B,axis = 0)

    B_matrix = tf.subtract(lower_triangel, tf.transpose(lower_triangel))
    
    B_matrix = MatrixExp(B_matrix,power_matrix,n)

    B_matrix = tf.tile ( tf.expand_dims(B_matrix,0),[batch_size,1,1] )

 

    Tresult = tf.matmul(B_matrix,A)                              # B * A

    Tresult = tf.matmul(Tresult,tf.transpose(B_matrix,[0,2,1]))      # B * A * B.T
    return Tresult


def Chol_de(A,n,batch_size):
    '''
    input matrix A and it's size n
    decomponent by Cholesky
    return a vector with size n*(n+1)/2
    '''
    #A = tf.add (A , 1e-10 * tf.diag(tf.random_uniform([n])) )
    # A = tf.cond( 
    #     tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) , 
    #     lambda: A, 
    #     lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
    #L = tf.cholesky(A)

    L = A
    result = tf.slice(L,[0,0,0],[-1,1,1])
    for i in range(1,n):
        j = i
        result = tf.concat( [result , tf.slice(L,[0,i,0],[-1,1,j+1])],axis = 2 )

    result = tf.reshape(result,[-1,n*(n+1)//2])
    return result


#def Chol_de(A,n,batch_size):
#    '''
#    input matrix A and it's size n
#    decomponent by Cholesky
#    return a vector with size n*(n+1)/2
#    '''
#    #A = tf.add (A , 1e-10 * tf.diag(tf.random_uniform([n])) )
#    # A = tf.cond( 
#    #     tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) , 
#    #     lambda: A, 
#    #     lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
#
#    A = tf.add(A , tf.tile(tf.expand_dims(tf.eye(n)*1e-3,axis=0),[batch_size,1,1]) )
#    L = tf.cholesky(A)
#    #L = A
#    result = tf.slice(L,[0,0,0],[-1,1,1])
#    for i in range(1,n):
#        j = i
#        result = tf.concat( [result , tf.slice(L,[0,i,0],[-1,1,j+1])],axis = 2 )
#    result = tf.reshape(result,[-1,n*(n+1)//2])
#    return result

#def Chol_com(l,n,eps,batch_size):
#    '''
#    input vector l and target shape n and eps to be the smallest value
#    return lower triangle matrix
#    '''
#    #batch_size = l.shape[0]
#    lower_triangle_ = []
#    for i in range(n):
#        #temp_ = tf.placeholder(tf.float32, shape=[batch_size, n-i-1])
#        #l[:,i*(i+1)//2] = 0.5*l[:,i*(i+1)//2]
#        lower_triangle_.append( tf.expand_dims ( tf.concat( [tf.slice(l,[0,i*(i+1)//2],[-1,i+1]) , tf.zeros((batch_size,n-i-1)) ] , axis = 1 ) , -1 ) )#
#
#    lower_triangle = tf.concat (lower_triangle_ , axis = 2)
#    #result = lower_triangle
#    result = []
#    #diag = tf.diag_part(tf.reshape(tf.slice(lower_triangle,[0,0,0],[-1,n,n]),[-1,n,n]))
#    #diag = tf.diag(diag)
#    #result = tf.subtract ( tf.slice(lower_triangle,[0,0,0],[-1,n,n]),diag )
#    for i in range(batch_size):
#        diag = tf.diag_part(tf.reshape(tf.slice(lower_triangle,[i,0,0],[1,n,n]),[n,n]))
#        #diag = tf.clip_by_value(diag,-np.inf,-eps)
#        diag = tf.diag(diag)
#        result.append( tf.subtract ( tf.slice(lower_triangle,[i,0,0],[1,n,n]),diag ))
#    return  tf.add(  tf.add(tf.concat(result,axis = 0) , tf.transpose(lower_triangle,[0,2,1]) )  , 0 * tf.eye(n) ) # make diag element to be eps or positive


def Chol_com(l,n,batch_size):
    '''
    input vector l and target shape n and eps to be the smallest value
    return lower trangle matrix
    '''
    lower_triangle_ = tf.expand_dims(tf.concat([tf.slice(l,[0,0],[-1,1]), tf.zeros((batch_size,n-1))],axis=1),1)
    for i in range(1, n):
        lower_triangle_ = tf.concat([lower_triangle_,tf.expand_dims(tf.concat([tf.slice(l,[0,i*(i+1)//2],[-1,i+1]), tf.zeros((batch_size,n-i-1))],axis=1),1)],axis=1)

    lower_triangle_ = tf.add(lower_triangle_ , tf.tile(tf.expand_dims(tf.eye(n)*1e-2,axis=0),[batch_size,1,1]) )
    result = tf.matmul(lower_triangle_,lower_triangle_,transpose_b=True)
    return result






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
in_channel = 5
out_channel = 15
tot_time_points = 20
class_num = 2
matrix_size = out_channel+1
epoch_num = 500
depth = 5
reduced_spatial_dim = 256
beta = 0.3

eps = 1e-10
n = matrix_size
a = [0.01, 0.25, 0.5, 0.9, 0.99]
a_num = len(a)

lr = 0.9
decay_steps = 1000
decay_rate = 0.99

matrix_length = tot_time_points - in_channel + 1
global_steps = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(lr, global_step = global_steps, decay_steps = decay_steps, decay_rate = decay_rate)
add_global = global_steps.assign_add(1)



X = tf.placeholder(np.float32,shape = (batch_size,matrix_length,height,width,in_channel)) 
y = tf.placeholder(np.float32,shape = (batch_size,class_num)) 


Weights_rnn = []
Bias_rnn = []

Weights_cnn = {
            'W1':tf.Variable(tf.random_normal([5,5,in_channel,10],stddev=1e-4)),
            'W2':tf.Variable(tf.random_normal([5,5,10,out_channel],stddev=1e-4))
            #'W3':tf.Variable(tf.random_normal([3,3,15,out_channel],stddev=1e-4))
            #'W4':tf.Variable(tf.random_normal([3,3,20,25],stddev=1e-4)),
            #'W5':tf.Variable(tf.random_normal([3,3,25,30],stddev=1e-4))
            }




for i in range(depth):
    Weights_rnn.append({
            'WR_root':tf.Variable(tf.random_uniform([a_num]),trainable=True),
            'Wt_root':tf.Variable(tf.random_uniform([1])),
            'Wphi_root':tf.Variable(tf.random_uniform([1])),
            'Ws_root':tf.Variable(tf.random_uniform([a_num])),
            #'wo':tf.Variable(tf.random_uniform([1, n*(n+1)//2])),
          })
    Bias_rnn.append({
            'Br':tf.Variable(tf.random_uniform([n*(n-1)//2,1])),
            'Bt':tf.Variable(tf.random_uniform([n*(n-1)//2,1])),
            'By':tf.Variable(tf.random_uniform([n*(n-1)//2,1])), # should be n*(n-1)/2, but when implementing it, I found that low - low.T will automatically give us n*(n-1)/2
            #'bo':tf.Variable(tf.random_uniform([1])),
          })


#W2_1 = tf.Variable(tf.random_normal([matrix_length*n*(n+1)//2, class_num],stddev=np.sqrt(2./(matrix_length*n*(n+1)//2*class_num))))  #following paper https://arxiv.org/pdf/1502.01852.pdf
#W2_1 = tf.Variable(tf.random_normal([n*(n+1)//2, matrix_length],stddev=np.sqrt(2./(matrix_length*n*(n+1)//2))))
W2_1 = tf.Variable(tf.random_normal([n*(n+1)//2, class_num],stddev=np.sqrt(2./(class_num*n*(n+1)//2))))
b2_1 = tf.Variable(tf.random_normal([1, class_num],stddev=np.sqrt(2./class_num)))
#W2_2 = tf.Variable(tf.random_normal([matrix_length, class_num],stddev=np.sqrt(2./(class_num*matrix_length))))
#b2_1 = tf.Variable(tf.random_normal([      1      , class_num],stddev=np.sqrt(2./class_num)))
#b2_2 = tf.Variable(tf.random_normal([      1      , class_num],stddev=np.sqrt(2./class_num)))



initMt = tf.placeholder(np.float32,[batch_size,a_num,n,n])

Mt_1 = initMt

output_series = None
inputs_series = tf.unstack(tf.transpose(X,[1,0,2,3,4]))


for current_X in inputs_series:
    cov_mat = None
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
    Fl = tf.reshape(P2,[batch_size,out_channel,reduced_spatial_dim])
    mean_batch = tf.reduce_mean(Fl,2)   #batch_size x out_channel
    mean_tensor = tf.tile(tf.expand_dims(mean_batch,axis=2),[1,1,reduced_spatial_dim]) #batch_size x out_channel x reduced_spatial_dim
    Fl_m = tf.subtract(Fl,mean_tensor)
    for i in range(batch_size):
        feat = tf.reshape(tf.slice(Fl_m,[i,0,0],[1,-1,-1]),[out_channel,reduced_spatial_dim])
        mean_vec = tf.reshape(tf.slice(mean_batch,[i,0],[1,-1]),[out_channel,1])
        mean_vec_t = tf.reshape(mean_vec, [1,out_channel])
        mean_cov = tf.matmul(mean_vec, mean_vec_t)
        #cov_feat = tf.add(tf.matmul(feat, feat, transpose_b=True)/reduced_spatial_dim, beta*beta*mean_cov)
        cov_feat = tf.add(tf.matmul(feat, feat, transpose_b=True), beta*beta*mean_cov)
        cov_feat = tf.concat([cov_feat, beta*mean_vec],axis=1)
        mean_vec_t = tf.concat([beta*mean_vec_t, tf.constant([1.],shape=[1,1])],axis=1)
        cov_feat = tf.concat([cov_feat, mean_vec_t],axis=0)
        cov_feat = tf.expand_dims(cov_feat, axis=0)
        if cov_mat is None:
           cov_mat = cov_feat
        else:
           cov_mat = tf.concat([cov_mat, cov_feat],axis=0)
    ##End of CNN and cov computation block
    ##RNN
    yt = cov_mat
    for i in range(depth):
        n_current_X = tf.reshape(yt,[batch_size,n,n])
        Yt =  NUS(Weights_rnn[i]['WR_root'], Mt_1, a_num, tf.reduce_sum(tf.pow(Weights_rnn[i]['WR_root'],2))+eps, n)
        Rt = Translation( Yt, Bias_rnn[i]['Br'] , n, batch_size )
        tt = FM(n_current_X, Rt, tf.pow(Weights_rnn[i]['Wt_root'],2)/(tf.reduce_sum([tf.pow(Weights_rnn[i]['Wt_root'],2), tf.pow(Weights_rnn[i]['Wphi_root'],2)])+eps), n)
        Phit = Translation ( tt, Bias_rnn[i]['Bt'] , n, batch_size )
        
        next_state = []
        for j in range(a_num):
            next_state.append (  tf.expand_dims ( FM ( tf.reshape ( tf.slice(Mt_1,[0,j,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) , Phit, a[j] , n ) , 1 ) )
        Mt = tf.concat(next_state,axis = 1)
        St =  NUS(Weights_rnn[i]['Ws_root'], Mt, a_num, tf.reduce_sum(tf.pow(Weights_rnn[i]['Ws_root'],2))+eps, n)
        yt = Translation ( St, Bias_rnn[i]['By'] , n, batch_size )
        #yt = Chol_com(tf.nn.relu(Chol_de(yt,n,batch_size)),n,batch_size)
        
        Mt_1 = Mt
    yt = tf.transpose( Chol_de ( yt, n,batch_size ) ,[1,0])
    #yt = tf.nn.tanh(tf.transpose( f( Chol_de ( yt, n )) ,[1,0]))
    if output_series is None:
        #output_series = ot
        output_series = yt
        #print output_series.shape
    else:
        #output_series = tf.concat([output_series,ot],axis = 0)
        output_series = tf.concat([output_series,yt],axis = 0)
    
output_series = tf.slice(output_series,[(matrix_length-1)*n*(n+1)//2,0],[n*(n+1)//2,-1])
output_series = tf.keras.layers.BatchNormalization()(tf.transpose(output_series,[1,0]))
#output_series_1 = tf.nn.relu( tf.add( tf.matmul ( output_series, W2_1 ), b2_1 ) )
predict_label = tf.nn.softmax( tf.add( tf.matmul ( output_series, W2_1 ), b2_1 ) )
#predict_label = tf.nn.softmax( tf.clip_by_value(tf.add( tf.matmul ( output_series_1, W2_2 ), b2_2 ), -50, 50) )
#predict_label = tf.nn.softmax( tf.nn.tanh(tf.add( tf.matmul ( output_series_1, W2_2 ), b2_2 )) )

#loss = tf.reduce_mean( tf.reduce_sum( -y * tf.log(predict_label+eps),1)) 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
     logits = predict_label,
     labels = y
))

correct_prediction = tf.equal(tf.argmax(predict_label, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#loss = tf.reduce_mean(tf.reduce_sum( tf.pow( y - predict_label , 2 ),1 ))

with tf.control_dependencies([add_global]):
    #opt = tf.train.AdagradOptimizer(learning_rate)
    #opt = tf.train.RMSPropOptimizer(learning_rate,momentum=0.9)
    opt = tf.train.AdadeltaOptimizer(learning_rate)
    train_step = opt.minimize(loss)
grad = tf.gradients(loss,[Weights_rnn[0]['WR_root'],Weights_rnn[0]['Wt_root'],Weights_rnn[0]['Ws_root'], Weights_rnn[0]['Wphi_root'],Weights_cnn['W1'],Weights_cnn['W2']])

'''
load data and label here
and make them randomly and group into batch
'''

data0,label0 = Readdata(file_address='../c1_data_10.npz',tot_time_points=tot_time_points,height=height,width=width,true_label=0,class_num=class_num, in_channel=in_channel)
data1,label1 = Readdata(file_address='../c1_data_15.npz',tot_time_points=tot_time_points,height=height,width=width,true_label=1,class_num=class_num, in_channel=in_channel)
data = np.append(data0,data1,axis = 0)
label = np.append(label0,label1,axis = 0)

batch_data,batch_label = shuffle_to_batch(data,label,batch_size)

batch_num = len(batch_data)

init_state = np.tile(np.eye(n)*1e-5,[batch_size,a_num,1,1]) 
#init_state =  np.tile(1e-5 * np.random.uniform(low=0.0,high=1.0,size=(n,n)),[batch_size,a_num,1,1]) 
loss_p = 0

training_batch_num = int(batch_num*0.9)

#CL = Chol_de(current_X,n)
#CC = Chol_com(CL,n,eps)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    for epoch in range(epoch_num):
        for batch_idx in range(training_batch_num):
            data_batch_in = np.reshape(batch_data[batch_idx],[batch_size,matrix_length,height,width,in_channel])
            label_batch_in = np.reshape(batch_label[batch_idx],[batch_size,class_num])
            #pdb.set_trace()
            #print batch_idx
            #CL_,CC_ ,current_X_= sess.run([CL,CC,current_X],
            #_, loss_ , predict_label_,Weights_,Rt_,Yt_,tt_,Phit_,Mt_,St_,yt_,ot_,Bias_,W2_,b2_,grad_= sess.run([train_step,loss,predict_label,Weights,Rt,Yt,tt,Phit,Mt,St,yt,ot,Bias,W2,b2,grad],
            _, loss_, Weights_, Bias_, grad_, predict_, W2_1_,y_, acc_ = sess.run([train_step,loss,Weights_rnn, Bias_rnn,grad,predict_label,W2_1,y,accuracy],
            #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                     feed_dict={
                           X:data_batch_in,
                           y:label_batch_in,
                           initMt:init_state,
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
    final_acc = 0.
    for batch_idx in range(training_batch_num, batch_num):
        data_batch_in = np.reshape(batch_data[batch_idx],[batch_size,matrix_length,height,width,in_channel])
        label_batch_in = np.reshape(batch_label[batch_idx],[batch_size,class_num])
        loss_, acc_ = sess.run([loss,accuracy],
            #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                     feed_dict={
                           X:data_batch_in,
                           y:label_batch_in,
                           initMt:init_state,
                            })
        final_acc = final_acc + 1.0*acc_/(batch_num-training_batch_num)
        print(loss_,acc_)
    print(final_acc)
    np.save('result_10_15.npy',final_acc)
