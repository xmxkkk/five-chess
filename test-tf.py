from record import Record
from sklearn.preprocessing import Normalizer
import numpy as np


data=Record()
x,board1,board2,board3,board4,board5,y=data.load(0,100000)
# print(y[0:100,0])
# print(np.mean(y))
# exit()

def handle(x):
    x[x==-99999]=0.1
    x[x==0]=0.5
    x[x==-1]=0

handle(x)
handle(board1)
handle(board2)
handle(board3)
handle(board4)
handle(board5)


import tensorflow as tf

batch_size=1000

x_input=tf.placeholder(tf.float32,shape=(None,8,5))
board1_input=tf.placeholder(tf.float32,shape=(None,3,3))
board2_input=tf.placeholder(tf.float32,shape=(None,5,5))
board3_input=tf.placeholder(tf.float32,shape=(None,7,7))
board4_input=tf.placeholder(tf.float32,shape=(None,9,9))
board5_input=tf.placeholder(tf.float32,shape=(None,11,11))

output=tf.placeholder(tf.float32,shape=(None,1))



def divide(input_shape):
    dense1=tf.layers.dense(input_shape,64,activation=tf.nn.sigmoid,use_bias=False)
    dense2=tf.layers.dense(dense1,128,activation=tf.nn.sigmoid,use_bias=False)
    return tf.layers.dense(dense2,1,use_bias=False)

div0=divide(x_input[:,0,:])
div1=divide(x_input[:,1,:])
div2=divide(x_input[:,2,:])
div3=divide(x_input[:,3,:])
div4=divide(x_input[:,4,:])
div5=divide(x_input[:,5,:])
div6=divide(x_input[:,6,:])
div7=divide(x_input[:,7,:])

loss0=tf.reduce_mean(tf.abs(div0-output),axis=-1)
loss1=tf.reduce_mean(tf.abs(div1-output),axis=-1)
loss2=tf.reduce_mean(tf.abs(div2-output),axis=-1)
loss3=tf.reduce_mean(tf.abs(div3-output),axis=-1)
loss4=tf.reduce_mean(tf.abs(div4-output),axis=-1)
loss5=tf.reduce_mean(tf.abs(div5-output),axis=-1)
loss6=tf.reduce_mean(tf.abs(div6-output),axis=-1)
loss7=tf.reduce_mean(tf.abs(div7-output),axis=-1)

x_loss=tf.reduce_mean([loss0,loss1,loss2,loss3,loss4,loss5,loss6,loss7],axis=0)

def board1_learn(input_shape):
    reshape1=tf.expand_dims(input_shape,-1)
    conv1=tf.layers.conv2d(reshape1,32,kernel_size=3,strides=1,padding='same',activation=None)
    conv2=tf.layers.conv2d(conv1,32,kernel_size=3,strides=1,padding='same',activation='relu')
    batch1=tf.layers.batch_normalization(conv2)
    conv3=tf.layers.conv2d(batch1,64,kernel_size=3,strides=1,padding='same',activation=None)

    flatten1=tf.layers.flatten(conv3)
    dense1=tf.layers.dense(flatten1,1)
    return dense1

dense1=board1_learn(board1_input)
board1_loss=tf.reduce_mean(tf.abs(dense1 - output), axis=-1)

def board2_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1=tf.layers.conv2d(reshape1,32,kernel_size=3,strides=1,padding='same',activation=None)
    conv2=tf.layers.conv2d(conv1,32,kernel_size=3,strides=1,padding='same',activation='relu')
    batch1=tf.layers.batch_normalization(conv2)
    conv3=tf.layers.conv2d(batch1,64,kernel_size=3,strides=1,padding='same',activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation=None)

    flatten1=tf.layers.flatten(conv4)
    dense1=tf.layers.dense(flatten1,1)
    return dense1

dense2=board2_learn(board2_input)
board2_loss=tf.reduce_mean(tf.abs(dense2 - output), axis=-1)

def board3_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1=tf.layers.conv2d(reshape1,32,kernel_size=3,strides=1,padding='same',activation=None)
    conv2=tf.layers.conv2d(conv1,32,kernel_size=3,strides=1,padding='same',activation='relu')
    batch1=tf.layers.batch_normalization(conv2)

    conv3=tf.layers.conv2d(batch1,64,kernel_size=3,strides=1,padding='same',activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=3, strides=1, padding='same', activation=None)

    flatten1=tf.layers.flatten(conv5)
    dense1=tf.layers.dense(flatten1,1)
    return dense1

dense3=board3_learn(board3_input)
board3_loss=tf.reduce_mean(tf.abs(dense3 - output), axis=-1)

def board4_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1=tf.layers.conv2d(reshape1,32,kernel_size=3,strides=1,padding='same',activation=None)
    conv2=tf.layers.conv2d(conv1,32,kernel_size=3,strides=1,padding='same',activation='relu')
    batch1=tf.layers.batch_normalization(conv2)
    conv3=tf.layers.conv2d(batch1,64,kernel_size=3,strides=1,padding='same',activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=3, strides=1, padding='same', activation='relu')
    conv6 = tf.layers.conv2d(conv5, 128, kernel_size=3, strides=1, padding='same', activation=None)

    flatten1=tf.layers.flatten(conv6)
    dense1=tf.layers.dense(flatten1,1)
    return dense1

dense4=board4_learn(board4_input)
board4_loss=tf.reduce_mean(tf.abs(dense4 - output), axis=-1)

def board5_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1=tf.layers.conv2d(reshape1,32,kernel_size=3,strides=1,padding='same',activation=None)
    conv2=tf.layers.conv2d(conv1,32,kernel_size=3,strides=1,padding='same',activation='relu')
    batch1=tf.layers.batch_normalization(conv2)
    conv3=tf.layers.conv2d(batch1,63,kernel_size=3,strides=1,padding='same',activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=3, strides=1, padding='same', activation='relu')
    conv6 = tf.layers.conv2d(conv5, 128, kernel_size=3, strides=1, padding='same', activation=None)

    flatten1=tf.layers.flatten(conv6)
    dense1=tf.layers.dense(flatten1,1)
    return dense1

dense5=board5_learn(board5_input)
board5_loss=tf.reduce_mean(tf.abs(dense5 - output), axis=-1)


loss=tf.reduce_mean([x_loss,board1_loss,board2_loss,board3_loss,board4_loss,board5_loss])


train_op=tf.train.AdamOptimizer().minimize(loss)


saver=tf.train.Saver()

n_split=len(x)-5000

x_train=x[:n_split]
board1_train=board1[:n_split]
board2_train=board2[:n_split]
board3_train=board3[:n_split]
board4_train=board4[:n_split]
board5_train=board5[:n_split]
y_train=y[:n_split]

x_test=x[n_split:]
board1_test=board1[n_split:]
board2_test=board2[n_split:]
board3_test=board3[n_split:]
board4_test=board4[n_split:]
board5_test=board5[n_split:]
y_test=y[n_split:]


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # saver.restore(sess=sess,save_path='./model/model.ckpt')

    for epoch in range(1000):

        total=len(x_train)
        page_size=1000
        page_no=int(total/page_size)

        loss_total=0

        for i in range(page_no):
            start=i*page_size % total
            end=(i*page_size+page_size) %total
            if end > total:
                end=total

            input_data=x_train[start:end]
            board1_data=board1_train[start:end]
            board2_data = board2_train[start:end]
            board3_data = board3_train[start:end]
            board4_data = board4_train[start:end]
            board5_data = board5_train[start:end]

            output_data=y_train[start:end]

            _,loss_val=sess.run([train_op,loss],feed_dict={
                x_input:input_data,
                board1_input:board1_data,
                board2_input: board2_data,
                board3_input: board3_data,
                board4_input: board4_data,
                board5_input: board5_data,
                output:output_data
            })

            print('loss_val={:.4f}'.format(loss_val))
            # print(div0_val)
            # print(output_val)

            loss_total+=loss_val

        saver.save(sess=sess,save_path='./model/model.ckpt')
        _,val_loss_val=sess.run([train_op,loss],feed_dict={
            x_input: x_test,
            board1_input: board1_test,
            board2_input: board2_test,
            board3_input: board3_test,
            board4_input: board4_test,
            board5_input: board5_test,
            output: y_test
        })
        print("epoch={}     loss_val={:.4f}     val_loss_val={:.4f}".format(epoch,loss_total/page_no,val_loss_val))