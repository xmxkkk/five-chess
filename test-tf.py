from record import Record
from sklearn.preprocessing import Normalizer
import numpy as np


data=Record()
x,y=data.load(0,100000)
y = y[:, np.newaxis]
y=Normalizer().fit_transform(y)


import tensorflow as tf

batch_size=1000

input=tf.placeholder(tf.float32,shape=(None,8,5))
output=tf.placeholder(tf.float32,shape=(None,1))

def divide(input_shape):
    dense1=tf.layers.dense(input_shape,64,activation=tf.nn.sigmoid,use_bias=False)
    dense2=tf.layers.dense(dense1,128,activation=tf.nn.sigmoid,use_bias=False)
    return tf.layers.dense(dense2,1,use_bias=False)


div0=divide(input[:,0,:])
div1=divide(input[:,1,:])
div2=divide(input[:,2,:])
div3=divide(input[:,3,:])
div4=divide(input[:,4,:])
div5=divide(input[:,5,:])
div6=divide(input[:,6,:])
div7=divide(input[:,7,:])

loss0=tf.reduce_mean(tf.abs(div0-output),axis=-1)
loss1=tf.reduce_mean(tf.abs(div1-output),axis=-1)
loss2=tf.reduce_mean(tf.abs(div2-output),axis=-1)
loss3=tf.reduce_mean(tf.abs(div3-output),axis=-1)
loss4=tf.reduce_mean(tf.abs(div4-output),axis=-1)
loss5=tf.reduce_mean(tf.abs(div5-output),axis=-1)
loss6=tf.reduce_mean(tf.abs(div6-output),axis=-1)
loss7=tf.reduce_mean(tf.abs(div7-output),axis=-1)

loss=tf.reduce_mean([loss0,loss1,loss2,loss3,loss4,loss5,loss6,loss7])

train_op=tf.train.AdamOptimizer().minimize(loss)


saver=tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(1000):

        total=len(x)
        page_size=1000
        page_no=int(total/page_size)

        loss_total=0

        for i in range(page_no):
            start=i*page_size % total
            end=(i*page_size+page_size) %total
            if end > total:
                end=total

            input_data=x[start:end]
            output_data=y[start:end]

            _,loss_val,div0_val,output_val=sess.run([train_op,loss,div0,output],feed_dict={input:input_data,
                                                                                        output:output_data})
            print('loss_val={:.4f}'.format(loss_val))
            # print(div0_val)
            # print(output_val)

            loss_total+=loss_val

        print("epoch={}     loss_val={:.4f}".format(epoch,loss_total/page_no))