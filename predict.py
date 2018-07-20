import numpy as np
import tensorflow as tf

board_list=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0,-1,-1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = -99999
    vector[-pad_width[1]:] = -99999
    return vector


board=np.array(board_list)

board_no=-99999

x=[]
board1=[]
board2=[]
board3=[]
board4=[]
board5=[]

result_map={}

idx=0

for i in range(15):
    for j in range(15):
        if board[i,j]==0:

            step=[i,j]

            result_map[str(idx)]=step
            idx=idx+1

            oldboard=board.copy()
            board[i,j]=1
            line0 = [board[step[0]][step[1]],
                     board[step[0]][step[1] + 1] if step[1] + 1 < 15 else board_no,
                     board[step[0]][step[1] + 2] if step[1] + 2 < 15 else board_no,
                     board[step[0]][step[1] + 3] if step[1] + 3 < 15 else board_no,
                     board[step[0]][step[1] + 4] if step[1] + 4 < 15 else board_no]

            line1 = [board[step[0]][step[1]],
                     board[step[0]][step[1] - 1] if step[1] - 1 > 0 else board_no,
                     board[step[0]][step[1] - 2] if step[1] - 2 > 0 else board_no,
                     board[step[0]][step[1] - 3] if step[1] - 3 > 0 else board_no,
                     board[step[0]][step[1] - 4] if step[1] - 4 > 0 else board_no]

            line2 = [board[step[0]][step[1]],
                     board[step[0] + 1][step[1]] if step[0] + 1 < 15 else board_no,
                     board[step[0] + 2][step[1]] if step[0] + 2 < 15 else board_no,
                     board[step[0] + 3][step[1]] if step[0] + 3 < 15 else board_no,
                     board[step[0] + 4][step[1]] if step[0] + 4 < 15 else board_no]

            line3 = [board[step[0]][step[1]],
                     board[step[0] - 1][step[1]] if step[0] - 1 > 0 else board_no,
                     board[step[0] - 2][step[1]] if step[0] - 1 > 0 else board_no,
                     board[step[0] - 3][step[1]] if step[0] - 1 > 0 else board_no,
                     board[step[0] - 4][step[1]] if step[0] - 1 > 0 else board_no]

            line4 = [board[step[0]][step[1]],
                     board[step[0] - 1][step[1] - 1] if step[0] - 1 > 0 and step[1] - 1 > 0 else board_no,
                     board[step[0] - 2][step[1] - 2] if step[0] - 2 > 0 and step[1] - 1 > 0 else board_no,
                     board[step[0] - 3][step[1] - 3] if step[0] - 3 > 0 and step[1] - 1 > 0 else board_no,
                     board[step[0] - 4][step[1] - 4] if step[0] - 4 > 0 and step[1] - 1 > 0 else board_no]

            line5 = [board[step[0]][step[1]],
                     board[step[0] - 1][step[1] + 1] if step[0] - 1 > 0 and step[1] + 1 < 15 else board_no,
                     board[step[0] - 2][step[1] + 2] if step[0] - 2 > 0 and step[1] + 2 < 15 else board_no,
                     board[step[0] - 3][step[1] + 3] if step[0] - 3 > 0 and step[1] + 3 < 15 else board_no,
                     board[step[0] - 4][step[1] + 4] if step[0] - 4 > 0 and step[1] + 4 < 15 else board_no]

            line6 = [board[step[0]][step[1]],
                     board[step[0] + 1][step[1] - 1] if step[0] + 1 < 15 and step[1] - 1 > 0 else board_no,
                     board[step[0] + 2][step[1] - 2] if step[0] + 2 < 15 and step[1] - 2 > 0 else board_no,
                     board[step[0] + 3][step[1] - 3] if step[0] + 3 < 15 and step[1] - 3 > 0 else board_no,
                     board[step[0] + 4][step[1] - 4] if step[0] + 4 < 15 and step[1] - 4 > 0 else board_no]

            line7 = [board[step[0]][step[1]],
                     board[step[0] + 1][step[1] + 1] if step[0] + 1 < 15 and step[1] + 1 < 15 else board_no,
                     board[step[0] + 2][step[1] + 2] if step[0] + 2 < 15 and step[1] + 2 < 15 else board_no,
                     board[step[0] + 3][step[1] + 3] if step[0] + 3 < 15 and step[1] + 3 < 15 else board_no,
                     board[step[0] + 4][step[1] + 4] if step[0] + 4 < 15 and step[1] + 4 < 15 else board_no]

            lines = [line0, line1, line2, line3, line4, line5, line6, line7]

            newboard = np.pad(board, 5, padwithtens)

            newx = step[0] + 5
            newy = step[1] + 5

            boards = []
            for i in range(1, 6):
                boards.append(newboard[newx - i:newx + i + 1, newy - i:newy + i + 1].tolist())

            x.append(lines)
            board1.append(boards[0])
            board2.append(boards[1])
            board3.append(boards[2])
            board4.append(boards[3])
            board5.append(boards[4])

            board=oldboard


x=np.array(x)
board1=np.array(board1)
board2=np.array(board2)
board3=np.array(board3)
board4=np.array(board4)
board5=np.array(board5)


print(x.shape)
print(board1.shape)
print(board2.shape)
print(board3.shape)
print(board4.shape)
print(board5.shape)

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


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    saver.restore(sess=sess,save_path='./model/model.ckpt')

    div0_val,div1_val,div2_val,div3_val,div4_val,div5_val,div6_val,div7_val,dense1_val,dense2_val,dense3_val,\
    dense4_val,dense5_val=sess.run([div0,div1,div2,div3,div4,div5,div6,div7,dense1,dense2,dense3,dense4,dense5],\
        feed_dict={
        x_input: x,
        board1_input: board1,
        board2_input: board2,
        board3_input: board3,
        board4_input: board4,
        board5_input: board5,
    })

    # div0_val = np.abs(div0_val)
    # div1_val = np.abs(div1_val)
    # div2_val = np.abs(div2_val)
    # div3_val = np.abs(div3_val)
    # div4_val = np.abs(div4_val)
    # div5_val = np.abs(div5_val)
    # div6_val = np.abs(div6_val)
    # div7_val = np.abs(div7_val)
    #
    # dense1_val = np.abs(dense1_val)
    # dense2_val = np.abs(dense2_val)
    # dense3_val = np.abs(dense3_val)
    # dense4_val = np.abs(dense4_val)
    # dense5_val = np.abs(dense5_val)

    div_val=div0_val+div1_val+div2_val+div3_val+div4_val+div5_val+div6_val+div7_val

    score=div_val/8+dense1_val+dense2_val+dense3_val+dense4_val+dense5_val

    score=np.reshape(score, [-1, ])
    result_idx_max=np.argmax(score)
    # result_idx_min = np.argmin(score)
    print(score)
    print(result_map[str(result_idx_max)])
    # print(result_map[str(result_idx_min)])
