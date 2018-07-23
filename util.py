import numpy as np
from sklearn.preprocessing import Normalizer
import random
import tensorflow as tf

def line_learn(input_shape):
    dense1 = tf.layers.dense(input_shape, 64, activation=tf.nn.sigmoid, use_bias=False)
    dense2 = tf.layers.dense(dense1, 128, activation=tf.nn.sigmoid, use_bias=False)
    return tf.layers.dense(dense2, 1, use_bias=False)

def board1_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=3, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=3, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv3)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board2_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=3, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=3, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=3, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv4)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board3_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=3, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=3, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)

    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=3, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=3, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv5)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board4_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=3, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=3, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 64, kernel_size=3, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=3, strides=1, padding='same', activation='relu')
    conv6 = tf.layers.conv2d(conv5, 128, kernel_size=3, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv6)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def board5_learn(input_shape):
    reshape1 = tf.expand_dims(input_shape, -1)
    conv1 = tf.layers.conv2d(reshape1, 32, kernel_size=3, strides=1, padding='same', activation=None)
    conv2 = tf.layers.conv2d(conv1, 32, kernel_size=3, strides=1, padding='same', activation='relu')
    batch1 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d(batch1, 63, kernel_size=3, strides=1, padding='same', activation='relu')
    conv4 = tf.layers.conv2d(conv3, 64, kernel_size=3, strides=1, padding='same', activation='relu')

    batch2 = tf.layers.batch_normalization(conv4)
    conv5 = tf.layers.conv2d(batch2, 128, kernel_size=3, strides=1, padding='same', activation='relu')
    conv6 = tf.layers.conv2d(conv5, 128, kernel_size=3, strides=1, padding='same', activation='relu')

    flatten1 = tf.layers.flatten(conv6)
    dense1 = tf.layers.dense(flatten1, 1)
    return dense1

def random_step(board,who_step):
    if board[7][7]==0:
        return (7,7,who_step)
    lst=[]
    for i in range(15):
        for j in range(15):
            lst.append((i, j, who_step))

    return random.choice(lst)

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = -99999
    vector[-pad_width[1]:] = -99999
    return vector

def norm_y(y):
    y = np.array(y, dtype='float32')
    y = Normalizer().fit_transform([y])
    y = y[0][:, np.newaxis]
    return y

def handle(x):
    x[x==-99999]=0.1
    x[x==0]=0.5
    x[x==-1]=0
    return x

def board_data(board,who_step):

    board=board*-1

    x = []
    board1 = []
    board2 = []
    board3 = []
    board4 = []
    board5 = []

    result_map = {}

    idx = 0

    for i in range(15):
        for j in range(15):
            if board[i, j] == 0:
                step = [i, j]

                result_map[str(idx)] = step
                idx = idx + 1

                oldboard = board.copy()
                board[i, j] = 1

                lines = board_line8(board, step)
                boards = board_shape(board, step)

                x.append(lines)
                board1.append(boards[0])
                board2.append(boards[1])
                board3.append(boards[2])
                board4.append(boards[3])
                board5.append(boards[4])

                board = oldboard
    x = handle(np.array(x))
    board1 = handle(np.array(board1))
    board2 = handle(np.array(board2))
    board3 = handle(np.array(board3))
    board4 = handle(np.array(board4))
    board5 = handle(np.array(board5))

    return result_map,x,board1,board2,board3,board4,board5

def board_shape(board,step):
    newboard = np.pad(board, 5, padwithtens)

    newx = step[0] + 5
    newy = step[1] + 5

    boards = []
    for i in range(1, 6):
        boards.append(newboard[newx - i:newx + i + 1, newy - i:newy + i + 1].tolist())

    return boards

def board_line8(board,step):
    board_no=-99999
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

    return lines