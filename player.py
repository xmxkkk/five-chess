import numpy as np
import random
import tensorflow as tf
from record import Record
import os
from util import padwithtens,handle,norm_y,board_shape,board_line8,board_data,random_step,line_learn,\
    board1_learn,board2_learn,board3_learn,board4_learn,board5_learn

class Player:
    def __init__(self,who_step,random_probability=0.9,model_name=None,weight_name=None):
        self.random_probability=random_probability
        self.who_step=who_step
        self.model_name=model_name
        self.weight_name=weight_name
        self.sess = None
        self.saver = None

        with tf.Graph().as_default():
            self._load_model()
            self.load_session()

    def _load_data(self,model_name):
        record=Record()
        self.x,self.board1,self.board2,self.board3,self.board4,self.board5,self.y=record.load(0,100000,model_name)

    def _load_model(self):
        self.x_input = tf.placeholder(tf.float32, shape=(None, 8, 5))
        self.board1_input = tf.placeholder(tf.float32, shape=(None, 3, 3))
        self.board2_input = tf.placeholder(tf.float32, shape=(None, 5, 5))
        self.board3_input = tf.placeholder(tf.float32, shape=(None, 7, 7))
        self.board4_input = tf.placeholder(tf.float32, shape=(None, 9, 9))
        self.board5_input = tf.placeholder(tf.float32, shape=(None, 11, 11))
        self.output = tf.placeholder(tf.float32, shape=(None, 1))

        self.div0 = line_learn(self.x_input[:, 0, :])
        self.div1 = line_learn(self.x_input[:, 1, :])
        self.div2 = line_learn(self.x_input[:, 2, :])
        self.div3 = line_learn(self.x_input[:, 3, :])
        self.div4 = line_learn(self.x_input[:, 4, :])
        self.div5 = line_learn(self.x_input[:, 5, :])
        self.div6 = line_learn(self.x_input[:, 6, :])
        self.div7 = line_learn(self.x_input[:, 7, :])

        loss0 = tf.reduce_mean(tf.abs(self.div0 - self.output), axis=-1)
        loss1 = tf.reduce_mean(tf.abs(self.div1 - self.output), axis=-1)
        loss2 = tf.reduce_mean(tf.abs(self.div2 - self.output), axis=-1)
        loss3 = tf.reduce_mean(tf.abs(self.div3 - self.output), axis=-1)
        loss4 = tf.reduce_mean(tf.abs(self.div4 - self.output), axis=-1)
        loss5 = tf.reduce_mean(tf.abs(self.div5 - self.output), axis=-1)
        loss6 = tf.reduce_mean(tf.abs(self.div6 - self.output), axis=-1)
        loss7 = tf.reduce_mean(tf.abs(self.div7 - self.output), axis=-1)

        x_loss = tf.reduce_mean([loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7], axis=0)

        self.dense1 = board1_learn(self.board1_input)
        board1_loss = tf.reduce_mean(tf.abs(self.dense1 - self.output), axis=-1)

        self.dense2 = board2_learn(self.board2_input)
        board2_loss = tf.reduce_mean(tf.abs(self.dense2 - self.output), axis=-1)

        self.dense3 = board3_learn(self.board3_input)
        board3_loss = tf.reduce_mean(tf.abs(self.dense3 - self.output), axis=-1)

        self.dense4 = board4_learn(self.board4_input)
        board4_loss = tf.reduce_mean(tf.abs(self.dense4 - self.output), axis=-1)

        self.dense5 = board5_learn(self.board5_input)
        board5_loss = tf.reduce_mean(tf.abs(self.dense5 - self.output), axis=-1)

        self.loss = tf.reduce_mean([x_loss, board1_loss, board2_loss, board3_loss, board4_loss, board5_loss])

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        self._load_data(self.model_name)

        n_split = len(self.x) - 1000

        x_train = self.x[:n_split]
        board1_train = self.board1[:n_split]
        board2_train = self.board2[:n_split]
        board3_train = self.board3[:n_split]
        board4_train = self.board4[:n_split]
        board5_train = self.board5[:n_split]
        y_train = self.y[:n_split]

        x_test = self.x[n_split:]
        board1_test = self.board1[n_split:]
        board2_test = self.board2[n_split:]
        board3_test = self.board3[n_split:]
        board4_test = self.board4[n_split:]
        board5_test = self.board5[n_split:]
        y_test = self.y[n_split:]

        for epoch in range(1000):

            total = len(x_train)
            page_size = 1000
            page_no = int(total / page_size)

            loss_total = 0

            for i in range(page_no):
                start = i * page_size % total
                end = (i * page_size + page_size) % total
                if end > total:
                    end = total

                input_data = x_train[start:end]
                board1_data = board1_train[start:end]
                board2_data = board2_train[start:end]
                board3_data = board3_train[start:end]
                board4_data = board4_train[start:end]
                board5_data = board5_train[start:end]

                output_data = y_train[start:end]

                _, loss_val = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.x_input: input_data,
                    self.board1_input: board1_data,
                    self.board2_input: board2_data,
                    self.board3_input: board3_data,
                    self.board4_input: board4_data,
                    self.board5_input: board5_data,
                    self.output: output_data
                })

                print('loss_val={:.4f}'.format(loss_val))

                loss_total += loss_val

            if self.weight_name is not None:
                self.saver.save(sess=self.sess, save_path=self.weight_name)

            _, val_loss_val = self.sess.run([self.train_op, self.loss], feed_dict={
                self.x_input: x_test[:1000],
                self.board1_input: board1_test[:1000],
                self.board2_input: board2_test[:1000],
                self.board3_input: board3_test[:1000],
                self.board4_input: board4_test[:1000],
                self.board5_input: board5_test[:1000],
                self.output: y_test[:1000]
            })
            print("epoch={}     loss_val={:.4f}     val_loss_val={:.4f}".format(epoch, loss_total / page_no,
                                                                                    val_loss_val))

    def load_session(self,reload=False):
        if self.sess is None or reload:
            self.saver = tf.train.Saver()
            self.sess=tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # self.saver.restore(sess=self.sess, save_path=self.weight_name)

    def predict(self,board,reload=False):

        result=[]
        self.result_map, x, board1, board2, board3, board4, board5 = board_data(board,self.who_step)

        div0_val, div1_val, div2_val, div3_val, div4_val, div5_val, div6_val, div7_val, dense1_val, dense2_val, dense3_val, \
        dense4_val, dense5_val = self.sess.run(
            [self.div0, self.div1, self.div2, self.div3, self.div4, self.div5, self.div6, self.div7, self.dense1,
             self.dense2, self.dense3, self.dense4, self.dense5], \
            feed_dict={
                self.x_input: x,
                self.board1_input: board1,
                self.board2_input: board2,
                self.board3_input: board3,
                self.board4_input: board4,
                self.board5_input: board5
            })

        div_val = div0_val + div1_val + div2_val + div3_val + div4_val + div5_val + div6_val + div7_val

        score = div_val / 8 + dense1_val + dense2_val + dense3_val + dense4_val + dense5_val

        score = np.reshape(score, [-1, ])

        idx = 0
        for k in score:
            result.append({"score": k, "idx": idx, "step": self.result_map[str(idx)]})
            idx += 1
        result=sorted(result,key=lambda item:item['score'], reverse=True)
        return result

    def step(self,board):
        if board[7][7] == 0:
            return (7, 7, self.who_step)

        if self.random_probability>random.random():
            result=self.predict(board)
            stepss=random.choice(result[0:5])
            return (stepss["step"][0],stepss["step"][1],self.who_step)

        return random_step(board,self.who_step)
    

player=Player(1,1.0,"pos_model","./model/model1/model.ckpt")
# playerb=Player(-1,1.0,"pos_model","./model/model1/model.ckpt")
player.train()
'''
board_list=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
board=np.array(board_list)


result1=player.predict(board)
print(result1)

board_list=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
board=np.array(board_list)
result2=player.predict(board)
print(result2)'''
