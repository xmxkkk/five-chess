from record import Record
from keras.models import Sequential
from keras.layers import Dense,Flatten,Reshape
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D,MaxPooling1D
import random
import os
import numpy as np

class Learn:
    record=None

    def learn(self,model_name,batch_size=1000,total_batch=1000,epochs=10,step_num=225,only_learn_num_0=False):
        self.record=Record()
        # random.seed(1337)

        model,weights_path=eval('self.build_model_'+model_name+'(model_name)')

        for i in range(total_batch):
            X,y=self.record.load(i,batch_size,epochs=epochs,step_num=step_num,only_learn_num_0=only_learn_num_0)
            if X is None:
                break

            model.fit(X,y,batch_size=len(y),epochs=epochs,verbose=1)
            model.save_weights(weights_path)

    def predict(self,model_name,board,who_step):

        board=board.copy()
        board=board*who_step
        who_step_pre=who_step*who_step

        model,_=eval('self.build_model_'+model_name+'(model_name)')

        blank_scores=[]
        for i in range(15):
            for j in range(15):
                if board[i][j]==0:
                    board_clone=board.copy()
                    board_clone[i][j]=who_step_pre
                    lst=board_clone[np.newaxis,:]
                    score=model.predict(lst)
                    blank_scores.append({"score":score[0][0],"x":i,"y":j})

        x,y=0,0
        max_score=-100
        for item in blank_scores:
            if item['score']>max_score:
                x=item['x']
                y=item['y']
                max_score=item['score']
        print(blank_scores)

        return (x,y,who_step)

    def build_model_001(self,model_name):
        model = Sequential()
        acti = 'tanh'
        model.add(Dense(32, input_shape=(15, 15), activation=acti))
        model.add(Dense(32, activation=acti))
        model.add(Dense(16, activation=acti))
        model.add(Flatten())
        model.add(Dense(8, activation=acti))
        model.add(Dense(4, activation=acti))
        model.add(Dense(1, activation=acti))
        model.compile(optimizer='adam', loss='mae')

        # modelName = eval('self.build_model_' + model_name + '(model)')
        weights_path = "./weights/" + model_name + ".h5"
        if os.path.exists(weights_path):
            model.load_weights(weights_path)

        return model,weights_path

    def build_model_002(self,model_name):
        model = Sequential()
        model.add(Conv1D(32,kernel_size=1,strides=1,padding='same',input_shape=(15,15)))
        model.add(MaxPooling1D(pool_size=1,padding='same'))
        model.add(Conv1D(64, kernel_size=1, strides=1, padding='same'))
        model.add(MaxPooling1D(pool_size=1,padding='same'))

        model.add(Flatten())
        model.add(Dense(128,activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(optimizer='adam', loss='mae')

        # modelName = eval('self.build_model_' + model_name + '(model)')
        weights_path = "./weights/" + model_name + ".h5"
        if os.path.exists(weights_path):
            model.load_weights(weights_path)

        return model,weights_path

    def build_model_003(self, model_name):
        model = Sequential()
        model.add(Conv1D(32, kernel_size=5, strides=1, padding='same', input_shape=(15, 15)))
        model.add(Conv1D(64, kernel_size=5, strides=1, padding='same'))
        model.add(Conv1D(96, kernel_size=5, strides=1, padding='same'))
        model.add(Conv1D(128, kernel_size=5, strides=1, padding='same'))

        model.add(Flatten())
        model.add(Dense(1000, activation='tanh'))
        model.add(Dense(200, activation='tanh'))
        model.add(Dense(50, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(optimizer='adam', loss='mae')

        # modelName = eval('self.build_model_' + model_name + '(model)')
        weights_path = "./weights/" + model_name + ".h5"
        if os.path.exists(weights_path):
            model.load_weights(weights_path)

        return model, weights_path

l=Learn()

# l.learn("003",batch_size=1000,total_batch=1000,epochs=2,step_num=120)

board=np.zeros((15,15))
board[0][0]=1
board[0][1]=1
board[0][2]=1
board[0][3]=1
board[1][0]=-1
board[1][1]=-1
board[1][2]=-1
board[1][3]=-1

# print(l.predict("001",board,1))# (11, 13, 1)
print(l.predict("003",board,1))# (11, 13, 1)
