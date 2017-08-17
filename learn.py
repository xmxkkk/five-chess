from record import Record
from keras.models import Sequential
from keras.layers import Dense,Flatten,Reshape
from keras.layers.recurrent import LSTM
import random

class Learn:
    record=None


    def learn(self):
        self.record=Record()
        random.seed(1337)

        model=Sequential()

        model.add(LSTM(
            256,
            input_shape=(225,225),
            activation='sigmoid',
            return_sequences=False
        ))
        model.add(Reshape((-1,)))
        model.add(Dense(1024, activation='sigmoid'))
        model.add(Dense(256,activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dense(1, activation='tanh'))
        model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

        model.load_weights("weights.h5")

        x,y=self.record.load(20,10)
        pred_y=model.predict(x)
        print(y)
        print(pred_y)

        '''
        print(model.summary())

        for i in range(100):
            print('roll in ', i)
            batch_size=100
            x, y = self.record.load(i, int(batch_size/2))
            model.fit(x,y,batch_size=batch_size,epochs=10,verbose=1)
            model.save_weights("weights.h5")'''


        # print(x)
        # (100, 225, 15, 15)




l=Learn()
l.learn()