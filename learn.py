from record import Record
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers.recurrent import LSTM


class Learn:
    record=None
    def learn(self):
        self.record=Record()

        x,y=self.record.load(0,20)
        print(x.shape)
        print(y.shape)
        # print(x)
        # (100, 225, 15, 15)


l=Learn()
l.learn()