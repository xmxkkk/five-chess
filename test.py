from keras.models import Model,Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense
from record import Record

record=Record()
x,y=record.load_3(0,10000,shuffle=True)
print(x.shape,y.shape)

model=Sequential()
model.add(SimpleRNN(
    256,
    input_shape=(15,15),
    unroll=True,
    dropout=0.5
))

model.add(Dense(64,activation='relu'))
model.add(Dense(1))

model.compile('adam',loss='mae')
model.fit(x,y,batch_size=1000,epochs=1000,verbose=1,validation_split=0.1)
