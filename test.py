from record import Record
from sklearn.preprocessing import Normalizer
import numpy as np


data=Record()
x,y=data.load(0,100000)
y = y[:, np.newaxis]
y=Normalizer().fit_transform(y)


print(x.shape)
print(y.shape)

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Input,Reshape,BatchNormalization

activate=None

model=Sequential()
model.add(Reshape(input_shape=(8,5),target_shape=(8,5,1)))
model.add(Conv2D(16,kernel_size=5,strides=1,padding='same',use_bias=False,activation=None))
model.add(Conv2D(32,kernel_size=5,strides=1,padding='same',use_bias=False,activation='relu'))
#
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=1,padding='same',use_bias=False,activation='relu'))
model.add(Conv2D(64,kernel_size=3,strides=1,padding='same',use_bias=False,activation='relu'))
model.add(Conv2D(64,kernel_size=1,strides=1,padding='same',use_bias=False,activation='relu'))

model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=1,padding='same',use_bias=False,activation='relu'))
model.add(Conv2D(64,kernel_size=3,strides=1,padding='same',use_bias=False,activation='relu'))
model.add(Conv2D(64,kernel_size=1,strides=1,padding='same',use_bias=False,activation='relu'))

model.add(Flatten())

model.add(Dense(1))

model.compile('adam',loss='mae')


model.fit(x,y,epochs=1000,batch_size=1000,verbose=1,validation_split=0.3)