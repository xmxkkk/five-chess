import numpy as np


board=np.linspace(0,4,num=4).reshape((2,2))

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = -99999
    vector[-pad_width[1]:] = -99999
    return vector

newboard=np.pad(board,2,padwithtens)

print(newboard)
