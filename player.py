import numpy as np

class Player:
    probability = 0.1
    def __init__(self,probability=0.9,):
        self.probability=probability

    def step(self,board):

