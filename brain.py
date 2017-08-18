import random
from learn import Learn

class Brain:
    probability=0.5
    learn=None
    def __init__(self,probability=0.5):
        self.probability=probability
        self.learn=Learn()
    def step(self,game):
        if game.who_step==1:
            return self.learn.predict("003",game.board,game.who_step)
        else:
            # return self.learn.predict("003", game.board, game.who_step)

            if len(game.steps) == 0:
                return (7, 7, game.who_step)
            lst = []
            for i in range(game.h):
                for j in range(game.w):
                    if game.board[i][j] == 0:
                        lst.append((i, j, game.who_step))
            return random.choice(lst)
            ''''''

        '''
        if random.random()<self.probability:
            if len(game.steps) == 0 :
                return (7,7,game.who_step)
            lst = []
            for i in range(game.h):
                for j in range(game.w):
                    if game.board[i][j] == 0:
                        lst.append((i, j, game.who_step))
            return random.choice(lst)
        else:
            # 只能选择
            pass'''
