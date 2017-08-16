import random


class Brain:
    probability=0.5
    def __init__(self,probability=0.5):
        self.probability=probability

    def step(self,game):
        if random.random()<self.probability:
            lst = []
            for i in range(game.h):
                for j in range(game.w):
                    if game.board[i][j] == 0:
                        lst.append((i, j, game.who_step))
            return random.choice(lst)
        else:
            # 只能选择
            pass
