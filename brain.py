import random

class Brain:
    probability=0.1
    def __init__(self):
        pass

    def step(self,game):
        if random.random()<0.1:
            lst = []
            for i in range(game.h):
                for j in range(game.w):
                    if game.board[i][j] == 0:
                        lst.append((i, j, game.who_step))
            return random.choice(lst)
        else:
            # 只能选择
            pass


