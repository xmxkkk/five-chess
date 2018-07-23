import numpy as np
import random
from record import Record
from player import PlayerFirst
from util import random_step

class Game:
    board=None
    w=15
    h=15
    steps=None
    who_step=1
    record=None
    def __init__(self,model_name,player_a,player_b):
        self.init()
        self.model_name=model_name
        self.player_a=player_a
        self.player_b=player_b
        self.record=Record()

    def init(self):
        self.board = np.zeros((self.w, self.h)).astype('int')
        self.steps = []

    def is_win(self):
        line1=np.ones((5,1)).astype('int')
        line2=np.ones((1,5)).astype('int')
        line3=np.eye(5).astype('int')
        line4=np.fliplr(line3)

        for i in range(self.h-5):
            for j in range(self.w-5):
                for line in np.array([line3,line4]):
                    matrix=self.board[i:i+5,j:j+5]
                    temp=(matrix*line).sum()
                    if temp==5 or temp==-5:
                        return int(temp/5)

        for i in range(self.h-5):
            for j in range(self.w):
                for line in np.array([line1]):
                    matrix=self.board[i:i+5,j:j+1]
                    temp=(matrix*line).sum()
                    if temp==5 or temp==-5:
                        return int(temp/5)

        for i in range(self.h):
            for j in range(self.w-5):
                for line in np.array([line2]):
                    matrix=self.board[i:i+1,j:j+5]
                    temp=(matrix*line).sum()
                    if temp==5 or temp==-5:
                        return int(temp/5)

        return 0

    def is_full(self):
        return (self.board==0).sum()==0

    def start(self,who_step=1,verbose=0):
        self.init()

        self.who_step=who_step
        while True:
            full=self.is_full()
            win=self.is_win()
            if full or win:
                if full:
                    if verbose==1:
                        print("draw.")
                if win:
                    if verbose == 1:
                        print(str(win)+" win.")
                break

            if self.who_step==1:
                s=self.player_a.step(self.board)
            else:
                s=self.player_b.step(self.board)

            self.board[s[0]][s[1]]=s[2]
            self.steps.append(s)
            self.who_step*=-1

            if verbose == 1:
                self.print()

        self.record.save_chess(self.steps,self.model_name)

        if verbose == 1:
            print("game over.step_num = ",len(self.steps))

    def print(self):
        for i in range(self.h):
            for j in range(self.w):
                if self.board[i][j]==1:
                    print("  1",end="")
                elif self.board[i][j]==-1:
                    print(" -1",end="")
                else:
                    print("  0",end="")
            print()
        print("----------------------------------------------------")

player_a=PlayerFirst(1,0.8,"./model/player_first/model1/model.ckpt",step_type=0,step_top_n=3)
player_b=PlayerFirst(-1,0.8,"./model/player_first/model2/model.ckpt",step_type=0,step_top_n=3)

game=Game("pos_model",player_a,player_b)

for i in range(1):
    game.start(1,1)
    print("no.{}".format(i))

# for i in range(10000):
#     game.start()
#     print("epochs =",i)

# print(game.isWin())
# print(game.isFull())


