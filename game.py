import numpy as np
import random
from brain import Brain

print(np.__version__)
class Game:
    board=None
    w=15
    h=15
    steps=None
    who_step=1
    brain=None

    def __init__(self,brain):
        self.board=np.zeros((self.w,self.h)).astype('int')
        self.steps=[]
        self.brain=brain

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

    def step(self):
        return self.brain.step(self)

    def start(self,who_step=1):
        self.who_step=who_step
        while True:
            full=self.is_full()
            win=self.is_win()
            if full or win:
                if full:
                    print("draw.")
                if win:
                    print(str(win)+" win.")
                break

            s=self.step()
            self.board[s[0]][s[1]]=s[2]
            self.steps.append(s)
            self.who_step*=-1

            self.print()

        print("game over.")

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

brain=Brain()

game=Game(brain)
# game.board[0][0]=-1
# game.board[1][1]=-1
# game.board[2][2]=-1
# game.board[3][3]=-1
# game.board[4][4]=-1

game.start()

# print(game.isWin())
# print(game.isFull())

