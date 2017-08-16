import numpy as np
import random

print(np.__version__)
class Game:
    board=None
    w=15
    h=15
    steps=None
    who_step=1

    def __init__(self):
        self.board=np.zeros((self.w,self.h)).astype('int')
        self.steps=[]

    def isWin(self):
        line1=np.ones((5,1)).astype('int')
        line2=np.ones((1,5)).astype('int')
        line3=np.eye(5).astype('int')
        line4=np.fliplr(line3)

        lines=np.array([line1,line2,line3,line4])

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
    def isFull(self):
        return (self.board==0).sum()==0

    def step(self):
        lst=[]
        for i in range(self.h):
            for j in range(self.w):
                if self.board[i][j]==0:
                    lst.append((i,j,self.who_step))
        return random.choice(lst)

    def start(self,who_step=1):
        while True:
            full=self.isFull()
            win=self.isWin()
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
        print("--------------------------")

game=Game()
# game.board[0][0]=-1
# game.board[1][1]=-1
# game.board[2][2]=-1
# game.board[3][3]=-1
# game.board[4][4]=-1

game.start()

# print(game.isWin())
# print(game.isFull())

