from db import Db
import json
import hashlib
import numpy as np

class Record:
    db=None
    def __init__(self):
        self.db=Db()

    def save(self,steps):
        winner=steps[-1][-1]

        jsonStr=json.dumps(steps)
        # print(type(jsonStr))

        md5=hashlib.md5(jsonStr.encode("utf-8")).hexdigest()

        row=self.db.query('select * from chess where md5=%s',md5)
        if row is None:
            self.db.update('insert into chess (steps,md5,create_time,winner) values (%s,%s,now(),%s)',[jsonStr,md5,winner])
        else:
            self.db.update('update chess set num=num+1 where md5=%s',md5)

    '''
    def load(self,pageNo=0,pageSize=1):
        datas = self.db.query("select * from chess order by learn_num asc,id asc limit " + str(pageNo * pageSize) + "," + str(pageSize))
        data=datas[0]

        one = []
        board = [[0 for i in range(15)] for j in range(15)]
        steps = json.loads(data[1])
        for step in steps:
            board[step[0]][step[1]] = step[2]
            one.append(board.copy())
        for step in range(225 - len(steps)):
            one.append([[0 for i in range(15)] for j in range(15)])

        return np.array(one),np.array(data[4])
    '''
    ''''''
    def load(self,pageNo=0,pageSize=100):
        datas=self.db.query("select * from chess order by learn_num asc,id asc limit "+str(pageNo*pageSize)+","+str(pageSize))

        result=[]
        y=[]
        for row in datas:
            one=[]
            board=[0 for i in range(225)]
            steps=json.loads(row[1])
            for step in steps:
                board[step[0]*15+step[1]]=step[2]
                one.append(board.copy())
            for step in range(225-len(steps)):
                one.append([0 for i in range(225)])

            result.append(one)
            y.append(row[4])

            result.append(np.array(one)*-1)
            y.append(row[4]*-1)

        return np.array(result),np.array(y)

