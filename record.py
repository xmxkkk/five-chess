from db import Db
import json
import hashlib
import numpy as np

class Record:
    db=None
    def __init__(self):
        self.db=Db()

    def save_chess(self,steps):
        winner=steps[-1][-1]

        jsonStr=json.dumps(steps)
        # print(type(jsonStr))

        md5=hashlib.md5(jsonStr.encode("utf-8")).hexdigest()

        row=self.db.query('select * from chess where md5=%s',md5)
        if row is None:
            self.db.update('insert into chess (steps,md5,create_time,winner,step_num) values (%s,%s,now(),%s,%s)',[jsonStr,md5,winner,len(steps)])
        else:
            self.db.update('update chess set num=num+1 where md5=%s',md5)

        row = self.db.query('select * from chess where md5=%s', md5)

        row = row[0]

        if winner==0:
            return

        scores=np.linspace(0,winner,len(steps))
        i=0
        for step in steps:
            board=[[0 for i in range(15)] for j in range(15)]
            board[step[0]][step[1]]=step[2]
            boardStr=json.dumps(board)
            boardStrMd5 = hashlib.md5(boardStr.encode("utf-8")).hexdigest()
            self.db.update('insert into step (chess_id,board,md5,score,idx,create_time,step_num) values (%s,%s,%s,%s,%s,now(),%s)'
                           ,[str(row[0]),boardStr,boardStrMd5,str(scores[i]),str(i),str(len(steps))])
            i+=1
    def dev_chess(self):
        datas=self.db.query('select * from chess where is_dev=0 order by step_num asc')
        for row in datas:
            chess_id=row[0]
            steps=json.loads(row[1])
            winner=steps[-1][-1]
            if winner==0:
                return

            scores=np.linspace(0,winner,len(steps))
            i = 0
            for step in steps:
                board = [[0 for i in range(15)] for j in range(15)]
                board[step[0]][step[1]] = step[2]
                boardStr = json.dumps(board)
                boardStrMd5 = hashlib.md5(boardStr.encode("utf-8")).hexdigest()
                self.db.update(
                    'insert into step (chess_id,board,md5,score,idx,create_time) values (%s,%s,%s,%s,%s,now())',
                    [str(row[0]), boardStr, boardStrMd5, str(scores[i]), str(i)])
                i += 1

    ''''''
    def load(self,pageNo=0,pageSize=100,epochs=0,step_num=90):

        data=self.db.query("select count(1) from step where step_num<"+str(step_num))
        cnt=data[0][0]
        maxPageSize=int(cnt/pageSize)

        pageNo = pageNo % maxPageSize

        datas=self.db.query("select * from step where step_num<"+str(step_num)+" order by learn_num asc,step_num asc,chess_id asc limit "+str(pageNo*pageSize)+","+str(pageSize))

        result=[]
        y=[]

        ids=[]

        for row in datas:
            data=json.loads(row[2])
            result.append(data)
            y.append(row[4])
            ids.append(str(row[0]))

        self.db.update('update step set learn_num=learn_num+'+str(epochs)+' where id in ('+','.join(ids)+')')

        return np.array(result),np.array(y)

