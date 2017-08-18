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
    def load_feature(self,pageNo=0,pageSize=100,epochs=10):
        datas = self.db.query("select * from step order by learn_num asc,step_num asc,chess_id asc limit " + str(pageNo * pageSize) + "," + str(pageSize))

        dct={}
        X=[]
        y=[]
        for row in datas:
            data = json.loads(row[2])
            data = np.array(data)
            winner=data[-1][-1]

            for no in range(2,8):
                line1 = np.ones((no, 1)).astype('int')
                line2 = np.ones((1, no)).astype('int')
                line3 = np.eye(no).astype('int')
                line4 = np.fliplr(line3)

                for i in range(15-no):
                    for j in range(15-no):
                        for line in np.array([line3, line4]):
                            matrix = data[i:i + no, j:j + no]
                            style=matrix * line
                            if str(style) in dct.keys():
                                dct[str(style)]+=winner
                            else:
                                dct[str(style)] =winner

                for i in range(15-no):
                    for j in range(15):
                        for line in np.array([line1]):
                            matrix = data[i:i + no, j:j + 1]
                            style = matrix * line
                            if str(style) in dct.keys():
                                dct[str(style)]+=winner
                            else:
                                dct[str(style)] =winner

                for i in range(15):
                    for j in range(15-no):
                        for line in np.array([line2]):
                            matrix = data[i:i + 1, j:j + no]
                            style = matrix * line
                            if str(style) in dct.keys():
                                dct[str(style)]+=winner
                            else:
                                dct[str(style)] =winner

                data=data*-1
                for i in range(15-no):
                    for j in range(15-no):
                        for line in np.array([line3, line4]):
                            matrix = data[i:i + no, j:j + no]
                            style=matrix * line
                            if str(style) in dct.keys():
                                dct[str(style)]+=-winner
                            else:
                                dct[str(style)] =-winner

                for i in range(15-no):
                    for j in range(15):
                        for line in np.array([line1]):
                            matrix = data[i:i + no, j:j + 1]
                            style = matrix * line
                            if str(style) in dct.keys():
                                dct[str(style)]+=-row[4]
                            else:
                                dct[str(style)] =-row[4]

                for i in range(15):
                    for j in range(15-no):
                        for line in np.array([line2]):
                            matrix = data[i:i + 1, j:j + no]
                            style = matrix * line
                            if str(style) in dct.keys():
                                dct[str(style)]+=-winner
                            else:
                                dct[str(style)] =-winner

            y.append(row[4])
        for k,v in dct.items():
            if v>0:
                print(k,'=',v)
        return
    def load(self,pageNo=0,pageSize=100,epochs=10,step_num=225,only_learn_num_0=False):
        if only_learn_num_0:
            data=self.db.query("select count(1) from step where learn_num=0 and step_num<"+str(step_num))
        else:
            data = self.db.query("select count(1) from step where step_num<" + str(step_num))

        cnt=data[0][0]

        if cnt==0:
            return None,None

        maxPageSize=int(cnt/pageSize)

        if maxPageSize==0:
            pageSize=cnt
            pageNo=0
        else:
            pageNo = pageNo % maxPageSize

        if only_learn_num_0:
           datas=self.db.query("select * from step where step_num<"+str(step_num)+" order by learn_num asc,step_num asc,chess_id asc limit "+str(pageNo*pageSize)+","+str(pageSize))
        else:
            datas = self.db.query("select * from step where learn_num=0 and step_num<" + str(
                step_num) + " order by learn_num asc,step_num asc,chess_id asc limit " + str(
                pageNo * pageSize) + "," + str(pageSize))
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

record=Record()
record.load_feature(0,100)
