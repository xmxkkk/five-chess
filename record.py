from db import Db
import json
import hashlib
import numpy as np
from sklearn.preprocessing import Normalizer
from util import padwithtens,handle,board_line8,board_shape,norm_y

class Record:
    db=None
    def __init__(self):
        self.db=Db()

    def save_chess(self,steps,model_name):
        winner=steps[-1][-1]

        if winner<=0:
            return

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

        pow_idx=0
        idx=0
        size=len(steps)

        pre_score=0

        board = [[0 for i in range(15)] for j in range(15)]
        for step in steps:
            board[step[0]][step[1]]=step[2]

            lines = board_line8(board, step)
            boards = board_shape(board, step)

            boardStr=json.dumps(board)
            boardStrMd5 = hashlib.md5(boardStr.encode("utf-8")).hexdigest()

            score=(1000000*winner/size)*(0.9**(size-pow_idx-1))

            if pre_score>0:
                add_score=score-pre_score
            else:
                add_score=0

            self.db.update('insert into step (chess_id,board,md5,score,idx,create_time,step_num,step_pos,add_shape,'
                           +'add_score,board1,board2,board3,board4,board5,model_name)'
                           +'values '
                           +'(%s,%s,%s,%s,%s,now(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                           ,[str(row[0]),boardStr,boardStrMd5,str(score),str(idx),str(len(steps))
                            ,json.dumps([step[0],step[1]])
                             ,json.dumps(lines)
                             ,str(add_score)
                             ,json.dumps(boards[0]),json.dumps(boards[1]),json.dumps(boards[2]),json.dumps(boards[3])
                               , json.dumps(boards[4])
                               , model_name])

            pre_score = score

            pow_idx+=1
            idx+=1

    ''''''
    def load(self,pageNo=0,pageSize=100,model_name=None):
        if model_name is None:
            data = self.db.query("select count(1) from step where score>0")
        else:
            data = self.db.query("select count(1) from step where score>0 and model_name='"+model_name+"'")

        cnt=data[0][0]

        if cnt==0:
            return None,None

        maxPageSize=int(cnt/pageSize)

        if maxPageSize==0:
            pageSize=cnt
            pageNo=0
        else:
            pageNo = pageNo % maxPageSize

        datas = self.db.query("select * from step limit " + str(pageNo * pageSize) + "," + str(pageSize))
        result=[]
        y=[]

        ids=[]
        board1=[]
        board2=[]
        board3=[]
        board4=[]
        board5=[]
        for row in datas:
            data=json.loads(row[9])
            result.append(data)
            y.append(row[10])
            board1.append(json.loads(row[11]))
            board2.append(json.loads(row[12]))
            board3.append(json.loads(row[13]))
            board4.append(json.loads(row[14]))
            board5.append(json.loads(row[15]))

            ids.append(str(row[0]))

        # self.db.update('update step set learn_num=learn_num+'+str(epochs)+' where id in ('+','.join(ids)+')')

        x=handle(np.array(result,dtype='float32'))
        board1 = handle(np.array(board1,dtype='float32'))
        board2 = handle(np.array(board2, dtype='float32'))
        board3 = handle(np.array(board3, dtype='float32'))
        board4 = handle(np.array(board4, dtype='float32'))
        board5 = handle(np.array(board5, dtype='float32'))
        y=norm_y(y)

        return x,board1,board2,board3,board4,board5,y


