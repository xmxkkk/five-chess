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

        # scores=np.linspace(0,winner,len(steps))

        board_no=-99999

        i=0
        size=len(steps)

        pre_score=0

        for step in steps:
            board=[[0 for i in range(15)] for j in range(15)]
            board[step[0]][step[1]]=step[2]

            line0 = [board[step[0]][step[1]],
                     board[step[0]][step[1] + 1] if step[1]+1<15 else board_no,
                     board[step[0]][step[1] + 2] if step[1]+2<15 else board_no,
                     board[step[0]][step[1] + 3] if step[1]+3<15 else board_no,
                     board[step[0]][step[1] + 4] if step[1]+4<15 else board_no]

            line1 = [board[step[0]][step[1]],
                    board[step[0]][step[1] - 1] if step[1]-1>0 else board_no,
                    board[step[0]][step[1] - 2] if step[1]-2>0 else board_no,
                    board[step[0]][step[1] - 3] if step[1]-3>0 else board_no,
                    board[step[0]][step[1] - 4] if step[1]-4>0 else board_no]

            line2 = [board[step[0]][step[1]],
                     board[step[0]+1][step[1]] if step[0]+1<15 else board_no,
                     board[step[0]+2][step[1]] if step[0]+2<15 else board_no,
                     board[step[0]+3][step[1]] if step[0]+3<15 else board_no,
                     board[step[0]+4][step[1]] if step[0]+4<15 else board_no]

            line3 = [board[step[0]][step[1]],
                     board[step[0] - 1][step[1]] if step[0]-1>0 else board_no,
                     board[step[0] - 2][step[1]] if step[0]-1>0 else board_no,
                     board[step[0] - 3][step[1]] if step[0]-1>0 else board_no,
                     board[step[0] - 4][step[1]] if step[0]-1>0 else board_no]

            line4 = [board[step[0]][step[1]],
                     board[step[0] - 1][step[1] - 1] if step[0]-1>0 and step[1]-1>0 else board_no,
                     board[step[0] - 2][step[1] - 2] if step[0]-2>0 and step[1]-1>0 else board_no,
                     board[step[0] - 3][step[1] - 3] if step[0]-3>0 and step[1]-1>0 else board_no,
                     board[step[0] - 4][step[1] - 4] if step[0]-4>0 and step[1]-1>0 else board_no]

            line5 = [board[step[0]][step[1]],
                     board[step[0] - 1][step[1] + 1] if step[0]-1>0 and step[1]+1<15 else board_no,
                     board[step[0] - 2][step[1] + 2] if step[0]-2>0 and step[1]+2<15 else board_no,
                     board[step[0] - 3][step[1] + 3] if step[0]-3>0 and step[1]+3<15 else board_no,
                     board[step[0] - 4][step[1] + 4] if step[0]-4>0 and step[1]+4<15 else board_no]

            line6 = [board[step[0]][step[1]],
                     board[step[0] + 1][step[1] - 1] if step[0]+1<15 and step[1]-1>0 else board_no,
                     board[step[0] + 2][step[1] - 2] if step[0]+2<15 and step[1]-2>0 else board_no,
                     board[step[0] + 3][step[1] - 3] if step[0]+3<15 and step[1]-3>0 else board_no,
                     board[step[0] + 4][step[1] - 4] if step[0]+4<15 and step[1]-4>0 else board_no]

            line7 = [board[step[0]][step[1]],
                     board[step[0] + 1][step[1] + 1] if step[0]+1<15 and step[1]+1<15 else board_no,
                     board[step[0] + 2][step[1] + 2] if step[0]+2<15 and step[1]+2<15 else board_no,
                     board[step[0] + 3][step[1] + 3] if step[0]+3<15 and step[1]+3<15 else board_no,
                     board[step[0] + 4][step[1] + 4] if step[0]+4<15 and step[1]+4<15 else board_no]

            lines = [line0, line1, line2, line3, line4, line5, line6, line7]

            boardStr=json.dumps(board)
            boardStrMd5 = hashlib.md5(boardStr.encode("utf-8")).hexdigest()

            score=(1000000*winner/size)*(0.9**(size-i-1))

            if pre_score>0:
                add_score=score-pre_score
            else:
                add_score=0

            self.db.update('insert into step (chess_id,board,md5,score,idx,create_time,step_num,step_pos,add_shape,add_score'
                           ') values (%s,%s,%s,'
                           '%s,%s,now(),%s,%s,%s,%s)'
                           ,[str(row[0]),boardStr,boardStrMd5,str(score),str(i),str(len(steps))
                            ,json.dumps([step[0],step[1]])
                             ,json.dumps(lines)
                             ,str(add_score)])

            pre_score = score

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
                    'insert into step (chess_id,board,md5,score,idx,create_time,step_pos) values (%s,%s,%s,%s,%s,'
                    'now(),%s)',
                    [str(row[0]), boardStr, boardStrMd5, str(scores[i]), str(i), json.dumps([step[0],step[1]])])
                i += 1

    ''''''
    def load(self,pageNo=0,pageSize=100):
        data = self.db.query("select count(1) from step where score>0")

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

        for row in datas:
            data=json.loads(row[9])
            result.append(data)
            y.append(row[10])
            ids.append(str(row[0]))

        # self.db.update('update step set learn_num=learn_num+'+str(epochs)+' where id in ('+','.join(ids)+')')

        return np.array(result),np.array(y)

