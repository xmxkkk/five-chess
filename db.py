import pymysql as MySQLdb

host='localhost'
user='root'
passwd='6232'
port=3306
dbname='five'

class Db:
    conn=None

    def connect(self):
        if self.conn is None:
            self.conn=MySQLdb.connect(host=host,port=port,user=user,passwd=passwd,db=dbname,charset='utf8')
        return self.conn

    def close(self):
        self.conn.close()
        self.conn=None

    def query(self,sql,params=None):
        conn=self.connect()

        with conn.cursor() as cur:
            if params is None:
                cur.execute(sql)
            else:
                cur.execute(sql,params)

            datas=cur.fetchall()
            result=[]
            for row in datas:
                result.append(row)
            if len(result)==0:
                result=None

        self.close()
        return result

    def update(self,sql,params=None):
        conn=self.connect()

        with conn.cursor() as cur:
            if params is None:
                cur.execute(sql)
            else:
                cur.execute(sql,params)
        conn.commit()
        self.close()

# db=Db()
# result=db.query("select * from chess")
# print(result)
# db.update('update chess set winner=2 where id=%s',"1")