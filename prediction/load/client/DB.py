import psycopg2
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('db.init')

host = config.get('DB', 'host')
dbname = config.get('DB', 'dbname')
user = config.get('DB', 'user')
password = config.get('DB', 'password')
port = config.get('DB', 'port')

class Database():
    def __init__(self):
        self.con = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
        self.cursor = self.con.cursor()

    def __del__(self):
        self.con.close()
        self.cursor.close()

    def execute(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchall()
        df = pd.DataFrame(row)
        df.columns = [desc[0] for desc in self.cursor.description]
        return df

    def select(self, table, column, cond=None):
        sql = "SELECT {} from {} {}".format(column, table, cond)
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            df = pd.DataFrame(result)
            df.columns = [desc[0] for desc in self.cursor.description]
            return df
        except Exception as e:
            result = ("select DB err", e)
            return result

    def insert(self, table, column):
        sql = "INSERT INTO {} values {}".format(table, column)
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            df = pd.DataFrame(result)
            df.columns = [desc[0] for desc in self.cursor.description]
            return df
        except Exception as e:
            result = ("insert DB err", e)
            return result

    def commit(self):
        self.cursor.commit()
