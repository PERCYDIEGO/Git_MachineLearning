#Este proyecto es para poder realizar análisis de datos utilizando chagpt
import sqlalchemy as sa
import pandas as pd


class Conexion:
    def __init__(self, server, database, username=None, password=None):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.driver = 'ODBC Driver 17 for SQL Server'

        if self.username is None:
            connection_string = f"mssql+pyodbc://{self.server}/{self.database}?driver={self.driver}&Trusted_Connection=yes"
        else:
            connection_string = f"mssql+pyodbc://{self.username}:{self.password}@{self.server}/{self.database}?driver={self.driver}"

        self.engine = sa.create_engine(connection_string)
        self.conn = self.engine.connect()


    def verificacion(self):
        print(self.conn)

  
    def query(self,query):

        big_query = sa.text(query)
        df = pd.read_sql_query(big_query,  self.conn)
        return df

    




