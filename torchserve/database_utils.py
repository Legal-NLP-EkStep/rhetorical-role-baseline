import numpy
import psycopg2


def addapt_numpy_float16(numpy_float16):
    return psycopg2.extensions.AsIs(numpy_float16)


psycopg2.extensions.register_adapter(numpy.float16, addapt_numpy_float16)


class PostgresDatabase:
    def __init__(self, host='', database='', user='', password=''):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = self.connect(self.host, self.database, self.user, self.password)
        self.cursor = self.connection.cursor()
        self.check_if_table_exist_otherwise_create()
        self.connection.commit()
        self.cursor.close()
        self.connection.close()

    @staticmethod
    def connect(host, database, user, password):
        """ Connect to the PostgreSQL database server """
        conn = None
        try:
            conn = psycopg2.connect(database=database, user=user, password=password, host=host)
            return conn
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def check_if_table_exist_otherwise_create(self):

        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS public.rhetorical_role_api_credentials (token varchar(100) NOT NULL,request_count bigint NOT NULL,used_quota bigint NOT NULL, allocated_to varchar(100));")

    def fetch(self, table_name="rhetorical_role_api_credentials"):
        self.connection = self.connect(self.host, self.database, self.user, self.password)
        self.cursor = self.connection.cursor()
        select_statement = f'select * from public.{table_name}'
        self.cursor.execute(select_statement)
        data = self.cursor.fetchall()
        self.cursor.close()
        self.connection.close()
        return data

    def update_request_count(self, token, request_count, quota_used, table_name='rhetorical_role_api_credentials'):
        self.connection = self.connect(self.host, self.database, self.user, self.password)
        self.cursor = self.connection.cursor()
        columns = ['token']
        values = [token]
        update_statement = f'UPDATE public.{table_name} SET request_count={str(request_count)}, used_quota={str(quota_used)}WHERE (%s)=%s'
        self.cursor.execute(update_statement, (psycopg2.extensions.AsIs(','.join(columns)), tuple(values)))
        self.connection.commit()
        self.cursor.close()
        self.connection.close()
