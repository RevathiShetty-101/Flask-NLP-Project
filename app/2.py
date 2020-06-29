import sqlalchemy as db
import pandas as pd
engine=db.create_engine('mysql://root:root@localhost/stud')
connection=engine.connect()
metadata=db.MetaData()
BigQuestion=db.Table('BigQuestion',metadata,autoload=True,autoload_with=engine)
query=db.select([BigQuestion])
ResultProxy=connection.execute(query)
ResultSet=ResultProxy.fetchall()
rf=pd.DataFrame(ResultSet,columns=[0,1,'hypothesis'])     
data = {'hypothesis':[rf]}  
data_frame=pd.DataFrame(data)
print(data_frame.head())