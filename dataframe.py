import pandas as pd
data=pd.read_csv('/Users/admin/Documents/frontend/premise.csv',usecols=["answer"])
data1=pd.read_csv('/Users/admin/Documents/frontend/hypothesis.csv',usecols=["ref"])
data_frame=pd.DataFrame(data)
data_frame=data_frame.rename(columns={'answer':'premise'})
data_frame['hypothesis']=data1
print(data_frame.head())
#data_frame.to_csv('a.csv')