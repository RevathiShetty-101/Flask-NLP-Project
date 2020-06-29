import pickle
import csv
data=open('/Users/admin/Documents/frontend/app/model/test_df.pickle','rb')
data_frame=pickle.load(data)
data_frame.to_csv('test.csv')