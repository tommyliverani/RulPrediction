import pandas as pd
from itables import show
from itables import options as opt
import numpy as np
import csv



def select_rul(data):
    last_node=data['node'][0]
    last_anomaly=0
    for index,row in data.iterrows():
        if data['is_rising_anomaly'][index]==1:
            last_anomaly=index
        if last_node != data['node'][index]:
            for i in range(last_anomaly+1,index):
                data=data.drop(i)
        last_node = data['node'][index]
    for i in range(last_anomaly,data.index[-1]):
                data=data.drop(i+1)
    return data
				

def compute_rul(data):
	data['rul']=0
	rlu=0
	last_index=0
	last_node=data['node'][0]
	for index,row in data.iterrows():
		if data['node'][index]!=last_node:
			last_node=data['node'][index]
			last_index=index
		if data['is_rising_anomaly'][index]==1.0:
			for i in range(last_index,index,1):
				data['rul'][i]=index-i
			last_index=index+1
		last_node=data['node'][index]
	return data

def max_rul(data):
	max=data['rul'][0]
	for index,row in data.iterrows():
		if data['rul'][index]>max:
			max=data['rul'][index]
	return max

def normalize_rul(data):
    max=max_rul(data)*1.0
    data['rul']=data['rul']*1.0
    for index,row in data.iterrows():
        data['rul'][index]=data['rul'][index]*1.0/max
    return data




def compute_final_rul_data(path):
	full_data = pd.read_csv(path+'/final_data_full.csv')
	full_data= select_rul(full_data)
	full_data= compute_rul(full_data)
	full_data.to_csv(path+'/final_rul_data.csv')

	

def split_data(data,ratio):
	np.random.seed(42)	
	nodes = data.node.unique()
	np.random.shuffle(nodes)
	sep = int(ratio * len(nodes))
	tr_nodes = set(nodes[:sep])
	tr_list, ts_list = [], []
	for node, gdata in data.groupby('node'):
        	if node in tr_nodes:
        		tr_list.append(gdata)
       		else:
        		ts_list.append(gdata)
	tr_data = pd.concat(tr_list)
	ts_data = pd.concat(ts_list)
	return tr_data, ts_data
	
	
		
	
	




	
