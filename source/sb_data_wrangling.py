import pandas as pd
import numpy as np
import math


def clean_portfolio(data):
	# create a new id column, combining the first four letters of the offer type, the duration and the difficulty columns
	data['my_id']  = np.array(['{}_{}d_{}'.format(val['offer_type'][:4], val['duration'], val['difficulty']) for ind, val in data.iterrows()])

	### create mappings between id and my_id and drop id column
	id_to_my_id = dict(zip(data.id, data.my_id))
	#my_id_to_id = dict(zip(data.my_id, data.id))

	data.drop(columns=['id'], inplace=True)
	data.set_index( 'my_id',  inplace=True)

	# create and add the one-hot encoded columns 
	data['web']  = np.array(['web' in channel for channel in data.channels]).astype(int)
	data['email'] = np.array(['email' in channel for channel in data.channels]).astype(int)
	data['mobile'] = np.array(['mobile' in channel for channel in data.channels]).astype(int)
	data['social']  = np.array(['social' in channel for channel in data.channels]).astype(int)

	data['bogo'] = np.array([offer == 'bogo'  for offer in data.offer_type]).astype(int)
	data['info'] = np.array([offer == 'informational'  for offer in data.offer_type]).astype(int)
	data['disc'] = np.array([offer == 'discount'  for offer in data.offer_type]).astype(int)

	# drop the categorical columns
	data.drop(columns=['channels', 'offer_type'], inplace=True)
	
	return id_to_my_id
	
	
def clean_profile(data):
	# use pandas to create one-hot encoded columns for the gender column
	df = pd.get_dummies(data.gender)

	# add the new columns to the profile dataframe
	data['female'] = df['F']
	data['male']   = df['M']
	data['other']  = df['O']

	data.drop(columns='gender', inplace=True)
	data.set_index('id', inplace=True)
	
	
def get_invalid_view_indices(data):
	### figure out how to get the earliest non-negative view_time column for each offer received
	data = data.sort_values(by=['person', 'offer_id', 'time_x', 'time_y'])

	pid = ''
	oid = ''
	tx  = 0
	vt  = 0
	prev_ind = 0
	drop_ind = []
	for ind, row in data.iterrows():
		# if this is a duplicate row
		if (row.person == pid) & (row.offer_id == oid) & (row.time_x == tx):
			# keep the smallest non-negative value
			if vt < 0:
				drop_ind.append(prev_ind)
				vt = row.view_time
			else:    
				drop_ind.append(ind)
		else:
			pid = row.person
			oid = row.offer_id
			tx  = row.time_x
			vt  = row.view_time
		prev_ind = ind
		
	return drop_ind	
					
					
def get_valid_transaction_indices(data):

	pid = ''
	ty  = ''
	am  = 0
	pt  = 0
	prev_ind = 0
	drop_ind = []

	for ind, row in data.iterrows():
		# if this is a duplicate row
		if (row.person == pid) & (row.time_y == ty) & (row.amount == am):
			# keep the smallest non-negative purchase time value
			if (pt < 0):
				drop_ind.append(prev_ind)    
				pt = row.purc_time
			else:    
				drop_ind.append(ind)
				
		else:
			pid = row.person
			ty  = row.time_y
			am  = row.amount
			pt  = row.purc_time
			
		prev_ind = ind
		
	return drop_ind