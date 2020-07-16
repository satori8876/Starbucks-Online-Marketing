import pandas as pd
import numpy as np
import math

import seaborn as sb
import matplotlib.pyplot as plt


def display_transcript(data):
	data['days'] = data['time']/ 24  

	base_color = sb.color_palette()[0]
	sb.set(rc={'figure.figsize':(9,5)})
	sb.violinplot(data=data, x='days', y='event', color = base_color, inner=None, figsize=(12,6));
	data.drop(columns='days', inplace=True)
	
def display_profile(data):
	
	data['gender_int'] = np.array([row.gender == 'M' for ind, row in data.iterrows() ]).astype(int)

	# create a large figure so we can plot multiple charts to zoom in and look at the most interesting areas 
	plt.figure(figsize=(25, 8))
	plt.suptitle('Age vs. Income', size='xx-large', weight='bold')

	plt.subplot(1,3,1)
	plt.scatter(x=data['age'], y=data['income'], c=data['gender_int'], 
				cmap='RdYlBu', s=7, alpha=0.15)
	plt.xticks(size='large')
	plt.yticks(size='large')

	plt.subplot(1,3,2)
	plt.scatter(x=data['age'], y=data['income'], c=data['gender_int'], 
				cmap='RdYlBu', alpha=0.15)
	plt.xlim(20, 70)
	plt.ylim(55000, 125000)
	plt.xticks(size='large')
	plt.yticks(size='large')

	plt.subplot(1,3,3)
	plt.scatter(x=data['age'], y=data['income'], c=data['gender_int'], 
				cmap='RdYlBu', alpha=0.25)
	plt.xlim(33, 51)
	plt.ylim(71000, 104000)
	plt.xticks(size='large')
	plt.yticks(size='large');	
	
	
def display_age_by_gender(data):
	# create a dummy dataframe with the necessary columns
	df = pd.DataFrame(data[['gender', 'age']])
	df['age_cat'] = (round(df.age) // 3) * 3

	# count the number of members by age category and gender
	df = df.groupby(['age_cat', 'gender']).size()
	df = df.reset_index(name='count')

	df = df.pivot(index='age_cat', columns='gender', values='count')
	df.fillna(0, inplace=True)

	df.plot.line(figsize=(10,4), colormap='plasma');	
	
def display_income_by_gender(data):
	# create a dummy dataframe with the necessary columns
	df = pd.DataFrame(data[['gender', 'income']])
	df['income_cat'] = (round(df.income) // 3000) * 3000

	# count the number of members by income category and gender
	df = df.groupby(['income_cat', 'gender']).size()
	df = df.reset_index(name='count')

	df = df.pivot(index='income_cat', columns='gender', values='count')
	df.fillna(0, inplace=True)
	df.plot.line(figsize=(10,4), colormap='plasma');	
	
def display_membership_by_gender(data, base_date):

	if 'member_days' not in data.columns:
		calculate_member_days(data, base_date)
	
	# create a dummy dataframe with the necessary columns
	df = pd.DataFrame()
	df['gender'] = data['gender']
	df['member_for_months'] = data['member_days'] // 30

	# count the number of members by income category and gender
	df = df.groupby(['member_for_months', 'gender']).size()
	df = df.reset_index(name='count')

	df = df.pivot(index='member_for_months', columns='gender', values='count')
	df.fillna(0, inplace=True)
	df.plot.line(figsize=(10,4), colormap='plasma', xlim=(65, 0));

def calculate_member_days(data, base_date):
	# convert the became_member_on numeric values to strings and format them
	data['member_date'] =  [str(row.became_member_on)[:4]+'-'+
							   str(row.became_member_on)[4:6]+'-'+
							   str(row.became_member_on)[6:8]
							   for ind, row in data.iterrows()]

	# create a new column for numbers of days passed between becoming a member and collecting this data
	data['member_days'] = data.apply(lambda x: (base_date - pd.Timestamp(x.member_date)).days, axis=1)

	data.drop(columns=['became_member_on', 'member_date'], inplace=True)
	
	
def display_principal_components(data):
	
	colors = sb.color_palette('colorblind') + sb.color_palette('muted')[3:9]
	index  = np.arange(len(data.index))
	bar_width = 0.7
	columns = data.columns

	x_offset = np.zeros(len(data.index))

	plt.figure(figsize=(20,6))

	for ii in range(len(columns)):
		plt.barh(index, data[columns[ii]], height=bar_width, left=x_offset, color=colors[ii])
		x_offset = x_offset + data[columns[ii]]
		
	plt.gca().invert_yaxis()
	plt.yticks(np.arange(10), data.index)

	plt.title('Composition of the First Ten Components')
	plt.legend(columns)    
	plt.show()    
	
def display_cluster_sizes(data):	
	f, ax = plt.subplots(figsize=(20, 6))
	sb.heatmap(data.astype(int), annot=True, fmt='^6d',  
			   linewidths=.5, ax=ax, cmap='nipy_spectral');	