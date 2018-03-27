############################ BBC DATA TASK ############################
# 																	  #
# Analysing traffic data from BBC Website over June and July of 2016. #
#																	  #
#######################################################################

# Libraries ###########################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster

# Useful Dictionaries converting strings to integers ##################

platform_dict = {'Mobile' : int(0), 
		 'Tablet' : int(1), 
		 'Computer': int(2), 
		 'Big screen': int(3)}

product_dict = {'sport': 0, 
		'news': 1, 
		'bbcthree': 2, 
		'iplayerradio': 3, 
		'weather': 4, 
		'homepageandsearch': 5, 
		'cbbc': 6, 
		'music': 7, 
		'tvandiplayer': 8, 
		'aboutthebbc': 9, 
		'knowledge': 10, 
		'cbeebies': 11, 
		'travel': 12, 
		'newsbeat': 13} 


#######################################################################

def load_data():

	# Loads data from the csv file with title 'web_usage_big.csv'

	df = pd.read_csv('web_usage_big.csv')

	# Pickles it for quicker loading later

	df.to_pickle('web_use.pkl')

	# Return dataframe

	return df

#######################################################################

def clean_data_for_analysis():

	# Cleans data for analysis.

	# Load data from pickle file

	data = pd.read_pickle('web_use.pkl')

	# Convert date_time entries to datetime python variables

	data['date_time'] = pd.to_datetime(data['date_time'], errors = 'coerce')

	# Encode platform data as integers

	data['platform'] = data['platform'].map(platform_dict)

	# Group products beginning with 'kl-' under the category 'Konwledge'

	data['product'] = data['product'].replace('kl-.*', 'knowledge', regex=True)

	# Encode products as integers

	data['product'] = data['product'].map(product_dict)

	# Replace NaNs in 'search_term' with empty strings

	data['search_term'].fillna('', inplace=True)

	# Keep only the columns corresponding to 'user_id', 'date_time', 'platform', and 'product'

	data.drop(['name_page','page_url','search_term','region', 'app_type'], axis = 1, inplace=True)

	# Drop rows that contain NaNs

	data = data.dropna()

	# Ensure 'platform' and 'product' are integers

	data[['platform','product']] = data[['platform','product']].astype(int)

	# Replace 'user_id' with single integer value for ease of
	# reading and faster evaluation

	# Get list of unique users

	users = data['user_id'].unique()

	# Create new dataframe of unique users

	users_df = pd.DataFrame(data = users, columns = ['user_id'])

	# Turn this dataframe into a dictionary

	users_dict = (users_df.reset_index()
			      .set_index('user_id')
			      .to_dict()
			      )

	# Map integers to user ids in original dataframe

	data['user_id'] = data['user_id'].map(users_dict['index'])

	# Pickle dataframe for easy access later

	data.to_pickle('cleaned_data.pkl')

	return data

#######################################################################

def visits_per_prod():

	# Creates plot of percentage of visits to each content type

	# Unpickle dataframe

	data = pd.read_pickle('cleaned_data.pkl')

	# Forget previous plots

	plt.clf()
	plt.cla()
	plt.close()

	# Count the number of visits to each type of product and store in dataframe

	sorted_data = (data.groupby('product')['platform']
			   .count()
			   .sort_values(ascending=False)
			   .reset_index()
			   )

	# Express number of visits to each platform as percentage of total visits

	sorted_data['platform'] = sorted_data['platform']/sorted_data['platform'].sum()*100

	# Map integers to strings using product_dict and dictionary comprehension

	sorted_data['product'] = sorted_data['product'].map({k:v for v,k in product_dict.items()})

	# Set index as 'product'

	sorted_data.set_index('product', inplace=True)

	# Plot data

	ax = sorted_data.plot(kind = 'bar', legend=False)

	# Plot options

	ax.set_title('Visits to types of content during June and July 2016')

	ax.set_ylabel('Percentage of visits')

	ax.set_xlabel('Product categories')

	plt.tight_layout()

	# Save plot

	plt.savefig('VisitsPerProduct.png')

	# Show plot

	plt.show()

	return

#######################################################################

def visits_per_platform():

	# Creates plot of percentage of visits made through each platform

	# Load data from pickle jar

	data = pd.read_pickle('cleaned_data.pkl')

	# Forget previous plots

	plt.clf()
	plt.cla()
	plt.close()

	# Count the number of visits that came from the different platforms

	sorted_data = (data.groupby('platform')['product']
			   .count()
			   .sort_values(ascending=False)
			   .reset_index()
			   )

	# Express it as a percentage of total visits

	sorted_data['product'] = sorted_data['product']/sorted_data['product'].sum()*100

	# map platform integers to strings 

	sorted_data['platform'] = sorted_data['platform'].map({k:v for v,k in platform_dict.items()})

	# Set index as 'platform'

	sorted_data.set_index('platform', inplace=True)

	# Plot the data in sorted_data()

	ax = sorted_data.plot(kind = 'bar', legend=False)

	# Plot options

	ax.set_title('Content views per platform during June and July 2016')

	ax.set_ylabel('Percentage of views')

	ax.set_xlabel('Platform')

	plt.tight_layout()

	# Save plot

	plt.savefig('VisitsPerPlatform.png')

	plt.show()

	return

#######################################################################

def visits_per_product_time_series():

	# Plots number of visits to top 5 content categories during 
	# June/July 2016 as time series

	# Unpickle data

	data = pd.read_pickle('cleaned_data.pkl')

	# Forget previous plots

	plt.clf()
	plt.cla()
	plt.close()

	# Set time column

	times = data['date_time']

	# collect data and group by the date, and the hour, and count the number of
	# of visits per platform per product, per hour.

	collated = data.groupby([times.dt.date, times.dt.hour, 'product'])['user_id'].count()

	# Rese the index of this dataframe

	collated = collated.reset_index(level=2)

	# Create series corresponding to the number of visits to each type of content per hour during the period

	sport = collated[collated['product']==0]['user_id'].rename('Sport')
	news = collated[collated['product']==1]['user_id'].rename('News')
	homepageAndSearch = collated[collated['product']==5]['user_id'].rename('Homepage and Search')
	tvAndiPlayer = collated[collated['product']==8]['user_id'].rename('TV and iPlayer')
	weather = collated[collated['product']==4]['user_id'].rename('Weather')

	# Put these separate series into one big dataframe.

	full_data = pd.DataFrame(data=[sport, news, homepageAndSearch, tvAndiPlayer, weather]).T

	# Plot dataframe

	ax = full_data.plot(figsize = (8,5))

	# Plot options

	plt.xticks(range(1,648,24),full_data.index.levels[0][1:].astype(str).tolist(),rotation=90)

	ax.set_ylabel('Visits')

	ax.set_xlabel('Time')

	ax.set_title('Visits to top 5 content categories per hour during entire period')

	plt.tight_layout()

	# Save plot

	plt.savefig('TimeSeriesJuneJulyTop5.png')

	plt.show()

	return

#######################################################################

def visits_per_product_weekday():

	# Generates plot of proportion of visits per hour to each content 
	# type during the week

	# Load pickle

	data = pd.read_pickle('cleaned_data.pkl')

	# Forget previous plots

	plt.clf()
	plt.cla()
	plt.close()

	# Extract times of dataframe

	times = data['date_time']

	# Ensure we only use weekdays

	data = data[data['date_time'].dt.dayofweek<5]

	# Count the number of visits per hour and per 'product'

	collated = data.groupby([times.dt.hour, 'product'])['user_id'].count()

	# Reset index

	collated = collated.reset_index(level=1)

	# create new series for each content type

	sport = collated[collated['product']==0]['user_id'].rename('Sport')
	news = collated[collated['product']==1]['user_id'].rename('News')
	homepageAndSearch = collated[collated['product']==5]['user_id'].rename('Homepage and Search')
	tvAndiPlayer = collated[collated['product']==8]['user_id'].rename('TV and iPlayer')
	weather = collated[collated['product']==4]['user_id'].rename('Weather')

	# Put them all into a data frame and divide by the number of visits 
	# to each content type during the week

	full_data = pd.DataFrame(data = [sport/sport.sum()*100, 
					 news/news.sum()*100, 
					 homepageAndSearch/homepageAndSearch.sum()*100, 
					 tvAndiPlayer/tvAndiPlayer.sum()*100, 
					 weather/weather.sum()*100]).T

	# Plot options

	ax = full_data.plot(grid=True)

	ax.set_ylabel('Percentage of visits')

	ax.set_xlabel('Hour (in 24h format)')

	ax.set_title('Visits to top 5 content categories per hour during weekdays')

	plt.xticks(range(0,24),[str(k)+" - "+str(k+1) for k in range(0,24)],rotation=90)

	plt.tight_layout()

	# Save plot

	plt.savefig('TimeSeriesWeekday.png')

	plt.show()

	return

#######################################################################

def visits_per_product_weekend():

	# Generates plot of proportion hourly visits to each content type per hour 
	# during the weekend (same as above apart from line 390)

	data = pd.read_pickle('cleaned_data.pkl')

	# Forget previous plots

	plt.clf()
	plt.cla()
	plt.close()

	times = data['date_time']

	# Ensure we only keep weekend days

	data = data[data['date_time'].dt.dayofweek>=5]

	collated = data.groupby([times.dt.hour, 'product'])['user_id'].count()

	collated = collated.reset_index(level=1)

	sport = collated[collated['product']==0]['user_id'].rename('Sport')
	news = collated[collated['product']==1]['user_id'].rename('News')
	homepageAndSearch = collated[collated['product']==5]['user_id'].rename('Homepage and Search')
	tvAndiPlayer = collated[collated['product']==8]['user_id'].rename('TV and iPlayer')
	weather = collated[collated['product']==4]['user_id'].rename('Weather')

	full_data = pd.DataFrame(data = [sport/sport.sum()*100, 
					 news/news.sum()*100, 
					 homepageAndSearch/homepageAndSearch.sum()*100, 
					 tvAndiPlayer/tvAndiPlayer.sum()*100, 
					 weather/weather.sum()*100]).T

	ax = full_data.plot(grid = True)

	ax.set_ylabel('Percentage of visits')

	ax.set_xlabel('Hour (in 24h format)')

	ax.set_title('Visits to top 5 content categories per hour during weekends')

	ax.set_ylim(0,10)

	plt.xticks(range(0,24),[str(k)+" - "+str(k+1) for k in range(0,24)],rotation=90)

	plt.tight_layout()

	plt.savefig('TimeSeriesWeekends.png')

	plt.show()

	return

#######################################################################

def user_prod_count():

	# Generates a dataframe that is indexed by users, with column entries
	# given by the number of times each user has accessed each of the 
	# top 5 content categories

	# Unpickle data

	data = pd.read_pickle('cleaned_data.pkl')

	# Count total number of visits each user made

	total_visits = data.groupby('user_id')['date_time'].count()

	# Count number of sports, news, homepage/search page, tv and iPlayer, 
	# and weather content views made per user

	sport = data.groupby(data[data['product']==0]['user_id'])['product'].count()

	news = data.groupby(data[data['product'] == 1]['user_id'])['product'].count()

	homepageAndSearch = data.groupby(data[data['product'] == 5]['user_id'])['product'].count()

	tvAndiPlayer = data.groupby(data[data['product'] == 8]['user_id'])['product'].count()

	weather = data.groupby(data[data['product'] == 4]['user_id'])['product'].count()

	# Put it all in a big data frame

	full_data = pd.DataFrame(index=total_visits.index.astype(int), data = {'Total Visits': total_visits,
										'Sport' : sport, 
										'News': news, 
										'Homepage and Search':homepageAndSearch,
										'TV and iPlayer':tvAndiPlayer,
										'Weather':weather})

	# Replace NaNs with 0.

	full_data.fillna(int(0),inplace=True)

	# Pickle data frame

	full_data.to_pickle('user_content.pkl')

	return full_data

#######################################################################

def k_means_data():

	# Generates dataframe for use with kMeans clustering algorithm

	# Unpickle data

	data = pd.read_pickle('user_content.pkl')

	# Loop over data.columns

	for col in data.columns: 
		if col != 'Total Visits':

			# Divide each column by total views to get the proportion of views per
			# product per user.
			
			data[col] = data[col]/data['Total Visits']

	# Get rid of total trips column

	data.drop('Total Visits', axis=1, inplace = True)

	return data


#######################################################################

def k_means_cluster_elbow():

	# Tries k_Means with increasing k from 1 to 10, then plots the learning curve

	# Call function from above

	data = k_means_data()

	# Convert ourput to an np.array

	mat = data.as_matrix()

	# Initialise empty list in which to store sum of squared errors

	errs = list()

	# Loop over potential number of clusters

	for k in range(1,11):

		# Fit model
		km = sklearn.cluster.KMeans(n_clusters = k, random_state=0).fit(mat)

		# Append sum squared error to errs
		errs.append(km.inertia_)

	# Forget previous plots

	plt.clf()
	plt.cla()
	plt.close()

	# Plot options

	plt.plot(range(1,11),errs)
	plt.xlabel('Number of clusters')
	plt.ylabel('Sum squared error')
	plt.title('Plot of Sum Squared Error of kMeans clustering')
	plt.xticks(range(0,11))

	plt.savefig('kMeansSSE.png')

	plt.show()

	return

#######################################################################

def k_means_clusters_bar():

	# Generates bar chart displaying profiles of average, typical users from the
	# four different clusters

	# Load data

	data = k_means_data()

	# Turn it into array

	mat = data.as_matrix()

	# Run ML clustering algorithm with k = 4 (the optimal number of clusters was 
	# selected by using the elbow method and looking at the plot in 'kMeansSSe.png')

	km = sklearn.cluster.KMeans(n_clusters = 4, random_state=0).fit(mat)

	# Create dataframe with cluster centorids from clustering algorithm

	user_clusters = pd.DataFrame(index = ['Sports fans', 'Homepage searchers','News fans', 'Telly fans'],
				     data = km.cluster_centers_*100, 
				     columns = data.columns)

	plt.close()

	# Plot chart

	ax = user_clusters.plot(kind='barh', xticks = range(0,110,10))

	plt.xlabel('Percentage of content viewed in period')

	plt.ylabel('Type of user')

	ax.set_title('Percentage of views per content type for the \nfour different types of users')

	plt.tight_layout()

	# Save plot

	plt.savefig('ClusterProfile.png')

	plt.show()

	# Set cluster labels for different users

	data['cluster_label'] = km.labels_

	# Pickle data

	data.to_pickle('clustered_data.pkl')

	return

#######################################################################

def cluster_pie():

	# Generates pie chart showing amount of users belonging to each 
	# group

	# Load data

	data = pd.read_pickle('clustered_data.pkl')

	# Create list of numbers of size of each cluster

	nums = [len(data[data['cluster_label'] == 0]), 
			len(data[data['cluster_label'] == 1]), 
			len(data[data['cluster_label'] == 2]), 
			len(data[data['cluster_label'] == 3])]

	# Set labels for pie slices

	labels = 'Sports fans (' + str(round(100*nums[0]/sum(nums),1))+'%)', 'Homepage searchers (' + str(round(100*nums[1]/sum(nums),1))+'%)', 'News fans (' + str(round(100*nums[2]/sum(nums),1))+'%)', 'Telly fans (' + str(round(100*nums[3]/sum(nums),1))+'%)'

	# Set colours

	colors = ['green', 'blue', 'orange','red']

	# Set other plot options

	explode = [0,0,0,0]

	plt.pie(nums,labels=labels, colors = colors, explode = explode, shadow=True)

	plt.axis('equal')

	plt.savefig('UserProportions.png')

	plt.show()

	return

#######################################################################

def visits_per_cluster(clust, wkd):

	# Calculates rate of visits per hour to each product for six hours at a time,
	# for users belonging to the group corresonding to clust (one of [0,1,2,3]), 
	# on the weekend if wkd == 1, weekday if wkd == 0

	# read clustered pickle

	clusters = pd.read_pickle('clustered_data.pkl')

	# read cleaned pickle

	data = pd.read_pickle('cleaned_data.pkl')

	# create dictionary from clusters

	clust_dict = clusters['cluster_label'].to_dict()

	# create column with entries given by the cluster label of the user_id

	data['cluster_label'] = data['user_id'].map(clust_dict)

	data.dropna(inplace=True)

	# Restrict to the cluster specified by clust

	data = data[data['cluster_label'] == clust]

	# Dictionary for use later

	user_dict = {0:'sports fans', 1:'homepage searchers', 2: 'news fans', 3:'telly fans'}

	# Time column

	times = data['date_time']

	if int(wkd) == 0:
		# Select data from weekdays if wkd == 0
		data = data[data['date_time'].dt.dayofweek < 5]
		day_string = 'a weekday'
	else:
		# Select data from weekend otherwise
		data = data[data['date_time'].dt.dayofweek > 4]
		day_string = 'the weekend'

	# Select only top 5 content categories
	
	data = data[(data['product'] == 0) |
				(data['product'] == 1) |
				(data['product'] == 5) |
				(data['product'] == 8) |
				(data['product'] == 4) ]

	# map integers in 'product' to string

	data['product'] = data['product'].map({v:k for k,v in product_dict.items()})

	# map integers in 'platform' to string

	data['platform'] = data['platform'].map({v:k for k,v in platform_dict.items()})

	# Count how many views there were per date, product, 
	# and platform, and divide by total number of views
	# of that type of content.

	collated = (data.groupby([times.dt.hour, 'product','platform'])['user_id']
				   .count()
				   )/len(data)*100

	# Set maximum range for plot

	col_max = collated.max()+5

	collated = collated.unstack('platform').fillna(0)
				   
	# Plot a set of 4 graphs covering six hour time slots accross day. 

	for i in range(0,4):

		# Forget previous plots

		plt.clf()
		plt.cla()
		plt.close()

		# Restrict to six hour period

		cl_sec = collated[(collated.index.get_level_values('date_time')<6*(i+1)) &
				 		  (collated.index.get_level_values('date_time')>=6*i)]

		# Plot data

		cl_sec.plot(kind='bar',stacked=True)

		# Plot options

		plt.title('Content views by '+ user_dict[clust] + ' between ' + str(6*i) + ' and '+ str(6*(i+1))+ ' on ' + day_string +'.')

		plt.ylim(0,col_max)

		plt.xlabel('(hour, content)')

		plt.ylabel('Percentage of visits')

		save_str = 'Cluster' + str(clust) + '-'+ str(i) + '-' + str(wkd) + '.png'

		plt.tight_layout()

		# Save figure

		plt.savefig(save_str)

	return

#######################################################################

def produce_all_plots():

	# Calls visits_per_cluster a number of times to get hourly visits
	# to top 5 types of content for the different groups.

	for i in range(0,4):
		for j in range(0,2):
			#
			visits_per_cluster(i,j)

	return

#######################################################################

# Pickling functions ##################################################

def save_obj(obj, name ):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)
