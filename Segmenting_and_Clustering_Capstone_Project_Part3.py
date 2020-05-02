#!/usr/bin/env python
# coding: utf-8

# ## Task 3: Explore and cluster the neighborhoods in Torondo¶
# 
# ## Import required library
# 

# In[3]:


import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library


# In[4]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

print('Libraries imported.')


# ## Read dataframe created from previous task

# In[6]:


import pandas as pd
df = pd.read_csv('task2.csv')
df.head()


# ## As per assignment, only work with boroughs that contain the word Toronto

# In[7]:


df3 = df[ df.Borough.str.contains('Toronto') ]


# In[7]:


df3


# In[8]:


#!pip install geocoder
import geocoder


# In[9]:


def get_latlng(postal_code):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, Toronto, Ontario'.format(postal_code))
        lat_lng_coords = g.latlng
    return lat_lng_coords
    
get_latlng('M4G')


# In[10]:


toronto_coords = get_latlng('')
toronto_coords


# ## Let's explore the first neighborhood in our dataframe. Replicate same analysis as New York City data
# 
# ## Get the neighborhood's name.
# 

# In[11]:


toronto_data = pd.DataFrame(df3)


# In[12]:


toronto_data = toronto_data.reset_index().drop('index', axis=1)


# In[13]:


toronto_data.head()


# In[14]:


toronto_data.loc[0, 'Neighborhood']


# ## Get the neighborhood's latitude and longitude values.

# In[15]:


neighborhood_latitude = toronto_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = toronto_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = toronto_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# ## Now, let's get the top 100 venues that are in East Toronto within a radius of 500 meters.¶
# 
# ## First, let's create the GET request URL. Name your URL url.
# 

# In[16]:


# type your answer here
VERSION = '20180604'
CLIENT_ID = 'DGOXJTOQUANFROSWHHHCQIVJ2NIOKEFISEZSQIJILV1VIIIC'
CLIENT_SECRET = 'FBZQJQ3DEKB1H0X3ATTLR1UHFMT0T2XUTAHQTKMQ2KX15X2C'
latitude = neighborhood_latitude
longitude = neighborhood_longitude
radius = 500
LIMIT = 100

url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION,radius, LIMIT)


# ## Send the GET request and examine the resutls

# In[17]:


results = requests.get(url).json()
results


# ## From the Foursquare lab in the previous module, we know that all the information is in the items key. Before we proceed, let's borrow the get_category_type function from the Foursquare lab.
# 

# In[18]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# ## Now we are ready to clean the json and structure it into a pandas dataframe.

# In[19]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# ### And how many venues were returned by Foursquare?
# 

# In[20]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## Explore Neighborhoods in Toronto
# 
# ### Let's create a function to repeat the same process to all the neighborhoods in Toronto
# 

# In[21]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# ### Now write the code to run the above function on each neighborhood and create a new dataframe called toronto_venues.

# In[22]:


toronto_venues = getNearbyVenues(toronto_data['Neighborhood'], toronto_data['Latitude'], toronto_data['Longitude'])


# In[23]:


toronto_venues.head()


# ### Let's check the size of the resulting dataframe

# In[24]:


toronto_venues.shape


# In[25]:


print(toronto_venues.shape)
toronto_venues.head()


# ### Let's check how many venues were returned for each neighborhood

# In[26]:


toronto_venues.groupby('Neighborhood').count()


# ### Let's find out how many unique categories can be curated from all the returned venues

# In[27]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ## Analyze Each Neighborhood

# In[28]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# ### And let's examine the new dataframe size.

# In[29]:


toronto_onehot.shape


# ### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[30]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# ### Let's confirm the new size

# In[31]:


toronto_grouped.shape


# ### Let's print each neighborhood along with the top 5 most common venues

# In[32]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ## Let's put that into a pandas dataframe¶
# 
# ## First, let's write a function to sort the venues in descending order.
# 

# In[33]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# ### Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[34]:


import numpy as np
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# ## Cluster Neighborhoods¶
# 
# ### Run k-means to cluster the neighborhood into 5 clusters.
# 

# In[35]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# ### Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[36]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns


# In[37]:


toronto_merged = toronto_merged.dropna()


# In[38]:


toronto_merged.head() 


# ### Finally, let's visualize the resulting clusters

# In[39]:


# create map
map_clusters = folium.Map(location=[toronto_coords[0], toronto_coords[1]], zoom_start=11)
# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters


# In[ ]:




