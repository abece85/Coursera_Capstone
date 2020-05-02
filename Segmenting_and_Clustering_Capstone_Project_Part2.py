#!/usr/bin/env python
# coding: utf-8

# ## Week 3 Assignment of Applied Data Science Capstone¶
# 
# 
# ## Task 2: Add the latitude and longitude coordinates to the dataframe

# In[1]:


import pandas as pd


# ## Read the dataframe created in task 1

# In[2]:


df = pd.read_csv('task1.csv')


# In[3]:


df.head()


# ## Install geocoder library

# In[4]:


get_ipython().system('pip install geocoder')
import geocoder


# ## Use Arcgis to get the coordinates

# In[5]:


def get_latlng(postal_code):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, Toronto, Ontario'.format(postal_code))
        lat_lng_coords = g.latlng
    return lat_lng_coords
    
get_latlng('M4G')


# ## Retrieve all Postal Code coordinates

# In[10]:


postal_codes = df['PostalCode']    
coords = [ get_latlng(postal_code) for postal_code in postal_codes.tolist() ]


# ## Add Latitude and Longitude columns

# In[11]:


df_coords = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])
df['Latitude'] = df_coords['Latitude']
df['Longitude'] = df_coords['Longitude']


# In[12]:


df.head()


# In[13]:


df[df.PostalCode == 'M5G']


# ## Save the dataframe for future

# In[14]:


df.to_csv('task2.csv', index=False)


# In[15]:


df


# In[ ]:




