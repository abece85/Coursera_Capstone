#!/usr/bin/env python
# coding: utf-8

# ## Week 3 Assignment of Applied Data Science Capstone
# 
# ## Task 1: Transform the data on Wiki page into pandas dataframe
# 
# # Import necessary library
# 

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# ## Group the wiki page content by using BeautifulSoup

# In[2]:


url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
wiki_html = requests.get(url).text
soup = BeautifulSoup(wiki_html, 'html.parser')


# ## Convert content of PostalCode HTML table as list of data
# 

# In[3]:


data = []
for tr in soup.tbody.find_all('tr'):
    data.append([ td.get_text().strip() for td in tr.find_all('td')])


# ## Create Dataframe
# 

# In[4]:


df = pd.DataFrame(data, columns=['PostalCode','Borough','Neighborhood'])


# In[5]:


df.head()




# # Drop "None" rows

# In[6]:


df = df.dropna()


# ## Drop any row which contains 'Not assigned' value

# In[7]:


empty = 'Not assigned'
df = df[(df.PostalCode != empty ) & (df.Borough != empty) & (df.Neighborhood != empty)]


# In[8]:


df.head()


# ## Group the dataframe by 'PostalCode' and 'Borough', convert the groupby value as string separated by commas and convert back to a new dataframe

# In[9]:


def neighborhood_list(grouped):    
    return ', '.join(sorted(grouped['Neighborhood'].tolist()))
                    
grp = df.groupby(['PostalCode', 'Borough'])
df2 = grp.apply(neighborhood_list).reset_index(name='Neighborhood')


# In[10]:


df2.head()


# ## Check some data

# In[11]:


df2[df2.Borough == 'East York']


# In[12]:


df2


# ## Save the dataframe to csv for future tasks

# In[13]:


df2.to_csv('task1.csv', index=False)


# ## Print the shape of new dataframe

# In[14]:


df2.shape


# In[ ]:




