#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px


# In[3]:


import plotly.graph_objects as go


# In[4]:


pd.set_option('display.max_columns',None)


# In[5]:


ac = pd.read_csv('US_Accidents_Dec19.csv')


# In[6]:


ac.head()


# In[7]:


ac.describe()


# In[8]:


#preprocessing data

#drop unnecessary columns
at=ac.drop(['End_Lat','End_Lng','Description','Number','Street','Airport_Code','ID','Timezone','Weather_Timestamp','Precipitation(in)','Wind_Direction'],axis=1)

#fill na values with corresponding mean value.
at['Wind_Chill(F)'].fillna(51.32685,inplace=True)
at['Temperature(F)'].fillna(62.35120,inplace=True)
at['Humidity(%)'].fillna(65.40542,inplace=True)
at['Pressure(in)'].fillna(29.83190,inplace=True)
at['Visibility(mi)'].fillna(9.150770,inplace=True)
at['Wind_Speed(mph)'].fillna(8.298064,inplace=True)
at['TMC'].fillna(207.8316,inplace=True)
at['Zipcode'].fillna(0,inplace=True)

#grouping weather conditions
at['Weather_Condition'].fillna('Clear',inplace=True)
at.loc[at['Weather_Condition'].str.contains('snow', case=False), 'Weather_Condition'] = 'Snow'
at.loc[at['Weather_Condition'].str.contains('Rain', case=False), 'Weather_Condition'] = 'Rain'
at.loc[at['Weather_Condition'].str.contains('Drizzle', case=False), 'Weather_Condition'] = 'Drizzle'
at.loc[at['Weather_Condition'].str.contains('Cloudy', case=False), 'Weather_Condition'] = 'Cloudy'
at.loc[at['Weather_Condition'].str.contains('Fair', case=False), 'Weather_Condition'] = 'Fair'
at.loc[at['Weather_Condition'].str.contains('Fog', case=False), 'Weather_Condition'] = 'Fog'
at.loc[at['Weather_Condition'].str.contains('Ice Pellets', case=False), 'Weather_Condition'] = 'Ice Pellets'
at.loc[at['Weather_Condition'].str.contains('Sleet', case=False), 'Weather_Condition'] = 'Sleet'
at.loc[at['Weather_Condition'].str.contains('Smoke', case=False), 'Weather_Condition'] = 'Smoke'
at.loc[at['Weather_Condition'].str.contains('T-Storm', case=False), 'Weather_Condition'] = 'Thunderstorms '
at.loc[at['Weather_Condition'].str.contains('Thunderstorms ', case=False), 'Weather_Condition'] = 'Thunderstorms '
at.loc[at['Weather_Condition'].str.contains('Sand', case=False), 'Weather_Condition'] = 'Sand'
at.loc[at['Weather_Condition'].str.contains('Cloud', case=False), 'Weather_Condition'] = 'Cloudy'
at.loc[at['Weather_Condition'].str.contains('Windy', case=False), 'Weather_Condition'] = 'Windy'
at.loc[at['Weather_Condition'].str.contains('Squalls', case=False), 'Weather_Condition'] = 'Squalls'
at.loc[at['Weather_Condition'].str.contains('Thunder', case=False), 'Weather_Condition'] = 'Thunder'
at.loc[at['Weather_Condition'].str.contains('Dust', case=False), 'Weather_Condition'] = 'Dust'
at.loc[at['Weather_Condition'].str.contains('N/A Precipitation', case=False), 'Weather_Condition'] = 'Clear'

#creating a column of time taken to clean up accident scene
at['Start_Time'] = pd.to_datetime(at['Start_Time'])
at['End_Time'] = pd.to_datetime(at['End_Time'])
at['Time_taken'] = (at['Start_Time'] - at['End_Time']).abs().dt.seconds
at.loc[(at['Time_taken'] > 0), 'Time_taken(min)'] = (at['Time_taken'] / 60)
at['Time_taken(min)'] = at['Time_taken(min)'].round()


# In[9]:


#graphing new columns NA values in each categories.
plt.figure(figsize=(10,8))
sns.heatmap(at.isna(),yticklabels=False,cbar=False,cmap='magma')


# In[10]:


at.head()


# In[11]:


#counts of accident severity level
plt.figure(figsize=(10,8))
sns.countplot(at['Severity'])


# In[152]:


#groupby accident scene time taken(min) and count of it
tttmean = at.groupby(pd.qcut(at['Time_taken(min)'],q=100,duplicates='drop')).count()
tttmean.index.name = 'Timetaken(min)'
tttmean=tttmean.reset_index()
plt.figure(figsize=(12,10))
sns.barplot(x=tttmean['Timetaken(min)'],y=tttmean['Source'])
plt.xticks(rotation=45)
plt.ylabel('Counts')


# In[13]:


#Time taken by severity level.
sevmean = at.groupby(at['Severity'])['Time_taken(min)'].mean()
plt.figure(figsize=(10,8))
sns.barplot(x=sevmean.index,y=sevmean.values)
plt.ylabel('Time taken to clean up accident scene(min)')


# In[14]:


#accidents count by weather condition.
wtct = at.groupby(at['Weather_Condition']).count()
wtct=wtct.sort_values('Source', ascending=False)
plt.figure(figsize=(12,8))
sns.barplot(x=wtct['Source'].index,y=wtct['Source'].values)
plt.xticks(rotation=90)
plt.show()


# In[15]:


#Average severity depends on weather condition
wtmean = at.groupby(at['Weather_Condition'])['Severity'].mean()
plt.figure(figsize=(10,8))
sns.barplot(x=wtmean.index,y=wtmean.values,palette='viridis')
plt.ylabel('Average severity depends on weather condition')
plt.xticks(rotation=90)
plt.show()


# In[16]:


#Accident counts by States
ctct = at.groupby(at['State']).count()
ctct=ctct.sort_values('Source', ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x=ctct['Source'].index,y=ctct['Source'].values,palette='magma')
plt.ylabel('Accident count by City')
plt.xticks(rotation=90)
plt.show()


# In[17]:


#Map accident counts by zipcode
zt = at.groupby(at['Zipcode']).count().reset_index()
fig = px.choropleth_mapbox(zt, geojson=counties, locations=zt['Zipcode'], color=zt['Source'],
                           color_continuous_scale="Viridis",
                           range_color=(0, 200),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'accident count by zipcode'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[18]:


#count true/false on each boolean columns
a1=at['Amenity'].value_counts().reset_index()
a2=at['Bump'].value_counts().reset_index()
a3=at['Crossing'].value_counts().reset_index()
a4=at['Give_Way'].value_counts().reset_index()
a5=at['Junction'].value_counts().reset_index()
a6=at['No_Exit'].value_counts().reset_index()
a7=at['Railway'].value_counts().reset_index()
a8=at['Roundabout'].value_counts().reset_index()
a9=at['Station'].value_counts().reset_index()
a10=at['Stop'].value_counts().reset_index()
a11=at['Traffic_Calming'].value_counts().reset_index()
a12=at['Traffic_Signal'].value_counts().reset_index()
a13=at['Turning_Loop'].value_counts().reset_index()
tfdata=pd.merge(a1,a2)
tfdata=pd.merge(tfdata,a3)
tfdata=pd.merge(tfdata,a4)
tfdata=pd.merge(tfdata,a5)
tfdata=pd.merge(tfdata,a6)
tfdata=pd.merge(tfdata,a7)
tfdata=pd.merge(tfdata,a8)
tfdata=pd.merge(tfdata,a9)
tfdata=pd.merge(tfdata,a10)
tfdata=pd.merge(tfdata,a11)
tfdata=pd.merge(tfdata,a12)


# In[19]:


#plot true false count on each category.
tfdata.plot.barh(figsize=(15,10),width=1)
plt.yticks([1,0],['True','False'])
plt.title('Accident count by accident scene surrounding',fontsize=15)
plt.show()


# In[20]:


#count day/night on each columns
b1=at['Sunrise_Sunset'].value_counts().reset_index()
b2=at['Civil_Twilight'].value_counts().reset_index()
b3=at['Nautical_Twilight'].value_counts().reset_index()
b4=at['Astronomical_Twilight'].value_counts().reset_index()
tfdata2=pd.merge(b1,b2)
tfdata2=pd.merge(tfdata2,b3)
tfdata2=pd.merge(tfdata2,b4)


# In[21]:


#plot day night count on each category.
tfdata2.plot.barh(figsize=(15,10),width=1)
plt.yticks([1,0],['Night','Day'])
plt.title('Accident count by day/night',fontsize=15)
plt.show()


# In[153]:


#plot average severity level depends on wind speed
at['Wind_Speed(mph)'] =at['Wind_Speed(mph)'].round()
wtt=at.groupby(pd.qcut(at['Wind_Speed(mph)'],q=20,duplicates='drop'))['Severity'].mean().reset_index()
plt.figure(figsize=(12,8))
sns.barplot(x=wtt['Wind_Speed(mph)'],y=wtt['Severity'])
plt.title('Average severity level by wind speed')
plt.ylabel('Average severity level')
plt.xlabel('Wind speed(mph)')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




