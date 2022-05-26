#!/usr/bin/env python
# coding: utf-8

# In[1]:


x=10
X=7
#print('The value of x',x)
#print('The value of X',X)
z=x+7
z=4
y=z**2


# In[2]:


import numpy as np
np.sqrt(y)
print('SQRT of y is',np.sqrt(y))


# In[11]:


import pandas as pd
df=pd.read_csv('E:\haberman.csv')
df.info() ## to know the data type
df.shape ## To know the data size (n= no. of observations, p=no. of variables)
df.describe() # Summary measures
df.status.mode()
df.status.value_counts()
df.mode()
df.median()
df.corr()


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


# In[14]:


df.boxplot()
plt.show()


# In[15]:


plt.boxplot(df.age)
plt.title('Boxplot for Age of Breast Cancer Patients')
plt.show()


# In[16]:


plt.hist(df.age, bins=10)
plt.xlabel('Age of Patients')
plt.ylabel('Number of Patients')
plt.title('Histogram of Age of Breast Cancer Patients')
plt.show()


# In[17]:


plt.boxplot(df.axil_nodes)
plt.title('Boxplot for Axillary Nodes of Breast Cancer Patients')
plt.show()


# In[18]:


plt.hist(df.axil_nodes, bins=10)
plt.xlabel('Axillary Nodes of Patients')
plt.ylabel('Number of Patients')
plt.title('Histogram of Axillary Nodes of Breast Cancer Patients')
plt.show()


# In[19]:


plt.boxplot(df.operation_year)
plt.title('Boxplot for Operation Year of Breast Cancer Patients')
plt.show()


# In[20]:


plt.boxplot(df.operation_year)
plt.title('Boxplot for Operation Year of Breast Cancer Patients')
plt.show()


# In[21]:


objects = ('Survived', 'Dead')
x_pos = np.arange(len(objects))
status_fre=[225, 81] # get the frequency value from df.status.value_counts()

plt.bar(x_pos, status_fre)
plt.xticks(x_pos, objects)
plt.ylabel('Number of Patients')
plt.title('Survival Status of Breast Cancer Patients')
plt.show()


# In[22]:


status_fre=[225, 81]
plt.pie(status_fre, labels=['Survived', 'Dead'], colors=['yellowgreen', 'lightcoral'],  autopct='%.1f%%')
plt.show()


# In[24]:


pip install stemgraphic


# In[25]:


import stemgraphic
fig, ax = stemgraphic.stem_graphic(df.age)
plt.show()


# In[26]:


#datarowsSeries = [pd.Series([30, 66, 2, 1], index=df.columns ), pd.Series([100, 3, 1, 0], index=df.columns ),]
#df = df.append(datarowsSeries , ignore_index=True)
#fig, ax = stemgraphic.stem_graphic(df.age)
#plt.show()


# In[27]:


plt.scatter(df['age'],df['axil_nodes'], color = 'g')
plt.xlabel('Age')
plt.ylabel('Axil Nodes')
plt.title('Axil_nodes vs Age')
plt.show()


# In[28]:


df['status'] = df['status'].map({1:'survived', 2:'dead'})


# In[29]:


sns.set_style('whitegrid');
sns.FacetGrid(df, hue = 'status', height = 6).map(plt.scatter, 'age', 'axil_nodes').add_legend();
plt.show();


# In[30]:


plt.scatter(df['age'],df['operation_year'], c = 'b')
plt.xlabel('Age')
plt.ylabel('Operation year')
plt.title('Operation year vs Age')
plt.show()


# In[31]:


plt.scatter(df['operation_year'],df['axil_nodes'], color = 'r')
plt.xlabel('Operation Year')
plt.ylabel('Axillary Nodes')
plt.title('Operation Year vs Axillary Nodes')
plt.show()


# In[32]:


sns.boxplot(x='status',y='axil_nodes', data=df)
plt.show()


# In[33]:


sns.boxplot(x='status',y='axil_nodes', data=df)
plt.show()


# In[34]:


sns.boxplot(x='status',y='operation_year', data=df)
plt.show()


# In[35]:


sns.set_style('whitegrid');
sns.pairplot(df, hue = 'status', height = 4)
plt.show()


# In[ ]:




