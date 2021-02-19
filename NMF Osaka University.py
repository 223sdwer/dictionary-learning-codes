#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image
import cv2
from sklearn.decomposition import NMF
import matplotlib.cm as cm
import patch_list2
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager
fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")


# In[16]:


height = 32
width = 32
N = 282
data = np.zeros(height*width*N).reshape([N,32,32])



#HCM    
for i in range(282):
    data[i] = patch_list2.imgs[i+5169]
    
#data = data.reshape(282,1024) 
#sc = StandardScaler()
#data = sc.fit_transform(data)
    
data = data/255
    

data2 = data.reshape(N,32*32)
data2.shape


cols = 10
rows = 5

N_com = 50

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(50):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax = 1.35,cmap = "gray")
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[23]:


N_com = 1

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)
plt.imshow(x.components_[0].reshape(32,32),cmap = "gray")


# In[15]:


x.components_


# In[17]:


xd[0]


# In[22]:


cols = 12
rows = 6

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(72):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
   # ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)


# In[14]:


cols = 9
rows = 5

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))


for z in range(45):
    a = z // cols
    b = z % cols
    ote[a, b].imshow(data2[z+47].reshape(32,32),vmax = 1.0,cmap = cm.Greys_r,interpolation = 'nearest')
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[13]:


np.max(data2[47:92])


# In[15]:


np.max(x.components_[0:49])


# In[17]:


np.min(x.components_[0:49])


# In[44]:


plt.figure()
plt.imshow(x.components_[0].reshape(32,32),cmap=cm.Greys_r)
plt.colorbar()
plt.clim(0,0.4)
plt.show()


# In[3]:


gosa = np.zeros([50,282])
for i in range(50):
    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(282):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)


# In[4]:


gosa2 = np.zeros(50)
for i in range(50):
    gosa2[i] = np.sum(gosa[i])


# In[5]:


gosa2 = gosa2/282


# In[9]:


plt.plot(gosa2)
plt.title("HCMの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[34]:


for i in range(282):
    plt.plot(gosa[:,i])
plt.title("HCMの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[48]:


np.min(gosa[0,:])


# In[42]:


labela = np.array(gosa[0,:] <= 0.0054,dtype=float)
labela


# In[44]:


np.where(labela==1)[0]


# In[11]:


label = np.array(gosa[0,:] > 0.02,dtype=float)
label


# In[29]:


np.where(label==0)[0]


# In[13]:


for i in np.where(label==0)[0]:
    print(i)
    plt.imshow(data2[i].reshape(32,32),cmap=cm.Greys_r)
    plt.show()


# In[25]:


gosa3 = np.zeros([282,282])
for i in range(282):
    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd= pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(282):
        gosa3[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)


# In[26]:


gosa4 = np.zeros(282)
for i in range(282):
    gosa4[i] = np.sum(gosa3[i])


# In[27]:


gosa4 = gosa4/282
plt.plot(gosa4)
plt.title("HCMの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[30]:


for i in range(282):
    plt.plot(gosa3[:,i])
plt.title("HCMの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[14]:


aiu, ote = plt.subplots(ncols=6, nrows=4, figsize=(20,10))

cols = 6
rows = 4

for i in range(12):
    r = i // cols
    c = i % cols
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = cm.Greys_r)
    ote[r, c].set_title("dimention reduction %d" % i)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


for z in range(12,24):
    a = z // cols
    b = z % cols
    k = z-12
    ote[a, b].imshow(data2[k].reshape(32,32),cmap = cm.Greys_r)
    ote[a, b].set_title("original %d" % k)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[6]:


#CON

height = 32
width = 32
N = 143
data = np.zeros(height*width*N).reshape([N,32,32])


data = patch_list2.imgs[5451:5594]

    
data = data.reshape(143,1024) 
#sc = StandardScaler()
#data = sc.fit_transform(data)
data = data/255
    

data2 = data.reshape(N,32*32)
data2.shape


    



cols = 5
rows = 5

N_com = 143

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)

Print_com=25

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax = 1.6,cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[54]:


np.max(x.components_[0:25])


# In[65]:


plt.figure()
plt.imshow(x.components_[0].reshape(32,32),cmap=cm.Greys_r)
plt.colorbar()
plt.clim(0,0.2)
plt.show()


# In[20]:


gosa = np.zeros([50,143])
for i in range(50):
    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(143):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)


# In[21]:


gosa2 = np.zeros(50)
for i in range(50):
    gosa2[i] = np.sum(gosa[i])


# In[22]:


gosa2 = gosa2/143


# In[23]:


plt.plot(gosa2)
plt.title("CONの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[24]:


for i in range(143):
    plt.plot(gosa[:,i])
plt.title("CONの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[13]:


xd2 = nmf2.transform(data4)
xe2 = nmf2.inverse_transform(xd2)

aiu, ote = plt.subplots(ncols=6, nrows=2, figsize=(20,10))

cols = 6
rows = 2

for i in range(6):
    r = i // cols
    c = i % cols
    ote[r, c].imshow(xe2[i].reshape(32,32),cmap = cm.Greys_r)
    ote[r, c].set_title("dimention reduction %d" % i)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


for z in range(6,12):
    a = z // cols
    b = z % cols
    k = z-6
    ote[a, b].imshow(data3[k].reshape(32,32),cmap = cm.Greys_r)
    ote[a, b].set_title("original %d" % k)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[7]:


#RET

N = 210

data = patch_list2.imgs[11574:11784]
    
data = data.reshape(N,1024) 

#sc = StandardScaler()
#data = sc.fit_transform(data)

data = data/255
data2 = data.reshape(N,32*32)



cols = 10
rows = 5

N_com = 210

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)

Print_com = 50


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[27]:


gosa = np.zeros([50,210])
for i in range(50):
    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(210):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)


# In[28]:


gosa2 = np.zeros(50)
for i in range(50):
    gosa2[i] = np.sum(gosa[i])
gosa2 = gosa2/210


# In[29]:


plt.plot(gosa2)
plt.title("RETの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[30]:


for i in range(210):
    plt.plot(gosa[:,i])
plt.title("RETの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[59]:


#GGO
N = 609

data = patch_list2.imgs[5594:6203]
    
data = data.reshape(N,1024) 
#sc = StandardScaler()
#data = sc.fit_transform(data)

data = data/255
data2 = data.reshape(N,32*32)
data2.shape


cols = 10
rows = 5

N_com = 609

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)

Print_com = 50

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax = 0.4,cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[64]:


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax = 0.2,cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[32]:


gosa = np.zeros([50,609])
for i in range(50):
    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(609):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)


# In[33]:


gosa2 = np.zeros(50)
for i in range(50):
    gosa2[i] = np.sum(gosa[i])
gosa2 = gosa2/609


# In[34]:


plt.plot(gosa2)
plt.title("GGOの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[35]:


for i in range(609):
    plt.plot(gosa[:,i])
plt.title("GGOの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[33]:


xd3 = nmf3.transform(data6)
xe3 = nmf3.inverse_transform(xd3)

aiu, ote = plt.subplots(ncols=14, nrows=4, figsize=(40,20))

cols = 14
rows = 4

for v in range(28):
    r = v // cols
    c = v % cols
    ote[r, c].imshow(xe3[v].reshape(32,32),cmap = cm.Greys_r)
    ote[r, c].set_title("dimention reduction %d" % v)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      
for m in range(28,56):
    a = m // cols
    b = m % cols
    k = m-28
    ote[a, b].imshow(data6[k].reshape(32,32),cmap = cm.Greys_r)
    ote[a, b].set_title("original %d" % k)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[21]:


plt.plot(xd3[:,5])
plt.show


# In[8]:


#HCM

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image
import cv2
from sklearn.decomposition import NMF
import matplotlib.cm as cm
import patch_list

height = 32
width = 32
N = 72
data = np.zeros(height*width*N).reshape([N,32,32])



for i in range(N):
    data[i] = patch_list.imgs[i+359]
    
data = data/255.0
data2 = data.reshape(N,32*32)
data2.shape


cols = 12
rows = 6

N_com = 72

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(N_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[3]:


gosa = np.zeros([72,72])
for i in range(72):
    nmf = NMF(n_components = i+1)
    x = nmf.fit(data2)
    xd = nmf.transform(data2)
    xe = nmf.inverse_transform(xd)
    for j in range(72):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j])).mean(axis=None)
        
gosa2 = np.zeros(72)
for i in range(72):
    gosa2[i] = np.sum(gosa[i])
    
gosa2 = gosa2/72


# In[4]:


plt.plot(gosa2)
plt.show


# In[44]:


gosa = np.zeros(72)
for i in range(72):

    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    gosa[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)

plt.plot(gosa)
plt.show()


# In[46]:


plt.plot(gosa)
plt.show()


# In[4]:


cols = 12
rows = 12

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(72):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    #ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


for z in range(72,144):
    a = z // cols
    b = z % cols
    k = z-72
    w = k+1
    ote[a, b].imshow(data2[k].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image
import cv2
from sklearn.decomposition import NMF
import matplotlib.cm as cm
import patch_list

#CON

height = 32
width = 32
N = 25
data = np.zeros(height*width*N).reshape([N,32,32])

patch_list.imgs.shape


for i in range(N):
    data[i] = patch_list.imgs[i+431]
    
data = data/255.0
data2 = data.reshape(N,32*32)
data2.shape


cols = 5
rows = 5

N_com = 25

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(N_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = cm.Greys_r)
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[7]:


gosa = np.zeros([25,25])
for i in range(25):
    nmf = NMF(n_components = i+1)
    x = nmf.fit(data2)
    xd = nmf.transform(data2)
    xe = nmf.inverse_transform(xd)
    for j in range(25):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j])).mean(axis=None)
        
gosa2 = np.zeros(25)
for i in range(25):
    gosa2[i] = np.sum(gosa[i])
    
gosa2 = gosa2/25


# In[7]:


gosa = np.zeros(25)
for i in range(25):

    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    gosa[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)


# In[8]:


plt.plot(gosa2)
plt.show()


# In[30]:


cols = 5
rows = 5

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(25):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = cm.Greys_r)
    #ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


# In[23]:


cols = 5
rows = 5

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))


for z in range(25):
    a = z // cols
    b = z % cols
    ote[a, b].imshow(data2[z].reshape(32,32),cmap = cm.Greys_r)
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image
import cv2
from sklearn.decomposition import NMF
import matplotlib.cm as cm
import patch_list

#RET

height = 32
width = 32
N = 70
data = np.zeros(height*width*N).reshape([N,32,32])

patch_list.imgs.shape


for i in range(N):
    data[i] = patch_list.imgs[i+866]
    
data = data/255.0
data2 = data.reshape(N,32*32)
data2.shape


cols = 10
rows = 7

N_com = 70

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(N_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = cm.Greys_r)
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[18]:


gosa = np.zeros([70,70])
for i in range(70):
    nmf = NMF(n_components = i+1)
    x = nmf.fit(data2)
    xd = nmf.transform(data2)
    xe = nmf.inverse_transform(xd)
    for j in range(70):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j])).mean(axis=None)
        
gosa2 = np.zeros(70)
for i in range(70):
    gosa2[i] = np.sum(gosa[i])
    
gosa2 = gosa2/70


# In[19]:


plt.plot(gosa2)
plt.show


# In[10]:


gosa = np.zeros(70)
for i in range(70):

    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    gosa[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)

plt.plot(gosa)
plt.show()


# In[10]:


cols = 14
rows = 10

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(70):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = cm.Greys_r)
    #ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


# In[7]:


cols = 7
rows = 10

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for z in range(70):
    a = z // cols
    b = z % cols
    ote[a, b].imshow(data2[z].reshape(32,32),cmap = cm.Greys_r)
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image
import cv2
from sklearn.decomposition import NMF
import matplotlib.cm as cm
import patch_list

#GGO

height = 32
width = 32
N = 50
data = np.zeros(height*width*N).reshape([N,32,32])

patch_list.imgs.shape


for i in range(N):
    data[i] = patch_list.imgs[i+457]
    
data = data/255.0
data2 = data.reshape(N,32*32)
data2.shape


cols = 10
rows = 5

N_com = 50

nmf = NMF(n_components = N_com)
x = nmf.fit(data2)
xd = nmf.transform(data2)
xe = nmf.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(N_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[15]:


gosa = np.zeros([50,50])
for i in range(50):
    nmf = NMF(n_components = i+1)
    x = nmf.fit(data2)
    xd = nmf.transform(data2)
    xe = nmf.inverse_transform(xd)
    for j in range(50):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j])).mean(axis=None)
        
gosa2 = np.zeros(50)
for i in range(50):
    gosa2[i] = np.sum(gosa[i])
    
gosa2 = gosa2/50


# In[16]:


plt.plot(gosa2)
plt.show


# In[15]:


gosa = np.zeros(50)
for i in range(50):

    pca = NMF(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    gosa[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)

plt.plot(gosa)
plt.show()


# In[12]:


cols = 10
rows = 10

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(50):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = cm.Greys_r)
    #ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


for z in range(50,100):
    a = z // cols
    b = z % cols
    k = z-50
    w = k+1
    ote[a, b].imshow(data2[k].reshape(32,32),cmap = cm.Greys_r)
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)

