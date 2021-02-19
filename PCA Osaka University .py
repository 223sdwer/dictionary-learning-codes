#!/usr/bin/env python
# coding: utf-8

# In[3]:


#HCM
#!/usr/bin/env python
# -*- coding: utf-8 -*-

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import patch_list2
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager
import patch_test_list
fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")


# In[3]:


N_nor = 5371

nor_data = patch_list2.imgs[6203:11574].reshape(N_nor,32*32)
sc = StandardScaler()
nor_data = sc.fit_transform(nor_data)

cols = 20
rows = 10

N_com = 1024

nor_pca = PCA(n_components = N_com)
x = nor_pca.fit(nor_data)
xd = nor_pca.transform(nor_data)
xe = nor_pca.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(200):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmin=-0.18, vmax=0.152,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[9]:


xd.shape


# In[12]:


x.components_.shape


# In[11]:


xe.shape


# In[20]:


xe


# In[18]:


np.dot(xd,x.components_)


# In[4]:


ev_ratio = nor_pca.explained_variance_ratio_
ev_ratio.cumsum()


# In[7]:


np.where(ev_ratio.cumsum()>=0.4)[0][0]


# In[16]:


height = 32
width = 32
N = 282
data = np.zeros(height*width*N).reshape([N,32,32])



#HCM    
for i in range(282):
    data[i] = patch_list2.imgs[i+5169]
    
data = data.reshape(282,1024) 
sc = StandardScaler()
data = sc.fit_transform(data)
    

    

data2 = data.reshape(N,32*32)
data2.shape


cols = 5
rows = 2

N_com = 282

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(10):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmin=-0.103, vmax=0.098,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[12]:


np.max(x.components_[0:9])


# In[13]:


np.min(x.components_[0:9])


# In[18]:


height = 32
width = 32
N = 282
data = np.zeros(height*width*N).reshape([N,32,32])



#HCM    
for i in range(282):
    data[i] = patch_list2.imgs[i+5169]
    
data = data.reshape(282,1024) 
sc = StandardScaler()
data = sc.fit_transform(data)
    

    

data2 = data.reshape(N,32*32)
data2.shape


cols = 20
rows = 10

N_com = 200

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(200):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmin=-0.18, vmax=0.152,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[45]:


np.min(x.components_[0:50])


# In[46]:


np.max(x.components_[0:50])


# In[20]:


plt.figure()
plt.colorbar()
plt.clim(-0.12,0.12)
plt.show()


# In[21]:


plt.figure()
plt.imshow(x.components_[0].reshape(32,32),cmap="bwr")
plt.colorbar()
plt.clim(0,0.05)
plt.show()


# In[46]:


height = 32
width = 32
N = 282
data = np.zeros(height*width*N).reshape([N,32,32])



#HCM    
for i in range(282):
    data[i] = patch_list2.imgs[i+5169]
    
data = data.reshape(282,1024) 
#sc = StandardScaler()
#for i in range(1024):
    #data[:,i] = sc.fit_transform(data[:,i].reshape(-1,1)).reshape(282,)

    
data = data/255    
data2 = data.reshape(N,32*32)
data2.shape


cols = 10
rows = 2

N_com = 20

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(20):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmin=-0.12, vmax=0.12,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[11]:


np.max(x.components_)


# In[12]:


np.min(x.components_)


# In[47]:


#test全部
height = 32
width = 32
P = 927
test = np.zeros(height*width*P).reshape([P,32,32])


for i in range(927):
    test[i] = patch_test_list.imgs[i]
    
#HCM
N_hcm_t = 73
hcm_test = test[361:434].reshape(N_hcm_t,32*32)
hcm_test = hcm_test/255


#x = pca.fit_transform(data2)
xd = pca.transform(hcm_test)
xe = pca.inverse_transform(xd)

cols = 12
rows = 6

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(72):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),vmin=0,vmax=1.0,cmap = "bwr",interpolation = 'nearest')
   # ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)


# In[7]:


np.max(xe)


# In[8]:


np.min(xe)


# In[22]:


x.components_


# In[31]:


aiu, ote = plt.subplots(ncols=10, nrows=2, figsize=(20,10))

for i in range(19):
    r = i // 10
    c = i % 10
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmin=0,vmax=1.0,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[56]:


x.components_


# In[42]:


x


# In[45]:


cols = 12
rows = 6

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))


for z in range(72):
    a = z // cols
    b = z % cols
    ote[a, b].imshow(hcm_test[z].reshape(32,32),vmin=0,vmax=1.0,cmap = "bwr",interpolation = 'nearest')
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[49]:


gosa = np.zeros(73)
for j in range(73):
    gosa[j] = np.square(xe[j].reshape(32,32) - hcm_test[j].reshape(32,32)).mean(axis=None)
plt.plot(gosa)
plt.title("再構成誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("データの番号",fontdict = {"fontproperties": fontprop})
plt.ylabel("再構成誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[54]:


gosa


# In[56]:


plt.plot(gosa)
plt.title("再構成誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("データの番号",fontdict = {"fontproperties": fontprop})
plt.ylabel("再構成誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[19]:


np.max(hcm_test)


# In[21]:


np.min(hcm_test)


# In[6]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio.cumsum()


# In[7]:


ev_ratio


# In[5]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.title("HCMの累積寄与率",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("累積寄与率",fontdict = {"fontproperties": fontprop})
plt.show()
#pca.explained_variance_ratio_


# In[18]:


np.max()


# In[7]:


plt.figure()
plt.imshow(x.components_[0].reshape(32,32),cmap="bwr")
plt.colorbar()
plt.clim(0,0.05)
plt.show()


# In[8]:


gosa = np.zeros([282,282])
for i in range(282):
    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(282):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)


# In[10]:


gosa2 = np.zeros(282)
for i in range(282):
    gosa2[i] = np.sum(gosa[i])


# In[11]:


gosa2 = gosa2/282


# In[13]:


plt.plot(gosa2)
plt.title("HCMの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[18]:


for i in range(282):
    plt.plot(gosa[:,i])
plt.title("HCMの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[19]:


aiu = np.zeros(72)

for i in range(72):
    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    aiu[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)


# In[20]:


aiu


# In[81]:


cols = 12
rows = 6

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(72):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
   # ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
    


# In[75]:


np.max(xe[0:71])


# In[76]:


np.min(xe[0:71])


# In[72]:


cols = 12
rows = 6

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))


for z in range(72):
    a = z // cols
    b = z % cols
    ote[a, b].imshow(data2[z].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[78]:


np.max(data2[0:71])


# In[79]:


np.min(data2[0:71])


# In[ ]:


ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.show()


# In[23]:


cols = 2
rows = 2

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(2):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
   # ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


for z in range(2,4):
    a = z // cols
    b = z % cols
    k = z-2
    w = k+1
    ote[a, b].imshow(data2[k].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[50]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio


# In[51]:


ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.show()


# In[11]:


#CON

height = 32
width = 32
N = 143
data = np.zeros(height*width*N).reshape([N,32,32])


data = patch_list2.imgs[5451:5594]

    
data = data.reshape(143,1024) 
sc = StandardScaler()
data = sc.fit_transform(data)
    
    

data2 = data.reshape(N,32*32)
data2.shape


    



cols = 5
rows = 2

N_com = 143

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)

Print_com=10

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax=0.123,vmin=-0.125,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


    
    


# In[ ]:





# In[83]:


#CON

height = 32
width = 32
N = 143
data = np.zeros(height*width*N).reshape([N,32,32])


data = patch_list2.imgs[5451:5594]

    
data = data.reshape(143,1024) 
data = data/255
    
    

data2 = data.reshape(N,32*32)
data2.shape


    



cols = 5
rows = 2

N_com = 10

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)

Print_com=10

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax=0.14,vmin=-0.15,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[82]:


#test全部
height = 32
width = 32
P = 927
test = np.zeros(height*width*P).reshape([P,32,32])


for i in range(927):
    test[i] = patch_test_list.imgs[i]
    
#CON
N_con_t = 26
con_test = test[434:460].reshape(N_con_t,32*32)
con_test = con_test/255


#x = pca.fit_transform(data2)
xd = pca.transform(con_test)
xe = pca.inverse_transform(xd)

cols = 13
rows = 2

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(26):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),vmin=0.13,vmax=1.0,cmap = "bwr",interpolation = 'nearest')
   # ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)


# In[77]:


cols = 13
rows = 2

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for z in range(26):
    a = z // cols
    b = z % cols
    #k=z+1
    ote[a, b].imshow(con_test[z].reshape(32,32),vmin=0,vmax=1.0,cmap = "bwr")
    #ote[a, b].set_title("original %d" % k)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[87]:


gosa = np.zeros(26)
for j in range(26):
    gosa[j] = np.square(xe[j].reshape(32,32) - con_test[j].reshape(32,32)).mean(axis=None)
plt.plot(gosa)
plt.title("再構成誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("データの番号",fontdict = {"fontproperties": fontprop})
plt.ylabel("再構成誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[78]:


np.max(xe)


# In[79]:


np.min(xe)


# In[80]:


np.max(con_test)


# In[81]:


np.min(con_test)


# In[84]:


np.max(x.components_)


# In[85]:


np.min(x.components_)


# In[47]:


plt.figure()
plt.imshow(x.components_[0].reshape(32,32),cmap="bwr")
plt.colorbar()
plt.clim(-0.125,0.123)
plt.show()


# In[29]:


np.sum(x.components_[0])


# In[21]:


gosa = np.zeros([143,143])
for i in range(143):
    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(143):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)
        
gosa2 = np.zeros(143)
for i in range(143):
    gosa2[i] = np.sum(gosa[i])
    
gosa2 = gosa2/143


# In[22]:


plt.plot(gosa2)
plt.title("CONの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[23]:


for i in range(143):
    plt.plot(gosa[:,i])
plt.title("CONの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[42]:


gosa = np.zeros(25)
for i in range(25):

    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    gosa[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)
    
    


# In[15]:


cols = 5
rows = 5

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(25):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(xe[i].reshape(32,32),cmap = "bwr")
    #ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)
      


# In[7]:


cols = 5
rows = 5

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for z in range(0,25):
    a = z // cols
    b = z % cols
    #k=z+1
    ote[a, b].imshow(data2[z].reshape(32,32),cmap = "bwr")
    #ote[a, b].set_title("original %d" % k)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[1]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio


# In[18]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.title("CONの累積寄与率",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("累積寄与率",fontdict = {"fontproperties": fontprop})
plt.show()
#pca.explained_variance_ratio_


# In[ ]:


#CON

height = 32
width = 32
N = 143
data = np.zeros(height*width*N).reshape([N,32,32])


data = patch_list2.imgs[5451:5594]

    
data = data.reshape(143,1024) 
sc = StandardScaler()
data = sc.fit_transform(data)
    
    

data2 = data.reshape(N,32*32)
data2.shape


    



cols = 5
rows = 5

N_com = 143

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)

Print_com=25

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[14]:


#RET

N = 210

data = patch_list2.imgs[11574:11784]
    
data = data.reshape(N,1024) 

sc = StandardScaler()
data = sc.fit_transform(data)
data2 = data.reshape(N,32*32)



cols = 5
rows = 2

N_com = 210

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)

Print_com = 10


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax=0.127,vmin=-0.124,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[16]:


#RET

N = 210

data = patch_list2.imgs[11574:11784]
    
data = data.reshape(N,1024) 

data=data/255
data2 = data.reshape(N,32*32)



cols = 5
rows = 2

N_com = 210

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)

Print_com = 10


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax=0.127,vmin=-0.124,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[52]:


np.max(x.components_[0:49])


# In[53]:


np.min(x.components_[0:49])


# In[55]:


plt.figure()
plt.imshow(x.components_[0].reshape(32,32),cmap="bwr")
plt.colorbar()
plt.clim(-0.124,0.127)
plt.show()


# In[25]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.title("RETの累積寄与率",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("累積寄与率",fontdict = {"fontproperties": fontprop})
plt.show()
#pca.explained_variance_ratio_


# In[25]:


gosa = np.zeros([210,210])
for i in range(210):
    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(210):
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)
        
gosa2 = np.zeros(210)
for i in range(210):
    gosa2[i] = np.sum(gosa[i])
    
gosa2 = gosa2/210


# In[28]:


plt.plot(gosa2)
plt.title("RETの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[29]:


for i in range(143):
    plt.plot(gosa[:,i])
plt.title("RETの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[16]:


plt.plot(gosa2)
plt.show


# In[57]:


gosa = np.zeros(70)
for i in range(70):

    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    gosa[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)


# In[39]:


plt.plot(gosa[:,0])
plt.show()


# In[62]:


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
      


for z in range(70,140):
    a = z // cols
    b = z % cols
    k = z-70
    w = k+1
    ote[a, b].imshow(data2[k].reshape(32,32),cmap = cm.Greys_r)
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[37]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio


# In[36]:


ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.show()


# In[17]:


#GGO
N = 609

data = patch_list2.imgs[5594:6203]
    
data = data.reshape(N,1024) 
sc = StandardScaler()
data = sc.fit_transform(data)

data2 = data.reshape(N,32*32)
data2.shape


cols = 5
rows = 2

N_com = 609

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)

Print_com = 10

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax=0.11363,vmin=-0.128,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[18]:


#GGO
N = 609

data = patch_list2.imgs[5594:6203]
    
data = data.reshape(N,1024) 
data = data/255
data2 = data.reshape(N,32*32)
data2.shape


cols = 5
rows = 2

N_com = 609

pca = PCA(n_components = N_com)
x = pca.fit(data2)
xd = pca.transform(data2)
xe = pca.inverse_transform(xd)

Print_com = 10

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),vmax=0.11363,vmin=-0.128,cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[65]:


np.max(x.components_[0:29])


# In[66]:


np.min(x.components_[0:29])


# In[19]:


plt.figure()
plt.imshow(x.components_[0].reshape(32,32),cmap="bwr")
plt.colorbar()
plt.clim(-0.128,0.11363)
plt.show()


# In[59]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])
ev_ratio


# In[22]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.title("GGOの累積寄与率",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("累積寄与率",fontdict = {"fontproperties": fontprop})
plt.show()
#pca.explained_variance_ratio_


# In[32]:


gosa = np.zeros([609,609])
for i in range(609):
    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    for j in range(609):
        
        gosa[i][j] = (np.square(xe[j].reshape(32,32) - data[j].reshape(32,32))).mean(axis=None)
        
gosa2 = np.zeros(609)
for i in range(609):
    gosa2[i] = np.sum(gosa[i])
    
gosa2 = gosa2/609


# In[33]:


plt.plot(gosa2)
plt.title("GGOの最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差の平均",fontdict = {"fontproperties": fontprop})
plt.show()


# In[34]:


for i in range(609):
    plt.plot(gosa[:,i])
plt.title("GGOの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("使用基底の数",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[8]:


plt.plot(gosa2)
plt.show


# In[17]:


data2


# In[60]:


gosa = np.zeros(50)
for i in range(50):

    pca = PCA(n_components = i+1)
    x = pca.fit(data2)
    xd = pca.transform(data2)
    xe = pca.inverse_transform(xd)
    gosa[i] = (np.square(xe[0].reshape(32,32) - data[0])).mean(axis=None)


# In[45]:


plt.plot(gosa[:,0])
plt.show()


# In[65]:


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
      


# In[18]:


cols = 10
rows = 5

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))


for z in range(50):
    a = z // cols
    b = z % cols
    ote[a, b].imshow(data2[z].reshape(32,32),cmap = cm.Greys_r)
    #ote[a, b].set_title("original %d" % w)
    ote[a, b].get_xaxis().set_visible(False)
    ote[a, b].get_yaxis().set_visible(False)


# In[46]:


ev_ratio = pca.explained_variance_ratio_
ev_ratio


# In[47]:


ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.plot(ev_ratio)
plt.show()


# In[35]:


data

