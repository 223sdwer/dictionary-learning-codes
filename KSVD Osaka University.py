#!/usr/bin/env python
# coding: utf-8

# In[1]:


#HCM
#!/usr/bin/env python
# -*- coding: utf-8 -*-

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import cv2
from PIL import Image
import matplotlib.cm as cm
import KSVD
import patch_list2
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[2]:


import matplotlib.font_manager
fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")


# In[3]:


A_1D = np.zeros((32, 11))
for k in range(11):
    for i in range(32):
        A_1D[i, k] = np.cos(i * k * np.pi / 11.)
    if k != 0:
        A_1D[:, k] -= A_1D[:, k].mean()

plt.figure()
plt.imshow(A_1D, cmap="gray", interpolation="Nearest")
plt.show()

A_2D = np.kron(A_1D, A_1D) #初期辞書


# In[4]:


#HCM

height = 32
width = 32
N = 282
data = np.zeros(height*width*N).reshape([N,32,32])


for i in range(N):
    data[i] = patch_list2.imgs[i+5169]
    
sc = StandardScaler()
data = data.reshape(282,1024) 
data = sc.fit_transform(data)
    

data2 = data.reshape(N,32*32).swapaxes(0, 1)


sig = 0
k0 = 1
N_com = 121


# In[20]:


np.max(data)


# In[5]:


ksvd_hcm, log_ksvd_hcm, X_hcm = KSVD.KSVD(data2, sig, N_com, k0, n_iter=1, initial_dictionary=A_2D.copy())


# In[6]:


X_hcm.shape


# In[7]:


ksvd_hcm.shape


# In[8]:


ksvd2_hcm = ksvd_hcm.swapaxes(0, 1)
recon_hcm = np.dot(ksvd_hcm,X_hcm)
recon2_hcm = recon_hcm.swapaxes(0, 1)
ksvd_hcm.shape

cols = 11
rows = 11


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(121):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(ksvd2_hcm[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[9]:


#再構成
cols = 10
rows = 10

aiu, ote = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(100):
    r = i // cols
    c = i % cols
    p = i+1
    ote[r, c].imshow(recon2_hcm[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    #ote[r, c].set_title("dimention reduction %d" % p)
    ote[r, c].get_xaxis().set_visible(False)
    ote[r, c].get_yaxis().set_visible(False)


# In[23]:


rew = np.zeros(height*width*N).reshape([N,32,32])

for i in range(282):
    rew[i] = patch_list2.imgs[i+5169]
rew = rew.reshape(282,32*32)
rew = sc.fit_transform(rew)


# In[27]:


#元データ
cols = 10
rows = 10

abc, yui = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in range(100):
    r = i // cols
    c = i % cols
    p = i+1
    yui[r, c].imshow(rew[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    #ote[r, c].set_title("dimention reduction %d" % p)
    yui[r, c].get_xaxis().set_visible(False)
    yui[r, c].get_yaxis().set_visible(False)


# In[85]:


hcm_gosa_desu = np.zeros(282)
for i in range(282):
    hcm_gosa_desu[i] = (np.square(rew[i].reshape(32,32) - recon2_hcm[i].reshape(32,32))).mean(axis=None)


# In[94]:


plt.plot(hcm_gosa_desu)
plt.title("HCMの最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.xlabel("データの番号",fontdict = {"fontproperties": fontprop})
plt.ylabel("最小二乗誤差",fontdict = {"fontproperties": fontprop})
plt.show()


# In[13]:


tmp = np.array(X_hcm != 0,dtype=float)
tmp.shape
tmp


# In[14]:


a = np.zeros(121)
b = np.zeros(121)

for i in range(121):
    a[i] = np.sum(tmp[i])
    b[i] = np.sum(tmp[i])


# In[15]:


np.where(b>10)[0].shape


# In[82]:


cols = 8
rows = 5
j = 0

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))

for i in np.where(a>1)[0]:
    r = j // cols
    c = j % cols
    axes[r, c].imshow(ksvd2_hcm[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)
    j = j+1


# In[16]:


for i in np.where(a>1)[0]:
    print(i)
    plt.imshow(ksvd2[i].reshape(32,32),cmap=cm.Greys_r)
    plt.show()


# In[5]:


hcm_gosa = np.zeros([121,282])
hcm_data = data2.swapaxes(0, 1)
hcm_data = hcm_data.reshape(282,32,32)

for i in range(121):
    a,b,c = KSVD.KSVD(data2, sig, N_com, i, n_iter=50,  initial_dictionary=A_2D.copy())
    d = np.dot(a,c)
    d = d.swapaxes(0, 1)
    for j in range(282):
        hcm_gosa[i][j] = (np.square(d[j].reshape(32,32) - hcm_data[j])).mean(axis=None)


# In[6]:


hcm_gosa


# In[40]:


data2.shape


# In[52]:


hcm_gosa_1 = np.zeros([10,282])
hcm_data_1 = data2.swapaxes(0, 1)
hcm_data_1 = hcm_data_1.reshape(282,32,32)


a,b,c = KSVD.KSVD(data2, sig, N_com, 20, n_iter=50,  initial_dictionary=A_2D.copy())
d = np.dot(a,c)
d = d.swapaxes(0, 1)
hcm_gosa_1[i][j] = (np.square(d[1].reshape(32,32) - hcm_data[1])).mean(axis=None)


# In[7]:


np.max(hcm_gosa)


# In[12]:


hcm_gosa2 = np.zeros(59)
for i in range(59):
    hcm_gosa2[i] = np.sum(hcm_gosa[i])
    
hcm_gosa2 = hcm_gosa2/282


# In[49]:


hcm_gosa2.shape


# In[13]:


hcm_gosa2


# In[18]:


plt.plot(hcm_gosa2)
plt.show()


# In[55]:


plt.plot(gosa2)
plt.xlim(1, 4)
plt.ylim(0, 0.000000000000000000005)
plt.show()


# In[11]:


hcmg = gosa2
hcmg#4枚目で大きく変化


# In[23]:


plt.plot(gosa)
plt.show()


# In[15]:





# In[35]:


X.shape


# In[17]:


ksvd.shape


# In[45]:


tmp = np.copy(X)


# In[48]:


tmp[1]


# In[47]:


tmp = np.array(tmp != 0,dtype=float)
tmp.shape
tmp[1]


# In[30]:


np.sum(tmp[1])


# In[49]:


plt.imshow(tmp[0:72])


# In[31]:


a = np.zeros(121)

for i in range(121):
    a[i] = np.sum(tmp[i])

a


# In[50]:


np.sum(tmp,axis=1)


# In[51]:


np.where(a>1)[0]
for i in np.where(a>1)[0]:
    print(i)
    plt.imshow(ksvd2[i].reshape(32,32))
    plt.show()


# In[ ]:


np.where(a>1)[0]
for i in np.where(a>1)[0]:
    print(i)
    plt.imshow(A_2D[:,i].reshape(32,32))
    plt.show()


# In[17]:


#CON

height = 32
width = 32
N = 143
con_data = np.zeros(height*width*N).reshape([N,32,32])


for i in range(N):
    con_data[i] = patch_list2.imgs[i+5451]
    
con_data = con_data.reshape(143,1024) 
sc = StandardScaler()
con_data = sc.fit_transform(con_data)
con_data2 = con_data.reshape(N,32*32).swapaxes(0, 1)

sig = 0
k0 = 4
N_com = 121


# In[18]:


con_ksvd, con_log_ksvd, con_X = KSVD.KSVD(con_data2, sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[22]:


con_ksvd2 = con_ksvd.swapaxes(0, 1)
con_recon = np.dot(con_ksvd,con_X)
con_recon2 = con_recon.swapaxes(0, 1)

cols = 11
rows = 11


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(121):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(con_ksvd2[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[19]:


X


# In[20]:


tmp = np.array(X != 0,dtype=float)
tmp.shape
tmp


# In[21]:




a = np.zeros(121)

for i in range(121):
    a[i] = np.sum(tmp[i])

a


# In[22]:


np.where(a>1)[0]
for i in np.where(a>1)[0]:
    print(i)
    plt.imshow(ksvd2[i].reshape(32,32))
    plt.show()


# In[58]:


cong = np.zeros([10,143])

for i in range(10):
    a,b,c = KSVD.KSVD(con_data2, sig, N_com, i, n_iter=50,  initial_dictionary=A_2D.copy())
    d = np.dot(a,c)
    d = d.swapaxes(0, 1)
    for j in range(143):
        cong[i][j] = (np.square(d[j].reshape(32,32) - con_data[j])).mean(axis=None)


# In[41]:


cong2 = np.zeros(10)
for i in range(10):
    cong2[i] = np.sum(cong[i])
    
cong2 = cong2/25


# In[44]:


cong2#3枚目で大きく変化


# In[43]:


plt.plot(cong2)
plt.show


# In[ ]:


Print_com=121

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[15]:


#RET

height = 32
width = 32
N = 210
ret_data = np.zeros(height*width*N).reshape([N,32,32])

for i in range(N):
    ret_data[i] = patch_list2.imgs[i+11574]

ret_data = ret_data.reshape(210,1024) 
sc = StandardScaler()
ret_data = sc.fit_transform(ret_data)
ret_data2 = ret_data.reshape(N,32*32).swapaxes(0, 1)


sig = 0
k0 = 3
N_com = 121


# In[16]:


ksvd, log_ksvd, X = KSVD.KSVD(ret_data2, sig, N_com, k0, n_iter=1, initial_dictionary=A_2D.copy())


# In[17]:


ksvd2 = ksvd.swapaxes(0, 1)
recon = np.dot(ksvd,X)
recon2 = recon.swapaxes(0, 1)

cols = 11
rows = 11


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(121):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(ksvd2[i].reshape(32,32),cmap = cm.Greys_r,interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[26]:


retg = np.zeros([10,70])

for i in range(10):
    a,b,c = KSVD.KSVD(ret_data2, sig, N_com, i, n_iter=50,  initial_dictionary=A_2D.copy())
    d = np.dot(a,c)
    d = d.swapaxes(0, 1)
    for j in range(70):
        retg[i][j] = (np.square(d[j].reshape(32,32) - ret_data[j])).mean(axis=None)


# In[52]:


retg2 = np.zeros(10)
for i in range(10):
    retg2[i] = np.sum(retg[i])
    
retg2 = retg2/70


# In[53]:


plt.plot(retg2)
plt.show


# In[54]:


retg2 #4枚目で大きく変化


# In[ ]:


Print_com = 1


fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)


# In[20]:


#RET

height = 32
width = 32
N = 609



ggo_data = patch_list2.imgs[5594:6203]

ggo_data = ggo_data.reshape(609,1024) 
sc = StandardScaler()
ggo_data = sc.fit_transform(ggo_data)
ggo_data2 = ggo_data.reshape(N,32*32).swapaxes(0, 1)


sig = 0
k0 = 3
N_com = 121


# In[21]:


ksvd,b,X = KSVD.KSVD(ggo_data2, sig, N_com, k0, n_iter=1,  initial_dictionary=A_2D.copy())


# In[51]:


ggog = np.zeros([10,50])

for i in range(10):
    a,b,c = KSVD.KSVD(ggo_data2, sig, N_com, i, n_iter=50,  initial_dictionary=A_2D.copy())
    d = np.dot(a,c)
    d = d.swapaxes(0, 1)
    for j in range(50):
        ggog[i][j] = (np.square(d[j].reshape(32,32) - ggo_data[j])).mean(axis=None)


# In[55]:


ggog2 = np.zeros(10)
for i in range(10):
    ggog2[i] = np.sum(ggog[i])
    
ggog2 = ggog2/50


# In[57]:


plt.plot(ggog2)
plt.show


# In[58]:


ggog2 #4枚目で大きく変化


# In[ ]:


Print_com = 1

fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
for i in range(Print_com):
    r = i // cols
    c = i % cols
    axes[r, c].imshow(x.components_[i].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)

