#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image
import cv2
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import patch_list
import patch_list2
import patch_test_list
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import KSVD
import matplotlib.font_manager
from sklearn.metrics import confusion_matrix
import seaborn as sns
fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[2]:


sc = StandardScaler()

#data全部
height = 32
width = 32
N = 11784
data = np.zeros(height*width*N).reshape([N,32,32])

for i in range(11784):
    data[i] = patch_list2.imgs[i]
    
data = data.reshape(N,32*32)
data = data/255   


# In[3]:


#test全部
height = 32
width = 32
P = 927
test = np.zeros(height*width*P).reshape([P,32,32])


for i in range(927):
    test[i] = patch_test_list.imgs[i]
    
test = test.reshape(P,32*32)
test = test/255


# In[4]:


#data

#NOD
N_nod = 763
nod_data = data[0:763].reshape(N_nod,32*32)


#EMP
N_emp = 4406
emp_data = data[763:5169].reshape(N_emp,32*32)

#HCM
N_hcm = 282
hcm_data = data[5169:5451].reshape(N_hcm,32*32)

#CON
N_con = 143
con_data = data[5451:5594].reshape(N_con,32*32)

#GGO
N_ggo = 609
ggo_data = data[5594:6203].reshape(N_ggo,32*32)

#NOR
N_nor = 5371
nor_data = data[6203:11574].reshape(N_nor,32*32)

#RET
N_ret = 210
ret_data = data[11574:11784].reshape(N_ret,32*32)

#test
#NOD
N_nod_t = 65
nod_test = test[0:65].reshape(N_nod_t,32*32)

#EMP
N_emp_t = 296
emp_test = test[65:361].reshape(N_emp_t,32*32)

#HCM
N_hcm_t = 73
hcm_test = test[361:434].reshape(N_hcm_t,32*32)

#CON
N_con_t = 26
con_test = test[434:460].reshape(N_con_t,32*32)

#GGO
N_ggo_t = 46
ggo_test = test[460:506].reshape(N_ggo_t,32*32)

#NOR
N_nor_t = 355
nor_test = test[506:861].reshape(N_nor_t,32*32)

#RET
N_ret_t = 66
ret_test = test[861:927].reshape(N_ret_t,32*32)


# In[5]:


#NORの辞書
#40% 24
N_nor = 512

nor_pca = NMF(n_components = N_nor)
nor_pca.fit(nor_data)
#nor_x = nor_pca.transform(nor_data)
#nor_dx = nor_pca.inverse_transform(nor_x)


# In[6]:


#HCMの辞書
#40% 20
N_hcm = 141

hcm_pca = NMF(n_components = N_hcm)
hcm_pca.fit(hcm_data)
#hcm_x = hcm_pca.transform(hcm_data)
#hcm_dx = hcm_pca.inverse_transform(hcm_x)


# In[7]:


#CONの辞書
#40% 9
N_con = 72

con_pca = NMF(n_components = N_con)
con_pca.fit(con_data)
#con_x = con_pca.transform(con_data)
#con_dx = con_pca.inverse_transform(con_x)


# In[8]:


#RETの辞書
#40% 9
N_ret = 105

ret_pca = NMF(n_components = N_ret)
ret_pca.fit(ret_data)
#ret_x = ret_pca.transform(ret_data)
#ret_dx = ret_pca.inverse_transform(ret_x)


# In[9]:


#GGOの辞書
#40% 8
N_ggo = 305

ggo_pca = NMF(n_components = N_ggo)
ggo_pca.fit(ggo_data)
#ggo_x = ggo_pca.transform(ggo_data)
#ggo_dx = ggo_pca.inverse_transform(ggo_x)


# In[10]:


#NODの辞書
#40% 25
N_nod = 382

nod_pca = NMF(n_components = N_nod)
nod_pca.fit(nod_data)
#nod_x = nod_pca.transform(nod_data)
#nod_dx = nod_pca.inverse_transform(nod_x)


# In[11]:


#EMPの辞書
#40% 25
N_emp = 512

emp_pca = NMF(n_components = N_emp)
emp_pca.fit(emp_data)
#emp_x = emp_pca.transform(emp_data)
#emp_dx = emp_pca.inverse_transform(emp_x)


# In[5]:


#全体の辞書

N_all = 512

all_pca = NMF(n_components = N_all)
all_pca.fit(data)
#all_x = all_pca.transform(data)
#all_dx = all_pca.inverse_transform(all_x)

alltest_x = all_pca.transform(test)
#alltest_dx = all_pca.inverse_transform(alltest_x)



# In[13]:


#NODの辞書で再構成

nod_x = nod_pca.transform(data)
nod_dx = nod_pca.inverse_transform(nod_x)

nodtest_x = nod_pca.transform(test)
nodtest_dx = nod_pca.inverse_transform(nodtest_x)


# In[14]:


#EMPの辞書で再構成

emp_x = emp_pca.transform(data)
emp_dx = emp_pca.inverse_transform(emp_x)

emptest_x = emp_pca.transform(test)
emptest_dx = emp_pca.inverse_transform(emptest_x)


# In[15]:


#HCMの辞書で再構成

hcm_x = hcm_pca.transform(data)
hcm_dx = hcm_pca.inverse_transform(hcm_x)

hcmtest_x = hcm_pca.transform(test)
hcmtest_dx = hcm_pca.inverse_transform(hcmtest_x)


# In[16]:


#CONの辞書で再構成

con_x = con_pca.transform(data)
con_dx = con_pca.inverse_transform(con_x)

contest_x = con_pca.transform(test)
contest_dx = con_pca.inverse_transform(contest_x)


# In[17]:


#GGOの辞書で再構成

ggo_x = ggo_pca.transform(data)
ggo_dx = ggo_pca.inverse_transform(ggo_x)

ggotest_x = ggo_pca.transform(test)
ggotest_dx = ggo_pca.inverse_transform(ggotest_x)


# In[18]:


#NORの辞書で再構成

nor_x = nor_pca.transform(data)
nor_dx = nor_pca.inverse_transform(nor_x)

nortest_x = nor_pca.transform(test)
nortest_dx = nor_pca.inverse_transform(nortest_x)


# In[19]:


#RETの辞書で再構成

ret_x = ret_pca.transform(data)
ret_dx = ret_pca.inverse_transform(ret_x)

rettest_x = ret_pca.transform(test)
rettest_dx = ret_pca.inverse_transform(rettest_x)


# In[20]:


ret_dx.shape


# In[25]:


gosa = np.zeros([11784,7])
gosa_test = np.zeros([927,7])
for i in range(11784):
    gosa[i][5] = (np.square(nod_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][4] = (np.square(emp_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)    
    gosa[i][2] = (np.square(hcm_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][0] = (np.square(con_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][1] = (np.square(ggo_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][6] = (np.square(nor_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][3] = (np.square(ret_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    
for i in range(927):
    gosa_test[i][5] = (np.square(nodtest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][4] = (np.square(emptest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)    
    gosa_test[i][2] = (np.square(hcmtest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][0] = (np.square(contest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][1] = (np.square(ggotest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][6] = (np.square(nortest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][3] = (np.square(rettest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    


# In[37]:


gosa_label = np.zeros(11784)
for i in range(11784):
    gosa_label[i] = np.where(gosa[i]==np.min(gosa[i]))[0][0]
    
gosa_test_label = [1] * 927
for i in range(927):
    gosa_test_label[i] = np.where(gosa_test[i]==np.min(gosa_test[i]))[0][0]


# In[38]:


for i in range(927):
    gosa_test[i][gosa_test_label[i]]=20


# In[39]:


gosa_test_label2 = [1] * 927
for i in range(927):
    gosa_test_label2[i] = np.where(gosa_test[i]==np.min(gosa_test[i]))[0][0]


# In[2]:


label = np.zeros(11784)
label[763:5169] = 4
label[5169:5451] = 2
label[:763] = 5
label[5594:6203] = 1
label[6203:11574] = 6
label[11574:11784] = 3

label_test = np.zeros(927)
label_test[65:361] = 4
label_test[361:434] = 2
label_test[:65] = 5
label_test[460:506] = 1
label_test[506:861] = 6
label_test[861:927] = 3


# In[21]:


labels = ['NOD','EMP','HCM','CON','GGO','NOR','RET']


# In[3]:


labels = ['CON','GGO','HCM','RET','EMP','NOD','NOR']


# In[33]:


cm = confusion_matrix(label,gosa_label)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[41]:


print(accuracy_score(label,gosa_label))


# In[36]:


cm_test = confusion_matrix(label_test,gosa_test_label)
print(cm_test)


sns.heatmap(cm_test, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[5]:


cm_test = np.array([[  0  , 0 ,  0 ,  0  , 1  , 0 , 25],
 [  0 ,  0 ,  0  , 0 ,  7  , 0 , 39],
 [  0 ,  0  , 0 ,  0 , 11  , 0 , 62],
 [  0  , 0  , 0  , 0  , 6 ,  0 , 60],
 [  0 ,  0  , 0 ,  0  ,75  , 0, 221],
 [  0  , 0 ,  0 ,  0 ,  9  , 0 , 56],
 [  0 ,  0 ,  0 ,  0 , 50  , 0 ,305]],dtype=float)


# In[6]:


def rate(a):
    b = np.copy(a)
    for i in range(7):
        for j in range(7):
            a[i][j] = a[i][j]/np.sum(b[i])


# In[7]:


rate(cm_test)


# In[8]:


sns.heatmap(cm_test, annot=True, cmap='Oranges',fmt='.2f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[9]:


sns.heatmap(cm_test, annot=True, cmap='Oranges',fmt='.2f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[40]:


cm_test = confusion_matrix(label_test,gosa_test_label2)
print(cm_test)


sns.heatmap(cm_test, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[11]:


cm_test = np.array([[  0 ,  0 ,  0 ,  0 , 25  , 0 ,  1],
 [  0 ,  0  , 0 ,  0 , 39 ,  0  , 7],
 [  0 ,  0 ,  0 ,  0 , 62 ,  0  ,11],
 [  0 ,  0   ,0  , 0,  60 ,  0 ,  6],
 [  0 ,  0,   0 ,  0, 221 ,  0,  75],
 [  0 ,  0 ,  0 ,  0 , 56 ,  0 ,  9],
 [  0  , 0  , 0  , 0 ,305  , 0 , 50]],dtype=float)


# In[13]:


rate(cm_test)


# In[14]:


sns.heatmap(cm_test, annot=True, cmap='Oranges',fmt='.2f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[42]:


print(accuracy_score(label_test,gosa_test_label))


# In[17]:


log_reg_nod_x = LogisticRegression().fit(nod_x,label)
print(log_reg_nod_x.score(nod_x,label))
print(log_reg_nod_x.score(nodtest_x,label_test))

cm2 = confusion_matrix(label,log_reg_nod_x.predict(nod_x))
cm = confusion_matrix(label_test,log_reg_nod_x.predict(nodtest_x))
print(cm2)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[12]:


log_reg_nod_x = LogisticRegression().fit(nod_x,label)
print(log_reg_nod_x.score(nod_x,label))
print(log_reg_nod_x.score(nodtest_x,label_test))

cm2 = confusion_matrix(label,log_reg_nod_x.predict(nod_x))
cm = confusion_matrix(label_test,log_reg_nod_x.predict(nodtest_x))
print(cm2)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[22]:


a = np.array([[  33  ,  7  ,  1  ,  0  ,  0 , 722 ,   0],
 [   1, 4002 ,  25  ,  0 ,   0 , 378  ,  0],
 [   0  , 15 , 226  ,  0  ,  0 ,  41  ,  0],
 [   0 ,   0  ,  0 , 134  ,  4  ,  0  ,  5],
 [   3 ,   0  ,  0  ,  1 , 511 ,  88  ,  6],
 [   5 , 260  ,  3  ,  0  ,  0 ,5103  ,  0],
 [   0  ,  0  ,  1  ,  1 ,  74  ,  6 , 128]])

b = np.array([[  2  , 0  , 3 ,  0 ,  0  ,60  , 0],
 [  0, 204  , 4 ,  0  , 0 , 88  , 0],
 [  1 ,  0 , 42  , 0  , 5 , 20 ,  5],
 [  0 ,  0 ,  0  ,18  , 3 ,  0 ,  5],
 [  0 ,  0 ,  2 ,  0 , 19 , 24  , 1],
 [  1 , 12  , 2 ,  0  , 0 ,340 ,  0],
 [  0 ,  0  , 2 ,  1 , 34  , 1 , 28]])

sns.heatmap(a, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[23]:


sns.heatmap(b, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[29]:


log_reg_nod_x = LogisticRegression().fit(nod_x,label)
print(log_reg_nod_x.score(nod_x,label))
print(log_reg_nod_x.score(nodtest_x,label_test))

cm2 = confusion_matrix(label,log_reg_nod_x.predict(nod_x))
cm = confusion_matrix(label_test,log_reg_nod_x.predict(nodtest_x))
print(cm2)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[30]:


log_reg_emp_x = LogisticRegression().fit(emp_x,label)
print(log_reg_emp_x.score(emp_x,label))
print(log_reg_emp_x.score(emptest_x,label_test))

cm = confusion_matrix(label_test,log_reg_emp_x.predict(emptest_x))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[31]:


log_reg_hcm_x = LogisticRegression().fit(hcm_x,label)
print(log_reg_hcm_x.score(hcm_x,label))
print(log_reg_hcm_x.score(hcmtest_x,label_test))

cm = confusion_matrix(label_test,log_reg_hcm_x.predict(hcmtest_x))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[32]:


log_reg_con_x = LogisticRegression().fit(con_x,label)
print(log_reg_con_x.score(con_x,label))
print(log_reg_con_x.score(contest_x,label_test))

cm = confusion_matrix(label_test,log_reg_con_x.predict(contest_x))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[33]:


log_reg_ggo_x = LogisticRegression().fit(ggo_x,label)
print(log_reg_ggo_x.score(ggo_x,label))
print(log_reg_ggo_x.score(ggotest_x,label_test))

cm = confusion_matrix(label_test,log_reg_ggo_x.predict(ggotest_x))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[34]:


log_reg_ret_x = LogisticRegression().fit(ret_x,label)
print(log_reg_ret_x.score(ret_x,label))
print(log_reg_ret_x.score(rettest_x,label_test))

cm = confusion_matrix(label_test,log_reg_ret_x.predict(rettest_x))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[35]:


log_reg_nor_x = LogisticRegression().fit(nor_x,label)
print(log_reg_nor_x.score(nor_x,label))
print(log_reg_nor_x.score(nortest_x,label_test))

cm = confusion_matrix(label_test,log_reg_nor_x.predict(nortest_x))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[16]:


log_reg_all_x = LogisticRegression().fit(all_x,label)
print(log_reg_all_x.score(all_x,label))
print(log_reg_all_x.score(alltest_x,label_test))

cm = confusion_matrix(label_test,log_reg_all_x.predict(alltest_x))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[ ]:




