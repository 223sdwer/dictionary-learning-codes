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

import pickle
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
data = sc.fit_transform(data)    


# In[3]:


#test全部
height = 32
width = 32
P = 927
test = np.zeros(height*width*P).reshape([P,32,32])


for i in range(927):
    test[i] = patch_test_list.imgs[i]
    
test = test.reshape(P,32*32)
test = sc.transform(test)


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


# In[73]:


a = ksvd_test_all_X.swapaxes(0, 1)


# In[34]:


hcm = a[361:434]
hcm = hcm.swapaxes(0, 1)
hcm.shape


# In[47]:


con = a[434:460]
con = con.swapaxes(0, 1)
con.shape


# In[61]:


ggo = a[460:506]
ggo = ggo.swapaxes(0, 1)
ggo.shape


# In[46]:


ret = a[861:927]
ret = ret.swapaxes(0, 1)
ret.shape


# In[74]:


nor = a[506:861]
nor = nor.swapaxes(0, 1)
nor.shape


# In[92]:


nod = a[0:65]
nod = nod.swapaxes(0, 1)
nod.shape


# In[93]:


emp = a[65:361]
emp = emp.swapaxes(0, 1)
emp.shape


# In[75]:


tmp_nr = np.array(nor != 0,dtype=float)
tmp_nd = np.array(nod != 0,dtype=float)
tmp_e = np.array(emp != 0,dtype=float)


# In[35]:


tmp_h = np.array(hcm != 0,dtype=float)


# In[48]:


tmp_c = np.array(con != 0,dtype=float)


# In[63]:


tmp_g = np.array(ggo != 0,dtype=float)


# In[64]:


tmp_r = np.array(ret != 0,dtype=float)


# In[76]:


nr = np.zeros(121)

for i in range(121):
    nr[i] = np.sum(tmp_nr[i])
    
nr


# In[99]:


nd = np.zeros(121)

for i in range(121):
    nd[i] = np.sum(tmp_nd[i])
    
nd


# In[100]:


e = np.zeros(121)

for i in range(121):
    e[i] = np.sum(tmp_e[i])
    
e


# In[70]:


g = np.zeros(121)

for i in range(121):
    g[i] = np.sum(tmp_g[i])


# In[71]:


r = np.zeros(121)

for i in range(121):
    r[i] = np.sum(tmp_r[i])


# In[13]:


all_com = ksvd_all_com.swapaxes(0, 1)
all_com.shape


# In[58]:


for i in con:
    print(i)
    plt.imshow(all_com[i].reshape(32,32),cmap = "bwr")
    plt.show()


# In[82]:


cols = 5
rows = 2
fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(20,10))
k=0
for i in nod:
    
    r = k // cols
    c = k % cols
    axes[r, c].imshow(all_com[i].reshape(32,32),cmap = "bwr",interpolation = 'nearest')
    axes[r, c].get_xaxis().set_visible(False)
    axes[r, c].get_yaxis().set_visible(False)
    k = k+1


# In[79]:


hcm = [1,21,40,60,111]
ggo = [21,28,34,40,60]
ret = [1,21,34,40,60]
nod = [21,34,60,69,72,101,110]
nor = [  1,  21,  30,  32,  40,  68, 101, 104]
con = [  0,  12,  25,  32,  34,  60,  68,  82, 110]
emp = [ 1, 21, 34, 40, 57, 60, 69]


# In[ ]:





# In[ ]:





# In[41]:


np.where(h>=9)[0]


# In[87]:


np.where(g>=6)[0]


# In[86]:


np.where(r>=9)[0]


# In[101]:


np.where(nd>=6)[0]


# In[78]:


np.where(nr>=23)[0]


# In[103]:


np.where(e>=20)[0]


# In[56]:


np.where(c>=3)[0]


# In[72]:


g


# In[75]:


r


# In[37]:


h = np.zeros(121)

for i in range(121):
    h[i] = np.sum(tmp_h[i])


# In[38]:


h


# In[50]:


c = np.zeros(121)

for i in range(121):
    c[i] = np.sum(tmp_c[i])


# In[51]:


c


# In[ ]:





# In[31]:


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

sig = 0
k0 = 4
N_com = 121


# In[6]:


ksvd_nor_com, log_ksvd_nor, ksvd_nor_X = KSVD.KSVD(nor_data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[7]:


ksvd_hcm_com, log_ksvd_hcm, ksvd_hcm_X = KSVD.KSVD(hcm_data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[8]:


ksvd_con_com, log_ksvd_con, ksvd_con_X = KSVD.KSVD(con_data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[9]:


ksvd_ggo_com, log_ksvd_ggo, ksvd_ggo_X = KSVD.KSVD(ggo_data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[10]:


ksvd_ret_com, log_ksvd_ret, ksvd_ret_X = KSVD.KSVD(ret_data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[11]:


ksvd_nod_com, log_ksvd_nod, ksvd_nod_X = KSVD.KSVD(nod_data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[12]:


ksvd_emp_com, log_ksvd_emp, ksvd_emp_X = KSVD.KSVD(emp_data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[6]:


ksvd_all_com, log_ksvd_all, ksvd_all_X = KSVD.KSVD(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=A_2D.copy())


# In[29]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import cv2


# In[ ]:


#A: 辞書
#b: 観測された事例
#k0: xの非ゼロの要素数
#eps: 誤差の閾値

def OMP(A, b, k0, eps):
    #初期化
    x = np.zeros(A.shape[1]) #解
    S = np.zeros(A.shape[1], dtype=np.uint8) #サポート
    r = b.copy() #残差
    
    for i in range(k0):
        #サポートへの追加対象となる列のインデックスリスト
        idx = np.where(S == 0)[0]
        
        #誤差の計算
        err = r.T.dot(r) - A[:, idx].T.dot(r)**2 #内積で割るのを省略(してもしなくても変わらないため?)
        
        #サポートの更新
        S[idx[err.argmin()]] = 1
        
        #暫定解の更新
        As = A[:, S == 1]
        #LA.inv(A.T.dot(A)).dot(A.T)
        x[S == 1] = LA.pinv(As).dot(b)
        
        #残差の更新
        r = b - A.dot(x)
        
        #終了判定
        if LA.norm(r) < eps:
            break
    
    return x, S


# In[ ]:


#Y: 観測事例集合
#sigma: ノイズレート
#m: 辞書の列数(アトム数)
#k0: 非ゼロ要素の個数
#n_iter: 訓練回数
#A0: 真の辞書(Yを生成した辞書)
#initial_dictionary: 初期辞書

def KSVD_OMP(Y, sigma, m, k0, n_iter, A0=None, initial_dictionary=None):
    if initial_dictionary is None:
        A = Y[:, :m] #信号事例のm列を取ってきて初期辞書を生成
        for i in range(A.shape[1]):
            A[:, i] /= LA.norm(A[:, i]) #各列を正規化
    else:
        A = initial_dictionary
    
    X = np.zeros((A.shape[1], Y.shape[1])) #スパース係数行列
    eps = A.shape[0]*(sigma**2) #誤差の閾値(どうしてこう求めるのか分からん...)

    log = [] #記録保存用リスト
    
    bar = tqdm(total=n_iter, desc="train_step")
    for k in range(n_iter):
        #スパース符号化
        for i in tqdm(range(Y.shape[1]), desc="sparse_coding", leave=False):
        #for i in range(Y.shape[1]):
            X[:, i], _ = OMP(A, Y[:, i], k0, eps=eps) #各観測事例に対してOMPを実行しスパース係数行列を生成
        
        
        
        ###似ているアトムを消去するようなコード###
        
        if A0 is not None:
            pass
            """
            真の辞書をどれだけ復元しているか(復元率)を記録
            per = percent_of_recovering_atom(A, A0, 0.99)
            mean_err = np.abs(Y - A.dot(X)).mean()
            log.append([mean_err, per])
            print("{}\t mean error: {}, percent: {}".format(k, mean_err, per))
            """
        else:
            mean_err = np.abs(Y - A.dot(X)).mean()
            log.append(mean_err)
            #print("{}\t mean error: {}".format(k, mean_err))
        
        #終了判定
        if LA.norm(Y - A.dot(X))**2 < eps:
            break
        bar.update(1)
    bar.close()
    return A, np.array(log), X


# In[ ]:


def show_dictionary(A, name=None):
    n = int(np.sqrt(A.shape[0])) #8
    m = int(np.sqrt(A.shape[1])) #11
    A_show = A.reshape((n, n, m, m)) #(8, 8, 11, 11)
    fig, ax = plt.subplots(m, m, figsize=(8, 8))
    for row in range(m):
        for col in range(m):
            ax[row, col].imshow(A_show[:, :, col, row], cmap="gray", interpolation="Nearest")
            ax[row, col].axis("off")
    if name is not None:
        plt.savefig(name, dpi=220)


# In[14]:


#NODの辞書で再構成
ksvd_data_nod_com, log_ksvd_data_nod, ksvd_data_nod_X = KSVD_OMP(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_nod_com)
ksvd_test_nod_com, log_ksvd_test_nod, ksvd_test_nod_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_nod_com)


# In[19]:


nod_dx = np.dot(ksvd_nod_com,ksvd_data_nod_X)
nod_dx = nod_dx.swapaxes(0, 1)

nodtest_dx = np.dot(ksvd_nod_com,ksvd_test_nod_X)
nodtest_dx = nodtest_dx.swapaxes(0, 1)


# In[15]:


#EMPの辞書で再構成
ksvd_data_emp_com, log_ksvd_data_emp, ksvd_data_emp_X = KSVD_OMP(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_emp_com)
ksvd_test_emp_com, log_ksvd_test_emp, ksvd_test_emp_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_emp_com)


# In[18]:


emp_dx = np.dot(ksvd_emp_com,ksvd_data_emp_X)
emp_dx = emp_dx.swapaxes(0, 1)

emptest_dx = np.dot(ksvd_emp_com,ksvd_test_emp_X)
emptest_dx = emptest_dx.swapaxes(0, 1)


# In[16]:


#HCMの辞書で再構成
ksvd_data_hcm_com, log_ksvd_data_hcm, ksvd_data_hcm_X = KSVD_OMP(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_hcm_com)
ksvd_test_hcm_com, log_ksvd_test_hcm, ksvd_test_hcm_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_hcm_com)


# In[17]:


hcm_dx = np.dot(ksvd_hcm_com,ksvd_data_hcm_X)
hcm_dx = hcm_dx.swapaxes(0, 1)

hcmtest_dx = np.dot(ksvd_hcm_com,ksvd_test_hcm_X)
hcmtest_dx = hcmtest_dx.swapaxes(0, 1)


# In[17]:


#CONの辞書で再構成
ksvd_data_con_com, log_ksvd_data_con, ksvd_data_con_X = KSVD_OMP(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_con_com)
ksvd_test_con_com, log_ksvd_test_con, ksvd_test_con_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_con_com)


# In[16]:


con_dx = np.dot(ksvd_con_com,ksvd_data_con_X)
con_dx = con_dx.swapaxes(0, 1)

contest_dx = np.dot(ksvd_con_com,ksvd_test_con_X)
contest_dx = contest_dx.swapaxes(0, 1)


# In[18]:


#GGOの辞書で再構成
ksvd_data_ggo_com, log_ksvd_data_ggo, ksvd_data_ggo_X = KSVD_OMP(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_ggo_com)
ksvd_test_ggo_com, log_ksvd_test_ggo, ksvd_test_ggo_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_ggo_com)


# In[15]:


ggo_dx = np.dot(ksvd_ggo_com,ksvd_data_ggo_X)
ggo_dx = ggo_dx.swapaxes(0, 1)

ggotest_dx = np.dot(ksvd_ggo_com,ksvd_test_ggo_X)
ggotest_dx = ggotest_dx.swapaxes(0, 1)


# In[19]:


#NORの辞書で再構成
ksvd_data_nor_com, log_ksvd_data_nor, ksvd_data_nor_X = KSVD_OMP(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_nor_com)
ksvd_test_nor_com, log_ksvd_test_nor, ksvd_test_nor_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_nor_com)


# In[14]:


nor_dx = np.dot(ksvd_nor_com,ksvd_data_nor_X)
nor_dx = nor_dx.swapaxes(0, 1)

nortest_dx = np.dot(ksvd_nor_com,ksvd_test_nor_X)
nortest_dx = nortest_dx.swapaxes(0, 1)


# In[20]:


#RETの辞書で再構成
ksvd_data_ret_com, log_ksvd_data_ret, ksvd_data_ret_X = KSVD_OMP(data.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_ret_com)
ksvd_test_ret_com, log_ksvd_test_ret, ksvd_test_ret_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_ret_com)


# In[12]:


ret_dx = np.dot(ksvd_ret_com,ksvd_data_ret_X)
ret_dx = ret_dx.swapaxes(0, 1)

rettest_dx = np.dot(ksvd_ret_com,ksvd_test_ret_X)
rettest_dx = rettest_dx.swapaxes(0, 1)


# In[32]:


#全ての辞書で再構成
ksvd_test_all_com, log_ksvd_test_all, ksvd_test_all_X = KSVD_OMP(test.swapaxes(0, 1), sig, N_com, k0, n_iter=50, initial_dictionary=ksvd_all_com)


# In[27]:


all_dx = np.dot(ksvd_all_com,ksvd_all_X)
all_dx = all_dx.swapaxes(0, 1)

alltest_dx = np.dot(ksvd_all_com,ksvd_test_all_X)
alltest_dx = alltest_dx.swapaxes(0, 1)


# In[12]:


ksvd_test_all_X


# In[20]:


gosa = np.zeros([11784,7])
gosa_test = np.zeros([927,7])
for i in range(11784):
    gosa[i][0] = (np.square(nod_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][1] = (np.square(emp_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)    
    gosa[i][2] = (np.square(hcm_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][3] = (np.square(con_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][4] = (np.square(ggo_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][5] = (np.square(nor_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    gosa[i][6] = (np.square(ret_dx[i].reshape(32,32) - data[i].reshape(32,32))).mean(axis=None)
    
for i in range(927):
    gosa_test[i][0] = (np.square(nodtest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][1] = (np.square(emptest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)    
    gosa_test[i][2] = (np.square(hcmtest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][3] = (np.square(contest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][4] = (np.square(ggotest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][5] = (np.square(nortest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    gosa_test[i][6] = (np.square(rettest_dx[i].reshape(32,32) - test[i].reshape(32,32))).mean(axis=None)
    


# In[22]:


gosa_label = np.zeros(11784)
for i in range(11784):
    gosa_label[i] = np.where(gosa[i]==np.min(gosa[i]))[0][0]
    
gosa_test_label = np.zeros(927)
for i in range(927):
    gosa_test_label[i] = np.where(gosa_test[i]==np.min(gosa_test[i]))[0][0]


# In[6]:


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


# In[7]:


labels = ['CON','GGO','HCM','RET','EMP','NOD','NOR']


# In[16]:


cm = confusion_matrix(label,gosa_label)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[17]:


print(accuracy_score(label,gosa_label))


# In[27]:


cm_test = confusion_matrix(label_test,gosa_test_label)
print(cm_test)


sns.heatmap(cm_test, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[19]:


print(accuracy_score(label_test,gosa_test_label))


# In[28]:


log_reg_nod_x = LogisticRegression().fit(ksvd_data_nod_X.swapaxes(0, 1),label)
print(log_reg_nod_x.score(ksvd_data_nod_X.swapaxes(0, 1),label))
print(log_reg_nod_x.score(ksvd_test_nod_X.swapaxes(0, 1),label_test))

cm = confusion_matrix(label_test,log_reg_nod_x.predict(ksvd_test_nod_X.swapaxes(0, 1)))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[9]:


log_reg_emp_x = LogisticRegression().fit(ksvd_data_emp_X.swapaxes(0, 1),label)
print(log_reg_emp_x.score(ksvd_data_emp_X.swapaxes(0, 1),label))
print(log_reg_emp_x.score(ksvd_test_emp_X.swapaxes(0, 1),label_test))

cm2 = confusion_matrix(label,log_reg_emp_x.predict(ksvd_data_emp_X.swapaxes(0, 1)))
cm = confusion_matrix(label_test,log_reg_emp_x.predict(ksvd_test_emp_X.swapaxes(0, 1)))
print(cm2)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[30]:


log_reg_con_x = LogisticRegression().fit(ksvd_data_con_X.swapaxes(0, 1),label)
print(log_reg_con_x.score(ksvd_data_con_X.swapaxes(0, 1),label))
print(log_reg_con_x.score(ksvd_test_con_X.swapaxes(0, 1),label_test))

cm = confusion_matrix(label_test,log_reg_con_x.predict(ksvd_test_con_X.swapaxes(0, 1)))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[31]:


log_reg_ggo_x = LogisticRegression().fit(ksvd_data_ggo_X.swapaxes(0, 1),label)
print(log_reg_ggo_x.score(ksvd_data_ggo_X.swapaxes(0, 1),label))
print(log_reg_ggo_x.score(ksvd_test_ggo_X.swapaxes(0, 1),label_test))

cm = confusion_matrix(label_test,log_reg_ggo_x.predict(ksvd_test_ggo_X.swapaxes(0, 1)))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[8]:


log_reg_ret_x = LogisticRegression().fit(ksvd_data_ret_X.swapaxes(0, 1),label)
print(log_reg_ret_x.score(ksvd_data_ret_X.swapaxes(0, 1),label))
print(log_reg_ret_x.score(ksvd_test_ret_X.swapaxes(0, 1),label_test))

cm = confusion_matrix(label_test,log_reg_ret_x.predict(ksvd_test_ret_X.swapaxes(0, 1)))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[33]:


log_reg_hcm_x = LogisticRegression().fit(ksvd_data_hcm_X.swapaxes(0, 1),label)
print(log_reg_hcm_x.score(ksvd_data_hcm_X.swapaxes(0, 1),label))
print(log_reg_hcm_x.score(ksvd_test_hcm_X.swapaxes(0, 1),label_test))

cm = confusion_matrix(label_test,log_reg_hcm_x.predict(ksvd_test_hcm_X.swapaxes(0, 1)))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[34]:


log_reg_nor_x = LogisticRegression().fit(ksvd_data_nor_X.swapaxes(0, 1),label)
print(log_reg_nor_x.score(ksvd_data_nor_X.swapaxes(0, 1),label))
print(log_reg_nor_x.score(ksvd_test_nor_X.swapaxes(0, 1),label_test))

cm = confusion_matrix(label_test,log_reg_nor_x.predict(ksvd_test_nor_X.swapaxes(0, 1)))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[94]:


log_reg_all_x = LogisticRegression().fit(ksvd_all_X.swapaxes(0, 1),label)
print(log_reg_all_x.score(ksvd_all_X.swapaxes(0, 1),label))
print(log_reg_all_x.score(ksvd_test_all_X.swapaxes(0, 1),label_test))

cm = confusion_matrix(label_test,log_reg_all_x.predict(ksvd_test_all_X.swapaxes(0, 1)))
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[4]:


cm = np.array([[ 19 ,  1 ,  0 ,  6 ,  0 ,  0 ,  0],
 [  2,  22 ,  5 ,  5 ,  0  ,11 ,  1],
 [  2,  18 , 25 ,  2  , 0,  19  , 7],
 [  5 , 42 ,  5 , 14  , 0 ,  0  , 0],
 [  0,   0 ,  2 ,  0 ,199 ,  1 , 94],
 [  0 ,  2  , 9  , 0 ,  0  ,15  ,39],
 [  0 ,  1  , 3  , 0 , 14 ,  3, 334]],dtype = float)


# In[5]:


def rate(a):
    b = np.copy(a)
    for i in range(7):
        for j in range(7):
            a[i][j] = a[i][j]/np.sum(b[i])


# In[6]:


rate(cm)


# In[8]:


sns.heatmap(cm, annot=True, cmap='Oranges',fmt='.2f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# In[95]:


log_reg_all_x = LogisticRegression().fit(ksvd_all_X.swapaxes(0, 1),label)
print(log_reg_all_x.score(ksvd_all_X.swapaxes(0, 1),label))
print(log_reg_all_x.score(ksvd_test_all_X.swapaxes(0, 1),label_test))

cm2 = confusion_matrix(label,log_reg_all_x.predict(ksvd_all_X.swapaxes(0, 1)))
cm = confusion_matrix(label_test,log_reg_all_x.predict(ksvd_test_all_X.swapaxes(0, 1)))
print(cm2)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues',fmt='.0f',xticklabels=labels,yticklabels=labels)
#plt.savefig('data/dst/sklearn_confusion_matrix.png')


# za = open("ksvd_all_com4","wb")
# zb = open("log_ksvd_all4","wb")
# zc = open("ksvd_all_X4","wb")
# 
# pickle.dump(ksvd_all_com,za)
# pickle.dump(log_ksvd_all,zb)
# pickle.dump(ksvd_all_X,zc)
# 
# za.close
# zb.close
# zc.close

# In[25]:


za = open("ksvd_all_com4","rb")
zb = open("log_ksvd_all4","rb")
zc = open("ksvd_all_X4","rb")




ksvd_all_com = pickle.load(za)
log_ksvd_all = pickle.load(zb)
ksvd_all_X = pickle.load(zc)


# a = open("ksvd_nor_com4","wb")
# b = open("log_ksvd_nor4","wb")
# c = open("ksvd_nor_X4","wb")
# 
# d = open("ksvd_hcm_com4","wb")
# e = open("log_ksvd_hcm4","wb")
# f = open("ksvd_hcm_X4","wb")
# 
# g = open("ksvd_con_com4","wb")
# h = open("log_ksvd_con4","wb")
# i = open("ksvd_con_X4","wb")
# 
# j = open("ksvd_ggo_com4","wb")
# k = open("log_ksvd_ggo4","wb")
# l = open("ksvd_ggo_X4","wb")
# 
# m = open("ksvd_ret_com4","wb")
# n = open("log_ksvd_ret4","wb")
# zz = open("ksvd_ret_X4","wb")
# 
# o = open("ksvd_emp_com4","wb")
# p = open("log_ksvd_emp4","wb")
# q = open("ksvd_emp_X4","wb")
# 
# r = open("ksvd_nod_com4","wb")
# s = open("log_ksvd_nod4","wb")
# t = open("ksvd_nod_X4","wb")
# 
# u = open("log_ksvd_data_nod4","wb")
# v = open("ksvd_data_nod_X4","wb")
# 
# w = open("log_ksvd_test_nod4","wb")
# x = open("ksvd_test_nod_X4","wb")
# 
# y = open("log_ksvd_data_emp4","wb")
# z = open("ksvd_data_emp_X4","wb")
# 
# aa = open("log_ksvd_test_emp4","wb")
# ab = open("ksvd_test_emp_X4","wb")
# 
# ac = open("log_ksvd_data_hcm4","wb")
# ad = open("ksvd_data_hcm_X4","wb")
# 
# ae = open("log_ksvd_test_hcm4","wb")
# af = open("ksvd_test_hcm_X4","wb")
# 
# ah = open("log_ksvd_data_con4","wb")
# ai = open("ksvd_data_con_X4","wb")
# 
# aj = open("log_ksvd_test_con4","wb")
# ak = open("ksvd_test_con_X4","wb")
# 
# al = open("log_ksvd_data_ggo4","wb")
# am = open("ksvd_data_ggo_X4","wb")
# 
# an = open("log_ksvd_test_ggo4","wb")
# ao = open("ksvd_test_ggo_X4","wb")
# 
# ap = open("log_ksvd_data_nor4","wb")
# aq = open("ksvd_data_nor_X4","wb")
# 
# ar = open("log_ksvd_test_nor4","wb")
# at = open("ksvd_test_nor_X4","wb")
# 
# au = open("log_ksvd_data_ret4","wb")
# av = open("ksvd_data_ret_X4","wb")
# 
# aw = open("log_ksvd_test_ret4","wb")
# #ax = open("ksvd_test_ret_X4","wb")
# 
# 
# pickle.dump(ksvd_nor_com,a)
# pickle.dump(log_ksvd_nor,b)
# pickle.dump(ksvd_nor_X,c)
# 
# 
# pickle.dump(ksvd_hcm_com,d)
# pickle.dump(log_ksvd_hcm,e)
# pickle.dump(ksvd_hcm_X,f)
# 
# 
# pickle.dump(ksvd_con_com,g)
# pickle.dump(log_ksvd_con,h)
# pickle.dump(ksvd_con_X,i)
# 
# 
# 
# pickle.dump(ksvd_ggo_com,j)
# pickle.dump(log_ksvd_ggo,k)
# pickle.dump(ksvd_ggo_X,l)
# 
# 
# pickle.dump(ksvd_ret_com,m)
# pickle.dump(log_ksvd_ret,n)
# pickle.dump(ksvd_ret_X,zz)
# 
# 
# pickle.dump(ksvd_emp_com,o)
# pickle.dump(log_ksvd_emp,p)
# pickle.dump(ksvd_emp_X,q)
# 
# 
# pickle.dump(ksvd_nod_com,r)
# pickle.dump(log_ksvd_nod,s)
# pickle.dump(ksvd_nod_X,t)
# 
# pickle.dump(log_ksvd_data_nod,u)
# pickle.dump(ksvd_data_nod_X,v)
# 
# pickle.dump(log_ksvd_test_nod,w)
# pickle.dump(ksvd_test_nod_X,x)
# 
# pickle.dump(log_ksvd_data_emp,y)
# pickle.dump(ksvd_data_emp_X,z)
# 
# pickle.dump(log_ksvd_test_emp,aa)
# pickle.dump(ksvd_test_emp_X,ab)
# 
# pickle.dump(log_ksvd_data_hcm,ac)
# pickle.dump(ksvd_data_hcm_X,ad)
# 
# pickle.dump(log_ksvd_test_hcm,ae)
# pickle.dump(ksvd_test_hcm_X,af)
# 
# pickle.dump(log_ksvd_data_con,ah)
# pickle.dump(ksvd_data_con_X,ai)
# 
# pickle.dump(log_ksvd_test_con,aj)
# pickle.dump(ksvd_test_con_X,ak)
# 
# pickle.dump(log_ksvd_data_ggo,al)
# pickle.dump(ksvd_data_ggo_X,am)
# 
# pickle.dump(log_ksvd_test_ggo,an)
# pickle.dump(ksvd_test_ggo_X,ao)
# 
# pickle.dump(log_ksvd_data_nor,ap)
# pickle.dump(ksvd_data_nor_X,aq)
# 
# pickle.dump(log_ksvd_test_nor,ar)
# pickle.dump(ksvd_test_nor_X,at)
# 
# pickle.dump(ksvd_test_ret_X,au)
# pickle.dump(ksvd_data_ret_X,av)
# 
# pickle.dump(ksvd_test_ret_X,aw)
# #pickle.dump(ksvd_test_ret_X,ax)
# 
# 
# a.close
# b.close
# c.close
# d.close
# e.close
# f.close
# g.close
# h.close
# i.close
# j.close
# k.close
# l.close
# m.close
# n.close
# zz.close
# o.close
# p.close
# q.close
# r.close
# s.close
# t.close
# u.close
# v.close
# w.close
# x.close
# y.close
# z.close
# aa.close
# ab.close
# ac.close
# ad.close
# ae.close
# af.close
# ah.close
# ai.close
# aj.close
# ak.close
# al.close
# am.close
# an.close
# ao.close
# ap.close
# aq.close
# ar.close
# at.close
# au.close
# av.close
# aw.close
# #ax.close

# In[5]:


a = open("ksvd_nor_com4","rb")
b = open("log_ksvd_nor4","rb")
c = open("ksvd_nor_X4","rb")

d = open("ksvd_hcm_com4","rb")
e = open("log_ksvd_hcm4","rb")
f = open("ksvd_hcm_X4","rb")

g = open("ksvd_con_com4","rb")
h = open("log_ksvd_con4","rb")
i = open("ksvd_con_X4","rb")

j = open("ksvd_ggo_com4","rb")
k = open("log_ksvd_ggo4","rb")
l = open("ksvd_ggo_X4","rb")

m = open("ksvd_ret_com4","rb")
n = open("log_ksvd_ret4","rb")
zz = open("ksvd_ret_X4","rb")

o = open("ksvd_emp_com4","rb")
p = open("log_ksvd_emp4","rb")
q = open("ksvd_emp_X4","rb")

r = open("ksvd_nod_com4","rb")
s = open("log_ksvd_nod4","rb")
t = open("ksvd_nod_X4","rb")

u = open("log_ksvd_data_nod4","rb")
v = open("ksvd_data_nod_X4","rb")

w = open("log_ksvd_test_nod4","rb")
x = open("ksvd_test_nod_X4","rb")

y = open("log_ksvd_data_emp4","rb")
z = open("ksvd_data_emp_X4","rb")

aa = open("log_ksvd_test_emp4","rb")
ab = open("ksvd_test_emp_X4","rb")

ac = open("log_ksvd_data_hcm4","rb")
ad = open("ksvd_data_hcm_X4","rb")

ae = open("log_ksvd_test_hcm4","rb")
af = open("ksvd_test_hcm_X4","rb")

ah = open("log_ksvd_data_con4","rb")
ai = open("ksvd_data_con_X4","rb")

aj = open("log_ksvd_test_con4","rb")
ak = open("ksvd_test_con_X4","rb")

al = open("log_ksvd_data_ggo4","rb")
am = open("ksvd_data_ggo_X4","rb")

an = open("log_ksvd_test_ggo4","rb")
ao = open("ksvd_test_ggo_X4","rb")

ap = open("log_ksvd_data_nor4","rb")
aq = open("ksvd_data_nor_X4","rb")

ar = open("log_ksvd_test_nor4","rb")
at = open("ksvd_test_nor_X4","rb")

au = open("log_ksvd_data_ret4","rb")
av = open("ksvd_data_ret_X4","rb")

aw = open("log_ksvd_test_ret4","rb")
ax = open("ksvd_test_ret_X4","rb")







ksvd_nor_com = pickle.load(a)
log_ksvd_nor = pickle.load(b)
ksvd_nor_X = pickle.load(c)

ksvd_hcm_com = pickle.load(d)
log_ksvd_hcm = pickle.load(e)
ksvd_hcm_X = pickle.load(f)

ksvd_con_com = pickle.load(g)
log_ksvd_con = pickle.load(h)
ksvd_con_X = pickle.load(i)

ksvd_ggo_com = pickle.load(j)
log_ksvd_ggo = pickle.load(k)
ksvd_ggo_X = pickle.load(l)

ksvd_ret_com = pickle.load(m)
log_ksvd_ret = pickle.load(n)
ksvd_ret_X = pickle.load(zz)

ksvd_emp_com = pickle.load(o)
log_ksvd_emp = pickle.load(p)
ksvd_emp_X = pickle.load(q)

ksvd_nod_com = pickle.load(r)
log_ksvd_nod = pickle.load(s)
ksvd_nod_X = pickle.load(t)

log_ksvd_data_nod = pickle.load(u)
ksvd_data_nod_X = pickle.load(v)

log_ksvd_test_nod = pickle.load(w)
ksvd_test_nod_X = pickle.load(x)

log_ksvd_data_emp = pickle.load(y)
ksvd_data_emp_X = pickle.load(z)

log_ksvd_test_emp = pickle.load(aa)
ksvd_test_emp_X = pickle.load(ab)

log_ksvd_data_hcm = pickle.load(ac)
ksvd_data_hcm_X = pickle.load(ad)

log_ksvd_test_hcm = pickle.load(ae)
ksvd_test_hcm_X = pickle.load(af)

log_ksvd_data_con = pickle.load(ah)
ksvd_data_con_X = pickle.load(ai)

log_ksvd_test_con = pickle.load(aj)
ksvd_test_con_X = pickle.load(ak)

log_ksvd_data_ggo = pickle.load(al)
ksvd_data_ggo_X = pickle.load(am)

log_ksvd_test_ggo = pickle.load(an)
ksvd_test_ggo_X = pickle.load(ao)

log_ksvd_data_nor = pickle.load(ap)
ksvd_data_nor_X = pickle.load(aq)

log_ksvd_test_nor = pickle.load(ar)
ksvd_test_nor_X = pickle.load(at)

ksvd_test_ret_X = pickle.load(au)
ksvd_data_ret_X = pickle.load(av)

#log_ksvd_test_ret = pickle.load(aw)
#ksvd_test_ret_X = pickle.load(ax)

