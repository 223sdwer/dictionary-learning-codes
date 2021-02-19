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

def KSVD(Y, sigma, m, k0, n_iter, A0=None, initial_dictionary=None):
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
        
        #K-SVD辞書更新
        for j0 in tqdm(range(m), desc="update_dictionary", leave=False):
        #for j0 in tqdm(range(m)):
            omega = X[j0, :] != 0 #a_j0を使用している事例があるかどうか
            if np.sum(omega) == 0: continue #a_j0が使用されていない場合は次のアトムへ
            X[j0, omega] = 0 #a_j0の影響を測るため,a_j0の係数を0にする(a_j0を引っこ抜く)
            E = Y - A.dot(X) #残差行列
            E_R = E[:, omega] #omegaに対応する列のみを取り出す(Xの非ゼロ位置を固定するため)
            U, S, V = LA.svd(E_R)
            
            #対象のアトムとスパース表現を更新(ランク1近似)
            A[:, j0] = U[:, 0]
            X[j0, omega] = S[0] * V.T[:, 0]
        
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

