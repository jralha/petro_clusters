#%% Libraries
########################################################

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import numpy as np
import os
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True)
plt.style.use('ggplot')
plt.style.use('dark_background')

#%%Functions
########################################################
def optimal_k(inputData):
    scaler = RobustScaler()

    data = scaler.fit_transform(inputData)

    max_k=15
    wcss=[]
    dbs=[]
    sil=[]
    k_range = range(2,max_k)
    for k in (k_range):
        kmeans = KMeans(n_clusters=k)
        temp_pred = kmeans.fit_predict(data)
        wcss_k = kmeans.inertia_
        wcss.append(wcss_k)
        dbs.append(davies_bouldin_score(data,temp_pred))
        sil.append(silhouette_score(data,temp_pred))

    ## Plotting DBScores and Silhouette for optimal k

    plt.figure(figsize=[8,6])
    plt.subplot(2,1,1)
    plt.plot(k_range,sil)
    plt.ylabel("Silhueta")
    plt.subplot(2,1,2)
    plt.plot(k_range,dbs)
    plt.ylabel("Davis-Bouldin")
    plt.xlabel('Número Clusters')

    # Making both measures into a single one towards finding optimal K
    db_std  = scaler.fit_transform(np.array(dbs).reshape(-1, 1))
    sil_std = scaler.fit_transform(np.array(sil).reshape(-1, 1))
    combined = db_std-sil_std

    # plt.figure(figsize=[8,3])
    # plt.plot(k_range,combined)
    # plt.ylabel("Silhueta - Davies-Bouldin (Escalado)")
    # plt.xlabel('Número Clusters')
    
    plt.figure(figsize=[8,3])
    plt.plot(k_range,wcss)
    plt.ylabel("WCSS")
    plt.xlabel('Número Clusters')

    # Optimal K
    opt_k = k_range[np.where(combined == np.min(combined))[0][0]]

    return opt_k

#%% Load Data
########################################################
plug = pd.read_excel('alyne\\dados_litologia\\Dados_Plugues_litologia.xls')
core = pd.read_excel('alyne\\dados_litologia\\Dados_testo_litologia_final.xlsx').iloc[1:,:]


core_feat = core[['GR (gAPI)','PHI total (%)','KTIM (mD)','PEF (b/e)','VNR_f (frac)','RHOB (g/cm³)','T2LM (ms)']]
core_md = core['MD (m)']
core_fac = core['Fácies']

fac = np.unique(core_fac)

enc = LabelEncoder()
fac_num = enc.fit_transform(core_fac)
core['Fácies_num'] = fac_num
core['1'] = 1

#Binary stromatolite/esferulitite
core_bin = []
for i in fac_num:
    if i == 2 or i == 1:
        core_bin.append(0)
    else:
          core_bin.append(1)

core['sferulite'] = pd.Series(core_bin)

#stromatolite/esferulitite/intercaltion
core_int = []
for i in fac_num:
    if i == 0 or i == 1 or i == 4:
        core_int.append(0)
    elif i == 2:
        core_int.append(1)
    else:
        core_int.append(2)

core['sph-str-int'] = pd.Series(core_int)

#Describe data
pd.pivot_table(core,values='1',index=['Fácies'],aggfunc='count')

#%% Plot each feature
########################################################
# num=1
# numt = len(core_feat.columns)
# plt.figure(figsize=[20,10])
# for feat in core_feat.columns:
#     plt.subplot(1,numt,num)
#     num += 1
#     plt.scatter(core_feat[feat],core_md)
#     plt.plot(core_feat[feat],core_md)
#     plt.title(feat)

#%% Determine Optimal number of clusters
########################################################
# opt_k = optimal_k(core_feat)
#opt_k = len(fac)
opt_k = 3

#%% Scaling, clustering and PCA
########################################################
scaler = RobustScaler()
ipt = scaler.fit_transform(core_feat)

pca = PCA()
pca.fit(ipt)   
ppp = pca.transform(ipt)
print(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]+pca.explained_variance_ratio_[2])

gmm = GaussianMixture(n_components=opt_k,random_state=123,max_iter=1000)
pred = gmm.fit_predict(ppp)

#%% 3D Plot of PCA Components
########################################################
c_map = plt.get_cmap('viridis',opt_k)
fig = plt.figure(figsize=[20,20])
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(ppp.T[0],ppp.T[1],ppp.T[2],c=pred,cmap=c_map,s=100)
cbar = fig.colorbar(p)
cbar.set_label('Cluster')
ax.view_init(40,40)
ax.set_ylabel('2nd Component')
ax.set_zlabel('3rd Component')
ax.set_xlabel('1st Component')
plt.show()

#%% 2D Plots of PCA Components
########################################################
plt.figure(figsize=[10,30])
plt.subplot(3,1,1)
plt.scatter(ppp.T[0],ppp.T[1],c=pred,cmap=c_map)
plt.xlabel('1st Component')
plt.ylabel('2nd Component')
plt.colorbar(label='Cluster')
plt.subplot(3,1,2)
plt.scatter(ppp.T[0],ppp.T[2],c=pred,cmap=c_map)
plt.xlabel('1st Component')
plt.ylabel('3rd Component')
plt.colorbar(label='Cluster')
plt.subplot(3,1,3)
plt.scatter(ppp.T[1],ppp.T[2],c=pred,cmap=c_map)
plt.xlabel('2nd Component')
plt.ylabel('3rd Component')
plt.colorbar(label='Cluster')


#%% Pivot Tables
########################################################
core['Cluster'] = pred
pivot0 = pd.pivot_table(core,values='1',index=['Cluster','Fácies'],aggfunc='count')
pivot1 = pd.pivot_table(core,values='1',index=['Fácies','Cluster'],aggfunc='count')
pivot2 = pd.pivot_table(core,values='1',index=['sferulite','Cluster'],aggfunc='count')
pivot3 = pd.pivot_table(core,values='1',index=['sferulite','Cluster'],aggfunc='count')

# %%
