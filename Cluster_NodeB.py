# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:22:02 2018

@author: dragonv
"""

import pandas as pd
import numpy as np
from scipy.cluster import vq
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from math import sin, asin, cos, radians, fabs, sqrt

class Cluster(object):
    def __init__(self,nodb):
        self.nodb=nodb    
        
    ######计算eps的值
    def Eps(self):
        nbrs = NearestNeighbors(n_neighbors=2).fit(self.nodb)
        # n_neighbors最近邻的数目，返回距离每个点k个最近的点和距离指数，indices可以理解为表示点的下标，distances为距离
        distances, indices = nbrs.kneighbors(self.nodb)
        eps = distances.mean()  # 均值
        print('eps is %f' % eps)
        return eps
  
    #####根据eps利用dbscan计算簇数量及簇分类
    def Dbscan(self):
        eps=self.Eps()
        labels = DBSCAN(eps, min_samples=3).fit_predict(self.nodb)
        label = set(labels)
        print('ClustersNumber is %d' % len(label))
        dataset = pd.DataFrame({'Lat': self.nodb.iloc[:, 1], 'Lng': self.nodb.iloc[:, 0], 'ClusterID': labels})
        return dataset
   
    #####根据dbscan簇分类在利用K-means计算每个簇的核心
    def Xres(self):
        eps=self.Eps()
        dataset=self.Dbscan()
        xres = np.empty(shape=[0, 3])
        labels = DBSCAN(eps, min_samples=5).fit_predict(self.nodb)
        label = set(labels)
        for i in label:
            res, idx = vq.kmeans2(dataset.loc[dataset['ClusterID'] == i], 1, iter=20, minit='points')
            xres = np.row_stack((xres, res))
            xres = xres
            print(res)
        return xres

    def df_res(self):
        xres=self.Xres()
        df_res = pd.DataFrame(xres)
        return df_res

    #######经纬度距离计算函数
    def Hav(self,theta):
        s = sin(theta / 2)
        return s * s

    def Cal_distance(self,lat0, lng0, lat1, lng1):
        #######用haversine公式计算球面两点间的距离。
        #######经纬度转换成弧度
        EARTH_RADIUS = 6371  # 地球平均半径，6371km
        lat0 = radians(lat0)
        lat1 = radians(lat1)
        lng0 = radians(lng0)
        lng1 = radians(lng1)
        dlng = fabs(lng0 - lng1)
        dlat = fabs(lat0 - lat1)
        h = self.Hav(dlat) + cos(lat0) * cos(lat1) * self.Hav(dlng)
        distance = 2 * EARTH_RADIUS * asin(sqrt(h))
        return distance
        #####计算每个簇心到相应簇的每个基站的距离

    def Nodb_center_distance(self):
        eps=self.Eps()
        dataset=self.Dbscan()
        df3 = pd.DataFrame({'distance(km)': '', 'lng': '', 'lat': '', 'idx': '', 'density': ''}, index=[0])
        labels = DBSCAN(eps, min_samples=5).fit_predict(self.nodb)
        label = set(labels)
        for i in label:
            df2 = dataset.loc[dataset['ClusterID'] == i]
            Xarr = df2.loc[:, ['Lat', 'Lng']].as_matrix()
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Xarr)  # 根据Kde(高斯核密度函数）计算密度
            Xgas = kde.score_samples(Xarr)  # 根据Kde(高斯核密度函数）计算密度
            xres = self.Xres()
            for j in range(len(df2)):
                d = self.Cal_distance(xres[i][1], xres[i][2], df2.iloc[j, 1], df2.iloc[j, 2])
                print('distance is %.3f' % d)
                df4 = pd.Series({'distance(km)': d, 'lng': df2.iloc[j, 1], 'lat': df2.iloc[j, 2], 'idx': i, 'density': Xgas[j]})
                print(df4)
                df3.append(df4, ignore_index=True)
                df3 = df3.append(df4, ignore_index=True)
          
        return df3
    
    
