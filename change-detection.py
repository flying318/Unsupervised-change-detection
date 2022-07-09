# -*- coding: UTF-8 -*-
# from scipy.misc import imresize
from imageio import imread,imsave
import numpy as np
from sklearn.decomposition._pca import PCA
from sklearn.cluster import KMeans
from collections import Counter
import cv2
import os.path as osp
from osgeo import gdal_array

def find_vector_set(diff_image, new_size):
    i,j = (0,0)
    vector_set = np.zeros((int(new_size[0]*new_size[1]/25),25))
    print('vector_set.shape:',vector_set.shape)
    while i<vector_set.shape[0]:
        while j<new_size[0]:
            k=0
            while k<new_size[1]:
                block=diff_image[j:j+5,k:k+5]
                feature=block.ravel()
                vector_set[i,:]=feature
                k=k+5
            j=j+5
        i=i+1
    mean_vec=np.mean(vector_set,axis=0)
    vector_set=vector_set-mean_vec
    return vector_set,mean_vec

def find_FVS(EVS,diff_image,mean_vec,new):
    i=2
    feature_vector_set=[]
    while i<new[0]-2:
        j=2
        while j<new[1]-2:
            block=diff_image[i-2:i+3,j-2:j+3]
            feature=block.flatten()
            feature_vector_set.append(feature)
            j+=1
        i+=1
    FVS=np.dot(feature_vector_set,EVS)
    FVS=FVS-mean_vec
    print('[info]feature vector sapce size:',FVS.shape)
    return FVS

def clustering(FVS,components,new):
    kmeans=KMeans(components,verbose=0)
    kmeans.fit(FVS)
    output=kmeans.predict(FVS)
    count=Counter(output)
    least_index=min(count,key=count.get)
    change_map=np.reshape(output,(new[0]-4,new[1]-4))
    return least_index,change_map

def find_PCAKmeans(t1_path,t2_path,out_path):
    # 注意：图像输入需为灰度图
    if osp.splitext(t1_path)[-1] == '.tif':
        image1 = gdal_array.LoadFile(t1_path).astype(np.int16)[0,:,:]
        image2 = gdal_array.LoadFile(t2_path).astype(np.int16)[0,:,:]
        image2 = np.resize(image2,image1.shape) if image1.shape != image2.shape else None
    else:
        image1 = imread(t1_path,pilmode='L').astype(np.int16)
        image2 = imread(t2_path,pilmode='L').astype(np.int16)
    
    new_size = np.asarray(image1.shape[-2:])/5
    new_size = new_size.astype(int)*5

    diff_image = abs(image1-image2)
    print(diff_image.shape)
    imsave(osp.join(out_path,'diff.png'),diff_image)
    vector_set,mean_vec=find_vector_set(diff_image,new_size)
    pca=PCA()
    pca.fit(vector_set)
    EVS=pca.components_
    FVS=find_FVS(EVS,diff_image,mean_vec,new_size)
    components=3
    least_index,change_map=clustering(FVS,components,new_size)
    change_map[change_map==least_index]=255
    change_map[change_map!=255]=0
    change_map=change_map.astype(np.uint8)
    kernel=np.asarray(((0,0,1,0,0),
                        (0,1,1,1,0),
                        (1,1,1,1,1),
                        (0,1,1,1,0),
                        (0,0,1,0,0)),dtype=np.uint8)
    cleanChangeMap=cv2.erode(change_map,kernel)
    imsave(osp.join(out_path,'changemap.png'),change_map)
    imsave(osp.join(out_path,'cleanChangeMap.png'),cleanChangeMap)
    return

if __name__=='__main__':
    t1_path = r"C:\Users\flying\Desktop\学习资料\SAR相关\长沙\GF2\T1\t1_clip.tif"
    t2_path = r"C:\Users\flying\Desktop\学习资料\SAR相关\长沙\GF2\T2\t2_clip.tif"
    out_dir = r"C:\Users\flying\Desktop\学习资料\SAR相关\长沙\GF2"
    find_PCAKmeans(t1_path,t2_path,out_dir)
