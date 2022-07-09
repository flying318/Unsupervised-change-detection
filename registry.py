from msilib.schema import Error
import py_compile
from osgeo import gdal,gdal_array,gdalconst
import numpy as np
import cv2
import cmath
from matplotlib import pyplot as plt
from time import *
from imageio import imwrite,imread
import os.path as osp
import os

class Co_registry(object):

    def __init__(self,
                t1_path:str,
                t2_path:str,
                out_dir:str):
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.out_dir = out_dir
        if not osp.exists(self.t1_path) or not osp.exists(self.t2_path):
            raise Exception('输入路径有误，请重新输入！')
        elif not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.out_name = osp.join(self.out_dir,osp.basename(self.t2_path).replace('.','_registry.'))

    def load_img(self,path):
        if osp.splitext(path)[-1] == '.tif':
            img = gdal.Open(path)
            print('路径:{}的tif文件无法打开！'.format(path)) if img == None else None
            img_width = img.RasterXSize #图像列数
            img_height = img.RasterYSize #图像列数
            img_proj = img.GetProjection() #获取投影信息
            img_geotrans = img.GetGeoTransform() #获取仿射矩阵
            img_arr = img.ReadAsArray(0,0,img_width,img_height)
            print('tif数据尺寸:{}'.format(img_arr.shape))
            del img
            return img_arr,img_proj,img_geotrans
        else:
            img = imread(path)
            return img

    def normalization(self,data):
        return (data-np.min(data))/(np.max(data)-np.min(data))
    
    def tif16_to_8bit(self,img):
        if np.max(img)-np.min(img) != 0:
            img_norm = self.normalization(img)*255
        return np.uint8(img_norm)
    
    def imagexy2geo(self,trans,row,col):
        px = trans[0] + col*trans[1] + row*trans[2]
        py = trans[3] + col*trans[4] + row*trans[5]
        return px,py
    
    def geo2imagexy(self,trans,x,y):
        a = np.array([[trans[1],trans[2]],[trans[4],trans[5]]])
        b = np.array([x-trans[0],y-trans[3]])
        return np.linalg.solve(a,b)

    def writeTif(self, arr, projection, geotrans, out_path):
        arr = arr.transpose(2,0,1)
        y,x = arr.shape[-2:] #注意行列顺序
        bands_num = arr.shape[0] #获取输出波段
        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_tif = driver.Create(out_path,x,y,bands_num,gdalconst.GDT_UInt16)
        # 为输出文件设置投影和仿射变换
        out_tif.SetProjection(projection)
        out_tif.SetGeoTransform(geotrans)
        for i in range(arr.shape[0]):
            out_tif.GetRasterBand(i+1).WriteArray(arr[i]) if arr[i].any() != None else print('波段{}为空！'.format(i+1))
        # del out_tif
        return

    def SIFT(self,img_t1,img_t2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_t1,None)
        kp2, des2 = sift.detectAndCompute(img_t2,None)
        # 创建设置FLASNN匹配
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
        search_params = dict(check=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < n.distance * 0.7:
                good.append(m)
        MIN_MATCH_COUNT = 10 #设置特征点最低匹配数量为10
        if len(good) > MIN_MATCH_COUNT:
            # 获取关键点坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            # 计算变换矩阵和MASK
            M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,3.0) #单应性矩阵
            # TODO write "img_out.tif"
            img_out = cv2.warpPerspective(img_t2,M,img_t1.transpose(1,0,2).shape[:2],flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            self.writeTif(img_out,self.t1_proj,self.t1_geotrans,self.out_name)
            matchMasks = mask.ravel().tolist()
            self.calRMSE(src_pts,dst_pts,M,mask) #计算匹配精度
        else:
            print('缺少足够的特征点-%d/%d'%(len(good),MIN_MATCH_COUNT))
            matchMasks = None
        draw_params = dict(matchColor=(0,255,255),
                            singlePointColor=None,
                            matchesMask = matchMasks,
                            flags=2)
        row_t1, col_t1 = img_t1.shape[:2]
        row_t2, col_t2 = img_t2.shape[:2]
        img_show = np.empty((max(row_t1,row_t2),col_t1+col_t2))
        img_show = cv2.drawMatches(img_t1,kp1,img_t2,kp2,good,None,**draw_params)
        plt.imshow(img_show),plt.show()
    
    def calRMSE(self,src_pts,dst_pts,M,mask):
        # 求残差
        sum_H = 0 #残差和
        num = 0 #参与统计的总个数
        for i,j,m in zip(src_pts,dst_pts,mask):
            P_src = np.float32([i[0][0],i[0][1],1]).reshape((-1,1))
            P = np.matmul(M,P_src) #通过计算出的矩阵预测点
            p = np.float32([P[0]/P[2],P[1]/P[2]])
            j = j.T
            distance = np.linalg.norm(p-j)
            if m == True:
                sum_H += distance
                num += 1
        rmse = cmath.sqrt(sum_H/num)
        print('rmse:',rmse)
        return rmse

    def run(self):
        load_img = self.load_img
        tif16_to_8bit = self.tif16_to_8bit
        SIFT = self.SIFT
        t1_arr, self.t1_proj, self.t1_geotrans = load_img(self.t1_path)
        t2_arr, self.t2_proj, self.t2_geotrans = load_img(self.t2_path)
        t1_arr = tif16_to_8bit(t1_arr.transpose(1,2,0))
        t2_arr = tif16_to_8bit(t2_arr.transpose(1,2,0))
        begin_time = time()
        SIFT(t1_arr,t2_arr)
        end_time = time()
        run_time = end_time-begin_time
        print('数据处理完毕，耗时：{}s'.format(run_time))

if __name__ == "__main__":
    t1_path = r''
    t2_path = r''
    out_dir = r''
    registry = Co_registry(t1_path,t2_path,out_dir)
    registry.run()
