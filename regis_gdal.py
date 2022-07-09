from osgeo import gdal,gdal_array
import numpy as np
import cv2
import cmath
from matplotlib import pyplot as plt
from time import *
from imageio import imwrite,imread
import os.path as osp

def loadtif(tifpath):
    # if osp.splitext(tifpath)[-1] == '.tif':
    #     return gdal_array.LoadFile(tifpath)
    # else:
    #     img = imread(tifpath)
    #     print(img.shape)
    #     return img
    return gdal_array.LoadFile(tifpath)

def readTIFF(tifpath, bandnum):
    """
    Use GDAL to read data and transform them into arrays.
    :param tifpath:tif文件的路径
    :param bandnum:需要读取的波段
    :return:该波段的数据, narray格式。len(narray)是行数, len(narray[0])列数
    """
    image = gdal.Open(tifpath)  # 打开影像
    if image == None:
        print(tifpath + "该tif不能打开!")
        return
    im_width = image.RasterXSize  # 栅格矩阵的列数
    im_height = image.RasterYSize  # 栅格矩阵的行数
    im_bands = image.RasterCount  # 波段数
    im_proj = image.GetProjection()  # 获取投影信息坐标系
    im_geotrans = image.GetGeoTransform()  # 仿射矩阵
    print('tif数据:{}个行，{}个列，{}层波段, 取出第{}层.'.format(im_width, im_height, im_bands, bandnum))
    im_data = image.ReadAsArray(0, 0,  im_width, im_height)
    del image  # 减少冗余
    return im_data,im_proj, im_geotrans
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def Tiff16to8bit(img_16):
    print(type(img_16))
    if (np.max(img_16) - np.min(img_16) != 0):
        # img_nrm = (img_16 - np.min(img_16)) / (np.max(img_16) - np.min(img_16)) #计算灰度范围,归一化
        img_nrm = normalization(img_16)
        img_8 = np.uint8(255 * img_nrm)
    return img_8

def imagexy2geo(trans, row, col):
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py
def geo2imagexy(trans, x, y):
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def detect_computate(img):
    sift = cv2.SIFT_create()
    print(img.shape)
    kp = sift.detect(img,None)
    des = sift.compute(img,kp)
    return kp,des

def SIFT(img_l, img_r):
    sift = cv2.SIFT_create()
    # kp1, des1 = detect_computate(img_l)
    # kp2, des2 = detect_computate(img_r)
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_r, None)
    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # print(des1.shape,des2.shape)
    matches = flann.knnMatch(des1, des2, k=2)
    print(len(matches))
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7的匹配,初步筛除
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)#单应性矩阵
        img_out = cv2.warpPerspective(img_r,M,img_l.transpose(1,0,2).shape[:2],flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        #M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1.0)#本征矩阵
        matchesMask = mask.ravel().tolist()
        calRMSE(src_pts, dst_pts, M, mask)#计算匹配算的精度)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 255),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    row_l, col_l = img_l.shape[:2]
    row_r, col_r = img_r.shape[:2]
    img_show = np.empty((max(row_l, row_r), col_l + col_r))
    img_show = cv2.drawMatches(img_l,kp1,img_r,kp2,good,None,**draw_params)
    plt.imshow(img_show), plt.show()
    img = cv2.addWeighted(img_l,0.5,img_out,0.5,gamma=1)
    plt.imshow(img),plt.show()

def calRMSE(src_pts,dst_pts,M,mask):
    # 求残差
    sum_H = 0 #残差和
    num = 0 #参与统计的总个数
    for i, j, m in zip(src_pts, dst_pts, mask):
        P_src = np.float32([i[0][0],i[0][1],1]).reshape((-1, 1))
        P = np.matmul(M, P_src) #通过计算出的矩阵预测点
        p = np.float32([P[0] / P[2], P[1] / P[2]]) #从齐次坐标变为2维点
        j = j.T
        distance = np.linalg.norm(p - j)
        if (m == True):
            sum_H += distance
            num += 1
    rmse = cmath.sqrt(sum_H/num)
    print("rmse : ",rmse)
    return rmse

B = loadtif(r'')
F = loadtif(r'')
# F = np.resize(F,B.shape) if B.shape != F.shape else None #这句话有他妈大病
# x_offset,y_offset = 500,600
# block_size_x = 1000
# block_size_y = 1000

# block_B = B[y_offset:y_offset + block_size_y,x_offset:x_offset + block_size_x]
# block_F = F[y_offset:y_offset + block_size_y,x_offset:x_offset + block_size_x]
# block_B = Tiff16to8bit(block_B)
# block_F = Tiff16to8bit(block_F)
block_B = Tiff16to8bit(B[:3,:,:].transpose(1,2,0))
block_F = Tiff16to8bit(F[:3,:,:].transpose(1,2,0))
# block_B = Tiff16to8bit(B)
# block_F = Tiff16to8bit(F)
begin_time = time()
SIFT(block_B,block_F)
end_time = time()
run_time = end_time-begin_time
print ('匹配耗时运行时间：',run_time,'s') #该循环程序运行时间： 1.4201874732