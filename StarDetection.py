import numpy as np
import rawpy
import cv2
import pywt 
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance as spd 

def readImage(imgPath, imgWidth, imgHeight, index):
    print('read image')
    if os.path.splitext(imgPath)[-1] == '.ARW':
        imgData = rawpy.imread(imgPath)
        imgData = imgData.postprocess(output_bps=8)
        imgPath = 'align'+ str(index) + '.jpg'
        cv2.imwrite(imgPath, imgData)
    print(imgPath)
    imgData = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('test'+str(index)+'.jpg', imgData)
    imgData = cv2.imread('test'+str(index)+'.jpg', cv2.IMREAD_GRAYSCALE)
    imgData = np.array(imgData)
    #print(imgData.shape)
    #imgData = imgData.reshape(imgHeight, imgWidth)
    return imgData

def processImage(image, index):
    print('process image')
    #小波变换
    coeffs=pywt.wavedec2(image,'db8',level=4)
    coeffs[0].fill(0)
    coeffs[-1][0].fill(0)
    coeffs[-1][1].fill(0)
    coeffs[-1][2].fill(0)
    #print(len(coeffs))
    img_rec = pywt.waverec2(coeffs,'db8') #小波变换复原
    cv2.imwrite('test_db8_'+str(index)+'.jpg', img_rec)

    #均值滤波
    img_rec = cv2.imread('test_db8_'+str(index)+'.jpg', cv2.IMREAD_GRAYSCALE)
    print(img_rec.dtype)
    img_rec = cv2.blur(img_rec, (3,3))
    cv2.imwrite('test_blur_'+str(index)+'.jpg', img_rec)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    #img_rec = cv2.dilate(img_rec, kernel, iterations=2)
    #二值化
    #img_rec = img_rec.astype(np.uint16)
    _, img_rec_bin = cv2.threshold(img_rec, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #img_rec_bin = img_rec_bin.astype(np.uint8)
    cv2.imwrite('test_bin_'+str(index)+'.jpg', img_rec_bin)
    return img_rec_bin

def extractFeature(image, mask_region, k, index): #k: 统计临近星星个数
    print('extract feature')
    #膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    bin_clo = cv2.dilate(image, kernel, iterations=2)
    cv2.imwrite('test_bin_clo'+str(index)+'.jpg', bin_clo)
    #print(index, ' ', bin_clo[1][2775])
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
    img_star = np.zeros_like(image, np.uint8)
    star_centroids=[]
    for i in range(1, num_labels):

        if stats[i][-1]>180 or stats[i][-1]<60:
            continue
        if mask_region[0][0]<centroids[i][0]<mask_region[0][1] \
        and mask_region[1][0]<centroids[i][1]<mask_region[1][1]:
            mask = labels == i
            img_star[mask] = 255
            star_centroids.append(centroids[i])
            #print(stats[i, 0], stats[i, 1])
    star_centroids = np.array(star_centroids)
    print(len(star_centroids))
    n_stars = len(star_centroids)
    dist_mat = spd.cdist(star_centroids, star_centroids, 'euclidean')
    vec_mat = star_centroids.repeat(n_stars,axis=0).reshape(n_stars, n_stars, star_centroids.shape[-1])-star_centroids
    dist_ind = np.argsort(dist_mat)
    dist_close_mat = dist_mat[np.array(range(n_stars))[:,np.newaxis], dist_ind[:,1:k+1]]
    vec_close_mat = vec_mat[np.array(range(n_stars))[:,np.newaxis], dist_ind[:,1:k+1]]
    angle_mat = np.zeros((n_stars, k))
    for i in range(n_stars):
        for j in range(k):
            angle_mat[i][j] = np.arccos(np.dot(vec_close_mat[i][j], vec_close_mat[i][0])/(np.linalg.norm(vec_close_mat[i][j])*np.linalg.norm(vec_close_mat[i][0])))

    feature={}
    feature['star_centroids'] = star_centroids #星点的中心位置
    feature['vec_mat']=vec_close_mat #距离最近的k个星星的方向
    feature['dist_mat']=dist_close_mat #距离最近的k个星星的距离
    feature['angle_mat']=angle_mat #距离最近的k个星星分别和距离最近的一个星星的夹角
    return feature

def starPointMatch(feature1, feature2, n):
    print('match star point')
    distMat1 = feature1['dist_mat']
    distMat2 = feature2['dist_mat']
    print(distMat1, distMat2)
    diffOfDistance = spd.cdist(distMat1, distMat2, 'cityblock')
    minIndicesOfEachStar = np.argmin(diffOfDistance, axis=1)
    print(minIndicesOfEachStar)
    topNIndices = np.argpartition(diffOfDistance.min(axis=1), n)[:n]
    matchPositions = []
    for _, index in enumerate(topNIndices):
        matchPositions.append([index, minIndicesOfEachStar[index]])
    return matchPositions # [pos_in_img1, pos_in_img2]

IMG_WIDTH = 4024
IMG_HEIGHT = 6024
K = 5
mask_region = [[0,4024], [0, 3012]]
MATCH_NUM = 3
imgPath1 = '**.ARW'
imgPath2 = '***.ARW'
img1 = readImage(imgPath1, IMG_WIDTH, IMG_HEIGHT, 1)
starExtractedImg1 = processImage(img1, 1)
cv2.imwrite('test1.jpg', starExtractedImg1)
feature1 = extractFeature(starExtractedImg1, mask_region, K, 1)

img2 = readImage(imgPath2, IMG_WIDTH, IMG_HEIGHT, 2)
starExtractedImg2 = processImage(img2, 2)
cv2.imwrite('test2.jpg', starExtractedImg2)
feature2 = extractFeature(starExtractedImg2, mask_region, K, 2)

matchedStars = starPointMatch(feature1, feature2, MATCH_NUM)
starCentroids1 = feature1['star_centroids']
starCentroids2 = feature2['star_centroids']
coordinates1 = np.array([list(starCentroids1[matchedStars[i][0]]) for i in range(MATCH_NUM)], dtype=np.float32)
coordinates2 = np.array([list(starCentroids2[matchedStars[i][1]]) for i in range(MATCH_NUM)], dtype=np.float32)
print(coordinates1, coordinates2)
affine_matrix = cv2.getAffineTransform(coordinates1, coordinates2)
print(affine_matrix)

raw = rawpy.imread(imgPath1)

rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=8)
cv2.imwrite('postprocess.jpg', rgb)
output_image = cv2.warpAffine(rgb, affine_matrix, (IMG_WIDTH, IMG_HEIGHT))
cv2.imwrite('output.jpg', output_image)

'''
height=6024
width=4024
channels=3
imgData=np.fromfile('D:\\Files\\照片\\星空\\DSC02434.ARW',dtype=np.uint16)
imgData = imgData.reshape(height, width, channels)
imgData=cv2.cvtColor(imgData,cv2.COLOR_BGR2RGB)
'''
'''
imgData = rawpy.imread('D:\\Files\\照片\\星空\\DSC02434.ARW')
imgData = imgData.postprocess()
imgData = cv2.cvtColor(imgData,cv2.COLOR_RGB2GRAY)
imgData = imgData.reshape(6024, 4024)
cv2.imwrite('test_gray.jpg', imgData)

# 对img进行haar小波变换：
coeffs=pywt.wavedec2(imgData,'db8',level=4)
coeffs[0].fill(0)
coeffs[-1][0].fill(0)
coeffs[-1][1].fill(0)
coeffs[-1][2].fill(0)
print(len(coeffs))
img_rec = pywt.waverec2(coeffs,'db8')
cv2.imwrite('test_db8.jpg', img_rec)
img_rec = cv2.imread('test_db8.jpg', cv2.IMREAD_GRAYSCALE)
img_rec = cv2.blur(img_rec, (3,3))
#img_rec = cv2.normalize(img_rec,None,0,255,cv2.NORM_MINMAX)
#img_rec = np.uint8(np.power(img_rec, 1.5))
cv2.imwrite('test_db8_2.jpg', img_rec)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
ret, img_rec_bin = cv2.threshold(img_rec, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#print(img_rec.shape)
#hist = cv2.calcHist([img_rec], [0], None, [256], [0,256])
#plt.plot(hist)
#  标记x轴的名称
#plt.xlabel('pixel value')
# 显示直方图
#plt.show()
mask_region = [[0,4024], [0, 3012]]
cv2.imwrite('test_bin.jpg', img_rec_bin)
bin_clo = cv2.dilate(img_rec_bin, kernel, iterations=2)
cv2.imwrite('test_dil.jpg', bin_clo)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
print('stats = ',stats)
# 连通域的中心点
print('centroids = ',centroids)
img_star = np.zeros_like(img_rec, np.uint8)
star_centroids=[]
for i in range(1, num_labels):

    if stats[i][-1]>40:
        continue
    if not( mask_region[0][0]<centroids[i][0]<mask_region[0][1] \
     and mask_region[1][0]<centroids[i][1]<mask_region[1][1]):
        continue
    mask = labels == i
    img_star[mask] = 255
    star_centroids.append(centroids)
cv2.imwrite('test_star.jpg', img_star)

k=5
print(len(star_centroids))
n_stars = len(star_centroids)
dist_mat = spd.cdist(star_centroids, star_centroids)
vec_mat = star_centroids.repeat(n_stars,axis=0).reshape(n_stars, n_stars, star_centroids.shape[-1])-star_centroids
dist_ind = np.argsort(dist_mat)
dist_close_mat = dist_mat[np.array(range(n_stars))[:,np.newaxis], dist_ind[:,1:k+1]]
vec_close_mat = vec_mat[np.array(range(n_stars))[:,np.newaxis], dist_ind[:,1:k+1]]
angle_mat = np.zeros((n_stars, k))
for i in range(n_stars):
    for j in range(k):
        angle_mat[i][j] = np.argcos(np.dot(vec_close_mat[i][j], vec_close_mat[i][0])/(np.linalg.norm(vec_close_mat[i][j])*np.linalg.norm(vec_close_mat[i][0])))

feature={}
feature['vec_mat']=vec_close_mat #距离最近的k个星星的方向
feature['dist_mat']=dist_close_mat #距离最近的k个星星的距离
feature['angle_mat']=angle_mat #距离最近的k个星星分别和距离最近的一个星星的夹角

#imgA = np.uint8(cA/np.max(cA)*255)
#imgA = imgA.reshape(6024, 4024, 1)
#imgH = np.uint8(cH/np.max(cH)*255)
#imgH = imgH.reshape(6024, 4024, 1)
#print(imgH.channels())
##cv2.imwrite('test_A.jpg', imgA)
#cv2.imwrite('test_h.jpg', imgH)    
#imgGray = cv2.cvtColor(imgData,cv2.COLOR_RGB2GRAY)
#imgGray = imgGray.reshape(6024,4024,1)
#cv2.imwrite('test_diff.jpg', imgGray-imgA)

#cv2.imshow('img', imgData)
#cv2.waitKey()

'''