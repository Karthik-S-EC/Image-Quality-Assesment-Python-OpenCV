import cv2
import skimage
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
import numpy as np
import sys

def averaging(img):
    return cv2.blur(img, (5,5))

def gaussianBlurr(img,k):
    return cv2.GaussianBlur(img, (k,k), 0)

def bilateralfilter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def motion_blur(img):
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(img, -1, kernel_motion_blur)

def detectBlur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def plotImages(img,ttl, r, c, i):
    plt.subplot(r,c,i)
    img1 = img.copy()
    img1[:, :, 0] = img[:, :, 2]
    img1[:, :, 2] = img[:, :, 0]
    plt.imshow(img1)
    plt.title(ttl)
    plt.axis('off')

def medianFilter(img):
    return cv2.medianBlur(img, 3)

def meanFilter(img):
    kernel = np.ones((3,3),np.float32)/9
    return cv2.filter2D(img, -1, kernel)

def main(img1):
    img = cv2.imread(img1)
    imglist=[]
    imglist.append(averaging(img))
    imglist.append(gaussianBlurr(img,5))
    imglist.append(gaussianBlurr(img,11)) 
    imglist.append(bilateralfilter(img))
    gauss_noise = skimage.util.random_noise(img, mode='gaussian',var=0.01)
    gauss_noise = (gauss_noise*255).astype(int)
    imglist.append(medianFilter(gauss_noise)) 
    #imglist.append(medianFilter(s_and_p))
    imglist.append(meanFilter(gauss_noise)) 
    #imglist.append(meanFilter(s_and_p))

    print(imglist)

main(sys.argv[1])
