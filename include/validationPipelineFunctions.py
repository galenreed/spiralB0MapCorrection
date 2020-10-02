
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2
from myshow import myshow
import sys
from scipy import linalg
from scipy.io import loadmat, savemat
import cardiacDicomGlobals as cdg

def closestIndex(f, faxis):
    return np.argmin(np.abs(faxis-f))



def floatToInt2(img):
    img = np.uint8(255 * (img / img.max()))
    return img

def imageGrad(img,kernelSize):
    #$laplacian = cv2.Laplacian(testImg, cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernelSize)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernelSize)
    sobelxy = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
    return sobelxy

def imageThresh(img):
    # global thresholding
    ret1,th1 = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
    
    # otsu thresholding
    #ret2,th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return th1

def imageAnd(im1,im2):
    return cv2.bitwise_and(im1,im2)

def meanUpperQuantileImage(img, mask, q):
    # select mask pixels
    img = img[mask == mask.max()]
    img = img.flatten()
    
    # pick the upper quartile of pixels
    uq = np.quantile(img, q)
    uqpixels = img[img > uq]
    return np.mean(uqpixels)

def sumImaginaryComponent(img, mask):
    imgAbsImag = np.abs(np.imag(img))
    imgMaskPixels = imgAbsImag[mask>0]
    return np.sum(imgMaskPixels)

# works ok
def gradientObjectiveFunction(img, kernelSize, mask):
    imgGrad = imageGrad(img,kernelSize)
    return meanUpperQuantileImage(imgGrad,mask, .75)

# doesn't work too well
def gradientObjectiveFunction2(img, kernelSize, mask):
    imgGrad = imageGrad(img,kernelSize)
    gradInt2 = floatToInt2(grad)
    thresh = imageThresh(gradInt2)
    jointMask = imageAnd(thresh,mask)
    imgMaskPixels = img[jointMask>0]
    uq = np.quantile(img, .75)
    return np.mean(imgMaskPixels)

    
def calcSNR(img, noise):
    noiseEst = np.std(noise[0:20,0:20])
    return (img / noiseEst)

def findPeakSignalImages(mfi, maskPixels, globalOnResInd):
    ntime = mfi.shape[2]
    numSkipEachMet = [2,0,1] # filter first numSkip images to eliminate pyruvate artifacts
    peakTimeSignal = np.zeros((3,ntime))

    # first find the image with the best signal for each metabolite
    for it in range(ntime):
        for im in range(3):                      
            numSkip = numSkipEachMet[im]
            if it > numSkip - 1: # filter first images to eliminate artifacts (pixel overrange, etc)    
                img = np.abs(mfi[:,:,it,im, globalOnResInd])
                peakTimeSignal[im, it] = meanUpperQuantileImage(img, maskPixels, .75)

    # find the peak
    peakTime = np.zeros(3)
    for im in range(3):   
        peakTime[im] = np.argmax(peakTimeSignal[im,:])
    
    return peakTime
