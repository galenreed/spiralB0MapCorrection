
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

def calcSNR(img, noise):
    noiseEst = np.std(noise[0:10,0:10])
    return (img / noiseEst)


def floatToInt2(img):
    img = np.uint8(255 * (img / img.max()))
    return img

def imageGrad(img,kernelSize):
    #$laplacian = cv2.Laplacian(testImg, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernelSize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernelSize)
    sobelxy = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
    return sobelxy

def imageThresh(img):
    imgint2 = floatToInt2(img)
    ret1,th1 = cv2.threshold(imgint2,50,255,cv2.THRESH_BINARY)
    #ret2,th2 = cv2.threshold(grdint2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    return opening


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
    return meanUpperQuantileImage(imgGrad, mask, .50)
    #return meanUpperQuantileImage(imgGrad,mask, .75)
    
def tenengradSharpness(img, kernelSize):
    imgGrad = imageGrad(img,kernelSize)
    th1 = imageThresh(imgGrad)
    
    pixels = imgGrad[th1>0]
    
    # tenengrad
    #return np.mean(pixels*pixels)
    
    # sobel variance
    return np.var(pixels)
    
    
def gradientObjectiveFunctionSNRConstraint(img, kernelSize, mask):
    snrCutoff = 5
    imgGrad = imageGrad(img, kernelSize)
    strongGradientPixels = imgGrad[(img>snrCutoff) & (mask>0)]
    #uq = np.quantile(strongGradientPixels, 0.5)
    return np.mean(strongGradientPixels) 
    

# doesn't work too well
def gradientObjectiveFunction2(img, kernelSize, mask):
    imgGrad = imageGrad(img,kernelSize)
    gradInt2 = floatToInt2(grad)
    thresh = imageThresh(img)
    jointMask = imageAnd(thresh,mask)
    imgMaskPixels = img[jointMask>0]
    uq = np.quantile(img, .75)
    return np.mean(imgMaskPixels)

    

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



def applyObjectiveMetric(freqAxis, globalFreqSearch, mfr, mfi, mf, peakTime, maskPixels):
    kernelSize = 3
    ngf = len(globalFreqSearch)

    peakSignalUncorr = np.zeros((3,ngf))
    peakSignalSeg = np.zeros((3,ngf))
    peakSignalMFI = np.zeros((3,ngf))

    maxSigUncorr = np.zeros((3,ngf))
    maxSigSeg = np.zeros((3,ngf))
    maxSigMFI = np.zeros((3,ngf))

    
    for gf in range(ngf):

        indf = closestIndex(globalFreqSearch[gf], freqAxis)

        for im in range(3):                      

            im_unc = mfr[:,:, int(peakTime[im]), im, indf]
            im_mfi = mfi[:,:, int(peakTime[im]), im, gf]
            im_seg =  mf[:,:, int(peakTime[im]), im, gf]
            noise = mf[:,:, 7, im, gf]
            im_unc  = calcSNR(im_unc, noise)
            im_mfi  = calcSNR(im_mfi, noise)
            im_seg  = calcSNR(im_seg, noise)


            maxSigUncorr[im,gf] = meanUpperQuantileImage(np.abs(im_unc), maskPixels, .75)
            maxSigSeg[im,gf]    = meanUpperQuantileImage(np.abs(im_mfi), maskPixels, .75)
            maxSigMFI[im,gf]    = meanUpperQuantileImage(np.abs(im_seg), maskPixels, .75)

            #gradient sharpness based processing
            #peakSignalUncorr[im,gf] = gradientObjectiveFunction(np.abs(im_unc), kernelSize, maskPixels)
            #peakSignalMFI[im,gf]    = gradientObjectiveFunction(np.abs(im_mfi), kernelSize, maskPixels)
            #peakSignalSeg[im,gf]    = gradientObjectiveFunction(np.abs(im_seg), kernelSize, maskPixels)
            
            peakSignalUncorr[im,gf] = tenengradSharpness(np.abs(im_unc), kernelSize)
            peakSignalMFI[im,gf]    = tenengradSharpness(np.abs(im_mfi), kernelSize)
            peakSignalSeg[im,gf]    = tenengradSharpness(np.abs(im_seg), kernelSize)
    

    
    
    return peakSignalUncorr, peakSignalMFI, peakSignalSeg
