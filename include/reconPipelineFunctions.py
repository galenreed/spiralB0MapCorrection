
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


def generateMFICoefficients(freqAxis, currentPatient):
    nf = len(freqAxis)
    J = 1j
    PI = np.pi
    timeAxis = loadmat(currentPatient.readout)['t']
    #timeAxis = loadmat(baseDirectory+readouts[0])['t']

    ft = np.outer(freqAxis, timeAxis)
    A = np.exp(-2*J*PI*ft)
    Ainv = linalg.pinv2(A)# seems more stable than np.linalg.pinv()
    Ainv = np.transpose(Ainv) # should be necessary? need to double check the above lines
    cjmat = np.zeros((nf, nf), dtype=np.cdouble)
    ind = 0
    for f0 in freqAxis:
        # create the column vector of exp(J_w_t_k)
        ft = f0*timeAxis
        y = np.transpose(np.exp(-2 * J * PI * ft))
        cj = np.matmul(Ainv, y)
        cjmat[ind,:] = np.squeeze(cj)
        ind += 1
    return cjmat

def getHeartMask(currentPatient, loc):
    mask = cv2.imread(currentPatient.mask, 0)
    
    #dilate mask
    kernel = np.ones((5,5),np.uint8)
    maskDilation = cv2.dilate(mask, kernel, iterations = 1)
    mask3D = np.expand_dims(maskDilation, axis=0)
    maskSITK = sitk.GetImageFromArray(mask3D)
    maskSITK.CopyInformation(loc) 
    return maskSITK

def filterB0Map(b0map):
    b0phasePixels = np.squeeze(sitk.GetArrayFromImage(b0map))
    medianFiltered = cv2.medianBlur(b0phasePixels,3)

    #make new sitk image from filtered pixel array
    medianFiltered3D = np.expand_dims(medianFiltered, axis=0)
    b0phasef = sitk.GetImageFromArray(medianFiltered3D)
    b0phasef.CopyInformation(b0map)
    return b0phasef, b0phasePixels, medianFiltered


def resampleSITKImage(static, floating):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(static)
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress: {0:03.1f}%...".format(100*resample.GetProgress()),end=''))
    resample.AddCommand(sitk.sitkProgressEvent, lambda: sys.stdout.flush())
    return resample.Execute(floating)

def getThresholdFromLoc(loc):
    threshInput = np.squeeze(sitk.GetArrayFromImage(loc))
    threshInput = np.abs(threshInput) # pixels go slightly negative somehow
    threshInput = np.uint8(255.0 * threshInput / threshInput.max())
    ret,thresh1 = cv2.threshold(threshInput,20,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    locMask = cv2.dilate(thresh1, kernel,iterations = 2)
    return locMask

def loadMultiFrequencyFiles(currentPatient, freqAxis, multiCoilProcessing):

    #load mat files for multi frequency recon
    matFileBase = currentPatient.mfr
    sampleFile= 'matFilesIntermediate/' + matFileBase + '_f' + str(1) + '.mat'
    array = loadmat(sampleFile)['bb']
    array = np.squeeze(np.cdouble(array))
    arrayShape = array.shape
    nx = arrayShape[0]
    ny = arrayShape[1]
    ntime = arrayShape[2]
    nmet = arrayShape[3]
    nf = len(freqAxis)

    #for storing abs images 
    mfr = np.zeros((nx, ny, ntime, nmet, nf), dtype = np.cdouble)

    #for storing complex images 
    if multiCoilProcessing:
        ncoils = arrayShape[4]
        mfrc = np.zeros((nx, ny, ntime, nmet, ncoils, nf), dtype = np.cdouble)
    else:
        mfrc = np.zeros((nx, ny, ntime, nmet, nf), dtype = np.cdouble)

    print("image array shape")
    print(mfr.shape)

    for f in range(len(freqAxis)):

        #print('on frequency '+str(f+1)+' of ' + str(len(freqAxis)))
        currentFile= 'matFilesIntermediate/' + matFileBase + '_f' + str(f+1) + '.mat'
        if multiCoilProcessing:
            pixelArrayComplex = loadmat(currentFile)['bb']
            pixelArrayComplex = np.squeeze(np.cdouble(pixelArrayComplex))
            mfrc[:,:,:,:,:,f] = pixelArrayComplex

            # do the SOS here in the magnitude channel
            for coil in range(ncoils):       
                if coil in channelList:
                      mfr[:,:,:,:,f] = mfr[:,:,:,:,f] + np.abs(pixelArrayComplex[:,:,:,:,coil])   
            mfr[:,:,:,:,f] = np.sqrt(mfr[:,:,:,:,f] / len(channelList) )

        else:
            #pixelArray = loadmat(currentFile)['bbabs']
            pixelArrayComplex = loadmat(currentFile)['bb']
            #mfr[:,:,:,:,f] = pixelArray
            mfr[:,:,:,:,f] = pixelArrayComplex
            mfrc[:,:,:,:,f] = pixelArrayComplex
    return mfr, mfrc


def runMFI(globalFreqSearch, freqAxis, b0phase_r, mask, mfr, mfrc, cjmat, multiCoilProcessing):


    globalOnResInd = cdg.closestIndex(0, globalFreqSearch)
    onResInd = closestIndex(0, freqAxis)

    arrayShape = mfr.shape
    nx = arrayShape[0]
    ny = arrayShape[1]
    ntime = arrayShape[2]
    nmet = arrayShape[3]
    nf = len(freqAxis) # = arrayShape[4]
    ngf = len(globalFreqSearch)
    mf = np.zeros((nx, ny, ntime, nmet, ngf), dtype = np.cdouble)
    mfi = np.zeros((nx, ny, ntime, nmet, ngf), dtype = np.cdouble)



    #grab the maps and masks, apply appropriate scaling
    b0map = np.squeeze(sitk.GetArrayFromImage(b0phase_r))
    b0map =  b0map * 1.070 / 4.257


    for it in range(ntime):
        for im in range(nmet):
            for ix in range(nx):
                for iy in range(ny):
                    for gf in range(ngf):

                        globalShift = globalFreqSearch[gf]

                        #if False:
                        if mask[ix,iy] == 0:
                            mf[ix,iy,it,im,gf]  = mfr[ix,iy,it,im,onResInd]
                            mfi[ix,iy,it,im,gf] = mfr[ix,iy,it,im,onResInd]

                        else:
                            localFreq = b0map[ix,iy] 
                            indF = cdg.closestIndex(localFreq + globalShift, freqAxis)
                            MFICoeffs = np.squeeze(cjmat[indF,:])             

                            # segmented
                            mf[ix,iy,it,im,gf] = mfr[ix,iy,it,im,indF]

                            # since MFI is phase sensitive, have to apply channelwise
                            if multiCoilProcessing:
                                for coil in channelList:
                                    pixelFrequencyList = np.squeeze(mfrc[ix,iy,it,im,coil,:])
                                    weightedSum = np.dot(MFICoeffs, pixelFrequencyList)
                                    mfi[ix,iy,it,im,gf] = mfi[ix,iy,it,im,gf] + np.abs(weightedSum)
                                mfi[ix,iy,it,im,gf] = np.sqrt(mfi[ix,iy,it,im,gf] / len(channelList))
                            else:
                                # MFI, single channel
                                pixelFrequencyList = np.squeeze(mfrc[ix,iy,it,im,:])
                                MFICoeffs = np.squeeze(cjmat[indF,:])             
                                weightedSum = np.dot(MFICoeffs, pixelFrequencyList)
                                #mfi[ix,iy,it,im,gf] = np.abs(weightedSum)
                                mfi[ix,iy,it,im,gf] = weightedSum

    return mf, mfi, globalFreqSearch

