
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import cv2
from myshow import myshow
import sys
from scipy import linalg
from scipy.io import loadmat, savemat
import cardiacDicomGlobals as cdg
import validationPipelineFunctions as validate
import scipy.ndimage


channelList = [5,6,7]


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
    resample.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress:{0:03.1f}%...".format(100*resample.GetProgress()),end=''))
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

def loadMultiFrequencyFiles(currentPatient, freqAxis, isExample):


    
    #load mat files for multi frequency recon
    matFileBase = currentPatient.mfr
    if isExample:
        matFileBase = 'exampleData/freqReconstructions/' + matFileBase
    else:
        matFileBase = 'matFilesIntermediate/' + matFileBase
        
    sampleFile = matFileBase + '_f' + str(1) + '.mat'
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
    if currentPatient.multichannel:
        ncoils = arrayShape[4]
        mfrc = np.zeros((nx, ny, ntime, nmet, ncoils, nf), dtype = np.cdouble)
    else:
        mfrc = np.zeros((nx, ny, ntime, nmet, nf), dtype = np.cdouble)

    print("image array shape")
    print(mfr.shape)

    for f in range(len(freqAxis)):

        #print('on frequency '+str(f+1)+' of ' + str(len(freqAxis)))
        currentFile = matFileBase + '_f' + str(f+1) + '.mat'
        if currentPatient.multichannel:
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


# creates numThetaBins masks at the resolution of img
# these masks are segments of a circle of width circleThickness 
def divideMyocardiumIntoBins(img, circleCoords, numThetaBins, circleThickness):
    circleCenterX = circleCoords[0]
    circleCenterY = circleCoords[1]
    circleRadius = circleCoords[2]
    startX = np.int(np.floor(circleCenterX - circleRadius - circleThickness/2))
    endX   = np.int(np.ceil(circleCenterX + circleRadius + circleThickness/2))
    startY = np.int(np.floor(circleCenterY - circleRadius - circleThickness/2))
    endY   = np.int(np.ceil(circleCenterY + circleRadius + circleThickness/2))
    dTheta = 2*np.pi/numThetaBins
    thetaSamples = np.linspace(-np.pi, np.pi-dTheta, numThetaBins)
    meanVals = np.zeros(numThetaBins)
    binImages = np.zeros([numThetaBins, img.shape[0], img.shape[1]])

    for thetaInc in range(0, numThetaBins):
        # brute force search to see what's inside the arc segment
        pixelsInArc = []
        for xloc in range(startX, endX):
            for yloc in range(startY, endY):
                relativeX = xloc-circleCenterX
                relativeY = yloc-circleCenterY
                thisRadius = np.sqrt(relativeX**2 + relativeY**2)
                thisAngle = np.arctan2(relativeY, relativeX)
            
                if thisRadius > (circleRadius - circleThickness/2):
                    if thisRadius < (circleRadius + circleThickness/2):
                        if(thisAngle > thetaSamples[thetaInc]):
                            if(thisAngle < thetaSamples[thetaInc] + dTheta):
                                pixelsInArc.append(img[xloc][yloc])
                                binImages[thetaInc][xloc][yloc] = 1                
    return binImages



# start with a BGR color loc
# binImages generated by divideMyocardium()
def drawCardiacBinContours(loc, binImages):
    numThetaBins = len(binImages)
    for x in range(0, numThetaBins): 
        singleBinImage = binImages[x][:][:]
        singleBinImage = np.transpose(singleBinImage)
        singleBinImage  = validate.floatToInt2(singleBinImage)
        ret,thresh = cv2.threshold(singleBinImage, 127, 255, 0)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contourlw = 1
        cv2.drawContours(loc, contours, -1, (0,255,0), contourlw)
    return loc


# binImages generated by divideMyocardium()
# use a floating and target sitk image for resampling each of binImages
# this is required since the circles are defined in loc coords but applied in c13 coords
def resampleCardiacBinImages(binImages, loc_sitk, target_sitk):
    numThetaBins = len(binImages)  
    binImages_res = []
    for x in range(0, numThetaBins): 
        # need to add a singleton dimension for unused slice axis
        singleBinImage = np.expand_dims(binImages[x][:][:], axis=0)

        # convert binaryImage segments to SITK format with header
        singleBinImage = sitk.GetImageFromArray(singleBinImage)
    
        # copy header from the localizer
        singleBinImage.CopyInformation(loc_sitk)

        # resample to C13 locations
        singleBinImage = resampleSITKImage(target_sitk, singleBinImage)

        # extract the pixels back from SITK structure 
        singleBinImage = np.squeeze(sitk.GetArrayFromImage(singleBinImage))
    
        # set approximately 0 to just 0
        singleBinImage[singleBinImage < .2 * singleBinImage.max()] = 0

        # float64 -> int2
        singleBinImage = validate.floatToInt2(singleBinImage)
    
        binImages_res.append(singleBinImage)
    return binImages_res


#
#this section of code is just to translate the center from loc coords to c13 coords
#this is done by making an impulse image at the center then resampling and finding the max
def resampleCircleCenterCoords(circleCenterCoords, imgsource, imgtarget):
    locPixels = np.squeeze(sitk.GetArrayFromImage(imgsource))
    centerImpulse = np.zeros(locPixels.shape, dtype=np.uint16)
    centerImpulse[circleCenterCoords[0], circleCenterCoords[1]] = 255
    centerImpulse = np.expand_dims(centerImpulse, axis=0)
    centerImpulse = sitk.GetImageFromArray(centerImpulse)
    centerImpulse.CopyInformation(imgsource)
    centerImpulse = resampleSITKImage(imgtarget, centerImpulse)
    centerImpulse = np.squeeze(sitk.GetArrayFromImage(centerImpulse))
    maxInd = centerImpulse.argmax()
    circleLocsC13Coords = np.unravel_index(maxInd, centerImpulse.shape)    
    return circleLocsC13Coords

def makeLineProfiles(startPoint, endPoint, img, num):
    x0 = startPoint[0]
    y0 = startPoint[1]
    x1 = endPoint[0]
    y1 = endPoint[1]
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    zi = scipy.ndimage.map_coordinates(img, np.vstack((x,y)))
    return zi


def generateRadialEndpoints(circleCenter, numThetaBins, segRadius):
    dTheta = 2*np.pi/numThetaBins
    theta = np.linspace(0, 2*np.pi-dTheta, numThetaBins)
    endPoints = np.zeros([numThetaBins, 2])
    for x in range(numThetaBins):
        endPoints[x,0] = circleCenter[0] + segRadius*np.cos(theta[x])
        endPoints[x,1] = circleCenter[1] + segRadius*np.sin(theta[x])
    return endPoints


def fwhm(signal):
    maxVal = signal.max()
    minVal = signal.min()
    indMax = np.argmax(signal)

    if maxVal < 5: # too low SNR for this to really work
        return -1
    
    # extremum is at the edge, so data probably not that interesting
    if (indMax == 0) or (indMax == len(signal)-1):
        return -1
    
    
    found = False
    upIndex = indMax
    downIndex = indMax
    
    while found == False:
        upIndex += 1
        if signal[upIndex] <= maxVal/2:
            found = True
        if upIndex == len(signal)-1:# never found half max
            return -1
            
    found = False
    while found == False:
        downIndex -= 1
        if signal[downIndex] <= maxVal/2:
            found = True
        if downIndex == 0: # never found half max
            return -1
    
    #print('half max found at '+str(upIndex))
    #print('half max found at '+str(downIndex))
    return(upIndex-downIndex)



def generateLineProfilesBinned(img, NBins, NRadialSamples, centerCoords, segRadius):
    #numDegrees = 360
    numDegrees = NBins
    img = img / np.std(img[1:20,1:20])
    zi = np.zeros([NBins, NRadialSamples])
    zi_fine = np.zeros([numDegrees, NRadialSamples])
    endPointsFine = generateRadialEndpoints(centerCoords, numDegrees, segRadius)

    # generate line profiles
    for ind in range(numDegrees):
        x1 = endPointsFine[ind,0]
        y1 = endPointsFine[ind,1]
        zi_fine[ind][:] = makeLineProfiles(centerCoords, (x1,y1), img, NRadialSamples)  
    
    # accumulate to coarses theta resolution
    lineAccumulator = np.zeros(NRadialSamples)
    slowCounter = 0
    projectionsPerBin = np.floor(numDegrees/NBins)
    for ind in range(numDegrees):
        lineAccumulator += zi_fine[ind][:]
        if(int(ind % projectionsPerBin) == projectionsPerBin-1):
            zi[slowCounter][:] = lineAccumulator
            lineAccumulator = lineAccumulator*0.0
            slowCounter += 1
            
    return zi

