import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt


class patientExam:
    def __init__(self):
        self.loc = []
        self.c13 = []
        self.b0mag = []
        self.b0phase = []
        self.mfr = []
        self.readout = []
         
def horosPath(num):
    outStr =  '/Users/galenreed/Documents/Horos Data/DATABASE.noindex/10000/' + str(num) + '.dcm'
    return outStr 

def itkImageToInt(im):
    im = sitk.GetArrayFromImage(im)
    im = np.squeeze(im)
    im = np.uint8(255 * (im / im.max()))
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    return im

def plotC13Mat(arr):
    arrShape = arr.shape
    fig, axes = plt.subplots(arrShape[3], arrShape[2], figsize=(20,9))
    for x in range(0, arrShape[2]): 
        for y in range(0, arrShape[3]):
            thisIm = arr[:,:,x,y]
            axes[y][x].imshow(thisIm, cmap='gray')
            axes[y][x].axis('off')
            
def closestIndex(f, faxis):
    return np.argmin(np.abs(faxis-f))
    

# mat files used for storing readout parameters
baseDirectory = '/Users/galenreed/Documents/UTSW/cardiacProject/volunteerScans/'
readouts = []
readouts.append(baseDirectory + 'waveformFiles/spiral_13C_fov400_npix66_arms1_ksamp8_gmax23_smax74.mat')
readouts.append(baseDirectory + 'waveformFiles/spiral1armshort.mat')
    
    

P20200721L = patientExam()
P20200721L.loc = horosPath(7370)
P20200721L.c13 = horosPath(7412)
P20200721L.b0phase = horosPath(7397)
P20200721L.b0mag = horosPath(7419)
P20200721L.mask = 'masks/P2020_0721_long.png'
P20200721L.mfr = 'P10'
P20200721L.readout = readouts[0]

P20200721S = patientExam()
P20200721S.loc = horosPath(7370)
P20200721S.c13 = horosPath(7396)
P20200721S.b0phase = horosPath(7397)
P20200721S.b0mag = horosPath(7419)
P20200721S.mask = 'masks/P2020_0721_long.png'
P20200721S.mfr = 'P11'
P20200721S.readout = readouts[1]

P20200715L = patientExam()
P20200715L.loc = horosPath(7086)
P20200715L.c13 = horosPath(7240)
P20200715L.b0phase = horosPath(7062)
P20200715L.b0mag  = horosPath(7043)
P20200715L.mask = 'masks/P2020_0715.png'
P20200715L.mfr = 'P8'
P20200715L.readout = readouts[0]


P20200715S = patientExam()
P20200715S.loc = horosPath(7086)
P20200715S.c13 = horosPath(7054)
P20200715S.b0phase = horosPath(7062)
P20200715S.b0mag  = horosPath(7043)
P20200715S.mask = 'masks/P2020_0715.png'
P20200715S.mfr = 'P9'
P20200715S.readout = readouts[1]


# low bicarb, repeated twice and slightly higher the second time. 
P20200123S = patientExam()
P20200123S.loc = horosPath(3566)
P20200123S.c13 = horosPath(3520)
P20200123S.b0mag = horosPath(3539)
P20200123S.b0phase = horosPath(3543)
P20200123S.mask = 'masks/P2020_0123.png'
P20200123S.mfr = 'P7'
P20200123S.readout = readouts[1]

#12/06 no bicarbonate, 2 subjects 

P20191120S = patientExam()
P20191120S.loc = horosPath(791)
P20191120S.c13 = horosPath(1240)
P20191120S.b0mag = horosPath(904)
P20191120S.b0phase = horosPath(815)
P20191120S.mask = 'masks/P2019_1120.png'
P20191120S.mfr = 'P3'
P20191120S.readout = readouts[1]

# hashoian coil 
P20191031 = patientExam()
P20191031.loc = horosPath(223)
P20191031.c13 = horosPath(341)
P20191031.b0mag = horosPath(294)
P20191031.b0phase = horosPath(305)
P20191031.mask = 'masks/P2019_1031.png'
P20191031.mfr = 'P2'
P20191031.readout = readouts[0]

# pulseteq coil
P20191031L = patientExam()
P20191031L.loc = horosPath(148)
P20191031L.c13 = horosPath(6665)
P20191031L.b0mag = horosPath(22)
P20191031L.b0phase = horosPath(81)
P20191031L.mask = 'masks/P2019_1031_pulseteq.png'
P20191031L.mfr = 'P1'
P20191031L.readout = readouts[0]


#patientList = [P20200721L, P20200721S, P20200715L, P20200715S, P20200123S, P20191120S, P20191031, P20191031L]
# just do acquisitions with pulseteq coil
patientList = [P20200721L, P20200721S, P20200715L, P20200715S, P20200123S, P20191120S, P20191031L]







