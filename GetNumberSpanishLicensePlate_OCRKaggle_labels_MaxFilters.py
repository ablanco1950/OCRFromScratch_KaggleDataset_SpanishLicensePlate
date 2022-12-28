# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################

from tensorflow import keras



# 14 HITS
modelCNN = keras.models.load_model('ModelOCRKaggle42Epoch15HITS.h5')

MinRowBlack=175

FactorColumBlack=54/58
  
dir="C:\\"

# Test images

dirname= dir + "test6Training\\images"

dirname_labels = dir +"test6Training\\labels"


######################################################################

import pytesseract

import numpy as np

import cv2

X_resize=220
Y_resize=70

import os
import re

import imutils

########################################################################
def RecortaPorArriba(img):
    Altura=int(len(img)/2)
    while (Altura >= 0):
        SwHay=0
        for i in range(len(img[0])):
            if img[Altura][i]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[Altura:,:])
        Altura=Altura - 1
    return(img)
def RecortaPorAbajo(img):
    Altura=int(len(img)/2)
    while (Altura <= len(img)-1):
        SwHay=0
        for i in range(len(img[0])):
            if img[Altura][i]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[:Altura,:])
        Altura=Altura + 1
    return(img)
def RecortaPorDerecha(img):
    Anchura=int(len(img[0])/2)
    while (Anchura < len(img[0])-1):
        SwHay=0
        for i in range(len(img)):
            if img[i][Anchura]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[:,:Anchura])
        Anchura=Anchura +1
    return(img)
def RecortaPorIzquierda(img):
    Anchura=int(len(img[0])/2)
    while (Anchura >=0):
        SwHay=0
        for i in range(len(img)):
            if img[i][Anchura]==0:
                SwHay=1
                break
        if SwHay==0:
            return(img[:,Anchura:])
        Anchura=Anchura -1
    return(img)



###################################################

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency

#####################################################################
def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco García
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
  
    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
 
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

#########################################################################
def ApplyCLAHE(gray):
#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
    
    gray_img_eqhist=cv2.equalizeHist(gray)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
    gray_img_clahe=clahe.apply(gray_img_eqhist)
    return gray_img_clahe



#########################################################################
def FindLicenseNumber (gray,x_center,y_center, width,heigh, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################

    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    
   
    x_start= x_center - width*0.5
    x_end=x_center + width*0.5
    
    y_start= y_center - heigh*0.5
    y_end=y_center + heigh*0.5
    
    X_start=int(x_start*416)
    X_end=int(x_end*416)
    
    Y_start=int(y_start*416)
    Y_end=int(y_end*416)
    
    
    
    # Clipping the boxes in two positions helps
    # in license plate reading
    X_start=X_start + x_offset   
    Y_start=Y_start + y_offset
    
    
    #print ("X_start " + str(X_start))
    #print ("X_end " + str(X_end))
    #print ("Y_start " + str(Y_start))
    #print ("Y_end " + str(Y_end))
    
    TotHits=0
         
    gray=gray[Y_start:Y_end, X_start:X_end]
    
       
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
      
        gray=imutils.rotate(gray,angle=rotation)
        
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    ##############################################################################
    # https://perez-aids.medium.com/introduction-to-image-processing-part-2-image-enhancement-1135a2198793
    # 
    from skimage import img_as_ubyte
   
    # Contrast Stretching
    from skimage.exposure import rescale_intensity
    dark_image_intensity = img_as_ubyte(gray)
    dark_image_contrast = rescale_intensity(dark_image_intensity ,
                      in_range=tuple(np.percentile(dark_image_intensity , (2, 90))))  
    text=ocr(dark_image_contrast) 
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text + "  Hit with Filter Contrast Stretching" )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with Filter Contrast Stretching "+ text) 
   ################################################
    img = cv2.GaussianBlur(gray,(3,3),0)
    thresh = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2) 
    text=ocr(thresh) 
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text + "  Hit with FILTRO KAGGLE" )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with FILTRO KAGGLE "+ text) 
    
    #################################################################
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Perform text extraction
    text=ocr(invert) 
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text + "  Hit with MORPH" )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with MORPH "+ text) 
    
    
    
    
    
    gray_img_clahe=ApplyCLAHE(gray)
    
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
   
    text=ocr(o3) 
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_Spanish_LicensePlate(text)== 1:
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with CLAHE and THRESH_TOZERO" )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text) 
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
  
    text=ocr(gray1)  
    
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
        if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
            TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)
    
    threshold=ThresholdStable(gray)
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
   
    text=ocr(gray1)  
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":    
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)    
        if text==Licenses[i]:
            print(text + "  Hit with Stable and THRESH_TRUNC" )
            TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)         
        
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
    
    text=ocr(gray1)  
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
  
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Stable and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text) 
        
     
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
    
    SumBrightness=np.sum(gray)  
    threshold=(SumBrightness/177600.00) 
    
    #####################################################
  
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO)
   
    text=ocr(gray1)  
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
   
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Brightness and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)
     
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_OTSU)
    
    text=ocr(gray1)  
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
   
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Brightness and THRESH_OTSU" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text) 
     
   
    for z in range(4,8):
    
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
              
      
       text=ocr(gray1)  
       text = ''.join(char for char in text if char.isalnum())
       text=ProcessText(text)
       if ProcessText(text) != "":
      
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text +  "  Hit with Sharpen filter"  )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected as "+ text) 
    
    gray2= cv2.bilateralFilter(gray,3, 75, 75)
    for z in range(5,11):
       kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
       gray1 = cv2.filter2D(gray2, -1, kernel)
      
       text=ocr(gray1)  
       text = ''.join(char for char in text if char.isalnum())
       text=ProcessText(text)
       if ProcessText(text) != "":
       
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text +  "  Hit with Sharpen filter modified"  )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected as "+ text) 
              
        
    return TabLicensesFounded, ContLicensesFounded
def loadlabelsRoboflow (dirname ):
 #########################################################################
 
 ########################################################################  
     lblpath = dirname + "\\"
     
     labels = []
    
     Conta=0
     print("Reading labels from ",lblpath)
     
     
     
     for root, dirnames, filenames in os.walk(lblpath):
         
                
         for filename in filenames:
             
             if re.search("\.(txt)$", filename):
                 Conta=Conta+1
                 # case test
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 if Detect_Spanish_LicensePlate(License)== -1: continue
               
                 f=open(filepath,"r")

                 ContaLin=0
                 for linea in f:
                     
                     lineadelTrain =linea.split(" ")
                     if lineadelTrain[0] == "0":
                         ContaLin=ContaLin+1
                         labels.append(linea)
                         break
                 f.close() 
                 if ContaLin==0:
                     print("Rare labels without tag 0 on " + filename )
                   
                 
     
     return labels
 ########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
     
     
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
         
         
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 
                 # Spanish license plate is NNNNAAA
                 if Detect_Spanish_LicensePlate(License)== -1: continue
                
                 image = cv2.imread(filepath)
                 
                 #from skimage import img_as_ubyte
                 #from skimage.exposure import rescale_intensity
                 #image = img_as_ubyte(image)
                 #image = rescale_intensity(image ,
                 #                  in_range=tuple(np.percentile(image , (2, 90)))) 
                 
                
                 
                 #Color Balance
                #https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05
                
                 img = image
                    
                 r, g, b = cv2.split(img)
                
                 r_avg = cv2.mean(r)[0]
                
                 g_avg = cv2.mean(g)[0]
                
                 b_avg = cv2.mean(b)[0]
                
                 
                 # Find the gain occupied by each channel
                
                 k = (r_avg + g_avg + b_avg)/3
                
                 kr = k/r_avg
                
                 kg = k/g_avg
                
                 kb = k/b_avg
                
                 
                 r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
                
                 g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
                
                 b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
                
                 
                 balance_img = cv2.merge([b, g, r])
                 
                 image=balance_img
                 
                 #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) 
                  
                 images.append(image)
                 Licenses.append(License)
                 
                 
                
                 Cont+=1
     
     return images, Licenses




def Detect_Spanish_LicensePlate(Text):
    
    if len(Text) != 7: return -1
    if (Text[0] < "0" or Text[0] > "9" ) : return -1 
    if (Text[1] < "0" or Text[2] > "9" ) : return -1   
    if (Text[2] < "0" or Text[2] > "9" ) : return -1   
    if (Text[3] < "0" or Text[3] > "9" ) : return -1     
    if (Text[4] < "A" or Text[4] > "Z" ) : return -1 
    if (Text[5] < "A" or Text[5] > "Z" ) : return -1 
    if (Text[6] < "A" or Text[6] > "Z" ) : return -1 
    return 1


def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded


def DetectCharacter(gray1Slice):
    labs={0: 0,1: 1, 2: 2, 3: 3,4: 4,5: 5,6: 6,7: 7,8: 8,9: 9,
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',20: 'K', 21: 'L',22: 'M', 23: 'N', 24: 'O', 25: 'P',26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',31: 'V', 32: 'W',33: 'X', 34: 'Y',35: 'Z',36: 'a',37: 'b',38: 'c',39: 'd',40: 'e',41: 'f',42: 'g',43: 'h',44: 'i',45: 'j',46: 'k',47: 'l',48: 'm', 49: 'n',50: 'o',51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}

    
    img=RecortaPorArriba(gray1Slice)
   
    img=RecortaPorAbajo(img)
    
    #img=RecortaPorDerecha(img)
    
    #img=RecortaPorIzquierda(img)
    
    if img.size ==0: return ""
    
    img=img/255.0
    
    img=cv2.resize(img,(28,28),cv2.INTER_CUBIC)
    
    imgTest=[]
    imgTest.append(img.flatten())
   
    
    a=np.asarray(imgTest)
    hehe=labs[np.argmax(modelCNN.predict([a.reshape(28,28).tolist()]))]
  
    return str(hehe)

def ocr(gray1):

    
       
        lvalid=[]
        
        ret2,gray1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
       
        ##############################################
        # remove the dark parts at the beginning
        for j in range(len(gray1)):
            SumRow=0
            for k in range(len(gray1[0])):
              SumRow= SumRow+  gray1[j][k]
            #print(SumColum)
            #if SumColum> 255*40:
            if SumRow > 255*MinRowBlack:
              gray1=gray1[j:,:]
              break
        
               
        ##############################################
        # remove the dark parts at the beginning
        j=len(gray1)-1
        while j > 10:
            SumRow=0
            for k in range(len(gray1[0])):
              SumRow= SumRow+  gray1[j][k]
            #print(SumColum)
            #if SumColum> 255*40:
            if SumRow > 255*MinRowBlack:
              gray1=gray1[:j,:]
              break
            j=j-1
       
        ##############################################
        for j in range(len(gray1[0])):
            #print ("PROFUNDIDAD=" + str(len(gray1)))
            Deep=len(gray1)
            SumColum=0
            for k in range(len(gray1)):
              SumColum= SumColum+  gray1[k][j]
            #print(SumColum)
            #if SumColum> 255*20:
            
            #if SumColum> 255*(Deep*55)/58:
            if SumColum> 255*Deep*FactorColumBlack:    
            #if SumColum> 255*55:
              for k in range(len(gray1)):
                gray1[k][j] =0
            else:
              lvalid.append(j)
       
        
        SwInicial=0
        
        gray1Slice=[]
        
        textCNN=""
        
        for i in range( len(lvalid)):
            
            if SwInicial==0:
               lvalidAnt= lvalid[i]
               lvalidIni=lvalid[i]
               SwInicial=1
            else:
               #print(lvalidIni)
               #print(lvalidAnt)
               if lvalid[i]!=lvalidAnt+1:
                  gray1Slice=gray1[:,lvalidIni:lvalidAnt]
                  #gray1Slice=gray1[:,47:65]
                  if gray1Slice.size==0:
                      lvalidAnt= lvalid[i]
                      lvalidIni=lvalid[i]
                     
                      text1CNN =""
                      
                  else:
                     
                      lvalidAnt= lvalid[i]
                      lvalidIni=lvalid[i]
                      
                      text1CNN= DetectCharacter(gray1Slice)
                  
                  textCNN=textCNN + text1CNN
                  
               else:
                  lvalidAnt= lvalid[i]
       
       
        print( License + " recognized as "+ textCNN )  
        #if len(text)  > 7:
        #   text=text[len(text)-7:]  
       
        TabLicensesFounded=[]
        ContLicensesFounded=[]
       
        
        textCNN=ProcessText(textCNN)
        if textCNN != "":
           TabLicensesFounded, ContLicensesFounded = ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, textCNN)
       
        ymax=-1
        contmax=0
        licensemax=""
      
        for y in range(len(TabLicensesFounded)):
            if ContLicensesFounded[y] > contmax:
                contmax=ContLicensesFounded[y]
                licensemax=TabLicensesFounded[y]
        #print("Matricula = "+ licensemax)  
        print( License + " "+ licensemax)
        return licensemax
def ProcessText(text):
    if len(text)  > 7:
       text=text[len(text)-7:] 
    if Detect_Spanish_LicensePlate(text)== -1: 
       return ""
    else:
       return text
       
###########################################################
# MAIN
##########################################################

labels=loadlabelsRoboflow(dirname_labels)

imagesComplete, Licenses=loadimagesRoboflow(dirname)



print("Number of imagenes : " + str(len(imagesComplete)))
print("Number of  labels : " + str(len(labels)))
print("Number of   licenses : " + str(len(Licenses)))

TotHits=0
TotFailures=0
with open( "LicenseResults.txt" ,"w") as  w:
    for i in range (len(imagesComplete)):
          
            gray=imagesComplete[i]
            
            License=Licenses[i]
           
            
            lineaLabel =labels[i].split(" ")
            
            # Meaning of fields in files labels
            #https://github.com/ultralytics/yolov5/issues/2293
            #
            x_center=float(lineaLabel[1])
            y_center=float(lineaLabel[2])
            width=float(lineaLabel[3])
            heigh=float(lineaLabel[4])
            
            Cont=1
            
            x_off=3
            y_off=2
            
            x_resize=220
            y_resize=70
            
            Resize_xfactor=1.78
            Resize_yfactor=1.78
            
            ContLoop=0
            
            SwFounded=0
            
            BilateralOption=0
            
            TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray,x_center,y_center, width,heigh, x_off, y_off,  License, x_resize, y_resize, \
                                   Resize_xfactor, Resize_yfactor, BilateralOption)
              
            
            print(TabLicensesFounded)
            print(ContLicensesFounded)
            
            ymax=-1
            contmax=0
            licensemax=""
          
            for y in range(len(TabLicensesFounded)):
                if ContLicensesFounded[y] > contmax:
                    contmax=ContLicensesFounded[y]
                    licensemax=TabLicensesFounded[y]
            
            if licensemax == License:
               print(License + " correctly recognized") 
               TotHits+=1
            else:
                print(License + " Detected but not correctly recognized")
                TotFailures +=1
            print ("")  
            lineaw=[]
            lineaw.append(License) 
            lineaw.append(licensemax)
            lineaWrite =','.join(lineaw)
            lineaWrite=lineaWrite + "\n"
            w.write(lineaWrite)
              
print("")           
print("Total Hits = " + str(TotHits ) + " from " + str(len(imagesComplete)) + " images readed")


      
                 
        